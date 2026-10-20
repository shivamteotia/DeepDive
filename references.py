"""Reference exports -> normalised records: PubMed/MEDLINE (.nbib) and BibTeX (Europe PMC and others).

Stdlib only. A record that cannot be read is reported in the error list, never dropped silently."""

import csv
import html
import io
import re
from typing import Literal

from pydantic import BaseModel

Source = Literal["pubmed", "europepmc", "bibtex", "scopus", "wos", "embase", "ris", "csv"]
EUROPEPMC_PREFIXES = ("PMID", "PMCID", "PPR", "AGR", "MED", "PMC")


class Record(BaseModel):
    source: Source
    source_key: str
    title: str
    abstract: str
    authors: list[str]
    first_author: str  # lower-cased surname of the first author ("von damm")
    year: int | None
    journal: str
    doi: str  # lower-cased, no resolver prefix
    pmid: str
    pmcid: str
    language: str
    pub_types: list[str]
    keywords: list[str]
    raw: dict


_HTML_RE = re.compile(r"<[^>]+>")
_YEAR_RE = re.compile(r"\b(1[89]\d\d|20\d\d)\b")
_DOI_PREFIX_RE = re.compile(r"^(?:(?:https?://)?(?:www\.)?(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)


def strip_html(s: str) -> str:
    return " ".join(html.unescape(_HTML_RE.sub(" ", s or "")).split())


def norm_doi(s: str) -> str:
    return _DOI_PREFIX_RE.sub("", (s or "").strip()).strip().lower()


def norm_title(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", strip_html(s).lower()).split())


def _surname(author: str) -> str:
    """ "von Damm, Isabela" -> von damm; MEDLINE "Smith J" / "Tussing-Humphreys LA" -> smith; "Jane Smith" -> smith."""
    a = author.strip()
    if "," in a:
        return a.split(",", 1)[0].strip().lower()
    parts = a.split()
    initials = parts[-1].replace(".", "").replace("-", "") if parts else ""
    if len(parts) > 1 and initials.isalpha() and initials.isupper() and len(initials) <= 3:  # "Smith J", "Wu Y.", "Li J.-P."
        return " ".join(parts[:-1]).lower()
    return parts[-1].lower() if parts else ""


def _year(s: str) -> int | None:
    m = _YEAR_RE.search(s or "")
    return int(m.group(1)) if m else None


def _record(**kw) -> Record:
    authors = kw.get("authors", [])
    return Record(first_author=_surname(authors[0]) if authors else "", **kw)


# --- PubMed / MEDLINE (.nbib) ---

_MEDLINE_LINE = re.compile(r"^([A-Z]{2,4})\s*-\s?(.*)$")


def _medline_blocks(text: str) -> list[dict[str, list[str]]]:
    blocks: list[dict[str, list[str]]] = []
    tag = None
    for line in text.splitlines():
        m = _MEDLINE_LINE.match(line)
        if m:
            tag, value = m.group(1), m.group(2).strip()
            if tag == "PMID":
                blocks.append({})
            if blocks:
                blocks[-1].setdefault(tag, []).append(value)
        elif line.startswith("      ") and blocks and tag:  # continuation of the previous field
            blocks[-1][tag][-1] = (blocks[-1][tag][-1] + " " + line.strip()).strip()
        elif not line.strip():
            tag = None
    return blocks


def _ids(values: list[str], kind: str) -> str:
    return next((v.rsplit(" [", 1)[0].strip() for v in values if v.endswith(f"[{kind}]")), "")


def parse_medline(text: str) -> tuple[list[Record], list[str]]:
    recs, errs = [], []
    for n, b in enumerate(_medline_blocks(text), 1):
        pmid = (b.get("PMID") or [""])[0]
        if not pmid:
            errs.append(f"record {n}: no PMID")
            continue
        def first(t: str, b: dict = b) -> str:
            return (b.get(t) or [""])[0]

        recs.append(_record(
            source="pubmed", source_key=f"PMID:{pmid}", title=strip_html(first("TI")), abstract=strip_html(first("AB")),
            authors=b.get("FAU") or b.get("AU") or [], year=_year(first("DP")), journal=first("JT") or first("TA"),
            doi=norm_doi(_ids(b.get("AID", []), "doi") or _ids(b.get("LID", []), "doi")), pmid=pmid,
            pmcid=first("PMC"), language="|".join(b.get("LA", [])), pub_types=b.get("PT", []),
            keywords=[k.lstrip("*") for k in b.get("MH", []) + b.get("OT", [])], raw=b))
    return recs, errs


# --- BibTeX ---

_AT_RE = re.compile(r"@(\w+)\s*\{")
_ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s{}]+)\s*,")
_FIELD_RE = re.compile(r"\s*,?\s*([A-Za-z][\w-]*)\s*=\s*")
_SKIP_TYPES = {"comment", "string", "preamble"}


def _braced(text: str, i: int) -> int:
    """Index just past the brace that closes the one at text[i], or -1 if it never closes. Escaped braces don't count."""
    depth = 0
    for j in range(i, len(text)):
        if j and text[j - 1] == "\\":
            continue
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    return -1


def _quoted(text: str, i: int) -> int:
    """Index just past the quote closing the one at text[i] (quotes inside braces don't close), or -1."""
    depth = 0
    for j in range(i + 1, len(text)):
        c = text[j]
        if text[j - 1] == "\\":
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == '"' and depth <= 0:
            return j + 1
    return -1


def _clean(value: str) -> str:
    value = value.replace("\\{", "").replace("\\}", "")
    value = re.sub(r"\\([%&$#_])", r"\1", value)
    return " ".join(value.replace("{", "").replace("}", "").split())


def _fields(body: str) -> tuple[dict[str, str], str | None]:
    """Fields of one entry, and a problem message if part of it could not be read (fields before it are kept)."""
    out: dict[str, str] = {}
    i = 0
    while (m := _FIELD_RE.match(body, i)) is not None:
        name, i = m.group(1).lower(), m.end()
        if i < len(body) and body[i] == "{":
            end = _braced(body, i)
            if end == -1:
                return out, f"value of {name!r} is not closed"
            value, i = body[i + 1:end - 1], end
        elif i < len(body) and body[i] == '"':
            end = _quoted(body, i)
            if end == -1:
                return out, f"value of {name!r} is not closed"
            value, i = body[i + 1:end - 1], end
        else:
            m2 = re.match(r"[^,}\s]+", body[i:])
            value, i = (m2.group(0), i + m2.end()) if m2 else ("", i)
        out[name] = _clean(value)
        rest = body[i:].lstrip()
        if rest and not rest.startswith(","):
            return out, f"cannot read after field {name!r}"
    if body[i:].strip(" \t\r\n,"):
        return out, "unreadable text in the entry"
    return out, None


def _bib_record(entry_type: str, key: str, f: dict[str, str]) -> Record:
    prefix, _, ident = key.partition(":")
    europe = prefix.upper() in EUROPEPMC_PREFIXES and bool(ident)
    authors = [a.strip() for a in f.get("author", "").split(" and ") if a.strip()]
    return _record(
        source="europepmc" if europe else "bibtex", source_key=key, title=strip_html(f.get("title", "")),
        abstract=strip_html(f.get("abstract", "")), authors=authors, year=_year(f.get("year", "")),
        journal=f.get("journal") or f.get("publisher") or f.get("booktitle", ""), doi=norm_doi(f.get("doi", "")),
        pmid=f.get("pmid") or (ident if europe and prefix.upper() in ("PMID", "MED") else ""),
        pmcid=f.get("pmcid") or (ident if europe and prefix.upper() in ("PMCID", "PMC") else ""),
        language=f.get("language", ""), pub_types=[entry_type.lower()],
        keywords=[k.strip() for k in re.split(r"[;,]", f.get("keywords", "")) if k.strip()], raw=f)


def parse_bibtex(text: str) -> tuple[list[Record], list[str]]:
    """Every "@type{" start is either a record or an error: nothing is skipped silently. Each step moves forward."""
    recs, errs = [], []
    pos = 0
    while (at := _AT_RE.search(text, pos)) is not None:
        line = text.count("\n", 0, at.start()) + 1
        brace = at.end() - 1
        end = _braced(text, brace)
        if at.group(1).lower() in _SKIP_TYPES:
            pos = end if end != -1 else at.end()
            continue
        m = _ENTRY_RE.match(text, at.start())
        if m is None:
            errs.append(f"entry at line {line}: no citation key")
            pos = at.end()
            continue
        key = m.group(2)
        if end == -1:
            errs.append(f"entry {key} (line {line}): not closed")
            pos = m.end()
            continue
        fields, problem = _fields(text[m.end():end - 1])
        if problem:
            errs.append(f"entry {key} (line {line}): {problem}")
        recs.append(_bib_record(m.group(1), key, fields))
        pos = end
    return recs, errs


# --- RIS (Embase, Cochrane, Scopus, Web of Science, reference managers) ---

_RIS_LINE = re.compile(r"^([A-Z][A-Z0-9])  -(?: (.*))?$")
_RIS_TITLE, _RIS_ABSTRACT, _RIS_AUTHORS = ("TI", "T1", "CT", "BT"), ("AB", "N2"), ("AU", "A1")
_RIS_YEAR, _RIS_JOURNAL = ("PY", "Y1", "DA", "Y2"), ("T2", "JO", "JF", "JA", "J2", "T3")


_DATABASE_NAMES = (("embase", "embase"), ("scopus", "scopus"), ("web of science", "wos"), ("clarivate", "wos"),
                   ("pubmed", "pubmed"), ("medline", "pubmed"), ("europe pmc", "europepmc"))


def _ris_source(db: str) -> Source:
    d = db.lower()
    return next((src for key, src in _DATABASE_NAMES if key in d), "ris")


def database_code(name: str) -> str | None:
    """The stored source code for a database name as people write it ("Web of Science" -> wos), or None."""
    d = " ".join(name.lower().split())
    codes = {c for c in Source.__args__}
    return d.replace(" ", "") if d.replace(" ", "") in codes else next((c for k, c in _DATABASE_NAMES if k in d), None)


def parse_ris(text: str) -> tuple[list[Record], list[str]]:
    recs, errs = [], []
    tags: dict[str, list[str]] = {}
    last = None
    n = 0

    def first(*keys: str) -> str:
        return next((tags[k][0].strip() for k in keys if tags.get(k) and tags[k][0].strip()), "")

    def flush():
        nonlocal tags
        if not tags:
            return
        title = strip_html(" ".join(first(*_RIS_TITLE).split()))
        if not title:
            errs.append(f"record {n}: no title")
        else:
            authors = next((tags[k] for k in _RIS_AUTHORS if tags.get(k)), [])
            key = first("AN", "ID", "U1", "M1") or f"ris:{n}:{norm_title(title)[:80]}"
            source = _ris_source(first("DB", "DP"))
            recs.append(_record(
                source=source, source_key=key, title=title,
                abstract=strip_html(" ".join(first(*_RIS_ABSTRACT).split())), authors=authors,
                year=_year(first(*_RIS_YEAR)), journal=first(*_RIS_JOURNAL), doi=norm_doi(first("DO")),
                pmid=first("AN") if source == "pubmed" and first("AN").isdigit() else "", pmcid="", language=first("LA"),
                pub_types=[first("M3")] if first("M3") else [first("TY").lower()] if first("TY") else [],
                keywords=tags.get("KW", []), raw={k: v for k, v in tags.items()}))
        tags = {}

    for line in text.splitlines():
        m = _RIS_LINE.match(line.rstrip())
        if m:
            tag, value = m.group(1), (m.group(2) or "").strip()
            if tag == "TY":
                flush()
                n += 1
            if tag == "ER":
                flush()
                last = None
                continue
            tags.setdefault(tag, []).append(value)
            last = tag
        elif last and line.strip():  # wrapped value
            tags[last][-1] = f"{tags[last][-1]} {line.strip()}".strip()
    flush()
    return recs, errs


# --- CSV / tab-delimited / Excel (Scopus, Web of Science, Embase, generic) ---

_COLUMNS = {  # field -> header names (lower-case), first present wins
    "title": ("title", "article title", "document title", "ti"),
    "abstract": ("abstract", "ab"),
    "authors_full": ("author full names", "af", "author names"),
    "authors": ("authors", "author", "au"),
    "year": ("year", "publication year", "py", "publication date", "date"),
    "journal": ("source title", "journal", "journal title", "so", "publication title", "journal/book"),
    "doi": ("doi", "di"),
    "pmid": ("pubmed id", "pmid", "pm", "medline pmid"),
    "language": ("language of original document", "language", "la"),
    "type": ("document type", "publication type", "dt", "type"),
    "keywords": ("author keywords", "keywords", "de"),
    "key": ("eid", "ut", "ut (unique wos id)", "accession number", "embase accession number", "id"),
    "database": ("source", "database", "db"),
}
_COMMA_AUTHORS = re.compile(r"[^,;]+ [A-Z][A-Z.\-]*(?:, [^,;]+ [A-Z][A-Z.\-]*)+")
_SCOPUS_ID = re.compile(r"\s*\(\d+\)\s*$")


def _split(value: str, sep: str = ";") -> list[str]:
    return [x.strip() for x in value.split(sep) if x.strip()]


MAX_ROWS = 100_000
MAX_COLS = 200
MAX_EMPTY_RUN = 500  # an Excel sheet formatted down to row 1,048,576 ends here


def _table_records(rows, name: str) -> tuple[list[Record], list[str]]:
    rows = iter(rows)
    first_row = next(rows, None)
    if first_row is None:
        return [], [f"{name}: empty file"]
    header = [str(h or "").replace("\x00", "").strip().lstrip("\ufeff").lower() for h in first_row[:MAX_COLS]]
    col = {f: next((header.index(h) for h in names if h in header), None) for f, names in _COLUMNS.items()}
    if col["title"] is None:
        return [], [f"{name}: no Title column found (columns: {', '.join(h for h in header if h)[:200]})"]
    recs, errs = [], []
    empty_run = 0
    for n, row in enumerate(rows, 2):
        row = ["" if c is None else str(c).replace("\x00", "") for c in row[:MAX_COLS]]
        if not any(c.strip() for c in row):
            empty_run += 1
            if empty_run >= MAX_EMPTY_RUN:
                break
            continue
        empty_run = 0
        if n - 1 > MAX_ROWS:
            errs.append(f"{name}: only the first {MAX_ROWS:,} rows were read")
            break
        def get(field: str, row: list = row) -> str:
            i = col[field]
            return str(row[i]).strip() if i is not None and i < len(row) and row[i] is not None else ""

        title = strip_html(get("title"))
        if not title:
            errs.append(f"{name} row {n}: no title")
            continue
        key, db = get("key"), get("database").lower()
        source: Source = ("scopus" if key.startswith("2-s2.0-") or db == "scopus" else
                          "wos" if key.upper().startswith("WOS:") else "embase" if "embase" in db else "csv")
        full = [_SCOPUS_ID.sub("", a) for a in _split(get("authors_full"))]
        short = get("authors")
        if ";" not in short and _COMMA_AUTHORS.fullmatch(short):  # older Scopus: "Wu Y., Huang B."
            short = short.replace(", ", "; ")
        authors = full or _split(short)
        recs.append(_record(
            source=source, source_key=key or f"{source}:{n}:{norm_title(title)[:80]}", title=title,
            abstract=strip_html(get("abstract")), authors=authors, year=_year(get("year")), journal=get("journal"),
            doi=norm_doi(get("doi")), pmid=get("pmid") if get("pmid").isdigit() else "", pmcid="",
            language=get("language"), pub_types=_split(get("type")), keywords=_split(get("keywords")),
            raw={h: str(row[i]) for i, h in enumerate(header) if h and i < len(row) and row[i] not in (None, "")}))
    return recs, errs


csv.field_size_limit(16 * 1024 * 1024)  # Scopus exports "with references" have very long cells


def parse_csv(text: str, name: str) -> tuple[list[Record], list[str]]:
    """Delimiter = the most frequent of tab / semicolon / comma in the header line; quotes are standard CSV ("" inside
    a quoted field), which is what Scopus, WoS and Excel write."""
    head = text.split("\n", 1)[0]
    delimiter = max(("\t", ";", ","), key=head.count)
    try:
        return _table_records(csv.reader(io.StringIO(text), delimiter=delimiter), name)
    except csv.Error as e:
        return [], [f"{name}: could not read the CSV ({e})"]


def parse_xlsx(data: bytes, name: str) -> tuple[list[Record], list[str]]:
    try:
        import openpyxl  # ponytail: installed with docling, not declared in pyproject; declare it if docling goes
    except ImportError:
        return [], [f"{name}: Excel files need openpyxl; export as CSV instead"]
    try:
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        rows = wb.worksheets[0].iter_rows(values_only=True, max_col=MAX_COLS)  # lazy: no full-sheet padding in memory
        return _table_records(rows, name)
    except Exception as e:  # noqa: BLE001 — a corrupt workbook is reported, not raised
        return [], [f"{name}: could not read the Excel file ({type(e).__name__})"]


def _decode(data: bytes) -> str:
    """UTF-8 (with or without BOM), UTF-16 (some Web of Science exports), else Windows-1252 (Excel's default "CSV")."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16").lstrip("\ufeff")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def parse_file(name: str, data: bytes) -> tuple[list[Record], list[str]]:
    """Records and error messages. The format is sniffed from the content (extensions lie: PubMed exports .txt)."""
    if data[:2] == b"PK":  # a zip container: .xlsx
        return parse_xlsx(data, name)
    text = _decode(data)
    if re.search(r"^PMID\s*-", text, re.MULTILINE):
        return parse_medline(text)
    if re.search(r"^TY  -", text, re.MULTILINE):
        return parse_ris(text)
    if _ENTRY_RE.search(text):
        return parse_bibtex(text)
    first_line = text.split("\n", 1)[0]
    if any(d in first_line for d in (",", "\t", ";")):
        return parse_csv(text, name)
    return [], [f"{name}: unknown format (expected PubMed .nbib, BibTeX .bib, RIS, CSV or Excel)"]
