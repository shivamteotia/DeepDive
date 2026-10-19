"""Reference exports -> normalised records: PubMed/MEDLINE (.nbib) and BibTeX (Europe PMC and others).

Stdlib only. A record that cannot be read is reported in the error list, never dropped silently."""

import html
import re
from typing import Literal

from pydantic import BaseModel

Source = Literal["pubmed", "europepmc", "bibtex"]
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
    if len(parts) > 1 and parts[-1].isalpha() and parts[-1].isupper() and len(parts[-1]) <= 3:
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


def parse_file(name: str, data: bytes) -> tuple[list[Record], list[str]]:
    """Records and error messages. The format is sniffed from the content (extensions lie: PubMed exports .txt)."""
    text = data.decode("utf-8-sig", errors="replace")
    if re.search(r"^PMID\s*-", text, re.MULTILINE):
        return parse_medline(text)
    if _ENTRY_RE.search(text):
        return parse_bibtex(text)
    return [], [f"{name}: unknown format (expected PubMed .nbib or BibTeX .bib)"]
