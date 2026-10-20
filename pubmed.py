"""PubMed (NCBI E-utilities) shaped as research evidence: the latest papers matching a query, abstracts only.

esearch (sort by publication date, newest first) -> efetch (XML) -> one evidence hit per paper that has an abstract,
labelled "First-author Year, Journal (PMID n)" and linked to its PubMed page. Abstracts are untrusted text, quoted
like PDF chunks. An NCBI key (PUBMED_API_KEY) raises the limit from 3 to 10 requests a second; without one it still works.
ponytail: stdlib ElementTree on NCBI's XML (expat's entity-expansion limits apply); defusedxml if sources widen."""

import logging
import xml.etree.ElementTree as ET

import httpx

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MAX_RESULTS = 20
MAX_TEXT = 4000  # one abstract
TIMEOUT = 20
REVIEW_PREFIX = "Literature summary on: "
log = logging.getLogger(__name__)


def _text(el) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def _year(article) -> int | None:
    date = article.find(".//Journal/JournalIssue/PubDate")
    raw = _text(date.find("Year")) if date is not None and date.find("Year") is not None else _text(date)
    digits = next((w[:4] for w in raw.split() if w[:4].isdigit()), "")
    return int(digits) if digits else None


def _hit(article, project_id: str) -> dict | None:
    pmid = _text(article.find(".//MedlineCitation/PMID"))
    parts = [(p.get("Label"), _text(p)) for p in article.findall(".//Abstract/AbstractText")]
    abstract = " ".join(f"{label}: {t}" if label else t for label, t in parts if t)
    if not pmid or not abstract:
        return None
    title = _text(article.find(".//ArticleTitle"))
    journal = _text(article.find(".//Journal/Title"))
    first = article.find(".//AuthorList/Author/LastName")
    year = _year(article)
    doi = next((_text(e) for e in article.findall(".//ELocationID") if e.get("EIdType") == "doi"), "")
    label = f"{_text(first) or 'Anonymous'} {year or 'n.d.'}, {journal or 'PubMed'} (PMID {pmid})"
    return {"point_id": f"pubmed:{pmid}", "project_id": project_id, "source_type": "pubmed", "pmid": pmid,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", "title": title, "source_filename": label,
            "year": year, "journal": journal, "doi": doi, "page_numbers": [], "section_headings": [],
            "text": f"{title}\n{abstract}"[:MAX_TEXT], "score": 1.0}


def search(query: str, *, api_key: str, project_id: str, transport: httpx.BaseTransport | None = None) -> list[dict]:
    """The MAX_RESULTS latest PubMed papers for the query, as evidence hits. Any failure returns []."""
    key = {"api_key": api_key} if api_key else {}
    query = query.split("\n")[0].removeprefix(REVIEW_PREFIX).strip()  # the topic, never the summary framing
    try:
        with httpx.Client(timeout=TIMEOUT, transport=transport) as client:
            r = client.get(f"{EUTILS}/esearch.fcgi", params={"db": "pubmed", "term": query, "retmax": MAX_RESULTS,
                                                              "sort": "pub_date", "retmode": "json", "tool": "maxrim", **key})
            r.raise_for_status()
            ids = (r.json().get("esearchresult") or {}).get("idlist") or []
            if not ids:
                return []
            r = client.get(f"{EUTILS}/efetch.fcgi", params={"db": "pubmed", "id": ",".join(ids[:MAX_RESULTS]),
                                                             "retmode": "xml", "tool": "maxrim", **key})
            r.raise_for_status()
            root = ET.fromstring(r.content)
    except (httpx.HTTPError, ValueError, ET.ParseError) as e:
        log.warning("pubmed search failed: %s", type(e).__name__)  # type only: messages can echo the key
        return []
    hits = [_hit(a, project_id) for a in root.findall("PubmedArticle")]
    return [h for h in hits if h]


def review_question(question: str) -> str:
    """What a PubMed run asks the research pipeline: a summary of the papers' findings, not one specific answer (the
    shared answer prompt stays as evaluated; only PubMed runs are framed this way)."""
    return (f"{REVIEW_PREFIX}{question}\nSummarise what these published studies found: for each study or group "
            "of similar studies, the method or model, the population or setting, and the main reported results (for "
            "example AUROC, accuracy, effect sizes), with the study's year when the evidence gives it.")
