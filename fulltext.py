"""Full text: find a legally free-to-read PDF (Europe PMC, then Unpaywall), download it safely, and match a user's
uploaded PDF to a paper by the DOI or title on its first page. Pure functions over an injected httpx.Client.
Nothing behind a login, paywall or shadow library: only copies these two services report as free."""

import ipaddress
import logging
import re
import socket
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal
from urllib.parse import quote, urljoin

import httpx

from references import norm_doi, norm_title

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
UNPAYWALL = "https://api.unpaywall.org/v2/"
TIMEOUT = 20.0  # per connect/read
DOWNLOAD_SECONDS = 120.0  # whole download, all redirects included: a trickling server can't hold the job
MAX_REDIRECTS = 5
MATCH_SIMILAR = 0.9  # upload ↔ title (spec): a wrong link is worse than asking the user
MIN_TITLE_WORDS = 4  # shorter titles ("PCOS and depression") turn up in other papers' pages
TITLE_REGION_CHARS = 1500  # a paper's own title is near the top of page 1; further down it is a citation
FIRST_PAGE_CHARS = 4000
# A notice about another article prints that article's DOI and title: never auto-link one (the user assigns it).
NOTICE_RE = re.compile(r"\b(correction|corrigendum|erratum|retraction|retracted|expression of concern|addendum|"
                       r"comment on|reply to|response to|in reply|letter to the editor)\b", re.IGNORECASE)
NOTICE_CHARS = 300
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Found:
    url: str
    source: Literal["europepmc", "unpaywall"]
    license: str | None


def client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """Redirects are followed by download() itself, one checked hop at a time."""
    return httpx.Client(timeout=TIMEOUT, follow_redirects=False, transport=transport, headers={"User-Agent": "MaxRIM/0.1"})


def public_ip(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_global
    except ValueError:
        return False


def host_ok(host: str) -> bool:
    """Every address the host resolves to is public: PDF links come from third-party metadata and must not reach
    this server, its network or cloud metadata endpoints.
    ponytail: resolve-then-connect leaves a DNS-rebinding window; pin the resolved IP in the transport if that matters."""
    try:
        addrs = {a[4][0] for a in socket.getaddrinfo(host, None)}
    except (OSError, UnicodeError):
        return False
    return bool(addrs) and all(public_ip(a.split("%")[0]) for a in addrs)


def _get_json(http: httpx.Client, url: str, params: dict) -> dict:
    try:
        r = http.get(url, params=params)
        if r.status_code != 200:
            return {}
        data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        log.warning("full-text lookup failed: %s", type(e).__name__)
        return {}
    return data if isinstance(data, dict) else {}


def _europepmc(http: httpx.Client, paper: dict) -> list[Found]:
    if paper.get("pmid"):
        query = f"EXT_ID:{paper['pmid']} AND SRC:MED"
    elif paper.get("pmcid"):
        query = f"PMCID:{paper['pmcid']}"
    elif paper.get("doi"):
        query = f'DOI:"{paper["doi"]}"'
    else:
        return []
    data = _get_json(http, EPMC_SEARCH, {"query": query, "resultType": "core", "format": "json"})
    results = (data.get("resultList") or {}).get("result") or []
    out = []
    for res in results[:1] if isinstance(results, list) else []:
        urls = ((res.get("fullTextUrlList") or {}).get("fullTextUrl") or []) if isinstance(res, dict) else []
        for u in urls if isinstance(urls, list) else []:
            if (isinstance(u, dict) and u.get("documentStyle") == "pdf" and u.get("availabilityCode") in ("OA", "F")
                    and str(u.get("url") or "").startswith("http")):
                out.append(Found(url=u["url"], source="europepmc", license=res.get("license") or None))
    return out


def _unpaywall(http: httpx.Client, paper: dict, email: str) -> list[Found]:
    if not email or not paper.get("doi"):
        return []
    data = _get_json(http, UNPAYWALL + quote(paper["doi"], safe="/()"), {"email": email})
    locs = [data.get("best_oa_location")] + (data.get("oa_locations") or [])
    out, seen = [], set()
    for loc in locs:
        url = str(loc.get("url_for_pdf") or "") if isinstance(loc, dict) else ""
        if url.startswith("http") and url not in seen:
            seen.add(url)
            out.append(Found(url=url, source="unpaywall", license=loc.get("license") or None))
    return out


def find_pdfs(http: httpx.Client, paper: dict, unpaywall_email: str) -> list[Found]:
    """Free PDF candidates, best first: Europe PMC / PMC, then Unpaywall (only with a contact email)."""
    return _europepmc(http, paper) + _unpaywall(http, paper, unpaywall_email)


def download(http: httpx.Client, url: str, max_bytes: int) -> bytes | None:
    """The PDF at url, or None: not http(s) or not a public host (checked on every redirect hop), not 200, not
    application/pdf (a landing page), over max_bytes, over DOWNLOAD_SECONDS, or not %PDF."""
    deadline = time.monotonic() + DOWNLOAD_SECONDS
    try:
        for _ in range(MAX_REDIRECTS + 1):
            u = httpx.URL(url)
            if u.scheme not in ("http", "https") or not u.host or not host_ok(u.host):
                return None
            with http.stream("GET", u) as r:
                if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                    url = urljoin(str(u), r.headers["location"])
                    continue
                if r.status_code != 200 or "application/pdf" not in r.headers.get("content-type", "").lower():
                    return None
                if int(r.headers.get("content-length") or 0) > max_bytes:
                    return None
                data = bytearray()
                for chunk in r.iter_bytes():
                    data += chunk
                    if len(data) > max_bytes or time.monotonic() > deadline:
                        return None
                if time.monotonic() > deadline:
                    return None
                return bytes(data) if data.startswith(b"%PDF") else None
    except (httpx.HTTPError, ValueError) as e:
        log.warning("full-text download failed: %s", type(e).__name__)
    return None


def _title_on_page(title: str, region: str) -> int:
    """2: the normalised title appears whole in the page's title region; 1: as one window >= MATCH_SIMILAR; 0: no."""
    if len(title.split()) < MIN_TITLE_WORDS:
        return 0
    if f" {title} " in f" {region} ":
        return 2
    words, n = region.split(), len(title.split())
    for k in range(max(1, len(words) - n + 1)):
        m = SequenceMatcher(None, title, " ".join(words[k:k + n]))
        if m.real_quick_ratio() >= MATCH_SIMILAR and m.quick_ratio() >= MATCH_SIMILAR and m.ratio() >= MATCH_SIMILAR:
            return 1
    return 0


def match_upload(first_page_text: str, papers: list[dict]) -> str | None:
    """Paper id for an uploaded PDF, or None unless exactly one paper matches. A paper matches by its DOI on the page
    (alone, or with its title when the page names several DOIs: errata cite the original), or by its title near the
    top of the page unless the page names DOIs and none is the paper's own. A whole-title match outranks fuzzy ones
    ("Study 3" is not "Study 27")."""
    if NOTICE_RE.search(first_page_text[:NOTICE_CHARS]):
        return None
    page_dois = {norm_doi(d.rstrip(".,;)]")) for d in _DOI_RE.findall(first_page_text)}
    region = norm_title(first_page_text[:TITLE_REGION_CHARS])
    hits: dict[str, int] = {}
    for p in papers:
        doi = norm_doi(p.get("doi") or "")
        own_doi = bool(doi) and doi in page_dois
        titled = _title_on_page(norm_title(p.get("title") or ""), region)
        contradicted = bool(doi) and bool(page_dois) and not own_doi
        if own_doi and (len(page_dois) == 1 or titled):
            hits[p["id"]] = 3
        elif titled and not contradicted:
            hits[p["id"]] = titled
    best = [i for i, rank in hits.items() if rank == max(hits.values(), default=0)]
    return best[0] if len(best) == 1 else None


def first_page_text(path: str) -> str:
    """Text of page 1 (for upload matching). Runs in the ingest child: a native PDF crash must not hit the API."""
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(path)
        return pdf[0].get_textpage().get_text_range()[:FIRST_PAGE_CHARS] if len(pdf) else ""
    except Exception:  # noqa: BLE001 — matching is best effort; the PDF still ingests
        return ""
