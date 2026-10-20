"""Web search (Firecrawl or Tavily) shaped as research evidence. Results are untrusted text, quoted like PDF chunks."""

import functools
import hashlib
import logging

import httpx

TAVILY_URL = "https://api.tavily.com/search"
FIRECRAWL_URL = "https://api.firecrawl.dev/v2/search"
MAX_TEXT = 2000
log = logging.getLogger(__name__)


def _hit(project_id: str, url: str, title: str, text: str, score: float) -> dict:
    return {"point_id": "web:" + hashlib.sha256(url.encode()).hexdigest()[:32], "project_id": project_id,
            "source_type": "web", "url": url, "title": title, "source_filename": title or url,
            "page_numbers": [], "section_headings": [], "text": text[:MAX_TEXT], "score": score}


def _post(url: str, api_key: str, body: dict, timeout: float, transport) -> dict | None:
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            r = client.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=body)
            r.raise_for_status()
            data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        log.warning("web search failed: %s", type(e).__name__)  # type only: messages can echo request details
        return None
    return data if isinstance(data, dict) else None


def search(query: str, *, api_key: str, project_id: str, timeout: float, max_results: int = 5,
           transport: httpx.BaseTransport | None = None) -> list[dict]:
    """Tavily: top web results as evidence hits for this run's project. Any failure returns []: research falls back
    to papers."""
    data = _post(TAVILY_URL, api_key, {"query": query, "max_results": max_results, "search_depth": "basic"}, timeout,
                 transport) or {}
    results = data.get("results")
    hits = []
    for x in results if isinstance(results, list) else []:
        if not isinstance(x, dict):
            continue
        url, text, title = (str(x.get(k) or "").strip() for k in ("url", "content", "title"))
        if not url or not text:
            continue
        try:
            score = float(x.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        hits.append(_hit(project_id, url, title, text, score))
    return hits


def firecrawl(query: str, *, api_key: str, project_id: str, timeout: float, max_results: int = 5,
              transport: httpx.BaseTransport | None = None) -> list[dict]:
    """Firecrawl v2: the same evidence hits. Each result's main content as markdown (~1 credit per page on top of 2
    per search), else its description; ranked by position. Any failure returns []."""
    body = {"query": query, "limit": max_results, "timeout": int(timeout * 1000),
            "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True}}
    data = _post(FIRECRAWL_URL, api_key, body, timeout + 5, transport) or {}
    results = (data.get("data") or {}).get("web") if isinstance(data.get("data"), dict) else None
    hits = []
    for i, x in enumerate(results if isinstance(results, list) else []):
        if not isinstance(x, dict):
            continue
        url, title = (str(x.get(k) or "").strip() for k in ("url", "title"))
        text = str(x.get("markdown") or x.get("description") or "").strip()
        if url and text:
            hits.append(_hit(project_id, url, title, text, 1.0 / (i + 1)))
    return hits


def searcher(settings) -> functools.partial | None:
    """The configured web search, called as fn(query, project_id=...): Firecrawl when its key is set, else Tavily;
    None when web search is off or no key is set."""
    if not settings.web_search_enabled:
        return None
    fn, key = (firecrawl, settings.firecrawl_api_key) if settings.firecrawl_api_key else (search, settings.tavily_api_key)
    return functools.partial(fn, api_key=key, timeout=settings.web_search_timeout_seconds) if key else None
