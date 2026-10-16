"""Web search (Tavily) shaped as research evidence. Results are untrusted text, quoted like PDF chunks."""

import hashlib
import logging

import httpx

TAVILY_URL = "https://api.tavily.com/search"
MAX_TEXT = 2000
log = logging.getLogger(__name__)


def search(query: str, *, api_key: str, project_id: str, timeout: float, max_results: int = 5,
           transport: httpx.BaseTransport | None = None) -> list[dict]:
    """Top web results as evidence hits for this run's project. Any failure returns []: research falls back to papers."""
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            r = client.post(TAVILY_URL, headers={"Authorization": f"Bearer {api_key}"},
                            json={"query": query, "max_results": max_results, "search_depth": "basic"})
            r.raise_for_status()
            data = r.json()
    except (httpx.HTTPError, ValueError) as e:
        log.warning("web search failed: %s", type(e).__name__)  # type only: messages can echo request details
        return []
    results = data.get("results") if isinstance(data, dict) else None
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
        hits.append({"point_id": "web:" + hashlib.sha256(url.encode()).hexdigest()[:32], "project_id": project_id,
                     "source_type": "web", "url": url, "title": title, "source_filename": title or url,
                     "page_numbers": [], "section_headings": [], "text": text[:MAX_TEXT], "score": score})
    return hits
