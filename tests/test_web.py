import httpx

import web


def transport(handler, seen):
    def h(request):
        seen.append(request)
        return handler(request)
    return httpx.MockTransport(h)


def test_results_become_project_scoped_web_evidence():
    seen = []
    body = {"results": [{"title": "WHO pneumonia fact sheet", "url": "https://who.int/p", "content": "x" * 3000, "score": 0.9},
                        {"title": "", "url": "https://ex.org/a", "content": "short", "score": 0.4}]}
    hits = web.search("pneumonia mortality", api_key="tv-secret", project_id="p1", timeout=5,
                      transport=transport(lambda r: httpx.Response(200, json=body), seen))
    assert [h["source_filename"] for h in hits] == ["WHO pneumonia fact sheet", "https://ex.org/a"]
    first = hits[0]
    assert first["source_type"] == "web" and first["project_id"] == "p1" and first["url"] == "https://who.int/p"
    assert first["point_id"].startswith("web:") and first["point_id"] != hits[1]["point_id"]
    assert len(first["text"]) == web.MAX_TEXT and first["page_numbers"] == [] and first["section_headings"] == []
    assert seen[0].headers["authorization"] == "Bearer tv-secret"
    assert b"pneumonia mortality" in seen[0].content


def test_failure_returns_no_hits_instead_of_raising():
    def boom(request):
        raise httpx.ReadTimeout("slow", request=request)
    assert web.search("q", api_key="k", project_id="p", timeout=1, transport=transport(boom, [])) == []
    assert web.search("q", api_key="k", project_id="p", timeout=1,
                      transport=transport(lambda r: httpx.Response(401, json={"detail": "bad key"}), [])) == []


def test_results_without_url_or_text_are_skipped():
    body = {"results": [{"title": "t", "url": "", "content": "c"}, {"title": "t", "url": "https://a", "content": ""}]}
    assert web.search("q", api_key="k", project_id="p", timeout=1,
                      transport=transport(lambda r: httpx.Response(200, json=body), [])) == []


def test_malformed_responses_fall_back_to_no_hits():
    for body in ([1, 2], {"results": "nope"}, {"results": ["x", {"url": "https://a", "content": "c", "score": "high"}]}):
        hits = web.search("q", api_key="k", project_id="p", timeout=1,
                          transport=transport(lambda r, b=body: httpx.Response(200, json=b), []))
        assert all(h["url"] == "https://a" for h in hits)  # bad items skipped, never an exception


def test_firecrawl_results_become_the_same_web_evidence():
    import json
    seen = []
    body = {"success": True, "data": {"web": [
        {"url": "https://who.int/p", "title": "WHO fact sheet", "description": "short", "markdown": "y" * 3000},
        {"url": "https://ex.org/a", "title": "", "description": "only a description"},
        {"url": "", "title": "no url", "description": "dropped"}]}}
    hits = web.firecrawl("pneumonia mortality", api_key="fc-secret", project_id="p1", timeout=5,
                         transport=transport(lambda r: httpx.Response(200, json=body), seen))
    req = json.loads(seen[0].content)
    assert seen[0].url == web.FIRECRAWL_URL and seen[0].headers["authorization"] == "Bearer fc-secret"
    assert req["query"] == "pneumonia mortality" and req["limit"] == 5
    assert req["scrapeOptions"] == {"formats": ["markdown"], "onlyMainContent": True}
    assert [(h["url"], len(h["text"])) for h in hits] == [("https://who.int/p", web.MAX_TEXT),
                                                          ("https://ex.org/a", len("only a description"))]
    assert hits[0]["source_type"] == "web" and hits[0]["project_id"] == "p1" and hits[1]["source_filename"] == "https://ex.org/a"
    for bad in (lambda r: httpx.Response(402, json={"success": False}), lambda r: httpx.Response(200, text="not json")):
        assert web.firecrawl("q", api_key="k", project_id="p", timeout=1, transport=transport(bad, [])) == []


def test_searcher_prefers_firecrawl_then_tavily_and_is_off_when_disabled():
    from tests.conftest import make_settings
    assert web.searcher(make_settings(web_search_enabled=False, firecrawl_api_key="fc")) is None
    assert web.searcher(make_settings(web_search_enabled=True)) is None
    fc = web.searcher(make_settings(web_search_enabled=True, firecrawl_api_key="fc", tavily_api_key="tv"))
    tv = web.searcher(make_settings(web_search_enabled=True, tavily_api_key="tv"))
    assert fc.func is web.firecrawl and fc.keywords["api_key"] == "fc"
    assert tv.func is web.search and tv.keywords["api_key"] == "tv"


def test_firecrawl_without_scrape_is_search_only():
    import json
    seen = []
    body = {"success": True, "data": {"web": [{"url": "https://who.int/p", "title": "WHO", "description": "The fact sheet."}]}}
    hits = web.firecrawl("q", api_key="k", project_id="p", timeout=5, scrape=False,
                         transport=transport(lambda r: httpx.Response(200, json=body), seen))
    assert "scrapeOptions" not in json.loads(seen[0].content) and hits[0]["text"] == "The fact sheet."
    assert web.search("q", api_key="k", project_id="p", timeout=1, scrape=False,  # Tavily accepts and ignores it
                      transport=transport(lambda r: httpx.Response(200, json={"results": []}), [])) == []
