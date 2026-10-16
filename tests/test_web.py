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
