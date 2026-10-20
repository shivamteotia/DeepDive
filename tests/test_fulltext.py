import re

import httpx
import pytest

import fulltext as ft
from tests.test_api_screening import all_papers, by_pmid, draft, screen, setup

PDF = b"%PDF-1.7\n" + b"x" * 100


@pytest.fixture(autouse=True)
def no_dns(monkeypatch):
    """Tests never resolve names: IP literals get the real check, test host names count as public."""
    monkeypatch.setattr(ft, "host_ok", lambda host: ft.public_ip(host) if re.fullmatch(r"[\d.]+|[\da-f]*:[\da-f:]*", host) else True)
EPMC_PDF = "https://europepmc.org/articles/PMC1/pdf"


def epmc(url=EPMC_PDF, code="OA", style="pdf"):
    return {"resultList": {"result": [{"license": "cc by", "fullTextUrlList": {"fullTextUrl": [
        {"availabilityCode": code, "documentStyle": style, "url": url}]}}]}}


class Web:
    """httpx MockTransport: routes by host; records every request."""

    def __init__(self, epmc_json=None, unpaywall_json=None, files=None, per_pmid=False):
        self.per_pmid = per_pmid  # Europe PMC lists a distinct free PDF for every PMID
        self.epmc_json = epmc_json or {"resultList": {"result": []}}
        self.unpaywall_json = unpaywall_json
        self.files = files or {EPMC_PDF: (PDF, "application/pdf")}
        self.seen = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(str(request.url))
        url = str(request.url)
        if "ebi.ac.uk" in url:
            m = re.search(r"EXT_ID%3A(\d+)", url)
            return httpx.Response(200, json=epmc(f"https://europepmc.org/articles/{m.group(1)}/pdf")
                                  if self.per_pmid and m else self.epmc_json)
        m = re.fullmatch(r"https://europepmc.org/articles/(\d+)/pdf", url)
        if self.per_pmid and m:
            return httpx.Response(200, content=PDF + m.group(1).encode(), headers={"content-type": "application/pdf"})
        if "api.unpaywall.org" in url:
            return httpx.Response(200, json=self.unpaywall_json) if self.unpaywall_json else httpx.Response(404)
        if url in self.files:
            body, ctype = self.files[url]
            if body == "REDIRECT":
                return httpx.Response(302, headers={"location": ctype})
            return httpx.Response(200, content=body, headers={"content-type": ctype})
        return httpx.Response(404)

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self))


PAPER = {"id": "p1", "title": "Metabolic profile of women with PCOS", "doi": "10.1/abc", "pmid": "111", "pmcid": ""}


def test_europe_pmc_pdf_is_found_first():
    web = Web(epmc_json=epmc(), unpaywall_json={"best_oa_location": {"url_for_pdf": "https://x/p.pdf", "license": None}})
    found = ft.find_pdfs(web.client(), PAPER, "me@example.com")
    assert found[0] == ft.Found(url=EPMC_PDF, source="europepmc", license="cc by")
    assert "EXT_ID%3A111" in web.seen[0]


def test_unpaywall_used_when_europe_pmc_has_no_free_pdf():
    web = Web(epmc_json=epmc(code="S"), unpaywall_json={
        "best_oa_location": {"url_for_pdf": "https://pub.org/a.pdf", "license": "cc-by"},
        "oa_locations": [{"url_for_pdf": "https://pub.org/a.pdf"}, {"url_for_pdf": "https://repo.org/b.pdf", "license": None}]})
    found = ft.find_pdfs(web.client(), PAPER, "me@example.com")
    assert [(f.url, f.source, f.license) for f in found] == [("https://pub.org/a.pdf", "unpaywall", "cc-by"),
                                                             ("https://repo.org/b.pdf", "unpaywall", None)]
    assert any("email=me%40example.com" in u or "email=me@example.com" in u for u in web.seen)


def test_no_unpaywall_email_skips_unpaywall():
    web = Web(unpaywall_json={"best_oa_location": {"url_for_pdf": "https://pub.org/a.pdf"}})
    assert ft.find_pdfs(web.client(), PAPER, "") == [] and not any("unpaywall" in u for u in web.seen)


def test_source_errors_mean_not_found():
    def boom(request):
        raise httpx.ConnectError("down")
    assert ft.find_pdfs(httpx.Client(transport=httpx.MockTransport(boom)), PAPER, "me@example.com") == []
    web = Web()
    web.epmc_json = {"unexpected": True}
    assert ft.find_pdfs(web.client(), {**PAPER, "doi": ""}, "me@example.com") == []


def test_download_requires_a_real_pdf_under_the_cap():
    web = Web(files={"https://a/ok": (PDF, "application/pdf"), "https://a/landing": (b"<html>", "text/html"),
                     "https://a/fake": (b"<html>", "application/pdf"), "https://a/big": (PDF + b"x" * 1000, "application/pdf")})
    c = web.client()
    assert ft.download(c, "https://a/ok", 1000) == PDF
    assert ft.download(c, "https://a/landing", 1000) is None  # a landing page, not a PDF
    assert ft.download(c, "https://a/fake", 1000) is None
    assert ft.download(c, "https://a/big", 1000) is None
    assert ft.download(c, "https://a/missing", 1000) is None


def test_match_upload_by_doi_then_title():
    papers = [{"id": "a", "title": "Metabolic profile of women with PCOS", "doi": "10.1234/abc"},
              {"id": "b", "title": "Ovarian morphology in adolescents with irregular cycles", "doi": ""}]
    assert ft.match_upload("Journal X. https://doi.org/10.1234/ABC. Received 2020", papers) == "a"
    assert ft.match_upload("Original article\nOvarian morphology in adolescents with\nirregular cycles\nJ. Kim", papers) == "b"
    assert ft.match_upload("Something else entirely about rats", papers) is None
    twins = papers + [{"id": "c", "title": "Ovarian morphology in adolescents with irregular cycles", "doi": ""}]
    assert ft.match_upload("Ovarian morphology in adolescents with irregular cycles", twins) is None  # not confident


# --- job and endpoints ---

def fetch(api, h, pid):
    a = api.post(f"/api/v1/projects/{pid}/actions", headers=h, json={"tool": "fetch_full_text"})
    assert a.status_code == 201, a.text
    return a.json()


def test_fetch_full_text_links_open_access_pdfs_for_included_papers(api, signup, runner):
    h, pid, _ = setup(api, signup)
    draft(api, h, pid)
    screen(api, h, pid)
    web = Web(per_pmid=True)
    api.maxrim.http_transport = httpx.MockTransport(web)
    a = fetch(api, h, pid)
    assert a["estimate"]["papers"] == 10 and runner.calls == []  # nothing runs before confirmation
    r = api.post(f"/api/v1/actions/{a['id']}/confirm", headers=h)
    assert r.status_code == 200, r.text
    done = api.get(f"/api/v1/actions/{a['id']}", headers=h).json()
    assert done["status"] == "done" and done["result"]["found"] == 10, done
    got = by_pmid(all_papers(api, h, pid))
    assert got["1000"]["fulltext_status"] == "open_access_found" and got["1000"]["fulltext_license"] == "cc by"
    assert got["1000"]["fulltext_url"] == "https://europepmc.org/articles/1000/pdf" and got["1000"]["document_id"]
    assert got["1004"]["fulltext_status"] == "not_tried" and got["1004"]["document_id"] is None  # excluded
    assert len(runner.calls) == 10 and len(api.get(f"/api/v1/projects/{pid}/documents", headers=h).json()) == 10
    assert not any("ebi.ac.uk" in u and "1004" in u for u in web.seen)


def test_fetch_full_text_marks_papers_without_a_free_copy(api, signup, runner):
    h, pid, _ = setup(api, signup)
    draft(api, h, pid)
    screen(api, h, pid)
    api.maxrim.http_transport = httpx.MockTransport(Web())
    a = fetch(api, h, pid)
    api.post(f"/api/v1/actions/{a['id']}/confirm", headers=h)
    assert api.get(f"/api/v1/actions/{a['id']}", headers=h).json()["result"]["not_found"] == 10
    assert by_pmid(all_papers(api, h, pid))["1000"]["fulltext_status"] == "not_found" and runner.calls == []
    assert all_papers(api, h, pid, tab="fulltext")["total"] == 10


def test_upload_a_pdf_for_a_paper(api, signup, runner):
    h, pid, _ = setup(api, signup)
    paper = by_pmid(all_papers(api, h, pid))["1000"]
    r = api.post(f"/api/v1/projects/{pid}/papers/{paper['id']}/pdf", headers=h, files={"file": ("a.pdf", PDF, "application/pdf")})
    assert r.status_code == 202, r.text
    got = by_pmid(all_papers(api, h, pid))["1000"]
    assert got["fulltext_status"] == "uploaded" and got["document_id"] == r.json()["id"] and len(runner.calls) == 1
    other = signup("b@example.com")
    assert api.post(f"/api/v1/projects/{pid}/papers/{paper['id']}/pdf", headers=other,
                    files={"file": ("a.pdf", PDF, "application/pdf")}).status_code == 404


def test_a_plain_upload_is_linked_when_its_first_page_matches(api, signup, runner):
    h, pid, _ = setup(api, signup)
    draft(api, h, pid)
    screen(api, h, pid)  # candidates are the included papers still waiting for a PDF
    runner.result = {"chunk_count": 3, "doc_type": "research_article", "first_page": "Original article\nStudy 3 of PCOS\nA. Author"}
    r = api.post(f"/api/v1/projects/{pid}/documents", headers=h, files={"file": ("x.pdf", PDF, "application/pdf")})
    assert r.status_code == 202
    got = by_pmid(all_papers(api, h, pid))
    assert got["1003"]["document_id"] == r.json()["id"] and got["1003"]["fulltext_status"] == "uploaded"
    runner.result = {"chunk_count": 3, "doc_type": "research_article", "first_page": "Unrelated rat study"}
    api.post(f"/api/v1/projects/{pid}/documents", headers=h, files={"file": ("y.pdf", PDF + b"2", "application/pdf")})
    assert sum(p["document_id"] is not None for p in all_papers(api, h, pid)["items"]) == 1


# --- code-checker findings ---

@pytest.mark.parametrize("ip,ok", [("8.8.8.8", True), ("127.0.0.1", False), ("10.0.0.5", False), ("169.254.169.254", False),
                                   ("192.168.1.1", False), ("::1", False), ("0.0.0.0", False)])
def test_public_ip(ip, ok):
    assert ft.public_ip(ip) is ok


def test_download_refuses_private_hosts_redirects_and_other_schemes():
    web = Web(files={"https://a/redir": ("REDIRECT", "http://127.0.0.1:8000/api/v1/projects"),
                     "https://a/hop": ("REDIRECT", "https://a/ok"), "https://a/ok": (PDF, "application/pdf")})
    c = web.client()
    assert ft.download(c, "https://a/hop", 1000) == PDF  # a redirect to a public host is fine
    assert ft.download(c, "https://a/redir", 1000) is None and not any("127.0.0.1" in u for u in web.seen)
    assert ft.download(c, "http://169.254.169.254/latest", 1000) is None
    assert ft.download(c, "file:///etc/passwd", 1000) is None


def test_download_has_an_overall_deadline(monkeypatch):
    monkeypatch.setattr(ft, "DOWNLOAD_SECONDS", -1)
    assert ft.download(Web().client(), EPMC_PDF, 1000) is None


def test_match_upload_refuses_doubtful_links():
    orig = {"id": "orig", "title": "Metformin in women with polycystic ovary syndrome", "doi": "10.1000/orig"}
    other = {"id": "o", "title": "Ovarian morphology in adolescents with irregular cycles", "doi": "10.1000/o2"}
    nodoi = {"id": "n", "title": "Ovarian morphology in adolescents with irregular cycles", "doi": ""}
    erratum = "Erratum. doi:10.1000/erratum. Correction to: https://doi.org/10.1000/orig"
    assert ft.match_upload(erratum, [orig]) is None  # several DOIs on the page, the title is not there
    assert ft.match_upload(erratum + " Metformin in women with polycystic ovary syndrome", [orig]) is None  # a notice
    two = "Metformin in women with polycystic ovary syndrome. doi:10.1000/orig. See also doi:10.1000/other"
    assert ft.match_upload(two, [orig]) == "orig"  # several DOIs, but the paper's own title is on the page
    page = "doi:10.1000/somethingelse Ovarian morphology in adolescents with irregular cycles"
    assert ft.match_upload(page, [other]) is None  # the page names another DOI: not this paper
    assert ft.match_upload(page, [nodoi]) == "n"  # no DOI to contradict the title
    assert ft.match_upload("PCOS and depression", [{"id": "s", "title": "PCOS and depression", "doi": ""}]) is None
    far = "x " * 1000 + "Ovarian morphology in adolescents with irregular cycles"
    assert ft.match_upload(far, [nodoi]) is None  # a title deep in the page is a citation, not this paper's title


def test_paper_pdf_upload_never_overwrites_or_shares(api, signup, runner):
    h, pid, _ = setup(api, signup)
    got = by_pmid(all_papers(api, h, pid))

    def up(paper_id, body):
        return api.post(f"/api/v1/projects/{pid}/papers/{paper_id}/pdf", headers=h,
                        files={"file": ("a.pdf", body, "application/pdf")})

    assert up(got["1000"]["id"], PDF).status_code == 202
    assert up(got["1000"]["id"], PDF + b"other").status_code == 409  # already has a PDF
    assert up(got["1003"]["id"], PDF).status_code == 409  # this PDF is another paper's
    assert by_pmid(all_papers(api, h, pid))["1003"]["document_id"] is None


def test_one_pdf_is_never_linked_to_several_papers_by_the_fetch(api, signup, runner):
    h, pid, _ = setup(api, signup)
    draft(api, h, pid)
    screen(api, h, pid)
    api.maxrim.http_transport = httpx.MockTransport(Web(epmc_json=epmc()))  # the same "PDF" for every paper
    a = fetch(api, h, pid)
    api.post(f"/api/v1/actions/{a['id']}/confirm", headers=h)
    result = api.get(f"/api/v1/actions/{a['id']}", headers=h).json()["result"]
    assert result["found"] == 1 and result["not_found"] == 9


def test_fetch_retries_a_paper_whose_pdf_failed_to_ingest(api, signup, runner):
    h, pid, _ = setup(api, signup)
    draft(api, h, pid)
    screen(api, h, pid)
    api.maxrim.http_transport = httpx.MockTransport(Web(per_pmid=True))
    runner.result = RuntimeError("docling crashed")
    a = fetch(api, h, pid)
    api.post(f"/api/v1/actions/{a['id']}/confirm", headers=h)
    assert len(runner.calls) == 10
    runner.result = {"chunk_count": 3, "doc_type": "research_article"}
    b = fetch(api, h, pid)
    assert b["estimate"]["papers"] == 10  # failed ingests count as still needing a PDF
    api.post(f"/api/v1/actions/{b['id']}/confirm", headers=h)
    docs = api.get(f"/api/v1/projects/{pid}/documents", headers=h).json()
    assert len(runner.calls) == 20 and {d["status"] for d in docs} == {"ready"} and len(docs) == 10


def test_correction_notices_are_never_auto_linked():
    orig = {"id": "orig", "title": "Metformin in women with polycystic ovary syndrome: a retrospective cohort study",
            "doi": "10.1000/orig"}
    with_title = ("Correction\nCorrection to: Metformin in women with polycystic ovary syndrome: a retrospective cohort "
                  "study\nhttps://doi.org/10.1000/erratum\nThe original article can be found online at https://doi.org/10.1000/orig")
    assert ft.match_upload(with_title, [orig]) is None
    assert ft.match_upload("Erratum\nThe original article can be found online at https://doi.org/10.1000/orig.", [orig]) is None
    assert ft.match_upload("Metformin in women with polycystic ovary syndrome: a retrospective cohort study\n"
                           "Abstract. Background. https://doi.org/10.1000/orig", [orig]) == "orig"


def test_a_refused_upload_still_ingests_the_requeued_document(api, signup, runner):
    h, pid, _ = setup(api, signup)
    got = by_pmid(all_papers(api, h, pid))
    runner.result = RuntimeError("docling crashed")

    def up(paper_id):
        return api.post(f"/api/v1/projects/{pid}/papers/{paper_id}/pdf", headers=h, files={"file": ("a.pdf", PDF, "application/pdf")})

    up(got["1000"]["id"])  # linked to 1000; ingest failed
    runner.result = {"chunk_count": 3, "doc_type": "research_article"}
    assert up(got["1003"]["id"]).status_code == 409  # same bytes: 1000's PDF, not 1003's
    docs = api.get(f"/api/v1/projects/{pid}/documents", headers=h).json()
    assert [d["status"] for d in docs] == ["ready"] and len(runner.calls) == 2  # re-queued and ingested, not stuck
