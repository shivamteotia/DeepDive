import uuid

import pytest
from conftest import FakeOpenAI, InlinePool, make_settings
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

import ask
import ingest
import jobs
from app import create_app

PDF_A = b"%PDF-1.4 A\n%%EOF\n"
PDF_B = b"%PDF-1.4 B\n%%EOF\n"


def project(api, h, name):
    return api.post("/api/v1/projects", headers=h, json={"name": name}).json()["id"]


def upload(api, h, pid, data):
    return api.post(f"/api/v1/projects/{pid}/documents", headers=h, files={"file": ("p.pdf", data, "application/pdf")}).json()


def seed_chunks(api, rows):
    q = api.maxrim.qdrant
    if not q.collection_exists("unit"):
        q.create_collection("unit", vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
    q.upsert("unit", [
        models.PointStruct(id=str(uuid.uuid4()), vector=[1, 0, 0], payload={
            "project_id": pid, "document_sha256": sha, "source_filename": name, "page_numbers": [1],
            "section_headings": [], "text": f"text from {name}"})
        for pid, sha, name in rows
    ])


@pytest.fixture
def fake_research(monkeypatch):
    seen = {}

    def fake_run(openai_client, *, search_fn, project_id, question, max_iterations, **kw):
        if question.startswith("fail"):
            raise RuntimeError("model exploded")
        hits = search_fn(question)
        seen.update(project_id=project_id, hits=hits, max_iterations=max_iterations, question=question)
        return {"question": question, "status": "verified", "gaps": "", "evidence": hits,
                "claims": [{"text": "c", "evidence_ids": ["E1"], "verdict": "SUPPORTED", "rationale": "r"}],
                "log": [{"step": "plan", "sub_questions": [question]},
                        {"step": "verify", "iteration": 1, "SUPPORTED": 1, "UNSUPPORTED": 1, "claims": [
                            {"text": "c", "evidence_ids": ["E1"], "verdict": "SUPPORTED", "rationale": "r"},
                            {"text": "a POEM", "evidence_ids": ["E1"], "verdict": "UNSUPPORTED", "rationale": "the POEM"}]}],
                "removed_claims": ["a POEM"]}

    monkeypatch.setattr(jobs.research, "run", fake_run)
    return seen


def test_research_needs_a_ready_document(api, signup, runner):
    h = signup()
    pid = project(api, h, "A")
    assert api.post(f"/api/v1/projects/{pid}/research-runs", headers=h, json={"question": "risks?"}).status_code == 409
    runner.result = RuntimeError("bad pdf")
    upload(api, h, pid, PDF_A)
    assert api.post(f"/api/v1/projects/{pid}/research-runs", headers=h, json={"question": "risks?"}).status_code == 409


def test_run_completes_only_searches_ready_docs_of_its_project(api, signup, fake_research):
    h = signup()
    a, b = project(api, h, "A"), project(api, h, "B")
    doc_a, doc_b = upload(api, h, a, PDF_A), upload(api, h, b, PDF_B)
    seed_chunks(api, [(a, doc_a["sha256"], "a.pdf"), (b, doc_b["sha256"], "b.pdf"), (a, "still-processing", "x.pdf")])
    r = api.post(f"/api/v1/projects/{a}/research-runs", headers=h, json={"question": "  risks?  ", "max_iterations": 2})
    assert r.status_code == 202 and r.json()["status"] == "queued" and r.json()["question"] == "risks?"
    rid = r.json()["id"]
    run = api.get(f"/api/v1/research-runs/{rid}", headers=h).json()
    assert run["status"] == "completed" and run["result"]["status"] == "verified" and run["finished_at"]
    assert fake_research["project_id"] == a and fake_research["max_iterations"] == 2
    assert {x["source_filename"] for x in fake_research["hits"]} == {"a.pdf"}  # not B's, not the unready doc
    evidence = api.get(f"/api/v1/research-runs/{rid}/evidence", headers=h).json()
    assert evidence and set(evidence[0]) == {"eid", "point_id", "source_filename", "page_numbers", "section_headings", "text"}
    ver = api.get(f"/api/v1/research-runs/{rid}/verification", headers=h).json()
    assert ver["claims"][0]["verdict"] == "SUPPORTED" and [i["iteration"] for i in ver["iterations"]] == [1]
    assert [c["verdict"] for c in ver["iterations"][0]["claims"]] == ["SUPPORTED", "UNSUPPORTED"]
    for suffix in ("", "/evidence", "/verification"):  # rejected text never reaches the client; the stored audit keeps it
        assert "POEM" not in api.get(f"/api/v1/research-runs/{rid}{suffix}", headers=h).text


def test_failed_run_is_recorded(api, signup, fake_research):
    h = signup()
    pid = project(api, h, "A")
    upload(api, h, pid, PDF_A)
    rid = api.post(f"/api/v1/projects/{pid}/research-runs", headers=h, json={"question": "fail please"}).json()["id"]
    run = api.get(f"/api/v1/research-runs/{rid}", headers=h).json()
    assert run["status"] == "failed" and run["error"] == "RuntimeError: model exploded" and run["result"] is None
    assert api.get(f"/api/v1/research-runs/{rid}/evidence", headers=h).json() == []


def test_run_input_validation(api, signup):
    h = signup()
    pid = project(api, h, "A")
    for body in ({"question": "   "}, {"question": "ok?", "max_iterations": 0}, {"question": "ok?", "max_iterations": 6},
                 {"question": "x" * 2001}):
        assert api.post(f"/api/v1/projects/{pid}/research-runs", headers=h, json=body).status_code == 422


def test_runs_are_private(api, signup, fake_research):
    alice, bob = signup("alice@example.com"), signup("bob@example.com")
    pid = project(api, alice, "A")
    upload(api, alice, pid, PDF_A)
    rid = api.post(f"/api/v1/projects/{pid}/research-runs", headers=alice, json={"question": "risks?"}).json()["id"]
    for suffix in ("", "/evidence", "/verification"):
        assert api.get(f"/api/v1/research-runs/{rid}{suffix}", headers=bob).status_code == 404
    assert api.get(f"/api/v1/research-runs/{uuid.uuid4()}", headers=alice).status_code == 404
    assert api.post(f"/api/v1/projects/{pid}/research-runs", headers=bob, json={"question": "risks?"}).status_code == 404


def test_api_research_uses_hybrid_search_when_enabled(tmp_path, runner, monkeypatch, fake_research):
    q = QdrantClient(":memory:")
    app = create_app(make_settings(hybrid_search=True), qdrant=q, openai_client=FakeOpenAI(), runner=runner,
                     upload_dir=tmp_path / "uploads", ingest_pool=InlinePool(), research_pool=InlinePool())
    calls = []
    real = ask.search
    monkeypatch.setattr(ask, "search", lambda *a, **kw: calls.append(kw) or real(*a, **kw))
    with TestClient(app) as api:
        api.maxrim = app.state.maxrim
        api.post("/auth/register", json={"email": "h@example.com", "password": "correct horse battery"})
        token = api.post("/auth/login", json={"email": "h@example.com", "password": "correct horse battery"}).json()["access_token"]
        h = {"Authorization": f"Bearer {token}"}
        a, b = project(api, h, "A"), project(api, h, "B")
        doc_a, doc_b = upload(api, h, a, PDF_A), upload(api, h, b, PDF_B)
        ingest.ensure_collection(q, "unit", 3)
        q.upsert("unit", [
            models.PointStruct(id=str(uuid.uuid4()), payload={
                "project_id": pid, "document_sha256": sha, "source_filename": name, "page_numbers": [1],
                "section_headings": [], "text": "missing values were imputed"},
                vector={"": [1, 0, 0], ingest.SPARSE_VECTOR: ingest.sparse_vector("missing values were imputed")})
            for pid, sha, name in [(a, doc_a["sha256"], "a.pdf"), (b, doc_b["sha256"], "b.pdf")]
        ])
        r = api.post(f"/api/v1/projects/{a}/research-runs", headers=h, json={"question": "missing values?"})
        assert api.get(f"/api/v1/research-runs/{r.json()['id']}", headers=h).json()["status"] == "completed"
    assert calls and all(kw["hybrid"] is True for kw in calls)
    assert {x["source_filename"] for x in fake_research["hits"]} == {"a.pdf"}  # isolation holds with hybrid on


def test_a_pubmed_run_needs_no_papers_and_searches_pubmed(api, signup, fake_research, monkeypatch):
    import pubmed
    h = signup()
    pid = project(api, h, "A")  # no documents at all
    seen = []

    def fake_pubmed(query, *, api_key, project_id, transport=None):
        seen.append((query, project_id))
        return [{"point_id": "pubmed:1", "project_id": project_id, "source_type": "pubmed", "pmid": "1",
                 "url": "https://pubmed.ncbi.nlm.nih.gov/1/", "title": "T", "source_filename": "Smith 2025, J (PMID 1)",
                 "page_numbers": [], "section_headings": [], "text": "abstract", "score": 1.0}]
    monkeypatch.setattr(pubmed, "search", fake_pubmed)
    r = api.post(f"/api/v1/projects/{pid}/research-runs", headers=h, json={"question": "ML for ICU mortality?", "pubmed": True})
    assert r.status_code == 202, r.text
    run = api.get(f"/api/v1/research-runs/{r.json()['id']}", headers=h).json()
    assert run["status"] == "completed" and run["pubmed"] is True
    assert seen == [("ML for ICU mortality?", pid)]  # one search, for the topic, never a sub-question
    assert fake_research["hits"][0]["source_type"] == "pubmed"


def test_a_pubmed_run_asks_for_a_literature_summary(api, signup, fake_research, monkeypatch):
    import pubmed
    h = signup()
    pid = project(api, h, "A")
    monkeypatch.setattr(pubmed, "search", lambda *a, **k: [])
    r = api.post(f"/api/v1/projects/{pid}/research-runs", headers=h,
                 json={"question": "ML for early ICU mortality", "pubmed": True})
    run = api.get(f"/api/v1/research-runs/{r.json()['id']}", headers=h).json()
    assert fake_research["question"] == pubmed.review_question("ML for early ICU mortality")
    assert "ML for early ICU mortality" in fake_research["question"] and "found" in fake_research["question"]
    assert run["question"] == "ML for early ICU mortality"  # the user's words are what the chat shows


def test_a_pubmed_run_rejects_web_too_and_reports_pubmed_failures(api, signup, fake_research, monkeypatch):
    import pubmed
    h = signup()
    pid = project(api, h, "A")
    both = api.post(f"/api/v1/projects/{pid}/research-runs", headers=h,
                    json={"question": "ML for ICU mortality?", "pubmed": True, "web_search": True})
    assert both.status_code == 422

    def down(*a, **k):
        raise pubmed.PubMedError("PubMed search failed (HTTPStatusError)")
    monkeypatch.setattr(pubmed, "search", down)
    r = api.post(f"/api/v1/projects/{pid}/research-runs", headers=h, json={"question": "ML for ICU mortality?", "pubmed": True})
    run = api.get(f"/api/v1/research-runs/{r.json()['id']}", headers=h).json()
    assert run["status"] == "failed" and "PubMed search failed" in run["error"]
    assert fake_research.get("max_iterations") is None  # research never started on an empty, wrong evidence set


def test_a_pubmed_run_has_at_most_two_rounds(api, signup, fake_research, monkeypatch):
    import pubmed
    h = signup()
    pid = project(api, h, "A")
    monkeypatch.setattr(pubmed, "search", lambda *a, **k: [])
    api.post(f"/api/v1/projects/{pid}/research-runs", headers=h,
             json={"question": "ML for ICU mortality?", "pubmed": True, "max_iterations": 5})
    assert fake_research["max_iterations"] == 2


# --- paper resolver (papers.py): route the search to the paper(s) the question names ---
def stub_resolver(monkeypatch, route, names=None, with_cards=None):
    """papers.cards -> a Card per ready paper (or only `with_cards`), named from `names`; papers.resolve -> route.
    Returns the questions the resolver was asked."""
    import papers
    calls = []
    names = names or {}
    monkeypatch.setattr(jobs.papers, "cards", lambda q, coll, pid, shas: {
        s: papers.Card(names.get(s, s), "card") for s in shas if with_cards is None or s in with_cards})
    monkeypatch.setattr(jobs.papers, "resolve", lambda parse, question, cards: calls.append(question) or route)
    return calls


def two_paper_project(api, h):
    pid = project(api, h, "A")
    a, b = upload(api, h, pid, PDF_A), upload(api, h, pid, PDF_B)
    seed_chunks(api, [(pid, a["sha256"], "a.pdf"), (pid, b["sha256"], "b.pdf")])
    return pid, a["sha256"], b["sha256"]


def test_a_named_paper_is_the_only_one_searched(api, signup, fake_research, monkeypatch):
    import db
    import papers
    h = signup()
    pid, sha_a, _ = two_paper_project(api, h)
    calls = stub_resolver(monkeypatch, papers.Route([sha_a], []), names={sha_a: "a.pdf"})
    r = api.post(f"/api/v1/projects/{pid}/research-runs", headers=h, json={"question": "How big was study A?"})
    run = api.get(f"/api/v1/research-runs/{r.json()['id']}", headers=h).json()
    assert calls == ["How big was study A?"] and run["status"] == "completed"
    assert {x["source_filename"] for x in fake_research["hits"]} == {"a.pdf"}
    with api.maxrim.sessions() as s:  # the API hides the log; the stored audit record keeps the routing step
        stored = s.get(db.ResearchRun, uuid.UUID(run["id"])).result
    assert {"step": "resolve", "matched": ["a.pdf"], "unmatched": []} in stored["log"]


def test_a_comparison_searches_every_named_paper(api, signup, fake_research, monkeypatch):
    import papers
    h = signup()
    pid, sha_a, sha_b = two_paper_project(api, h)
    stub_resolver(monkeypatch, papers.Route([sha_a, sha_b], []))
    api.post(f"/api/v1/projects/{pid}/research-runs", headers=h, json={"question": "Compare A and B"})
    assert {x["source_filename"] for x in fake_research["hits"]} == {"a.pdf", "b.pdf"}


def test_a_paper_the_project_lacks_is_not_answered_from_another(api, signup, fake_research, monkeypatch):
    import papers
    h = signup()
    pid, _, _ = two_paper_project(api, h)
    stub_resolver(monkeypatch, papers.Route([], ["the stroke study"]))
    r = api.post(f"/api/v1/projects/{pid}/research-runs", headers=h, json={"question": "AUC of the stroke study?"})
    run = api.get(f"/api/v1/research-runs/{r.json()['id']}", headers=h).json()
    assert run["status"] == "completed" and fake_research.get("hits") is None  # research never ran
    assert run["result"]["claims"] == [] and "no paper matching: the stroke study" in run["result"]["gaps"]


def test_no_refusal_while_a_ready_paper_has_no_card(api, signup, fake_research, monkeypatch):
    # The resolver cannot see a paper without a card, so "the project has no such paper" would be a guess.
    import papers
    h = signup()
    pid, sha_a, _ = two_paper_project(api, h)
    stub_resolver(monkeypatch, papers.Route([], ["the stroke study"]), with_cards={sha_a})
    api.post(f"/api/v1/projects/{pid}/research-runs", headers=h, json={"question": "AUC of the stroke study?"})
    assert {x["source_filename"] for x in fake_research["hits"]} == {"a.pdf", "b.pdf"}  # searched everything


def test_a_refusal_still_passes_the_output_rail(api, signup, fake_research, monkeypatch):
    import papers
    h = signup()
    pid, _, _ = two_paper_project(api, h)
    stub_resolver(monkeypatch, papers.Route([], ["the stroke study"]))
    seen = []
    monkeypatch.setattr(jobs.rails, "check_output", lambda oa, m, q, answer: seen.append(answer) or None)
    api.post(f"/api/v1/projects/{pid}/research-runs", headers=h, json={"question": "AUC of the stroke study?"})
    assert seen and "no paper matching: the stroke study" in seen[0]


def test_a_resolver_failure_searches_every_paper(api, signup, fake_research, monkeypatch):
    h = signup()
    pid, _, _ = two_paper_project(api, h)

    def boom(*a, **k):
        raise RuntimeError("qdrant down")
    monkeypatch.setattr(jobs.papers, "cards", boom)
    r = api.post(f"/api/v1/projects/{pid}/research-runs", headers=h, json={"question": "risks?"})
    assert api.get(f"/api/v1/research-runs/{r.json()['id']}", headers=h).json()["status"] == "completed"
    assert {x["source_filename"] for x in fake_research["hits"]} == {"a.pdf", "b.pdf"}


def test_web_and_pubmed_runs_skip_the_resolver(api, signup, fake_research, monkeypatch):
    import pubmed
    h = signup()
    pid, _, _ = two_paper_project(api, h)
    calls = stub_resolver(monkeypatch, None)
    monkeypatch.setattr(pubmed, "search", lambda *a, **k: [])
    api.post(f"/api/v1/projects/{pid}/research-runs", headers=h, json={"question": "ML for ICU?", "pubmed": True})
    assert calls == []


def test_the_plan_is_made_while_papers_are_resolved(api, signup, monkeypatch):
    # Latency: planning (fast model) and the resolver (main model) don't depend on each other, so they overlap.
    import threading

    import papers
    h = signup()
    pid, _, _ = two_paper_project(api, h)
    both = threading.Barrier(2, timeout=5)  # each side waits for the other: passes only if they run concurrently

    def resolve(parse, question, cards):
        both.wait()
        return papers.Route(None, [])

    def plan(oa, model, question):
        both.wait()
        return ["planned sub-question"]
    seen = {}
    stub_resolver(monkeypatch, None)
    monkeypatch.setattr(jobs.papers, "resolve", resolve)
    monkeypatch.setattr(jobs.research, "plan", plan)
    monkeypatch.setattr(jobs.research, "run", lambda oa, **kw: seen.update(kw) or {
        "question": kw["question"], "status": "verified", "claims": [], "gaps": "", "evidence": [], "log": [],
        "removed_claims": []})
    r = api.post(f"/api/v1/projects/{pid}/research-runs", headers=h, json={"question": "risks?"})
    assert api.get(f"/api/v1/research-runs/{r.json()['id']}", headers=h).json()["status"] == "completed"
    assert seen["sub_questions"] == ["planned sub-question"]
