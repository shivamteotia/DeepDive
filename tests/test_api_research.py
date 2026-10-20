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
