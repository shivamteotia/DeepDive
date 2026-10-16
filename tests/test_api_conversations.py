import uuid

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

import db
import jobs
from app import create_app
from tests.conftest import FakeOpenAI, InlinePool, make_settings
from tests.test_api_research import PDF_A, project, seed_chunks, upload


@pytest.fixture
def fake_research(monkeypatch):
    seen = {"calls": []}

    def fake_run(openai_client, *, search_fn, project_id, question, max_iterations, **kw):
        hits = search_fn(question)
        seen["calls"].append({"question": question, "hits": hits})
        return {"question": question, "status": "verified", "gaps": "", "evidence": hits, "log": [],
                "claims": [{"text": f"answer to {question}", "evidence_ids": ["E1"], "verdict": "SUPPORTED",
                            "rationale": "r"}]}

    def fake_rewrite(openai_client, model, history, question):
        seen["history"] = history
        return f"standalone({question})" if history else question

    monkeypatch.setattr(jobs.research, "run", fake_run)
    monkeypatch.setattr(jobs.research, "rewrite_followup", fake_rewrite)
    return seen


def ready_project(api, h, name="A"):
    pid = project(api, h, name)
    doc = upload(api, h, pid, PDF_A)
    seed_chunks(api, [(pid, doc["sha256"], "a.pdf")])
    return pid


def ask_in(api, h, pid, question, **body):
    r = api.post(f"/api/v1/projects/{pid}/research-runs", headers=h, json={"question": question, **body})
    assert r.status_code == 202, r.text
    return api.get(f"/api/v1/research-runs/{r.json()['id']}", headers=h).json()


def test_conversation_crud_and_title_from_first_question(api, signup, fake_research):
    h = signup()
    pid = ready_project(api, h)
    c = api.post(f"/api/v1/projects/{pid}/conversations", headers=h, json={}).json()
    assert c["title"] == "New chat" and c["project_id"] == pid
    ask_in(api, h, pid, "What was the C-index of the nomogram in external validation of the SCAP model study?",
           conversation_id=c["id"])
    listed = api.get(f"/api/v1/projects/{pid}/conversations", headers=h).json()
    assert [x["id"] for x in listed] == [c["id"]] and listed[0]["title"].startswith("What was the C-index")
    assert len(listed[0]["title"]) <= 80
    assert api.delete(f"/api/v1/conversations/{c['id']}", headers=h).status_code == 204
    assert api.get(f"/api/v1/conversations/{c['id']}", headers=h).status_code == 404
    assert api.get(f"/api/v1/projects/{pid}/conversations", headers=h).json() == []


def test_followup_uses_earlier_turns_and_stores_standalone_question(api, signup, fake_research):
    h = signup()
    pid = ready_project(api, h)
    cid = api.post(f"/api/v1/projects/{pid}/conversations", headers=h, json={"title": "SCAP"}).json()["id"]
    first = ask_in(api, h, pid, "What was the nomogram C-index?", conversation_id=cid)
    assert first["standalone_question"] is None  # no history: asked as-is
    second = ask_in(api, h, pid, "and SOFA?", conversation_id=cid)
    assert fake_research["history"] == [{"question": "What was the nomogram C-index?",
                                         "answer": ["answer to What was the nomogram C-index?"]}]
    assert second["standalone_question"] == "standalone(and SOFA?)"
    assert fake_research["calls"][-1]["question"] == "standalone(and SOFA?)"  # research ran on the rewritten question
    convo = api.get(f"/api/v1/conversations/{cid}", headers=h).json()
    assert [r["question"] for r in convo["runs"]] == ["What was the nomogram C-index?", "and SOFA?"]
    assert convo["runs"][1]["result"]["claims"]


def test_runs_without_a_conversation_still_work(api, signup, fake_research):
    h = signup()
    pid = ready_project(api, h)
    run = ask_in(api, h, pid, "risks?")
    assert run["conversation_id"] is None and run["web_search"] is False and run["status"] == "completed"


def test_conversations_are_owned_and_project_scoped(api, signup, fake_research):
    alice, bob = signup("a@example.com"), signup("b@example.com")
    a1, a2 = ready_project(api, alice, "A1"), ready_project(api, alice, "A2")
    cid = api.post(f"/api/v1/projects/{a1}/conversations", headers=alice, json={}).json()["id"]
    assert api.get(f"/api/v1/conversations/{cid}", headers=bob).status_code == 404
    assert api.delete(f"/api/v1/conversations/{cid}", headers=bob).status_code == 404
    r = api.post(f"/api/v1/projects/{a2}/research-runs", headers=alice, json={"question": "why q?", "conversation_id": cid})
    assert r.status_code == 404  # a conversation from another project can't be attached
    assert api.get(f"/api/v1/projects/{a1}/conversations", headers=bob).status_code == 404


def test_web_search_is_refused_when_disabled(api, signup, fake_research):
    h = signup()
    pid = ready_project(api, h)
    r = api.post(f"/api/v1/projects/{pid}/research-runs", headers=h, json={"question": "why q?", "web_search": True})
    assert r.status_code == 400 and "web search" in r.json()["detail"].lower()


def test_web_hits_join_the_evidence_when_enabled(tmp_path, runner, fake_research, monkeypatch):
    calls = []

    def fake_web(query, *, api_key, project_id, timeout, **kw):
        calls.append((query, api_key, project_id))
        return [{"point_id": "web:1", "project_id": project_id, "source_type": "web", "url": "https://who.int/p",
                 "title": "WHO", "source_filename": "WHO", "page_numbers": [], "section_headings": [], "text": "w"}]

    monkeypatch.setattr(jobs.web, "search", fake_web)
    settings = make_settings(web_search_enabled=True, tavily_api_key="tv-secret")
    app = create_app(settings, qdrant=QdrantClient(":memory:"), openai_client=FakeOpenAI(), runner=runner,
                     upload_dir=tmp_path / "u", ingest_pool=InlinePool(), research_pool=InlinePool())
    with TestClient(app) as api:
        api.maxrim = app.state.maxrim
        api.post("/auth/register", json={"email": "w@example.com", "password": "correct horse battery"})
        tok = api.post("/auth/login", json={"email": "w@example.com", "password": "correct horse battery"}).json()
        h = {"Authorization": f"Bearer {tok['access_token']}"}
        pid = ready_project(api, h)
        run = ask_in(api, h, pid, "pneumonia mortality?", web_search=True)
        assert run["web_search"] is True
        sources = {x["source_filename"] for x in fake_research["calls"][-1]["hits"]}
        assert sources == {"a.pdf", "WHO"} and calls == [("pneumonia mortality?", "tv-secret", pid)]
        ask_in(api, h, pid, "papers only?")
        assert {x["source_filename"] for x in fake_research["calls"][-1]["hits"]} == {"a.pdf"}  # off unless asked


def test_ui_is_served_at_root(api):
    r = api.get("/")
    assert r.status_code == 200 and "<title>MaxRIM" in r.text
    assert api.get("/static/app.js").status_code == 200


def test_config_tells_the_ui_whether_web_search_is_available(api, signup):
    assert api.get("/api/v1/config").status_code == 401
    assert api.get("/api/v1/config", headers=signup()).json() == {"web_search": False}


def test_followup_waits_for_the_running_turn(api, signup, fake_research):
    h = signup()
    pid = ready_project(api, h)
    cid = api.post(f"/api/v1/projects/{pid}/conversations", headers=h, json={}).json()["id"]

    with api.maxrim.sessions() as s:  # a turn still in flight: its answer isn't history yet
        s.add(db.ResearchRun(project_id=uuid.UUID(pid), question="first?", conversation_id=uuid.UUID(cid),
                             status="running"))
        s.commit()
    r = api.post(f"/api/v1/projects/{pid}/research-runs", headers=h, json={"question": "and SOFA?", "conversation_id": cid})
    assert r.status_code == 409 and "still" in r.json()["detail"]


def test_history_includes_a_turn_with_the_same_timestamp(api, signup):
    h = signup()
    pid = uuid.UUID(ready_project(api, h))
    with api.maxrim.sessions() as s:
        c = db.Conversation(project_id=pid)
        s.add(c)
        s.flush()
        same = db.utcnow()  # coarse clocks (Windows) can stamp two quick runs identically
        first = db.ResearchRun(project_id=pid, question="q1?", conversation_id=c.id, status="completed", created_at=same,
                               result={"claims": [{"text": "a1"}]})
        second = db.ResearchRun(project_id=pid, question="q2?", conversation_id=c.id, status="queued", created_at=same)
        s.add_all([first, second])
        s.commit()
        assert jobs.conversation_history(s, second) == [{"question": "q1?", "answer": ["a1"]}]


@pytest.fixture
def web_api(tmp_path, runner, monkeypatch):
    """An API with web search enabled and a fake Tavily returning one WHO page."""
    def fake_web(query, *, api_key, project_id, timeout, **kw):
        return [{"point_id": "web:1", "project_id": project_id, "source_type": "web", "url": "https://who.int/p",
                 "title": "WHO", "source_filename": "WHO", "page_numbers": [], "section_headings": [], "text": "w"}]

    monkeypatch.setattr(jobs.web, "search", fake_web)
    settings = make_settings(web_search_enabled=True, tavily_api_key="tv-secret")
    app = create_app(settings, qdrant=QdrantClient(":memory:"), openai_client=FakeOpenAI(), runner=runner,
                     upload_dir=tmp_path / "u", ingest_pool=InlinePool(), research_pool=InlinePool())
    with TestClient(app) as api:
        api.maxrim = app.state.maxrim
        api.post("/auth/register", json={"email": "w@example.com", "password": "correct horse battery"})
        tok = api.post("/auth/login", json={"email": "w@example.com", "password": "correct horse battery"}).json()
        api.h = {"Authorization": f"Bearer {tok['access_token']}"}
        yield api


def test_web_only_chat_works_without_papers(web_api, fake_research):
    pid = project(web_api, web_api.h, "No papers")
    run = ask_in(web_api, web_api.h, pid, "pneumonia mortality?", web_search=True)
    assert run["status"] == "completed" and run["web_search"] is True
    assert {x["source_filename"] for x in fake_research["calls"][-1]["hits"]} == {"WHO"}


def test_no_papers_and_no_web_is_refused_with_a_hint(web_api, fake_research):
    pid = project(web_api, web_api.h, "No papers")
    r = web_api.post(f"/api/v1/projects/{pid}/research-runs", headers=web_api.h, json={"question": "risks?"})
    assert r.status_code == 409 and "web search" in r.json()["detail"].lower()


def test_web_only_request_is_refused_when_the_server_has_no_web_search(api, signup, fake_research):
    h = signup()
    pid = project(api, h, "No papers")
    r = api.post(f"/api/v1/projects/{pid}/research-runs", headers=h, json={"question": "risks?", "web_search": True})
    assert r.status_code == 400


def test_papers_uploaded_mid_chat_are_searched_by_later_turns(web_api, fake_research):
    h = web_api.h
    pid = project(web_api, h, "Grows")
    cid = web_api.post(f"/api/v1/projects/{pid}/conversations", headers=h, json={}).json()["id"]
    ask_in(web_api, h, pid, "pneumonia mortality?", conversation_id=cid, web_search=True)
    assert {x["source_filename"] for x in fake_research["calls"][-1]["hits"]} == {"WHO"}
    doc = upload(web_api, h, pid, PDF_A)  # FakeRunner marks it ready
    seed_chunks(web_api, [(pid, doc["sha256"], "a.pdf")])
    ask_in(web_api, h, pid, "and in the paper?", conversation_id=cid)
    assert {x["source_filename"] for x in fake_research["calls"][-1]["hits"]} == {"a.pdf"}
