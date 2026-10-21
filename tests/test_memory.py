"""Assistant memory: a free project briefing every turn, and project notes that last across chats. The assistant can
only propose a note; it is kept only when the researcher presses Save (like Confirm for paid steps)."""
import json
import uuid

from sqlalchemy import select

import db
from tests.test_agent import Script, call, chat, outputs, say
from tests.test_api_screening import draft, setup


def memory_of(request) -> dict:
    """The project memory the model was given at the start of its input."""
    first = request["input"][0]
    assert first["role"] == "user" and first["content"].startswith("Project memory")
    return json.loads(first["content"].split("\n", 1)[1])


def briefed(api, h, pid) -> dict:
    script = Script(say("ok"))
    chat(api, h, pid, script)
    return memory_of(script.requests[0])


def propose(api, h, pid, note, message="remember this"):
    s = Script([call("remember", {"note": note})], say("ok"))
    chat(api, h, pid, s, message)
    return outputs(s.requests[1])[0]


def test_every_turn_starts_with_a_free_project_briefing(api, signup):
    h, pid, _ = setup(api, signup)  # 30 papers
    draft(api, h, pid)
    m = briefed(api, h, pid)
    assert m["funnel"]["all"] == 30 and m["criteria"]["version"] == 1
    assert [c["id"] for c in m["criteria"]["items"]] == ["c1", "c2"] and m["notes"] == [] and m["columns"] == []
    assert {"id", "name", "definition", "kind", "check"} <= set(m["criteria"]["items"][0])  # full, for edits


def test_a_note_is_kept_only_after_the_researcher_saves_it(api, signup):
    h, pid, _ = setup(api, signup)
    out = propose(api, h, pid, "Flag AUROC below 0.85 as weak, e.g. in tables.")
    proposal = out["note_proposal"]
    assert proposal["text"] == "Flag AUROC below 0.85 as weak, e.g. in tables." and proposal["status"] == "proposed"
    assert briefed(api, h, pid)["notes"] == []  # an injected or careless "remember" does nothing by itself
    r = api.post(f"/api/v1/notes/{proposal['id']}/save", headers=h)
    assert r.status_code == 200 and r.json()["status"] == "saved"
    assert [n["text"] for n in briefed(api, h, pid)["notes"]] == ["Flag AUROC below 0.85 as weak, e.g. in tables."]
    assert api.delete(f"/api/v1/notes/{proposal['id']}", headers=h).status_code == 204
    assert briefed(api, h, pid)["notes"] == []


def test_the_assistant_can_forget_a_saved_note(api, signup):
    h, pid, _ = setup(api, signup)
    nid = propose(api, h, pid, "only adults")["note_proposal"]["id"]
    api.post(f"/api/v1/notes/{nid}/save", headers=h)
    s = Script([call("forget", {"note_id": nid})], say("Forgotten."))
    chat(api, h, pid, s, "forget the adults note")
    assert outputs(s.requests[1])[0] == {"forgotten": nid} and briefed(api, h, pid)["notes"] == []


def test_notes_belong_to_their_project_and_their_owner(api, signup):
    h, pid, _ = setup(api, signup)
    nid = propose(api, h, pid, "Project A secret preference")["note_proposal"]["id"]
    other = signup("b@example.com")
    assert api.post(f"/api/v1/notes/{nid}/save", headers=other).status_code == 404
    assert api.delete(f"/api/v1/notes/{nid}", headers=other).status_code == 404
    api.post(f"/api/v1/notes/{nid}/save", headers=h)
    b = api.post("/api/v1/projects", headers=h, json={"name": "B"}).json()["id"]
    assert briefed(api, h, b)["notes"] == []
    s = Script([call("forget", {"note_id": nid})], say("ok"))  # project A's real note, from project B
    chat(api, h, b, s)
    assert "error" in outputs(s.requests[1])[0]
    assert [n["text"] for n in briefed(api, h, pid)["notes"]] == ["Project A secret preference"]


def test_notes_are_bounded_and_refused_not_cut(api, signup, monkeypatch):
    import agent
    h, pid, _ = setup(api, signup)
    assert "at most" in propose(api, h, pid, "word " * 200)["error"]
    monkeypatch.setattr(agent, "MAX_NOTES", 1)
    nid = propose(api, h, pid, "only adults")["note_proposal"]["id"]
    api.post(f"/api/v1/notes/{nid}/save", headers=h)
    assert "forget one first" in propose(api, h, pid, "report AUROC")["error"]


def test_a_broken_briefing_does_not_stop_the_turn(api, signup, monkeypatch):
    import agent
    h, pid, _ = setup(api, signup)
    monkeypatch.setattr(agent, "briefing", lambda *a: (_ for _ in ()).throw(RuntimeError("db down")))
    r = chat(api, h, pid, Script(say("still here")), "hi")[1].json()
    assert "Memory unavailable" in r["trace"] and r["messages"][-1]["content"]["text"] == "still here"


def test_criteria_are_never_cut_from_the_briefing_and_columns_say_when_they_are(api, signup, monkeypatch):
    import agent
    h, pid, _ = setup(api, signup)
    draft(api, h, pid)
    monkeypatch.setattr(agent, "BRIEF_COLUMNS", 0)
    r = api.post(f"/api/v1/projects/{pid}/columns", headers=h,
                 json={"name": "Size", "instruction": "Number of participants", "kind": "number", "options": None})
    assert r.status_code == 201, r.text
    m = briefed(api, h, pid)
    assert len(m["criteria"]["items"]) == 2 and m["columns"] == [] and m["columns_total"] == 1


def test_proposed_notes_are_stored_but_not_in_the_briefing(api, signup):
    h, pid, _ = setup(api, signup)
    nid = propose(api, h, pid, "only adults")["note_proposal"]["id"]
    with api.maxrim.sessions() as s:
        n = s.get(db.ProjectNote, uuid.UUID(nid))
        assert n.status == "proposed" and s.scalar(select(db.ProjectNote.id)) is not None
    assert briefed(api, h, pid)["notes"] == []
