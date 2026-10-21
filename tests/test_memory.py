"""Assistant memory: a free project briefing every turn, and project notes that last across chats."""
import json

from tests.test_agent import Script, call, chat, outputs, say
from tests.test_api_screening import draft, setup


def memory_of(request) -> dict:
    """The project memory the model was given at the start of its input."""
    first = request["input"][0]
    assert first["role"] == "user" and first["content"].startswith("Project memory")
    return json.loads(first["content"].split("\n", 1)[1])


def test_every_turn_starts_with_a_free_project_briefing(api, signup):
    h, pid, _ = setup(api, signup)  # 30 papers
    draft(api, h, pid)
    script = Script(say("hello"))
    chat(api, h, pid, script, "hi")
    m = memory_of(script.requests[0])
    assert m["funnel"]["all"] == 30 and m["criteria"]["version"] == 1
    assert [c["id"] for c in m["criteria"]["items"]] == ["c1", "c2"] and m["notes"] == [] and m["columns"] == []


def test_notes_are_remembered_across_chats_and_can_be_forgotten(api, signup):
    h, pid, _ = setup(api, signup)
    first = Script([call("remember", {"note": "Only adults; always report AUROC."})], say("Noted."))
    chat(api, h, pid, first, "remember: only adults, always report AUROC")
    saved = outputs(first.requests[1])[0]
    assert saved["note"]["text"] == "Only adults; always report AUROC."
    second = Script([call("forget", {"note_id": saved["note"]["id"]})], say("Forgotten."))
    chat(api, h, pid, second, "forget that")  # a new chat
    assert [n["text"] for n in memory_of(second.requests[0])["notes"]] == ["Only adults; always report AUROC."]
    third = Script(say("ok"))
    chat(api, h, pid, third, "hi again")
    assert memory_of(third.requests[0])["notes"] == []


def test_notes_belong_to_their_project_and_are_bounded(api, signup):
    import agent
    h, pid, _ = setup(api, signup)
    chat(api, h, pid, Script([call("remember", {"note": "Project A secret preference"})], say("ok")))
    other = api.post("/api/v1/projects", headers=h, json={"name": "B"}).json()["id"]
    script = Script(say("ok"))
    chat(api, h, other, script)
    assert memory_of(script.requests[0])["notes"] == []
    long = Script([call("remember", {"note": "x" * 1000})], say("ok"))
    chat(api, h, pid, long)
    assert len(outputs(long.requests[1])[0]["note"]["text"]) == agent.NOTE_CHARS
    bad = Script([call("forget", {"note_id": "not-a-note"})], say("ok"))
    chat(api, h, other, bad)
    assert "error" in outputs(bad.requests[1])[0]
