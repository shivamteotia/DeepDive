import json
import threading
from types import SimpleNamespace

import screening as sc
from tests.test_api_references import project, upload

YEAR = {"id": "c1", "name": "2015 or later", "kind": "include_if", "definition": "Published 2015 or later",
        "check": "metadata", "field": "year", "value": ">=2015"}
DESIGN = {"id": "c2", "name": "Retrospective", "kind": "include_if", "definition": "Retrospective design, not an RCT",
          "check": "ai", "field": None, "value": None}
ADULTS = {"id": "c3", "name": "Adults", "kind": "include_if", "definition": "Participants are adults",
          "check": "ai", "field": None, "value": None}


def nbib(n=30):
    """n papers: every 3rd is retrospective, the rest RCTs; paper 1 is from 2010; paper 2's title says FAIL."""
    out = []
    for i in range(n):
        ab = ("A retrospective chart review of women with PCOS." if i % 3 == 0
              else "A randomized controlled trial of women with PCOS.")
        title = f"Study {i} of PCOS" + (" FAIL" if i == 2 else "")
        out.append(f"PMID- {1000 + i}\nDP  - {2010 if i == 1 else 2019}\nTI  - {title}\nAB  - {ab}\n"
                   f"FAU - Author{i}, A\nPT  - Journal Article\n")
    return "\n".join(out)


class FakeLLM:
    """Stands in for openai.responses.parse: drafts criteria; screens by the abstract; 'FAIL' titles raise."""

    def __init__(self, draft=None):
        self.draft = draft or [YEAR, DESIGN]
        self.screen_calls = []

    def parse(self, model, instructions, input, text_format):
        if text_format is sc.CriteriaDraft:
            return SimpleNamespace(output_parsed=sc.CriteriaDraft(criteria=[sc.Criterion(**c) for c in self.draft]))
        data = json.loads(input)
        self.screen_calls.append(data)
        if "FAIL" in data["title"]:
            raise RuntimeError("model down")
        retro = "retrospective" in data["abstract"]
        verdicts = []
        for c in data["criteria"]:
            if c["id"] == "c3":
                verdicts.append(sc.Verdict(criterion_id="c3", verdict="yes", quote="women", reason="r"))
            else:
                verdicts.append(sc.Verdict(criterion_id=c["id"], verdict="yes" if retro else "no",
                                           quote="retrospective chart review" if retro else "randomized controlled trial",
                                           reason="r"))
        return SimpleNamespace(output_parsed=sc.AiScreen(verdicts=verdicts))


def setup(api, signup, n=30, **kw):
    h = signup()
    pid = project(api, h)
    upload(api, h, pid, "p.nbib", nbib(n))
    llm = FakeLLM(**kw)
    api.maxrim.openai.responses = SimpleNamespace(parse=llm.parse)
    return h, pid, llm


def draft(api, h, pid, sentence="retrospective studies since 2015"):
    return api.post(f"/api/v1/projects/{pid}/criteria", headers=h, json={"sentence": sentence})


def screen(api, h, pid):
    a = api.post(f"/api/v1/projects/{pid}/actions", headers=h, json={"tool": "screen_all"})
    assert a.status_code == 201, a.text
    return api.post(f"/api/v1/actions/{a.json()['id']}/confirm", headers=h)


def by_pmid(listing):
    return {p["pmid"]: p for p in listing["items"]}


def all_papers(api, h, pid, **params):
    return api.get(f"/api/v1/projects/{pid}/papers", headers=h, params={"limit": 500, **params}).json()


def test_draft_creates_version_1_and_editing_creates_version_2(api, signup):
    h, pid, _ = setup(api, signup)
    assert api.get(f"/api/v1/projects/{pid}/criteria", headers=h).status_code == 404
    v1 = draft(api, h, pid)
    assert v1.status_code == 201 and v1.json()["version"] == 1 and [c["id"] for c in v1.json()["criteria"]] == ["c1", "c2"]
    v2 = api.put(f"/api/v1/projects/{pid}/criteria", headers=h, json={"criteria": [YEAR, DESIGN, ADULTS]})
    assert v2.status_code == 201 and v2.json()["version"] == 2
    got = api.get(f"/api/v1/projects/{pid}/criteria", headers=h).json()
    assert got["version"] == 2 and got["sentence"] == "retrospective studies since 2015" and len(got["criteria"]) == 3


def test_invalid_criteria_are_rejected(api, signup):
    h, pid, _ = setup(api, signup)
    put = lambda cs: api.put(f"/api/v1/projects/{pid}/criteria", headers=h, json={"criteria": cs})
    assert put([YEAR, YEAR]).status_code == 422  # duplicate ids
    assert put([{**YEAR, "value": "2020-2015"}]).status_code == 422  # a metadata check the code cannot run
    assert put([]).status_code == 422


def test_failed_draft_is_reported_not_saved(api, signup):
    h, pid, llm = setup(api, signup)
    llm.draft = []
    assert draft(api, h, pid).status_code == 502
    assert api.get(f"/api/v1/projects/{pid}/criteria", headers=h).status_code == 404


def test_test_run_writes_20_test_rows_and_no_real_decisions(api, signup):
    h, pid, llm = setup(api, signup)
    draft(api, h, pid)
    t = api.post(f"/api/v1/projects/{pid}/screening/test", headers=h)
    assert t.status_code == 200, t.text
    body = t.json()
    assert body["version"] == 1 and len(body["items"]) == sc.TEST_SAMPLE and body["estimate"]["calls"] == 30
    assert {i["ai_decision"] for i in body["items"]} <= {"include", "exclude", "review"}
    assert all(p["decision"] is None for p in all_papers(api, h, pid)["items"])
    calls = len(llm.screen_calls)
    again = api.post(f"/api/v1/projects/{pid}/screening/test", headers=h).json()  # same version: stored rows, no cost
    assert len(llm.screen_calls) == calls and [i["paper"]["id"] for i in again["items"]] == [i["paper"]["id"] for i in body["items"]]

    paper_id = body["items"][0]["paper"]["id"]
    fb = api.patch(f"/api/v1/projects/{pid}/screening/test/{paper_id}", headers=h, json={"feedback": "down"})
    assert fb.status_code == 200 and fb.json()["feedback"] == "down"
    stored = api.post(f"/api/v1/projects/{pid}/screening/test", headers=h).json()
    assert next(i for i in stored["items"] if i["paper"]["id"] == paper_id)["feedback"] == "down"
    other = next(p["id"] for p in all_papers(api, h, pid)["items"] if p["id"] not in {i["paper"]["id"] for i in body["items"]})
    assert api.patch(f"/api/v1/projects/{pid}/screening/test/{other}", headers=h, json={"feedback": "up"}).status_code == 404


def test_actions_need_criteria_and_start_pending(api, signup):
    h, pid, llm = setup(api, signup)
    assert api.post(f"/api/v1/projects/{pid}/actions", headers=h, json={"tool": "screen_all"}).status_code == 409
    draft(api, h, pid)
    a = api.post(f"/api/v1/projects/{pid}/actions", headers=h, json={"tool": "screen_all"}).json()
    assert a["status"] == "pending" and a["criteria_version"] == 1 and a["estimate"]["calls"] == 30
    assert llm.screen_calls == [] and all(p["decision"] is None for p in all_papers(api, h, pid)["items"])
    assert api.get(f"/api/v1/actions/{a['id']}", headers=h).json()["status"] == "pending"


def test_confirmed_screen_all_screens_every_paper(api, signup):
    h, pid, llm = setup(api, signup)
    draft(api, h, pid)
    r = screen(api, h, pid)
    assert r.status_code == 200, r.text
    assert api.get(f"/api/v1/actions/{r.json()['id']}", headers=h).json()["status"] == "done"
    got = by_pmid(all_papers(api, h, pid))
    assert got["1000"]["decision"] == "include" and got["1003"]["decision"] == "include"
    assert got["1004"]["decision"] == "exclude"  # RCT
    assert got["1001"]["decision"] == "exclude"  # 2010: metadata
    assert got["1002"]["decision"] == "review"  # model call failed: review, never exclude
    assert all(p["decided_by"] == "ai" and p["verdicts"] for p in got.values())
    assert "Study 1 of PCOS" not in {c["title"] for c in llm.screen_calls}  # excluded by metadata: never sent
    listing = all_papers(api, h, pid, tab="included")
    assert listing["total"] == 10 == len(listing["items"]) and listing["counts"]["review"] == 1  # papers 0, 3, .., 27
    assert api.post(f"/api/v1/actions/{r.json()['id']}/confirm", headers=h).status_code == 409  # only once


def test_confirm_after_criteria_changed_is_expired(api, signup):
    h, pid, llm = setup(api, signup)
    draft(api, h, pid)
    a = api.post(f"/api/v1/projects/{pid}/actions", headers=h, json={"tool": "screen_all"}).json()
    api.put(f"/api/v1/projects/{pid}/criteria", headers=h, json={"criteria": [YEAR, DESIGN, ADULTS]})
    r = api.post(f"/api/v1/actions/{a['id']}/confirm", headers=h)
    assert r.status_code == 409 and "expired" in r.json()["detail"]
    assert api.get(f"/api/v1/actions/{a['id']}", headers=h).json()["status"] == "expired" and llm.screen_calls == []


def test_another_users_action_is_404(api, signup):
    h, pid, _ = setup(api, signup)
    draft(api, h, pid)
    a = api.post(f"/api/v1/projects/{pid}/actions", headers=h, json={"tool": "screen_all"}).json()
    other = signup("b@example.com")
    assert api.post(f"/api/v1/actions/{a['id']}/confirm", headers=other).status_code == 404
    assert api.get(f"/api/v1/actions/{a['id']}", headers=other).status_code == 404
    assert api.get(f"/api/v1/projects/{pid}/criteria", headers=other).status_code == 404


def test_human_override_keeps_ai_verdicts(api, signup):
    h, pid, _ = setup(api, signup)
    draft(api, h, pid)
    screen(api, h, pid)
    p = by_pmid(all_papers(api, h, pid))["1000"]
    r = api.patch(f"/api/v1/projects/{pid}/papers/{p['id']}/decision", headers=h,
                  json={"decision": "exclude", "reason": "wrong population"})
    assert r.status_code == 200, r.text
    got = by_pmid(all_papers(api, h, pid))["1000"]
    assert got["decision"] == "exclude" and got["ai_decision"] == "include" and got["decided_by"] == "human"
    assert got["human_reason"] == "wrong population" and got["verdicts"] == p["verdicts"] and got["disagrees"]
    cleared = api.patch(f"/api/v1/projects/{pid}/papers/{p['id']}/decision", headers=h, json={"decision": None})
    assert cleared.status_code == 200 and by_pmid(all_papers(api, h, pid))["1000"]["decision"] == "include"


def test_rescreen_asks_only_changed_criteria_and_keeps_human_decisions(api, signup):
    h, pid, llm = setup(api, signup)
    draft(api, h, pid)
    screen(api, h, pid)
    p = by_pmid(all_papers(api, h, pid))["1000"]
    api.patch(f"/api/v1/projects/{pid}/papers/{p['id']}/decision", headers=h, json={"decision": "exclude", "reason": "x"})
    llm.screen_calls.clear()
    api.put(f"/api/v1/projects/{pid}/criteria", headers=h, json={"criteria": [YEAR, DESIGN, ADULTS]})
    assert screen(api, h, pid).status_code == 200
    asked = {call["title"]: [c["id"] for c in call["criteria"]] for call in llm.screen_calls}
    assert asked.pop("Study 2 of PCOS FAIL") == ["c2", "c3"]  # its earlier call failed: unclear verdicts are re-asked
    assert asked and set(map(tuple, asked.values())) == {("c3",)}  # unchanged yes/no verdicts are copied
    assert "Study 4 of PCOS" not in asked  # still excluded by a copied verdict: not paid for again
    got = by_pmid(all_papers(api, h, pid))
    assert got["1000"]["decision"] == "exclude" and got["1000"]["decided_by"] == "human" and got["1000"]["disagrees"]
    assert got["1003"]["decision"] == "include" and [v["criterion_id"] for v in got["1003"]["verdicts"]] == ["c1", "c2", "c3"]


def test_blind_mode_hides_ai_fields_and_stores_labels(api, signup):
    h, pid, _ = setup(api, signup)
    draft(api, h, pid)
    screen(api, h, pid)
    blind = all_papers(api, h, pid, blind=1)
    assert blind["items"] and not any(k in item for item in blind["items"]
                                      for k in ("verdicts", "ai_decision", "decision", "decided_by", "disagrees"))
    assert set(blind["counts"]) == {"all", "duplicates", "fulltext"}  # decision counts would leak the AI's answers
    assert api.get(f"/api/v1/projects/{pid}/papers", headers=h, params={"blind": 1, "tab": "included"}).status_code == 422
    pid0 = blind["items"][0]["id"]
    assert api.patch(f"/api/v1/projects/{pid}/papers/{pid0}/decision", headers=h,
                     json={"blind_label": "include"}).status_code == 200
    assert next(i for i in all_papers(api, h, pid, blind=1)["items"] if i["id"] == pid0)["blind_label"] == "include"
    assert api.patch(f"/api/v1/projects/{pid}/papers/{pid0}/decision", headers=h, json={}).status_code == 422


# --- code-checker findings ---

def decide_on(api, h, pid, paper_id, **body):
    return api.patch(f"/api/v1/projects/{pid}/papers/{paper_id}/decision", headers=h, json=body)


def test_papers_labelled_before_screening_still_get_ai_verdicts(api, signup):
    h, pid, _ = setup(api, signup)
    draft(api, h, pid)
    p = by_pmid(all_papers(api, h, pid))["1000"]
    decide_on(api, h, pid, p["id"], blind_label="exclude")
    q = by_pmid(all_papers(api, h, pid))["1003"]
    decide_on(api, h, pid, q["id"], decision="exclude", reason="x")
    assert screen(api, h, pid).status_code == 200
    got = by_pmid(all_papers(api, h, pid))
    assert got["1000"]["ai_decision"] == "include" and got["1000"]["blind_label"] == "exclude" and got["1000"]["verdicts"]
    assert got["1003"]["ai_decision"] == "include" and got["1003"]["decision"] == "exclude" and got["1003"]["disagrees"]


def test_human_work_survives_edits_under_a_new_version(api, signup):
    h, pid, llm = setup(api, signup)
    draft(api, h, pid)
    screen(api, h, pid)
    got = by_pmid(all_papers(api, h, pid))
    decide_on(api, h, pid, got["1000"]["id"], decision="exclude", reason="x")
    decide_on(api, h, pid, got["1003"]["id"], blind_label="include")
    api.put(f"/api/v1/projects/{pid}/criteria", headers=h, json={"criteria": [YEAR, DESIGN, ADULTS]})  # v2
    decide_on(api, h, pid, got["1000"]["id"], blind_label="exclude")  # a v2 row before v2 is screened
    api.put(f"/api/v1/projects/{pid}/criteria", headers=h, json={"criteria": [YEAR, DESIGN]})  # v3, never screened v2
    llm.screen_calls.clear()
    assert screen(api, h, pid).status_code == 200
    now = by_pmid(all_papers(api, h, pid))
    assert now["1000"]["decision"] == "exclude" and now["1000"]["human_reason"] == "x" and now["1000"]["blind_label"] == "exclude"
    assert now["1003"]["blind_label"] == "include" and now["1003"]["decision"] == "include"
    assert [c["title"] for c in llm.screen_calls] == ["Study 2 of PCOS FAIL"]  # v1 yes/no verdicts copied, not re-asked


def test_a_decision_made_during_a_run_does_not_fail_it(api, signup, monkeypatch):
    import funnel
    h, pid, llm = setup(api, signup)
    draft(api, h, pid)
    target = by_pmid(all_papers(api, h, pid))["1027"]["id"]
    real, once = llm.parse, threading.Lock()

    def parse_and_decide(*a, **kw):  # the user decides on a later paper while the first batch is being screened
        if once.acquire(blocking=False):  # exactly one request: the test DB is one SQLite connection shared by threads
            decide_on(api, h, pid, target, decision="include", reason="mid-run")
        return real(*a, **kw)

    api.maxrim.openai.responses = SimpleNamespace(parse=parse_and_decide)
    monkeypatch.setattr(funnel, "SCREEN_BATCH", 5)
    r = screen(api, h, pid)
    assert api.get(f"/api/v1/actions/{r.json()['id']}", headers=h).json()["status"] == "done"
    got = next(p for p in all_papers(api, h, pid)["items"] if p["id"] == target)
    assert got["decision"] == "include" and got["human_reason"] == "mid-run" and got["ai_decision"] == "include"


def test_blind_label_response_has_no_ai_fields(api, signup):
    h, pid, _ = setup(api, signup)
    draft(api, h, pid)
    screen(api, h, pid)
    p = all_papers(api, h, pid)["items"][0]
    out = decide_on(api, h, pid, p["id"], blind_label="include").json()
    assert out["blind_label"] == "include" and not set(out) & {"verdicts", "ai_decision", "decision", "disagrees"}


def test_queued_run_for_old_criteria_does_not_start(api, signup):
    import jobs
    h, pid, llm = setup(api, signup)
    draft(api, h, pid)
    queued = []
    api.maxrim.research_pool = SimpleNamespace(submit=lambda fn, *a: queued.append((fn, a)), shutdown=lambda **kw: None)
    a = api.post(f"/api/v1/projects/{pid}/actions", headers=h, json={"tool": "screen_all"}).json()
    api.post(f"/api/v1/actions/{a['id']}/confirm", headers=h)
    api.put(f"/api/v1/projects/{pid}/criteria", headers=h, json={"criteria": [YEAR, DESIGN, ADULTS]})
    fn, args = queued[0]
    assert fn is jobs.screen_all
    fn(*args)
    assert llm.screen_calls == [] and api.get(f"/api/v1/actions/{a['id']}", headers=h).json()["status"] == "expired"
