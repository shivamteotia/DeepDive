from pathlib import Path
from types import SimpleNamespace

import pytest

import ask
import research as r


class FakeOpenAI:
    """responses.parse pops the next canned output for the requested text_format."""

    def __init__(self, outputs: dict):
        self.outputs = {k: list(v) for k, v in outputs.items()}
        self.inputs = {k: [] for k in outputs}
        self.responses = SimpleNamespace(parse=self._parse)

    def _parse(self, *, model, instructions, input, text_format):
        self.inputs.setdefault(text_format, []).append(input)
        queue = self.outputs.get(text_format, [])
        return SimpleNamespace(output_parsed=queue.pop(0) if queue else None)


def hit(pid, text="t", name="a.pdf", page=1):
    return {"eid": "E1", "point_id": pid, "project_id": "demo", "score": 1.0, "text": text, "source_filename": name,
            "page_numbers": [page], "section_headings": []}


def rounds(*vs):
    """Verifier replies for research.run: with a question every round asks twice, here with the same answer."""
    return [v for v in vs for _ in range(2)]


def claim(text, *ids):
    return ask.Claim(text=text, evidence_ids=list(ids))


def verdicts(*vs):
    return r.Verification(verdicts=[r.Verdict(answers_question=True, claim_index=i, verdict=v, rationale="why") for i, v in enumerate(vs)])


def run(oa, search_fn=None, max_iterations=3):
    search_fn = search_fn or (lambda q: [hit("p1", "privacy risk"), hit("p2", "guidance issued")])
    return r.run(oa, search_fn=search_fn, project_id="demo", model="m", fast_model="f", guard_model="g", question="risks?",
                 max_iterations=max_iterations)


def test_plan_caps_and_falls_back():
    oa = FakeOpenAI({r.Plan: [r.Plan(sub_questions=["a", " ", "b", "c", "d", "e"]), None]})
    assert r.plan(oa, "f", "q") == ["a", "b", "c", "d"]
    assert r.plan(oa, "f", "q") == ["q"]  # refusal -> original question


def test_add_evidence_dedupes_and_keeps_ids_stable():
    ev = []
    assert r.add_evidence(ev, [hit("p1"), hit("p2"), hit("p1")], "demo") == 2
    assert r.add_evidence(ev, [hit("p2"), hit("p3")], "demo") == 1
    assert [(e["eid"], e["point_id"]) for e in ev] == [("E1", "p1"), ("E2", "p2"), ("E3", "p3")]


def test_verify_fails_closed():
    ev = [{**hit("p1"), "eid": "E1"}]
    claims = [claim("a", "E1"), claim("b", "E1")]
    partial_v = r.Verification(verdicts=[r.Verdict(answers_question=True, claim_index=0, verdict="SUPPORTED", rationale="ok"),
                                         r.Verdict(answers_question=True, claim_index=7, verdict="SUPPORTED", rationale="bogus index")])
    out = r.verify(FakeOpenAI({r.Verification: [partial_v]}), "g", claims, ev)
    assert [v.verdict for v in out] == ["SUPPORTED", "UNSUPPORTED"]
    out = r.verify(FakeOpenAI({r.Verification: [None]}), "g", claims, ev)
    assert [v.verdict for v in out] == ["UNSUPPORTED", "UNSUPPORTED"]


def test_a_claim_that_the_evidence_is_silent_never_answers_the_question():
    # u02/u05/u08: "The evidence does not report X" approved by both replies; silence belongs in gaps, so it is a
    # decline, not an answer. A claim about the study itself ("was not validated in children") is still judged.
    ev = [{**hit("p1"), "eid": "E1"}]
    silent = ["The evidence does not report what it cost to implement the Criticality Index.",
              "In children, the cited evidence does not report SCAP nomogram performance.",
              "The passages do not state a hydrocortisone dose for the SCAP patients."]
    study = "The SCAP nomogram was developed and validated in adult ICU cohorts, not in children."
    ok = r.Verification(verdicts=[r.Verdict(answers_question=True, claim_index=i, verdict="SUPPORTED", rationale="ok")
                                  for i in range(4)])
    out = r.verify(FakeOpenAI({r.Verification: [ok, ok]}), "g", [claim(c, "E1") for c in [*silent, study]], ev, question="q?")
    assert [v.verdict for v in out] == ["UNSUPPORTED"] * 3 + ["SUPPORTED"]
    assert out[0].rationale.startswith("only says the evidence is silent")
    # Only "the evidence/passages/sources do not ..." as the subject: study descriptions keep their verdict
    for text in ("The cohort, drawn from three data sources, did not include children.",
                 "The study's sources of data did not contain imaging.",
                 "Patients with evidence of sepsis did not include those on dialysis.",
                 "Evidence-based guidelines do not specify a hydrocortisone dose.",
                 "The study did not include children."):
        assert not r._SILENT.search(text), text


def test_verifier_sees_each_cited_source():
    # The study-name rule needs the source: chunk text rarely names its own paper.
    oa = FakeOpenAI({r.Verification: [verdicts("SUPPORTED")]})
    r.verify(oa, "g", [claim("Li2024 AUC 0.85", "E1")], [{**hit("p1", "AUC 0.85", name="Li2024.pdf"), "eid": "E1"}])
    assert '"source": "Li2024.pdf"' in oa.inputs[r.Verification][0]


def test_verifier_sees_the_question_in_a_run():
    # A true claim about the wrong study or time window ("in-hospital" for "90-day") is only catchable with the question.
    oa = FakeOpenAI({r.Verification: [verdicts("SUPPORTED")]})
    r.verify(oa, "g", [claim("In-hospital mortality was 21%", "E1")], [{**hit("p1", "21%"), "eid": "E1"}],
             question="What was the 90-day mortality?")
    assert '"question": "What was the 90-day mortality?"' in oa.inputs[r.Verification][0]
    oa = FakeOpenAI({r.Verification: [verdicts("SUPPORTED")]})
    r.verify(oa, "g", [claim("a", "E1")], [{**hit("p1"), "eid": "E1"}])  # claim-only check (evaluate.py) unchanged
    assert '"question"' not in oa.inputs[r.Verification][0]


def test_a_claim_that_does_not_answer_the_question_is_unsupported():
    # A prose rule alone was ignored (u07: in-hospital mortality for a 90-day question was SUPPORTED): the verdict
    # carries its own answers_question field, and False rejects the claim whatever the text says.
    ev = [{**hit("p1", "in-hospital mortality 21%"), "eid": "E1"}]
    off = r.Verification(verdicts=[r.Verdict(claim_index=0, rationale="text says in-hospital",
                                             answers_question=False, verdict="SUPPORTED")])
    (v,) = r.verify(FakeOpenAI({r.Verification: [off, off]}), "g", [claim("In-hospital mortality was 21%", "E1")], ev,
                    question="What was the 90-day mortality?")
    assert v.verdict == "UNSUPPORTED" and "question" in v.rationale
    (v,) = r.verify(FakeOpenAI({r.Verification: [off]}), "g", [claim("In-hospital mortality was 21%", "E1")], ev)
    assert v.verdict == "SUPPORTED"  # no question (evaluate.py's claim check): the field is ignored


def test_with_a_question_a_claim_needs_two_agreeing_verdicts():
    # Measured: the same model passed "in-hospital 21%" for a 90-day question in one run and rejected it in another.
    # Two independent replies, the stricter wins; claim-only checks (evaluate.py) stay one call.
    ev = [{**hit("p1", "in-hospital mortality 21%"), "eid": "E1"}]
    ok = r.Verification(verdicts=[r.Verdict(claim_index=0, rationale="r", answers_question=True, verdict="SUPPORTED")])
    off = r.Verification(verdicts=[r.Verdict(claim_index=0, rationale="r", answers_question=False, verdict="SUPPORTED")])
    oa = FakeOpenAI({r.Verification: [ok, off]})
    (v,) = r.verify(oa, "g", [claim("In-hospital mortality was 21%", "E1")], ev, question="90-day mortality?")
    assert v.verdict == "UNSUPPORTED" and len(oa.inputs[r.Verification]) == 2
    oa = FakeOpenAI({r.Verification: [ok, ok]})
    (v,) = r.verify(oa, "g", [claim("In-hospital mortality was 21%", "E1")], ev, question="in-hospital mortality?")
    assert v.verdict == "SUPPORTED"
    oa = FakeOpenAI({r.Verification: [ok, off]})
    r.verify(oa, "g", [claim("x", "E1")], ev)
    assert len(oa.inputs[r.Verification]) == 1


def test_a_claim_one_reply_skipped_fails_closed():
    ev = [{**hit("p1", "t"), "eid": "E1"}]
    ok = r.Verification(verdicts=[r.Verdict(claim_index=0, rationale="r", answers_question=True, verdict="SUPPORTED")])
    no = r.Verification(verdicts=[r.Verdict(claim_index=0, rationale="the text says 30-day", answers_question=True,
                                            verdict="UNSUPPORTED")])
    (v,) = r.verify(FakeOpenAI({r.Verification: [ok, None]}), "g", [claim("x", "E1")], ev, question="q?")
    assert v.verdict == "UNSUPPORTED"  # a reply that never judged the claim can't approve it
    (v,) = r.verify(FakeOpenAI({r.Verification: [no, None]}), "g", [claim("x", "E1")], ev, question="q?")
    assert v.verdict == "UNSUPPORTED" and v.rationale == "the text says 30-day"  # the real reason survives


def test_run_passes_the_question_to_the_verifier(monkeypatch):
    seen = []
    monkeypatch.setattr(r, "verify", lambda oa, m, claims, ev, question="": seen.append(question) or
                        [r.Verdict(answers_question=True, claim_index=i, verdict="SUPPORTED", rationale="ok") for i in range(len(claims))])
    oa = FakeOpenAI({r.Plan: [r.Plan(sub_questions=["q"])],
                     ask.Answer: [ask.Answer(claims=[claim("x", "E1")], gaps="")]})
    r.run(oa, search_fn=lambda q: [hit("p1")], project_id="demo", model="m", fast_model="f", guard_model="g",
          question="What was the 90-day mortality?")
    assert seen == ["What was the 90-day mortality?"]


def test_run_can_take_sub_questions_planned_elsewhere():
    # The API plans while the paper resolver runs (they don't depend on each other); run must not plan again.
    searched = []
    oa = FakeOpenAI({r.Plan: [r.Plan(sub_questions=["never used"])],
                     ask.Answer: [ask.Answer(claims=[], gaps="nothing")]})
    out = r.run(oa, search_fn=lambda q: searched.append(q) or [hit("p1")], project_id="demo", model="m",
                fast_model="f", guard_model="g", question="q", sub_questions=["s1", "s2"])
    assert searched == ["q", "s1", "s2"] and oa.inputs[r.Plan] == []
    assert out["log"][0] == {"step": "plan", "sub_questions": ["s1", "s2"]}


def test_run_searches_the_question_itself_besides_its_sub_questions():
    # u03: sub-questions can drop what the question specifies ("in children"); the question itself is searched
    # first, and only once when the planner returns it unchanged.
    searched = []
    for subs in (["SCAP nomogram performance"], ["How did it do in children?"]):
        searched.clear()
        oa = FakeOpenAI({r.Plan: [r.Plan(sub_questions=subs)], ask.Answer: [ask.Answer(claims=[], gaps="none")]})
        r.run(oa, search_fn=lambda q: searched.append(q) or [hit("p1")], project_id="demo", model="m",
              fast_model="f", guard_model="g", question="How did it do in children?")
        assert searched == list(dict.fromkeys(["How did it do in children?", *subs]))


def test_all_supported_first_pass_is_verified():
    oa = FakeOpenAI({
        r.Plan: [r.Plan(sub_questions=["risks"])],
        ask.Answer: [ask.Answer(claims=[claim("privacy risk", "E1")], gaps="")],
        r.Verification: rounds(verdicts("SUPPORTED")),
    })
    res = run(oa)
    assert res["status"] == "verified"
    assert len(oa.inputs[ask.Answer]) == 1
    assert [c["text"] for c in res["claims"]] == ["privacy risk"]


def test_injected_unsupported_claim_is_caught_revised_and_logged():
    oa = FakeOpenAI({
        r.Plan: [r.Plan(sub_questions=["risks"])],
        ask.Answer: [
            ask.Answer(claims=[claim("privacy risk", "E1"), claim("banned by the FDA", "E2")], gaps=""),
            ask.Answer(claims=[claim("privacy risk", "E1"), claim("FDA issued guidance", "E2")], gaps=""),
        ],
        r.Verification: rounds(verdicts("SUPPORTED", "UNSUPPORTED"), verdicts("SUPPORTED", "SUPPORTED")),
    })
    res = run(oa)
    assert "banned by the FDA" in oa.inputs[ask.Answer][1]  # rejection fed back to synthesizer
    assert [c["text"] for c in res["claims"]] == ["privacy risk", "FDA issued guidance"]
    assert res["status"] == "verified"
    assert [e.get("UNSUPPORTED", 0) for e in res["log"] if e["step"] == "verify"] == [1, 0]


def test_budget_exhausted_never_auto_approves():
    searches = []

    def search_fn(q):
        searches.append(q)
        return [hit("p1", "privacy risk"), hit(f"x{len(searches)}", "more")]

    oa = FakeOpenAI({
        r.Plan: [r.Plan(sub_questions=["risks"])],
        ask.Answer: [ask.Answer(claims=[claim("big risk", "E1"), claim("made up", "E1")], gaps="")] * 2,
        r.Verification: rounds(verdicts("THINLY_SUPPORTED", "UNSUPPORTED"), verdicts("THINLY_SUPPORTED", "UNSUPPORTED")),
    })
    res = run(oa, search_fn=search_fn, max_iterations=2)
    assert "big risk" in searches  # thin claim triggered extra research
    assert res["status"] == "needs_review"
    assert [(c["text"], c["verdict"]) for c in res["claims"]] == [("big risk", "THINLY_SUPPORTED")]
    assert "made up" not in res["gaps"] and "1 unsupported claim" in res["gaps"]  # rejected text never reaches the user
    assert res["removed_claims"] == ["made up"]  # ...but stays in the audit record
    out = r.render(res)
    assert "NEEDS REVIEW" in out and "big risk (thinly supported)" in out


def test_no_evidence_short_circuits():
    oa = FakeOpenAI({r.Plan: [r.Plan(sub_questions=["q"])]})
    res = run(oa, search_fn=lambda q: [])
    assert res["status"] == "needs_review" and "No sources found" in res["gaps"]
    assert ask.Answer not in oa.inputs or not oa.inputs[ask.Answer]


def test_rejects_bad_max_iterations():
    with pytest.raises(ValueError):
        run(FakeOpenAI({}), max_iterations=0)


def test_save_writes_audit_record(tmp_path: Path):
    res = r.finalize("q", ask.Answer(claims=[], gaps="g"), [], [{**hit("p1"), "eid": "E1"}], [{"step": "plan", "sub_questions": ["q"]}])
    path = r.save(res, tmp_path)
    data = r.json.loads(path.read_text(encoding="utf-8"))
    assert data["question"] == "q" and data["evidence"][0]["point_id"] == "p1" and "score" not in data["evidence"][0]


@pytest.mark.live
def test_live_verifier_rejects_false_claim():
    """README exit criterion: a deliberately false claim against real evidence must not pass."""
    from openai import OpenAI
    from qdrant_client import QdrantClient

    from config import Settings

    s = Settings()
    oa = OpenAI(api_key=s.openai_api_key)
    q = QdrantClient(url=s.qdrant_cluster_endpoint, api_key=s.qdrant_api_key)
    ev = ask.search(q, oa, collection=s.qdrant_collection, embed_model=s.openai_embed_model, project_id="demo",
                    question="study limitations", k=3)
    if not ev:
        pytest.skip("project 'demo' has no indexed documents")
    false = claim("This study proves the model is approved by the FDA for clinical use in all hospitals.", "E1")
    (v,) = r.verify(oa, s.openai_guard_model, [false], ev)
    assert v.verdict == "UNSUPPORTED", v.rationale


def test_duplicate_verdicts_most_severe_wins():
    ev = [{**hit("p1"), "eid": "E1"}]
    dup = r.Verification(verdicts=[r.Verdict(answers_question=True, claim_index=0, verdict="UNSUPPORTED", rationale="no"),
                                   r.Verdict(answers_question=True, claim_index=0, verdict="SUPPORTED", rationale="yes")])
    (v,) = r.verify(FakeOpenAI({r.Verification: [dup]}), "g", [claim("a", "E1")], ev)
    assert v.verdict == "UNSUPPORTED"


def test_cross_project_hit_is_rejected():
    with pytest.raises(RuntimeError):
        r.add_evidence([], [{**hit("p1"), "project_id": "other"}], "demo")


def test_empty_answer_mid_loop_keeps_earlier_claims():
    oa = FakeOpenAI({
        r.Plan: [r.Plan(sub_questions=["risks"])],
        ask.Answer: [ask.Answer(claims=[claim("privacy risk", "E1"), claim("made up", "E1")], gaps=""), None, None],
        r.Verification: rounds(verdicts("SUPPORTED", "UNSUPPORTED")),
    })
    res = run(oa)
    assert len(oa.inputs[ask.Answer]) == 3  # NO_ANSWER retried while budget remained
    assert [c["text"] for c in res["claims"]] == ["privacy risk"]
    assert res["status"] == "needs_review" and "made up" not in res["gaps"]
    assert res["removed_claims"] == ["made up"]


def test_audit_log_and_feedback_carry_rejected_claim_details():
    oa = FakeOpenAI({
        r.Plan: [r.Plan(sub_questions=["risks"])],
        ask.Answer: [ask.Answer(claims=[claim('banned "by" FDA', "E2")], gaps=""),
                     ask.Answer(claims=[claim("other", "E2")], gaps=""),
                     ask.Answer(claims=[claim("guidance", "E2")], gaps="")],
        r.Verification: rounds(verdicts("UNSUPPORTED"), verdicts("UNSUPPORTED"), verdicts("SUPPORTED")),
    })
    res = run(oa)
    first = next(e for e in res["log"] if e["step"] == "verify")
    assert first["claims"][0] == {"text": 'banned "by" FDA', "evidence_ids": ["E2"], "verdict": "UNSUPPORTED", "rationale": "why"}
    third_prompt = oa.inputs[ask.Answer][2]
    assert r.json.dumps('banned "by" FDA') in third_prompt and '"other"' in third_prompt  # JSON-encoded, cumulative


def test_invented_citation_forces_review():
    ans = ask.Answer(claims=[claim("ok", "E1")], gaps=ask.UNCITED_PREFIX + "fake")
    res = r.finalize("q", ans, [r.Verdict(answers_question=True, claim_index=0, verdict="SUPPORTED", rationale="")], [], [])
    assert res["status"] == "needs_review"


def test_keeps_round_with_most_supported_claims():
    oa = FakeOpenAI({
        r.Plan: [r.Plan(sub_questions=["risks"])],
        ask.Answer: [ask.Answer(claims=[claim("a", "E1"), claim("b", "E1"), claim("x", "E1")], gaps=""),
                     ask.Answer(claims=[claim("c", "E1"), claim("y", "E1")], gaps="")],
        r.Verification: rounds(verdicts("SUPPORTED", "SUPPORTED", "UNSUPPORTED"), verdicts("UNSUPPORTED", "UNSUPPORTED")),
    })
    res = run(oa, max_iterations=2)
    assert [c["text"] for c in res["claims"]] == ["a", "b"]


def test_all_invented_citations_are_retried():
    oa = FakeOpenAI({
        r.Plan: [r.Plan(sub_questions=["risks"])],
        ask.Answer: [ask.Answer(claims=[claim("fake", "E99")], gaps=""), ask.Answer(claims=[claim("ok", "E1")], gaps="")],
        r.Verification: rounds(verdicts("SUPPORTED")),
    })
    res = run(oa)
    assert len(oa.inputs[ask.Answer]) == 2 and [c["text"] for c in res["claims"]] == ["ok"]
    assert ["fake"] in [e.get("uncited") for e in res["log"]]  # hidden from the user, kept for the auditor
    assert "fake" not in res["gaps"]


def test_followup_without_history_is_used_as_is():
    oa = FakeOpenAI({})
    assert r.rewrite_followup(oa, "f", [], "what about SOFA?") == "what about SOFA?"
    assert not oa.inputs  # no LLM call


def test_followup_is_rewritten_from_quoted_history():
    oa = FakeOpenAI({r.Standalone: [r.Standalone(question="How did SOFA perform in the SCAP validation cohort?")]})
    history = [{"question": "What was the nomogram C-index?", "answer": ["C-index 0.903 in validation"]}]
    out = r.rewrite_followup(oa, "f", history, "and SOFA?")
    assert out == "How did SOFA perform in the SCAP validation cohort?"
    sent = oa.inputs[r.Standalone][0]
    assert '"C-index 0.903 in validation"' in sent and "and SOFA?" in sent  # history goes in as JSON data


def test_followup_rewrite_failure_falls_back_to_the_question():
    oa = FakeOpenAI({r.Standalone: [None]})
    assert r.rewrite_followup(oa, "f", [{"question": "q", "answer": []}], "and SOFA?") == "and SOFA?"


def test_audit_record_keeps_web_source_fields_only_for_web_evidence():
    pdf = {**hit("p1"), "eid": "E1"}
    page = {**hit("web:1", name="WHO"), "eid": "E2", "source_type": "web", "url": "https://who.int/p", "title": "WHO"}
    out = r.audit_record({"evidence": [pdf, page], "claims": []})["evidence"]
    assert "url" not in out[0] and out[1]["url"] == "https://who.int/p" and out[1]["source_type"] == "web"
    assert "score" not in out[1]


def test_verifier_schema_has_no_defaults():
    import json

    from openai.lib._pydantic import to_strict_json_schema
    assert '"default"' not in json.dumps(to_strict_json_schema(r.Verification))  # OpenAI strict mode forbids them
