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


def claim(text, *ids):
    return ask.Claim(text=text, evidence_ids=list(ids))


def verdicts(*vs):
    return r.Verification(verdicts=[r.Verdict(claim_index=i, verdict=v, rationale="why") for i, v in enumerate(vs)])


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
    partial_v = r.Verification(verdicts=[r.Verdict(claim_index=0, verdict="SUPPORTED", rationale="ok"),
                                         r.Verdict(claim_index=7, verdict="SUPPORTED", rationale="bogus index")])
    out = r.verify(FakeOpenAI({r.Verification: [partial_v]}), "g", claims, ev)
    assert [v.verdict for v in out] == ["SUPPORTED", "UNSUPPORTED"]
    out = r.verify(FakeOpenAI({r.Verification: [None]}), "g", claims, ev)
    assert [v.verdict for v in out] == ["UNSUPPORTED", "UNSUPPORTED"]


def test_verifier_sees_each_cited_source():
    # The study-name rule needs the source: chunk text rarely names its own paper.
    oa = FakeOpenAI({r.Verification: [verdicts("SUPPORTED")]})
    r.verify(oa, "g", [claim("Li2024 AUC 0.85", "E1")], [{**hit("p1", "AUC 0.85", name="Li2024.pdf"), "eid": "E1"}])
    assert '"source": "Li2024.pdf"' in oa.inputs[r.Verification][0]


def test_all_supported_first_pass_is_verified():
    oa = FakeOpenAI({
        r.Plan: [r.Plan(sub_questions=["risks"])],
        ask.Answer: [ask.Answer(claims=[claim("privacy risk", "E1")], gaps="")],
        r.Verification: [verdicts("SUPPORTED")],
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
        r.Verification: [verdicts("SUPPORTED", "UNSUPPORTED"), verdicts("SUPPORTED", "SUPPORTED")],
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
        r.Verification: [verdicts("THINLY_SUPPORTED", "UNSUPPORTED")] * 2,
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
    dup = r.Verification(verdicts=[r.Verdict(claim_index=0, verdict="UNSUPPORTED", rationale="no"),
                                   r.Verdict(claim_index=0, verdict="SUPPORTED", rationale="yes")])
    (v,) = r.verify(FakeOpenAI({r.Verification: [dup]}), "g", [claim("a", "E1")], ev)
    assert v.verdict == "UNSUPPORTED"


def test_cross_project_hit_is_rejected():
    with pytest.raises(RuntimeError):
        r.add_evidence([], [{**hit("p1"), "project_id": "other"}], "demo")


def test_empty_answer_mid_loop_keeps_earlier_claims():
    oa = FakeOpenAI({
        r.Plan: [r.Plan(sub_questions=["risks"])],
        ask.Answer: [ask.Answer(claims=[claim("privacy risk", "E1"), claim("made up", "E1")], gaps=""), None, None],
        r.Verification: [verdicts("SUPPORTED", "UNSUPPORTED")],
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
        r.Verification: [verdicts("UNSUPPORTED"), verdicts("UNSUPPORTED"), verdicts("SUPPORTED")],
    })
    res = run(oa)
    first = next(e for e in res["log"] if e["step"] == "verify")
    assert first["claims"][0] == {"text": 'banned "by" FDA', "evidence_ids": ["E2"], "verdict": "UNSUPPORTED", "rationale": "why"}
    third_prompt = oa.inputs[ask.Answer][2]
    assert r.json.dumps('banned "by" FDA') in third_prompt and '"other"' in third_prompt  # JSON-encoded, cumulative


def test_invented_citation_forces_review():
    ans = ask.Answer(claims=[claim("ok", "E1")], gaps=ask.UNCITED_PREFIX + "fake")
    res = r.finalize("q", ans, [r.Verdict(claim_index=0, verdict="SUPPORTED", rationale="")], [], [])
    assert res["status"] == "needs_review"


def test_keeps_round_with_most_supported_claims():
    oa = FakeOpenAI({
        r.Plan: [r.Plan(sub_questions=["risks"])],
        ask.Answer: [ask.Answer(claims=[claim("a", "E1"), claim("b", "E1"), claim("x", "E1")], gaps=""),
                     ask.Answer(claims=[claim("c", "E1"), claim("y", "E1")], gaps="")],
        r.Verification: [verdicts("SUPPORTED", "SUPPORTED", "UNSUPPORTED"), verdicts("UNSUPPORTED", "UNSUPPORTED")],
    })
    res = run(oa, max_iterations=2)
    assert [c["text"] for c in res["claims"]] == ["a", "b"]


def test_all_invented_citations_are_retried():
    oa = FakeOpenAI({
        r.Plan: [r.Plan(sub_questions=["risks"])],
        ask.Answer: [ask.Answer(claims=[claim("fake", "E99")], gaps=""), ask.Answer(claims=[claim("ok", "E1")], gaps="")],
        r.Verification: [verdicts("SUPPORTED")],
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


def test_plan_question_reads_scope_metric_and_target():
    out = r.Plan(sub_questions=["AUC of each model"], scope="all_values", metric="AUC",
                 metric_terms=["area under the curve"], target_paper="the SCAP study")
    p = r.plan_question(FakeOpenAI({r.Plan: [out]}), "f", "What AUC values were reported in the SCAP study?")
    assert (p.scope, p.metric, p.metric_terms, p.target_paper) == ("all_values", "AUC", ["area under the curve"], "the SCAP study")
    assert p.sub_questions == ["AUC of each model"]


def test_plan_question_defaults_to_standard():
    p = r.plan_question(FakeOpenAI({r.Plan: [r.Plan(sub_questions=["q1"])]}), "f", "q")
    assert (p.scope, p.metric, p.metric_terms, p.target_paper, p.sub_questions) == ("standard", "", [], "", ["q1"])
    p = r.plan_question(FakeOpenAI({}), "f", "orig")  # planner failed
    assert p.scope == "standard" and p.sub_questions == ["orig"]


def test_plan_schema_has_no_non_null_defaults():
    from openai.lib._pydantic import to_strict_json_schema
    assert '"default"' not in __import__("json").dumps(to_strict_json_schema(r.Plan))


DOCS = [{"sha256": "s1", "filename": "JPan2023.pdf", "title": "Development and validation of an in-hospital mortality "
         "risk prediction model for patients with severe community-acquired pneumonia"},
        {"sha256": "s2", "filename": "EA2022.pdf", "title": "Dynamic Mortality Risk Predictions for Children in ICU"}]


def plan_out(scope, metric="AUC", target=""):
    return r.Plan(sub_questions=["q"], scope=scope, metric=metric, metric_terms=["area under the curve"], target_paper=target)


class RecordingSearch:
    def __init__(self, docs=DOCS):
        self.calls, self.docs = [], {d["sha256"] for d in docs}

    def __call__(self, q, document=None, k=None):
        assert document is None or document in self.docs  # only server-provided documents
        self.calls.append((q, document, k))
        return [hit(f"{document or 'all'}-{len(self.calls)}", "AUC 0.850", name=document or "a.pdf")]


def run_scoped(scope, target="", docs=DOCS):
    oa = FakeOpenAI({r.Plan: [plan_out(scope, target=target)], ask.Answer: [ask.Answer(claims=[], gaps="none")]})
    search = RecordingSearch(docs)
    r.run(oa, search_fn=search, project_id="demo", model="m", fast_model="f", guard_model="g", question="q",
          documents=docs)
    return search.calls


def test_standard_scope_searches_the_project_once_per_sub_question():
    assert [(d, k) for _, d, k in run_scoped("standard")] == [(None, None)]


def test_all_values_without_target_searches_each_document():
    calls = run_scoped("all_values")
    assert [(d, k) for _, d, k in calls] == [("s1", r.ROW_K), ("s2", r.ROW_K)]
    assert all("area under the curve" in q and "AUC" in q for q, _, _ in calls)


def test_named_paper_is_searched_alone():
    assert [d for _, d, _ in run_scoped("best_model", target="the severe community-acquired pneumonia study")] == ["s1"]


def test_unmatched_target_falls_back_to_every_document():
    assert [d for _, d, _ in run_scoped("all_values", target="Smith2019.pdf")] == ["s1", "s2"]


def test_across_papers_uses_across_k():
    assert [(d, k) for _, d, k in run_scoped("across_papers")] == [("s1", r.ACROSS_K), ("s2", r.ACROSS_K)]


def test_too_many_documents_fall_back_to_one_project_search():
    many = [{"sha256": f"s{i}", "filename": f"p{i}.pdf", "title": f"paper {i}"} for i in range(r.MAX_COVERAGE_DOCS + 1)]
    assert [d for _, d, _ in run_scoped("across_papers", docs=many)] == [None]


def test_resolve_target_needs_real_overlap():
    assert r.resolve_target("the SCAP pneumonia mortality study", DOCS)["sha256"] == "s1"
    assert r.resolve_target("children ICU mortality", DOCS)["sha256"] == "s2"
    assert r.resolve_target("cardiology trial", DOCS) is None and r.resolve_target("", DOCS) is None


def test_render_shows_the_coverage_step():
    oa = FakeOpenAI({r.Plan: [plan_out("across_papers")], ask.Answer: [ask.Answer(claims=[], gaps="none")]})
    res = r.run(oa, search_fn=RecordingSearch(), project_id="demo", model="m", fast_model="f", guard_model="g",
                question="q", documents=DOCS)
    assert "coverage: 2 paper(s) searched" in r.render(res)


def row(model="predictive model", best=False, paper="JPan2023.pdf"):
    return ask.MetricRow(paper=paper, model=model, condition="validation cohort", metric="AUC", value="0.893", best=best)


def scoped_result(scope, claims, vs, docs=DOCS, hits=None):
    oa = FakeOpenAI({r.Plan: [plan_out(scope)], ask.Answer: [ask.Answer(claims=claims, gaps="")], r.Verification: [vs]})
    by_doc = hits or {"s1": [hit("p1", "our predictive model outperformed APACHE II", "JPan2023.pdf")], "s2": []}
    search = lambda q, document=None, k=None: by_doc.get(document, [])
    return r.run(oa, search_fn=search, project_id="demo", model="m", fast_model="f", guard_model="g", question="q",
                 documents=docs, max_iterations=1)


def test_rows_survive_into_the_result():
    res = scoped_result("all_values", [ask.Claim(text="AUC 0.893", evidence_ids=["E1"], row=row())], verdicts("SUPPORTED"))
    assert res["claims"][0]["row"]["model"] == "predictive model" and res["scope"] == "all_values"


def test_best_row_needs_a_supported_designation_in_the_same_paper():
    designation = ask.Claim(text="The authors' predictive model outperformed APACHE II and SOFA.", evidence_ids=["E1"])
    best = ask.Claim(text="AUC 0.893", evidence_ids=["E1"], row=row(best=True))
    ok = scoped_result("best_model", [designation, best], verdicts("SUPPORTED", "SUPPORTED"))
    assert ok["claims"][1]["row"]["best"] is True and ok["status"] == "verified"
    bad = scoped_result("best_model", [designation, best], verdicts("UNSUPPORTED", "SUPPORTED"))
    assert bad["claims"][0]["row"]["best"] is False and bad["status"] == "needs_review"


def test_searched_paper_without_rows_gets_a_coverage_line():
    res = scoped_result("across_papers", [ask.Claim(text="AUC 0.893", evidence_ids=["E1"], row=row())], verdicts("SUPPORTED"))
    assert "Not reported in: EA2022.pdf (searched; no AUC found)" in res["gaps"]
    assert "JPan2023.pdf" not in res["gaps"]


def test_web_hits_never_count_as_papers_in_coverage():
    hits = {"s1": [hit("p1", "AUC 0.893", "JPan2023.pdf")],
            "s2": [hit("w1", "AUC on the web", "Some page") | {"source_type": "web", "url": "https://x.org"}]}
    res = scoped_result("across_papers", [ask.Claim(text="AUC on web", evidence_ids=["E2"], row=row(paper="Some page"))],
                        verdicts("SUPPORTED"), hits=hits)
    assert "Not reported in: EA2022.pdf" in res["gaps"] and "JPan2023.pdf (searched" in res["gaps"]


def test_standard_scope_adds_no_coverage_lines():
    res = run(FakeOpenAI({ask.Answer: [ask.Answer(claims=[claim("c", "E1")], gaps="")],
                          r.Verification: [verdicts("SUPPORTED")]}))
    assert "Not reported in" not in res["gaps"] and "row" not in res["claims"][0]


def test_too_many_papers_says_so_in_gaps():
    many = [{"sha256": f"s{i}", "filename": f"p{i}.pdf", "title": f"paper {i}"} for i in range(r.MAX_COVERAGE_DOCS + 1)]
    res = scoped_result("across_papers", [claim("AUC 0.893", "E1")], verdicts("SUPPORTED"), docs=many,
                        hits={None: [hit("p1", "AUC 0.893")]})
    assert f"Searched the project as a whole: more than {r.MAX_COVERAGE_DOCS} papers." in res["gaps"]


def test_metric_question_with_no_evidence_keeps_scope_and_coverage_lines():
    res = scoped_result("across_papers", [], verdicts(), hits={"s1": [], "s2": []})
    assert res["scope"] == "across_papers" and "Not reported in: JPan2023.pdf" in res["gaps"]
    assert "Not reported in: EA2022.pdf" in res["gaps"]


def test_best_row_needs_a_designation_that_names_its_model():
    context = ask.Claim(text="JPan2023 enrolled 523 patients with SCAP.", evidence_ids=["E1"])  # supported, designates nothing
    best = ask.Claim(text="AUC 0.893", evidence_ids=["E1"], row=row(best=True))
    res = scoped_result("best_model", [context, best], verdicts("SUPPORTED", "SUPPORTED"))
    assert res["claims"][1]["row"]["best"] is False and res["status"] == "needs_review"


def test_verifier_sees_row_fields():
    oa = FakeOpenAI({r.Verification: [verdicts("SUPPORTED")]})
    ev = [hit("p1", "AUC 0.893") | {"eid": "E1"}]
    r.verify(oa, "g", [ask.Claim(text="AUC 0.893", evidence_ids=["E1"], row=row())], ev)
    sent = oa.inputs[r.Verification][0]
    assert '"row"' in sent and '"condition": "validation cohort"' in sent
    assert "row" in r.VERIFIER


@pytest.mark.parametrize("model", ["", "  ", "SOFA"])
def test_designation_match_is_whole_word_and_non_empty(model):
    context = ask.Claim(text="The authors' qSOFA model was proposed as final.", evidence_ids=["E1"])
    best = ask.Claim(text="AUC 0.893", evidence_ids=["E1"], row=row(model=model, best=True))
    res = scoped_result("best_model", [context, best], verdicts("SUPPORTED", "SUPPORTED"))
    assert res["claims"][1]["row"]["best"] is False
