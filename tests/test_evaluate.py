import json
from pathlib import Path

import pytest

import evaluate as ev
import research


def hit(text, name="a.pdf", eid="E1"):
    return {"eid": eid, "source_filename": name, "page_numbers": [1], "text": text}


def test_relevance_is_whitespace_and_case_insensitive_and_source_bound():
    exp = [{"source": "a.pdf", "contains": "501(1.8%)"}]
    assert ev.is_relevant(hit("Hospital Mortality b, Data = 501 (1.8 %)."), exp)
    assert not ev.is_relevant(hit("501(1.8%)", name="b.pdf"), exp)
    assert ev.first_relevant_rank([hit("x"), hit("y 501(1.8%)")], exp) == 2
    assert ev.first_relevant_rank([hit("x")], exp) is None


def test_retrieval_metrics():
    m = ev.retrieval_metrics([1, 2, None, 4])
    assert m["recall_at_k"] == 0.75
    assert m["mrr"] == pytest.approx((1 + 0.5 + 0.25) / 4)


def test_verifier_metrics_counts_thin_as_not_supported():
    row = lambda label, verdict, got=True: {"label": label, "verdict": verdict, "evidence_retrieved": got}
    rows = [row("false", "UNSUPPORTED"), row("false", "THINLY_SUPPORTED"), row("false", "SUPPORTED"),
            row("true", "SUPPORTED"), row("true", "THINLY_SUPPORTED"),
            row("false", "UNSUPPORTED", got=False)]  # retrieval miss: must not count as a catch
    m = ev.verifier_metrics(rows)
    assert m["catch_rate"] == pytest.approx(2 / 3)
    assert m["true_rejection_rate"] == 0.5
    assert m["n_excluded_no_evidence"] == 1


def test_empty_metrics_report_na_instead_of_crashing():
    res = {"retrieval": {**ev.retrieval_metrics([]), "items": []}, "verifier": {**ev.verifier_metrics([]), "items": []}}
    assert "recall@k=n/a" in ev.report(res) and "catch_rate=n/a" in ev.report(res)


def test_evaluate_end_to_end_with_fakes():
    exp = [{"source": "a.pdf", "contains": "needle"}]
    gold = {"retrieval": [{"id": "r1", "question": "q", "expected": exp}],
            "claims": [{"id": "f1", "label": "false", "search": "s", "claim": "wrong", "expected": exp},
                       {"id": "t1", "label": "true", "search": "s", "claim": "right", "expected": exp},
                       {"id": "t2", "label": "true", "search": "none", "claim": "right", "expected": exp}]}
    search = lambda q: [] if q == "none" else [hit("hay"), hit("needle", eid="E2")]
    verify = lambda c, evd: research.Verdict(answers_question=True, claim_index=0, verdict="UNSUPPORTED", rationale="no")
    res = ev.evaluate(gold, search_fn=search, verify_fn=verify)
    assert res["retrieval"]["items"][0]["rank"] == 2
    assert res["verifier"]["catch_rate"] == 1.0
    assert res["verifier"]["true_rejection_rate"] == 1.0  # t2 excluded: nothing retrieved
    assert res["verifier"]["items"][2]["verdict"] == "NO_EVIDENCE"
    out = ev.report(res)
    assert "WRONG t1" in out and "NO-EVIDENCE t2" in out and "WRONG t2" not in out


def test_gold_file_is_well_formed():
    gold = json.loads((Path(__file__).parent.parent / "eval" / "gold.json").read_text(encoding="utf-8"))
    ids = [x["id"] for x in gold["retrieval"] + gold["claims"]]
    assert len(ids) == len(set(ids))
    assert len(gold["retrieval"]) >= 20
    assert {c["label"] for c in gold["claims"]} == {"true", "false"}
    assert all(e["source"].endswith(".pdf") and e["contains"]
               for r in gold["retrieval"] + gold["claims"] for e in r["expected"])


@pytest.mark.live
def test_live_eval_meets_baseline():
    res = ev.run_live(Path(__file__).parent.parent / "eval" / "gold.json", k=6)
    print(ev.report(res))
    assert res["retrieval"]["recall_at_k"] >= 0.8
    assert res["verifier"]["catch_rate"] >= 0.9
    assert res["verifier"]["true_rejection_rate"] <= 0.2
