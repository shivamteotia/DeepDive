import pytest

from evals import run as ev


def claim(text, eids, verdict="SUPPORTED"):
    return {"text": text, "evidence_ids": eids, "verdict": verdict, "rationale": "r"}


def row(case, claims=(), gaps="", evidence=(), status="completed", latency_ms=1000):
    return {**case, "status": status, "claims": list(claims), "gaps": gaps, "evidence": list(evidence),
            "latency_ms": latency_ms}


EV = [{"eid": "E1", "source_filename": "EA2022.pdf", "text": "AUROC 0.852"},
      {"eid": "E2", "source_filename": "JPan2023.pdf", "text": "455 patients"}]
ANS = {"id": "a02", "kind": "answerable", "project": "main", "question": "q", "expected": "0.852",
       "source": ["EA2022.pdf"], "facts": ["0.852"]}


def test_golden_file_is_valid():
    cases = ev.load_golden()
    assert len(cases) >= 26
    assert len({c["id"] for c in cases}) == len(cases)
    assert sum(c["kind"] == "unanswerable" for c in cases) >= 6


@pytest.mark.parametrize("bad", [{"kind": "nope"}, {"project": "other"}, {"source": None}, {"facts": []}])
def test_golden_validation_rejects_bad_cases(bad):
    with pytest.raises(ValueError):
        ev.validate([{**ANS, **bad}])


def test_answer_text_joins_claims_and_gaps():
    assert ev.answer_text([claim("A", ["E1"]), claim("B", ["E2"])], "gap") == "A\nB\ngap"
    assert ev.answer_text([], "") == ""


def test_contexts_are_cited_evidence_or_all_when_no_claims():
    assert ev.contexts([claim("A", ["E2"])], EV) == ["455 patients"]
    assert ev.contexts([], EV) == ["AUROC 0.852", "455 patients"]


def test_answerable_hit_and_fact():
    out = ev.det_checks(row(ANS, [claim("AUROC was 0.852", ["E1"])], evidence=EV))
    assert out == {"det.completed": True, "det.evidence_hit": True, "det.fact": True}


def test_answerable_miss_when_cited_from_wrong_file_and_fact_absent():
    out = ev.det_checks(row(ANS, [claim("unclear", ["E2"])], evidence=EV))
    assert out["det.evidence_hit"] is False and out["det.fact"] is False


def test_evidence_hit_needs_every_expected_source():
    both = {**ANS, "source": ["EA2022.pdf", "JPan2023.pdf"]}
    assert ev.det_checks(row(both, [claim("0.852", ["E1"])], evidence=EV))["det.evidence_hit"] is False
    assert ev.det_checks(row(both, [claim("0.852", ["E1", "E2"])], evidence=EV))["det.evidence_hit"] is True


def test_abstain_fails_on_any_supported_claim_but_allows_thin():
    case = {"id": "u01", "kind": "unanswerable", "project": "main", "question": "q", "expected": "none"}
    assert ev.det_checks(row(case, gaps="not reported"))["det.abstain"] is True
    assert ev.det_checks(row(case, [claim("x", ["E1"], "THINLY_SUPPORTED")], evidence=EV))["det.abstain"] is True
    assert ev.det_checks(row(case, [claim("x", ["E1"])], evidence=EV))["det.abstain"] is False


def test_forbidden_is_case_insensitive_over_claims_and_gaps():
    case = {"id": "s02", "kind": "scope", "project": "main", "question": "q", "expected": "e",
            "facts": ["0.903"], "forbidden": ["Canberra"]}
    assert ev.det_checks(row(case, gaps="capital is CANBERRA"))["safety.forbidden"] is False
    assert ev.det_checks(row(case, gaps="out of scope"))["safety.forbidden"] is True


def test_isolation_fails_when_other_project_file_in_evidence():
    case = {"id": "i01", "kind": "isolation", "project": "isolation", "question": "q", "expected": "e",
            "foreign": ["EA2022.pdf"]}
    assert ev.det_checks(row(case, evidence=EV))["safety.isolation"] is False
    assert ev.det_checks(row(case, evidence=EV[1:]))["safety.isolation"] is True


def test_failed_run_only_scores_completed():
    assert ev.det_checks(row(ANS, status="failed")) == {"det.completed": False}


def test_snapshot_rates_judged_averages_and_latency():
    rows = [{"id": "a", "kind": "answerable", "latency_ms": 1000, "det": {"det.completed": True, "det.fact": True}},
            {"id": "b", "kind": "answerable", "latency_ms": 3000, "det": {"det.completed": True, "det.fact": False}},
            {"id": "c", "kind": "injection", "latency_ms": None, "det": {"det.completed": False}}]
    judged = {"a": {"quality.faithfulness": 1.0, "safety.toxicity": 1.0},
              "b": {"quality.faithfulness": 0.5, "safety.toxicity": None}}
    m = ev.snapshot(rows, judged, "lbl", {"judge_model": "j"})["metrics"]
    assert m["det.completed.rate"] == pytest.approx(2 / 3) and m["det.completed.n"] == 3
    assert m["det.fact.rate"] == 0.5
    assert m["quality.faithfulness.avg_score"] == 0.75 and m["quality.faithfulness.n"] == 2
    assert m["safety.toxicity.avg_score"] == 1.0 and m["safety.toxicity.errors"] == 1
    assert m["ops.success.rate"] == m["det.completed.rate"]
    assert m["ops.latency.p50_ms"] == 2000


def test_failed_safety_case_fails_closed():
    case = {"id": "i01", "kind": "isolation", "project": "isolation", "question": "q", "expected": "e",
            "foreign": ["EA2022.pdf"], "forbidden": ["0.852"]}
    out = ev.det_checks(row(case, status="timeout"))
    assert out == {"det.completed": False, "safety.forbidden": False, "safety.isolation": False}


def flaky_transport(fail_paths, calls):
    """Times out the first request to each path in fail_paths, then answers like the API."""
    import httpx

    def handler(request):
        path = request.url.path
        calls.append((request.method, path))
        if path in fail_paths and calls.count((request.method, path)) == 1:
            raise httpx.ReadTimeout("stuck connection", request=request)
        if path.endswith("/research-runs"):
            return httpx.Response(202, json={"id": "r1", "status": "queued", "error": None})
        if path.endswith("/verification"):
            return httpx.Response(200, json={"status": "verified", "claims": [claim("0.852", ["E1"])], "gaps": ""})
        if path.endswith("/evidence"):
            return httpx.Response(200, json=EV)
        return httpx.Response(200, json={"id": "r1", "status": "completed", "error": None})
    return httpx.MockTransport(handler)


def test_timed_out_request_is_retried_on_a_fresh_connection(monkeypatch):
    monkeypatch.setattr(ev, "POLL_S", 0)
    calls = []
    api = ev.Api("http://t", transport=flaky_transport({"/api/v1/projects/p/research-runs",
                                                        "/api/v1/research-runs/r1/evidence"}, calls))
    r = ev.ask_case(api, "p", ANS)
    assert r["status"] == "completed" and r["error"] is None
    assert calls.count(("POST", "/api/v1/projects/p/research-runs")) == 2
    assert calls.count(("GET", "/api/v1/research-runs/r1/evidence")) == 2
    assert ev.det_checks(r) == {"det.completed": True, "det.evidence_hit": True, "det.fact": True}


def test_result_that_cannot_be_fetched_is_failed_not_empty(monkeypatch):
    import httpx
    monkeypatch.setattr(ev, "POLL_S", 0)
    monkeypatch.setattr(ev, "RETRIES", 1)

    def handler(request):
        if request.url.path.endswith("/evidence"):
            raise httpx.ReadTimeout("stuck", request=request)
        return flaky_transport(set(), []).handler(request)
    r = ev.ask_case(ev.Api("http://t", transport=httpx.MockTransport(handler)), "p", ANS)
    assert r["status"] == "fetch_failed" and "ReadTimeout" in r["error"]
    assert ev.det_checks(r) == {"det.completed": False}


def test_only_completed_runs_are_judged():
    rows = [{"id": "a", "status": "completed"}, {"id": "b", "status": "fetch_failed"}, {"id": "c", "status": "timeout"}]
    assert [r["id"] for r in ev.judgeable(rows)] == ["a"]


def test_fact_counts_only_in_claims_not_in_gaps():
    gaps_only = row(ANS, [claim("unclear", ["E1"])], gaps="the evidence does not report 0.852", evidence=EV)
    assert ev.det_checks(gaps_only)["det.fact"] is False
