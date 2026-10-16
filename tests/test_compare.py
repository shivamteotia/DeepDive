import pytest

from evals.compare import compare

BASE = {"metrics": {"safety.toxicity.avg_score": 0.95, "quality.faithfulness.avg_score": 0.90,
                    "ops.latency.p95_ms": 4000, "safety.forbidden.rate": 1.0, "quality.faithfulness.n": 8,
                    "det.fact.rate": 0.9}}


def cand(**changes):
    return {"metrics": {**BASE["metrics"], **{k.replace("__", "."): v for k, v in changes.items()}}}


@pytest.mark.parametrize(("changes", "verdict"), [
    ({}, "PASS"),
    ({"quality__faithfulness__avg_score": 0.87}, "PASS"),  # within judge noise
    ({"quality__faithfulness__avg_score": 0.80}, "REVIEW"),
    ({"ops__latency__p95_ms": 5100}, "REVIEW"),
    ({"ops__latency__p95_ms": 2000}, "PASS"),  # faster
    ({"safety__toxicity__avg_score": 0.90}, "FAIL"),
    ({"safety__forbidden__rate": 0.96}, "FAIL"),  # a rate gate has zero tolerance
    ({"det__fact__rate": 0.8}, "REVIEW"),
    ({"quality__faithfulness__n": 3}, "PASS"),  # info
    ({"quality__faithfulness__errors": 2}, "PASS"),  # info
])
def test_verdicts(changes, verdict):
    assert compare(BASE, cand(**changes))[0] == verdict


def test_drop_of_exactly_the_tolerance_is_flat():
    assert compare({"metrics": {"safety.x.avg_score": 0.92}}, {"metrics": {"safety.x.avg_score": 0.90}})[0] == "PASS"


def test_dropped_gate_fails_and_dropped_guardrail_needs_review():
    gone = {k: v for k, v in BASE["metrics"].items() if k != "safety.forbidden.rate"}
    assert compare(BASE, {"metrics": gone})[0] == "FAIL"
    gone = {k: v for k, v in BASE["metrics"].items() if k != "quality.faithfulness.avg_score"}
    assert compare(BASE, {"metrics": gone})[0] == "REVIEW"
    gone = {k: v for k, v in BASE["metrics"].items() if k != "quality.faithfulness.n"}
    assert compare(BASE, {"metrics": gone})[0] == "PASS"
