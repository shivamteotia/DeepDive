"""The metric registry (one definition per eval metric) and tolerances set from measured run-to-run spread."""
import json
from pathlib import Path

from evals import compare, metrics, spread

BASE = json.loads((Path(__file__).parent.parent / "evals" / "baselines" / "baseline.json").read_text(encoding="utf-8"))


def test_every_metric_the_eval_produces_is_registered():
    missing = [m for m in BASE["metrics"] if metrics.get(m) is None]
    assert missing == []
    for m in metrics.REGISTRY.values():
        assert m.description and m.direction in ("higher", "lower") and m.kind in ("gate", "guardrail", "info")


def test_counts_are_info_and_safety_is_a_gate():
    assert metrics.get("quality.correctness.n").kind == "info"
    assert metrics.get("safety.isolation.rate").kind == "gate"
    assert metrics.get("ops.latency.p95_ms").direction == "lower"
    assert metrics.get("not.a.metric") is None


def snap(**values):
    return {"metadata": {"label": "x"}, "metrics": values}


def test_spread_reports_mean_stdev_and_noise_per_metric():
    runs = [snap(**{"quality.correctness.avg_score": v, "ops.latency.p95_ms": l})
            for v, l in ((0.98, 15000), (0.99, 14000), (0.92, 16000), (0.97, 15000), (0.99, 15000))]
    out = {r["id"]: r for r in spread.spread(runs)}
    c = out["quality.correctness.avg_score"]
    assert c["runs"] == [0.98, 0.99, 0.92, 0.97, 0.99] and round(c["mean"], 3) == 0.97
    assert round(c["spread"], 3) == 0.07 and round(c["noise"], 3) == 0.05  # max |run - mean|
    mean = spread.mean_snapshot(runs, label="mean of 5")
    assert round(mean["metrics"]["quality.correctness.avg_score"], 3) == 0.97 and mean["metadata"]["runs"] == 5


def test_compare_never_flags_a_drop_inside_the_measured_noise():
    noise = {"quality.correctness.avg_score": 0.08}
    base, cand = snap(**{"quality.correctness.avg_score": 0.98}), snap(**{"quality.correctness.avg_score": 0.91})
    assert compare.compare(base, cand)[0] == "REVIEW"  # the fixed 0.05 tolerance alone flags it
    verdict, rows = compare.compare(base, cand, noise=noise)
    assert verdict == "PASS" and rows[0]["status"] == "flat" and rows[0]["tolerance"] == 0.08
    worse = snap(**{"quality.correctness.avg_score": 0.85})
    assert compare.compare(base, worse, noise=noise)[0] == "REVIEW"  # beyond the noise still regresses


def test_a_safety_gate_keeps_its_own_tolerance_however_noisy():
    noise = {"safety.isolation.rate": 0.5}
    verdict, _ = compare.compare(snap(**{"safety.isolation.rate": 1.0}), snap(**{"safety.isolation.rate": 0.9}),
                                 noise=noise)
    assert verdict == "FAIL"  # safety never widens with noise
