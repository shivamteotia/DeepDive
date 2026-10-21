"""Regression gate (ported from disha-hr): diff the candidate snapshot against the baseline and give a verdict.

    uv run python -m evals.compare [--all]     # PASS=0  FAIL=1  REVIEW=2 (exit code, for CI)

Every metric id resolves to a rule (rule_for): which direction is better, and whether it is
  gate      - any drop beyond tolerance BLOCKS (safety.*),
  guardrail - a drop beyond tolerance is flagged for a human (quality, det, latency, reliability),
  info      - shown, never affects the verdict (counts, errors, p50).
Judged metrics are compared on their average score, not a pass rate.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from evals import metrics

BASELINES = Path(__file__).resolve().parent / "baselines"
EXIT_CODE = {"PASS": 0, "FAIL": 1, "REVIEW": 2}


def rule_for(mid):
    """-> (direction, kind, tolerance as (absolute, relative)), from the metric registry (evals/metrics.py)."""
    m = metrics.get(mid)
    return ("higher", "info", (0, 0)) if m is None else (m.direction, m.kind, (m.tol_abs, m.tol_rel))


def classify(mid, base, cand, noise=0.0):
    """noise: the metric's measured run-to-run noise (spread.json); it widens a guardrail's tolerance, never a gate's."""
    direction, kind, (tol, rel) = rule_for(mid)
    tolerance = max(tol, rel * abs(base or 0), noise if kind == "guardrail" else 0)
    row = {"id": mid, "baseline": base, "candidate": cand, "kind": kind, "delta": None, "tolerance": tolerance}
    if base is None or cand is None:
        if base is None:
            return {**row, "status": "new"}
        # a metric that vanished (every case failed, or the judge errored) must not read as a pass
        return {**row, "status": {"gate": "blocked", "guardrail": "regressed"}.get(kind, "dropped")}
    row["delta"] = delta = cand - base
    if kind == "info":
        return {**row, "status": "info"}
    if (delta > 0) == (direction == "higher") and delta != 0:
        return {**row, "status": "improved"}
    if round(abs(delta), 9) <= round(tolerance, 9):  # round: 0.92 - 0.90 is 0.0200000000000000018
        return {**row, "status": "flat"}
    return {**row, "status": "blocked" if kind == "gate" else "regressed"}


def compare(baseline, candidate, noise=None):
    b, c, noise = baseline["metrics"], candidate["metrics"], noise or {}
    rows = [classify(mid, b.get(mid), c.get(mid), noise.get(mid, 0.0)) for mid in sorted(set(b) | set(c))]
    statuses = {r["status"] for r in rows}
    return ("FAIL" if "blocked" in statuses else "REVIEW" if "regressed" in statuses else "PASS"), rows


def load_noise(path: Path) -> dict:
    """{metric id: noise} from evals.spread's output; {} when there is none yet."""
    if not path.exists():
        return {}
    return {r["id"]: r["noise"] for r in json.loads(path.read_text(encoding="utf-8"))}


def _fmt(v):
    return "" if v is None else f"{v:.4g}" if isinstance(v, float) and not math.isnan(v) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=BASELINES / "baseline.json")
    ap.add_argument("--candidate", type=Path, default=BASELINES / "candidate.json")
    ap.add_argument("--all", action="store_true", help="also show info metrics")
    ap.add_argument("--spread", type=Path, default=BASELINES / "spread.json",
                    help="measured run-to-run noise (evals.spread): widens guardrail tolerances")
    args = ap.parse_args()
    if not args.baseline.exists():
        print(f"no baseline at {args.baseline}; run `python -m evals.run --baseline` first")
        sys.exit(0)
    baseline, candidate = (json.loads(p.read_text(encoding="utf-8")) for p in (args.baseline, args.candidate))

    for name, snap in (("baseline", baseline), ("candidate", candidate)):
        meta = snap["metadata"]
        print(f"{name:10} {meta.get('label') or '(no label)'}  judge {meta['judge_model']}  {meta['created_at']}")
    if baseline["metadata"]["judge_model"] != candidate["metadata"]["judge_model"]:
        print("  ! judge models differ - judged scores are not comparable")

    noise = load_noise(args.spread)
    if noise:
        print(f"  tolerances widened to the measured noise in {args.spread.name} (guardrails only)")
    verdict, rows = compare(baseline, candidate, noise)
    order = ["blocked", "regressed", "dropped", "new", "improved", "flat", "info"]
    print("=" * 92)
    print(f"{'metric':<46} {'baseline':>10} {'candidate':>10} {'delta':>10} {'tol':>7}  status")
    print("-" * 92)
    for r in sorted(rows, key=lambda r: (order.index(r["status"]), r["id"])):
        if r["kind"] == "info" and not args.all:
            continue
        mark = "  <<<" if r["status"] in ("blocked", "regressed") else ""
        delta = f"{r['delta']:+.4g}" if r["delta"] is not None else ""
        print(f"{r['id']:<46} {_fmt(r['baseline']):>10} {_fmt(r['candidate']):>10} {delta:>10} "
              f"{_fmt(r['tolerance']):>7}  {r['status']}{mark}")
    counts = Counter(r["status"] for r in rows)
    print("-" * 92)
    print("  ".join(f"{k}={counts[k]}" for k in order if counts[k]))
    print({"PASS": "VERDICT: PASS   - no gate blocked, no guardrail regressed.",
           "REVIEW": "VERDICT: REVIEW - a guardrail regressed beyond tolerance; a human decides.",
           "FAIL": "VERDICT: FAIL   - a safety gate regressed. Blocked."}[verdict])
    sys.exit(EXIT_CODE[verdict])


if __name__ == "__main__":
    main()
