"""Run-to-run spread of the end-to-end eval: the same code evaluated N times shows how much each metric moves by
chance (LLM answers, the judge, latency). compare.py widens a guardrail's tolerance to that noise, but only against
the mean baseline of the same runs (spread.json records how many runs and which judge).

    uv run python -m evals.spread evals/baselines/runs/rep-<date>/run-*.json            # -> baselines/spread.json
    uv run python -m evals.spread ... --mean-baseline evals/baselines/baseline.json      # also write the mean snapshot

noise = the full spread (max - min) of the runs: comparing one new run against the mean, max |run - mean| flags ~13%
of unchanged guardrails as regressed, the full spread ~4% (simulated) — fit for a REVIEW that asks a human to look.
"""

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from evals import metrics

BASELINES = Path(__file__).resolve().parent / "baselines"


def _numbers(snapshots: list[dict]) -> list[str]:
    return sorted({k for s in snapshots for k, v in s["metrics"].items() if isinstance(v, (int, float))})


def _same_setup(snapshots: list[dict]) -> dict:
    """The shared judge and app models; runs from different setups cannot be averaged."""
    setups = {json.dumps([s["metadata"].get("judge_model"), s["metadata"].get("app_models")], sort_keys=True)
              for s in snapshots}
    if len(setups) > 1:
        raise ValueError("the runs used different judge or app models: their spread and mean mean nothing")
    return snapshots[0]["metadata"]


def spread(snapshots: list[dict]) -> dict:
    """{"metadata": {runs, judge_model, ...}, "metrics": [per metric: runs, mean, stdev, spread, noise, partial]}."""
    meta = _same_setup(snapshots)
    rows = []
    for mid in _numbers(snapshots):
        runs = [s["metrics"][mid] for s in snapshots if isinstance(s["metrics"].get(mid), (int, float))]
        m = metrics.get(mid)
        rows.append({"id": mid, "kind": m.kind if m else "unregistered", "runs": runs, "mean": statistics.fmean(runs),
                     "stdev": statistics.stdev(runs) if len(runs) > 1 else 0.0, "spread": max(runs) - min(runs),
                     "noise": max(runs) - min(runs), "partial": len(runs) < len(snapshots)})
    return {"metadata": {"runs": len(snapshots), "judge_model": meta.get("judge_model"),
                         "app_models": meta.get("app_models"), "labels": [s["metadata"].get("label") for s in snapshots],
                         "created_at": datetime.now(UTC).isoformat()}, "metrics": rows}


def mean_snapshot(snapshots: list[dict], label: str) -> dict:
    """A snapshot of per-metric means: a steadier baseline than any single run. A metric missing from some run is left
    out (its worst runs would be the missing ones); a missing error count is 0 errors."""
    meta = _same_setup(snapshots)
    n, out = len(snapshots), {}
    for mid in _numbers(snapshots):
        vals = [s["metrics"].get(mid) for s in snapshots]
        if mid.endswith(".errors"):
            out[mid] = round(sum(v or 0 for v in vals) / n, 6)
        elif all(isinstance(v, (int, float)) for v in vals):
            out[mid] = statistics.fmean(vals)
    return {"metadata": {**meta, "label": label, "runs": n, "created_at": datetime.now(UTC).isoformat()},
            "metrics": out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshots", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=BASELINES / "spread.json")
    ap.add_argument("--mean-baseline", type=Path, help="also write the mean of the runs as a snapshot here")
    args = ap.parse_args()
    snaps = [json.loads(p.read_text(encoding="utf-8")) for p in args.snapshots]
    result = spread(snaps)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{'metric':<44} {'kind':<10} {'mean':>9} {'spread':>9} {'stdev':>9}  runs")
    for r in result["metrics"]:
        if not r["id"].endswith((".n", ".errors")):
            flag = "  (missing in some runs)" if r["partial"] else ""
            print(f"{r['id']:<44} {r['kind']:<10} {r['mean']:>9.4g} {r['spread']:>9.3g} {r['stdev']:>9.3g}  "
                  + " ".join(f"{v:.3g}" for v in r["runs"]) + flag)
    print(f"-> {args.out} ({result['metadata']['runs']} runs)")
    if args.mean_baseline:
        args.mean_baseline.write_text(json.dumps(mean_snapshot(snaps, f"mean of {len(snaps)} runs"), indent=2,
                                                 sort_keys=True), encoding="utf-8")
        print(f"-> {args.mean_baseline}")


if __name__ == "__main__":
    main()
