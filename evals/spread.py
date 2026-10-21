"""Run-to-run spread of the end-to-end eval: the same code evaluated N times shows how much each metric moves by
chance (LLM answers, the judge, latency). compare.py uses the measured noise as a guardrail's tolerance.

    uv run python -m evals.spread evals/baselines/runs/rep-<date>/run-*.json            # -> baselines/spread.json
    uv run python -m evals.spread ... --mean-baseline evals/baselines/baseline.json      # also write the mean snapshot
"""

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from evals import metrics

BASELINES = Path(__file__).resolve().parent / "baselines"


def spread(snapshots: list[dict]) -> list[dict]:
    """Per metric: the runs, mean, stdev, spread (max - min) and noise (max |run - mean|)."""
    ids = sorted({k for s in snapshots for k, v in s["metrics"].items() if isinstance(v, (int, float))})
    out = []
    for mid in ids:
        runs = [s["metrics"][mid] for s in snapshots if isinstance(s["metrics"].get(mid), (int, float))]
        mean = statistics.fmean(runs)
        m = metrics.get(mid)
        out.append({"id": mid, "kind": m.kind if m else "unregistered", "runs": runs, "mean": mean,
                    "stdev": statistics.stdev(runs) if len(runs) > 1 else 0.0, "spread": max(runs) - min(runs),
                    "noise": max(abs(r - mean) for r in runs)})
    return out


def mean_snapshot(snapshots: list[dict], label: str) -> dict:
    """A snapshot whose metrics are the per-metric means: a steadier baseline than any single run."""
    first = snapshots[0]["metadata"]
    return {"metadata": {**first, "label": label, "runs": len(snapshots), "created_at": datetime.now(UTC).isoformat()},
            "metrics": {r["id"]: r["mean"] for r in spread(snapshots)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshots", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=BASELINES / "spread.json")
    ap.add_argument("--mean-baseline", type=Path, help="also write the mean of the runs as a snapshot here")
    args = ap.parse_args()
    snaps = [json.loads(p.read_text(encoding="utf-8")) for p in args.snapshots]
    rows = spread(snaps)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"{'metric':<44} {'kind':<10} {'mean':>9} {'spread':>9} {'noise':>9}  runs")
    for r in rows:
        if not r["id"].endswith((".n", ".errors")):
            print(f"{r['id']:<44} {r['kind']:<10} {r['mean']:>9.4g} {r['spread']:>9.3g} {r['noise']:>9.3g}  "
                  + " ".join(f"{v:.3g}" for v in r["runs"]))
    print(f"-> {args.out}")
    if args.mean_baseline:
        args.mean_baseline.write_text(json.dumps(mean_snapshot(snaps, f"mean of {len(snaps)} runs"), indent=2,
                                                 sort_keys=True), encoding="utf-8")
        print(f"-> {args.mean_baseline}")


if __name__ == "__main__":
    main()
