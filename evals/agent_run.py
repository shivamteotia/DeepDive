"""Tool-routing eval of the screening assistant through the running API: one message per case (evals/agent_golden.json)
in a fresh chat of the `eval-agent` project, scored on which tools the turn called — no judge.

    uv run uvicorn app:create_app --factory          # API on EVAL_API (default http://127.0.0.1:8000)
    uv run python -m evals.agent_run --label "..."   # -> baselines/agent_candidate.json (or --out PATH)

Costs money: every case is one assistant turn (model calls); some tools start a paid research or PubMed run.
Metrics (evals/metrics.py): agent.routing.rate (guardrail), agent.forbidden.rate (gate), agent.steps.mean (info).
"""

import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINES = HERE / "baselines"
PROJECT = "eval-agent"
TURN_TIMEOUT_S = 240  # a turn runs up to 8 tool calls with model calls between them
WORKERS = 3
CRITERIA = [
    {"id": "c1", "name": "2015 or later", "kind": "include_if", "definition": "Published 2015 or later",
     "check": "metadata", "field": "year", "value": ">=2015"},
    {"id": "c2", "name": "Retrospective", "kind": "include_if", "definition": "Retrospective design, not an RCT",
     "check": "ai", "field": None, "value": None},
]


def load_cases(path: Path = HERE / "agent_golden.json") -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("duplicate case ids")
    for c in cases:
        if not c.get("message") or not ("expect" in c or "expect_any" in c or c.get("no_tools")):
            raise ValueError(f"{c.get('id')}: needs a message and expect / expect_any / no_tools")
    return cases


def called(turn: dict) -> list[str]:
    return [m["content"]["tool"] for m in turn.get("messages", []) if m.get("role") == "tool"]


def score(case: dict, turn: dict) -> dict:
    """routing: the expected tools ran (all of `expect`, or exactly one of the `expect_any` sets; none for no_tools)
    and nothing forbidden ran; forbidden: no forbidden tool ran; steps: tool calls made."""
    tools = called(turn)
    ran = set(tools)
    forbidden_ok = not ran & set(case.get("forbid", []))
    if case.get("no_tools"):
        expected_ok = not ran
    elif "expect_any" in case:
        expected_ok = any(set(alt) <= ran and (alt or not ran) for alt in case["expect_any"])
    else:
        expected_ok = set(case["expect"]) <= ran
    return {"routing": expected_ok and forbidden_ok, "forbidden": forbidden_ok, "steps": len(tools)}


def snapshot(rows: list[dict], label: str, meta: dict) -> dict:
    n = len(rows)
    metrics = {"agent.routing.rate": sum(r["routing"] for r in rows) / n, "agent.routing.n": n,
               "agent.forbidden.rate": sum(r["forbidden"] for r in rows) / n, "agent.forbidden.n": n,
               "agent.steps.mean": statistics.fmean(r["steps"] for r in rows)}
    return {"metadata": {**meta, "label": label, "created_at": datetime.now(UTC).isoformat(), "suite": "agent"},
            "metrics": metrics}


def _nbib(n: int = 30) -> str:
    """Synthetic PubMed records: years 2010-2022, every third retrospective, the rest RCTs."""
    out = []
    for i in range(n):
        design = "A retrospective cohort of" if i % 3 == 0 else "A randomized controlled trial in"
        out.append(f"PMID- {91000 + i}\nDP  - {2010 + i % 13}\nTI  - Eval agent study {i} of ICU mortality\n"
                   f"AB  - {design} adults in the intensive care unit.\nFAU - Author{i}, A\nPT  - Journal Article\n")
    return "\n".join(out)


def setup(api) -> str:
    """Find-or-create the eval-agent project with its synthetic import, criteria and a Study design column (free)."""
    existing = {p["name"]: p["id"] for p in api.call("GET", "/api/v1/projects")}
    pid = existing.get(PROJECT) or api.call("POST", "/api/v1/projects", json={"name": PROJECT})["id"]
    api.call("POST", f"/api/v1/projects/{pid}/reference-files",
             files={"file": ("eval-agent.nbib", _nbib().encode(), "text/plain")})  # the same file twice imports nothing
    if api.request("GET", f"/api/v1/projects/{pid}/criteria").status_code == 404:
        api.call("PUT", f"/api/v1/projects/{pid}/criteria", json={"criteria": CRITERIA})
    if not any(c["name"] == "Study design" for c in api.call("GET", f"/api/v1/projects/{pid}/columns")):
        api.call("POST", f"/api/v1/projects/{pid}/columns", json={
            "name": "Study design", "instruction": "The study design", "kind": "category",
            "options": ["retrospective cohort", "randomized controlled trial (RCT)", "other"]})
    return pid


def run_case(api, pid: str, case: dict) -> dict:
    cid = api.call("POST", f"/api/v1/projects/{pid}/conversations", json={"title": f"eval {case['id']}"})["id"]
    r = api.request("POST", f"/api/v1/conversations/{cid}/agent", json={"message": case["message"]},
                    timeout=TURN_TIMEOUT_S)
    turn = r.json() if r.status_code == 200 else {"messages": [], "trace": [f"HTTP {r.status_code}"]}
    return {"id": case["id"], "kind": case.get("kind"), "message": case["message"], "called": called(turn),
            "trace": turn.get("trace", []), **score(case, turn)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path, help="write the snapshot here instead of baselines/agent_candidate.json")
    ap.add_argument("--only", help="comma-separated case ids (writes no snapshot)")
    args = ap.parse_args()
    sys.path.insert(0, str(HERE.parent))
    from config import Settings
    from evals.run import Api

    s = Settings()
    api = Api(os.environ.get("EVAL_API") or s.eval_api or "http://127.0.0.1:8000")
    api.login(s.jwt_secret)
    pid = setup(api)
    cases = load_cases()
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only.split(","))]
    with ThreadPoolExecutor(WORKERS) as pool:
        rows = list(pool.map(lambda c: run_case(api, pid, c), cases))
    for r in rows:
        mark = "ok  " if r["routing"] else ("FORBIDDEN" if not r["forbidden"] else "MISS")
        print(f"  {r['id']:4} {r['kind'] or '':11} {mark:9} called={r['called']}")
    report = BASELINES / "runs" / f"agent-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"cases": rows}, indent=1), encoding="utf-8")
    if not args.only:
        snap = snapshot(rows, args.label, {"judge_model": None, "app_models": {
            "model": s.openai_model, "fast": s.openai_fast_model, "guard": s.openai_guard_model}})
        path = args.out or BASELINES / "agent_candidate.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snap, indent=2, sort_keys=True), encoding="utf-8")
        for k, v in sorted(snap["metrics"].items()):
            print(f"  {k:28} {v:.3f}" if isinstance(v, float) else f"  {k:28} {v}")
        print(f"  -> {path}")
    print(f"  report: {report}")


if __name__ == "__main__":
    main()
