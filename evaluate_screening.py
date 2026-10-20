"""Screening evaluation against the user's blind labels (the gold set).

  uv run python evaluate_screening.py export --project <id>   # free: blind labels + criteria -> eval/screening_gold.json
  uv run python evaluate_screening.py run                     # paid: re-screen the gold papers with the current model

Primary metric: recall on includes = gold includes the AI does not exclude (include or review; review is a human's
pile, so it is kept). Secondary: precision of the AI's include, review-pile share, agreement. Run before and after any
screening prompt / model / threshold change; results go to data/evals/screening_<ts>.json."""

import argparse
import json
import sys
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

import db
import funnel
import screening as sc

ROOT = Path(__file__).parent
GOLD = ROOT / "eval" / "screening_gold.json"
OUT_DIR = ROOT / "data" / "evals"
PAPER_FIELDS = ("id", "title", "abstract", "year", "journal", "doi", "pmid", "language", "pub_types")


def export(sessions, project_id: uuid.UUID) -> dict:
    """The project's blind-labelled papers (with the fields screening reads) and its latest criteria."""
    with sessions() as s:
        cs = funnel.criteria_set(s, project_id)
        rows = s.scalars(select(db.ScreeningDecision).where(
            db.ScreeningDecision.project_id == project_id, db.ScreeningDecision.test.is_(False))
            .order_by(db.ScreeningDecision.criteria_version))
        newest = {d.paper_id: d.blind_label for d in rows}  # each paper's newest row wins, even when it cleared the label
        sample = {x.paper_id: x for x in s.scalars(select(db.LabelSample).where(db.LabelSample.project_id == project_id))}
        latest = funnel.latest_version(s, project_id)
        papers = []
        for pid, label in newest.items():
            p = s.get(db.Paper, pid) if label else None
            if p is not None and p.superseded_by is None:  # screening skips preprints replaced by their article
                out = funnel.paper_out(p)
                x = sample.get(pid)
                papers.append({**{k: out[k] for k in PAPER_FIELDS}, "label": label, "stratum": x.stratum if x else None,
                               "stratum_size": x.stratum_size if x else None})
        labelled: dict[str, int] = {}  # weights divide by the papers actually labelled per pile, not those drawn
        for p in papers:
            if p["stratum"]:
                labelled[p["stratum"]] = labelled.get(p["stratum"], 0) + 1
        for p in papers:  # with a sample, only sampled papers enter the weighted estimate (outside labels would
            size = p.pop("stratum_size")  # count twice: a sampled paper already stands for its whole pile)
            p["weight"] = round(size / labelled[p["stratum"]], 4) if p["stratum"] else (None if sample else 1.0)
        info = {"size": len(sample), "labelled": sum(labelled.values()),
                "stale": any(x.criteria_version != latest for x in sample.values())} if sample else None
        return {"project_id": str(project_id), "criteria_version": cs.version if cs else None,
                "criteria": cs.criteria if cs else [], "exported_at": datetime.now(UTC).isoformat(), "papers": papers,
                "sample": info}


def weighted_recall(papers: list[dict], decisions: list[str]) -> float | None:
    """Recall on includes with each labelled paper standing for its sampled pile (weight = pile size / sampled), so a
    stratified sample estimates the whole project's recall."""
    w = [p.get("weight", 1.0) for p in papers]
    inc = [i for i, p in enumerate(papers) if p["label"] == "include" and w[i] is not None]  # None: outside the sample
    total = sum(w[i] for i in inc)
    return round(sum(w[i] for i in inc if decisions[i] != "exclude") / total, 3) if total else None


def evaluate(gold: dict, parse: sc.Parse) -> dict:
    criteria = [sc.Criterion(**c) for c in gold["criteria"]]
    papers = gold["papers"]
    verdicts = sc.screen_many(parse, papers, criteria)
    decisions = [sc.decide(criteria, vs, bool(p.get("abstract"))) for p, vs in zip(papers, verdicts, strict=True)]
    includes = [i for i, p in enumerate(papers) if p["label"] == "include"]
    kept = [i for i in includes if decisions[i] != "exclude"]
    ai_inc = [i for i, d in enumerate(decisions) if d == "include"]
    decided = [i for i, d in enumerate(decisions) if d != "review"]
    n = len(papers)
    metrics = {
        "n": n, "includes": len(includes), "includes_kept": len(kept),
        "recall_includes": round(len(kept) / len(includes), 3) if includes else None,
        "recall_includes_weighted": weighted_recall(papers, decisions),  # the project-wide estimate from a sample
        "ai_include": len(ai_inc),
        "precision_include": round(sum(papers[i]["label"] == "include" for i in ai_inc) / len(ai_inc), 3) if ai_inc else None,
        "review_share": round(decisions.count("review") / n, 3) if n else None,
        "agreement_decided": round(sum(decisions[i] == papers[i]["label"] for i in decided) / len(decided), 3)
        if decided else None,
        "ai_decisions": dict(Counter(decisions)),
    }
    missed = [{"id": papers[i]["id"], "title": papers[i]["title"], "verdicts": [v.model_dump() for v in verdicts[i]]}
              for i in includes if decisions[i] == "exclude"]
    return {"metrics": metrics, "missed_includes": missed}


def _sessions():
    from config import Settings
    return db.make_sessions(Settings().database_url)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="write the gold file from a project's blind labels (free)")
    ex.add_argument("--project", required=True, type=uuid.UUID)
    ex.add_argument("--out", type=Path, default=GOLD)
    ex.add_argument("--force", action="store_true", help="replace an existing gold file")
    run = sub.add_parser("run", help="re-screen the gold papers (paid: one model call per paper)")
    run.add_argument("--gold", type=Path, default=GOLD)
    args = ap.parse_args(argv)

    if args.cmd == "export":
        out = export(_sessions(), args.project)
        if not out["criteria"]:
            raise SystemExit("This project has no screening criteria: nothing to evaluate against.")
        if not out["papers"]:
            raise SystemExit("This project has no blind labels yet: label papers in the Screening view's blind mode first.")
        if args.out.exists() and not args.force:
            raise SystemExit(f"{args.out} already exists (a frozen gold set); pass --force to replace it.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        labels = Counter(p["label"] for p in out["papers"])
        print(f"{args.out}: {len(out['papers'])} labelled papers ({dict(labels)}), criteria version {out['criteria_version']}")
        return 0

    from openai import OpenAI

    from config import Settings
    cfg = Settings()

    import ask
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    client, model = OpenAI(api_key=cfg.openai_api_key), cfg.openai_screen_model or cfg.openai_fast_model
    result = evaluate(gold, lambda i, inp, fmt: ask.parse_llm(client, model, i, inp, fmt))
    result |= {"model": model, "gold": str(args.gold), "criteria_version": gold.get("criteria_version"),
               "at": datetime.now(UTC).isoformat()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"screening_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=1))
    print(f"{len(result['missed_includes'])} missed includes; full result: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
