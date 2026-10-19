"""Dedup evaluation on the reference exports in bibliography/ (free: no network, no model calls).

Ground truth: records that share a PMID are the same paper. The blind run hides DOI/PMID/PMCID so only the
title + first author + year step can pair them; it reports false merges (different PMIDs joined: must stay ~0) and
missed pairs. Usage: uv run python evaluate_dedup.py [files...]"""

import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import dedup
import references

OUT_DIR = Path(__file__).parent / "data" / "evals"


def load(paths: list[Path]) -> list[references.Record]:
    recs = []
    for p in paths:
        r, errs = references.parse_file(p.name, p.read_bytes())
        if errs:
            print(f"{p.name}: {len(errs)} parse errors, e.g. {errs[:2]}")
        recs += r
    return recs


def pairs_of(groups: list[dedup.Group]) -> set[tuple[int, int]]:
    return {(a, b) for g in groups for a, b in combinations(sorted(g.members), 2)}


def evaluate(recs: list[references.Record]) -> dict:
    full, near_full = dedup.group(recs)
    blind_recs = [r.model_copy(update={"doi": "", "pmid": "", "pmcid": ""}) for r in recs]
    blind, near_blind = dedup.group(blind_recs)
    by_pmid: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(recs):
        if r.pmid:
            by_pmid[r.pmid].append(i)
    truth = {(a, b) for idx in by_pmid.values() for a, b in combinations(sorted(idx), 2)}
    with_pmid = {i for idx in by_pmid.values() for i in idx}
    predicted = {(a, b) for a, b in pairs_of(blind) if a in with_pmid and b in with_pmid}
    false_merges = sorted(p for p in predicted if recs[p[0]].pmid != recs[p[1]].pmid)
    missed = sorted(truth - predicted)
    return {
        "records": len(recs), "papers": len(full),
        "merged_by": {r: sum(1 for g in full if g.reason == r) for r in ("doi", "pmid", "title_year_author")},
        "possible_duplicates": len(near_full),
        "blind": {"truth_pairs": len(truth), "found": len(truth & predicted), "missed": len(missed),
                  "false_merges": len(false_merges), "recall": round(len(truth & predicted) / max(len(truth), 1), 4),
                  "possible_duplicates": len(near_blind)},
        "false_merge_examples": [[recs[a].title, recs[b].title] for a, b in false_merges[:10]],
        "missed_examples": [[recs[a].title, recs[a].year, recs[a].first_author, recs[b].title, recs[b].year,
                             recs[b].first_author] for a, b in missed[:10]],
    }


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or sorted(Path("bibliography").glob("*.*"))
    result = evaluate(load(paths))
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("examples")}, indent=1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"dedup_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps({"files": [str(p) for p in paths], **result}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
