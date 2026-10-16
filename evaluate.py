"""Evaluate retrieval (recall@k, MRR) and the verifier (false-claim catch rate, true-claim rejection rate).

Usage: uv run python evaluate.py [--gold eval/gold.json] [-k 6]
Results are saved to data/evals/ with the models and settings used.
"""

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import ask
import research

ROOT = Path(__file__).parent


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()  # whitespace-insensitive: PDF extraction spacing varies


def is_relevant(hit: dict, expected: list[dict]) -> bool:
    return any(hit["source_filename"] == e["source"] and norm(e["contains"]) in norm(hit["text"]) for e in expected)


def first_relevant_rank(hits: list[dict], expected: list[dict]) -> int | None:
    return next((i for i, h in enumerate(hits, 1) if is_relevant(h, expected)), None)


def retrieval_metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    if not n:
        return {"n": 0, "recall_at_k": None, "mrr": None}
    return {"n": n, "recall_at_k": sum(r is not None for r in ranks) / n, "mrr": sum(1 / r for r in ranks if r) / n}


def verifier_metrics(rows: list[dict]) -> dict:
    """Catch = a false claim not marked SUPPORTED. Rejection = a true claim not marked SUPPORTED.
    Only rows whose expected evidence was actually retrieved count, so retrieval misses can't inflate the catch rate."""
    scored = [r for r in rows if r["evidence_retrieved"]]
    false = [r for r in scored if r["label"] == "false"]
    true = [r for r in scored if r["label"] == "true"]
    return {
        "n_false": len(false), "n_true": len(true), "n_excluded_no_evidence": len(rows) - len(scored),
        "catch_rate": sum(r["verdict"] != "SUPPORTED" for r in false) / len(false) if false else None,
        "true_rejection_rate": sum(r["verdict"] != "SUPPORTED" for r in true) / len(true) if true else None,
    }


def evaluate(gold: dict, *, search_fn, verify_fn) -> dict:
    """search_fn(question) -> hits; verify_fn(claim, evidence) -> Verdict."""
    retrieval = []
    for item in gold["retrieval"]:
        hits = search_fn(item["question"])
        rank = first_relevant_rank(hits, item["expected"])
        retrieval.append({"id": item["id"], "rank": rank,
                          "top": [f'{h["source_filename"]} p{h["page_numbers"]}' for h in hits[:3]]})
    claims = []
    for item in gold["claims"]:
        evidence = search_fn(item["search"])
        retrieved = any(is_relevant(e, item["expected"]) for e in evidence)
        if evidence:
            v = verify_fn(ask.Claim(text=item["claim"], evidence_ids=[e["eid"] for e in evidence]), evidence)
            verdict, rationale = v.verdict, v.rationale
        else:
            verdict, rationale = "NO_EVIDENCE", "search returned nothing"
        claims.append({"id": item["id"], "label": item["label"], "evidence_retrieved": retrieved,
                       "verdict": verdict, "rationale": rationale})
    return {
        "retrieval": {**retrieval_metrics([r["rank"] for r in retrieval]), "items": retrieval},
        "verifier": {**verifier_metrics(claims), "items": claims},
    }


def fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def report(result: dict) -> str:
    r, v = result["retrieval"], result["verifier"]
    lines = [f'Retrieval  recall@k={fmt(r["recall_at_k"])}  MRR={fmt(r["mrr"])}  (n={r["n"]})']
    lines += [f'  MISS {i["id"]}: top={i["top"]}' for i in r["items"] if i["rank"] is None]
    lines.append(f'Verifier   catch_rate={fmt(v["catch_rate"])} (n={v["n_false"]})  '
                 f'true_rejection_rate={fmt(v["true_rejection_rate"])} (n={v["n_true"]})  '
                 f'excluded (evidence not retrieved)={v["n_excluded_no_evidence"]}')
    lines += [f'  NO-EVIDENCE {i["id"]}: expected passage not retrieved' for i in v["items"] if not i["evidence_retrieved"]]
    lines += [f'  WRONG {i["id"]} ({i["label"]} -> {i["verdict"]}): {i["rationale"]}' for i in v["items"]
              if i["evidence_retrieved"] and (i["label"] == "false") == (i["verdict"] == "SUPPORTED")]
    return "\n".join(lines)


def run_live(gold_path: Path, k: int, rerank_model: str | None = None) -> dict:
    """rerank_model: None = use RERANKER_MODEL from settings; "" = no reranking; otherwise that model."""
    from functools import partial

    from openai import OpenAI
    from qdrant_client import QdrantClient

    from config import Settings

    s = Settings()
    rerank_model = s.reranker_model if rerank_model is None else rerank_model
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    oa = OpenAI(api_key=s.openai_api_key)
    qdrant = QdrantClient(url=s.qdrant_cluster_endpoint, api_key=s.qdrant_api_key)
    search = partial(ask.search, qdrant, oa, collection=s.qdrant_collection, embed_model=s.openai_embed_model,
                     project_id=gold["project_id"], k=k, rerank_model=rerank_model, rerank_api_key=s.jena_ai_api_key,
                     hybrid=s.hybrid_search)
    result = evaluate(gold, search_fn=lambda q: search(question=q),
                      verify_fn=lambda c, ev: research.verify(oa, s.openai_guard_model, [c], ev)[0])
    result["config"] = {"gold": gold_path.name, "gold_version": gold.get("version"), "k": k,
                        "rerank_model": rerank_model or None, "hybrid": s.hybrid_search,
                        "embed_model": s.openai_embed_model, "guard_model": s.openai_guard_model,
                        "collection": s.qdrant_collection, "run_at": datetime.now(UTC).isoformat()}
    out = ROOT / "data" / "evals"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    result["saved_to"] = str(path)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", type=Path, default=ROOT / "eval" / "gold.json")
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--rerank-model", default=None, help='override RERANKER_MODEL; "none" disables reranking')
    args = ap.parse_args()
    rerank = "" if args.rerank_model == "none" else args.rerank_model
    result = run_live(args.gold.resolve(), args.k, rerank)
    print(f'rerank_model: {result["config"]["rerank_model"]}')
    print(report(result))
    print(f"\nSaved: {result['saved_to']}")


if __name__ == "__main__":
    main()
