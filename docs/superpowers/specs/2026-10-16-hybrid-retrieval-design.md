# Hybrid keyword + vector retrieval — design

Date: 2026-10-16. Status: approved in chat; awaiting written-spec review.

## Goal

Add keyword (BM25-style sparse) retrieval next to the existing dense OpenAI-embedding search, fused inside Qdrant,
to fix retrieval misses where the question shares exact terms with the evidence but the embedding ranks it below k.

Motivating misses: gold `r19` (`eval/gold.json`) and e2e `a11` (`evals/golden.json`), the same question —
"How were missing values handled in the SCAP study?" — whose target chunk contains "missing values greater than 20%".
Neither plain vector search nor any of the three rerankers tried on 2026-10-15 retrieved it.

**Success:** at k=6 on `evaluate.py`, recall@k > 0.90 with MRR >= 0.71, and `evals.compare` against the run 6
baseline is PASS (or REVIEW with every regression explained). Otherwise the feature stays off and the negative
result is logged in `evals/baselines/RESULTS.md`.

## Decisions

- Sparse vectors are computed in plain Python (stdlib only). No `fastembed`, no Qdrant Cloud Inference.
- Fusion is server-side Qdrant RRF over two prefetches (dense + sparse).
- Behind a config flag, off by default, like `RERANKER_MODEL`.
- Existing points are migrated by copying to a new collection; no re-chunking, no re-embedding.

## Sparse vector (`ingest.py`)

`sparse_vector(text: str, *, query: bool = False) -> models.SparseVector`

- Tokens: `re.findall(r"[a-z0-9]+(?:[./,][a-z0-9]+)*", text.lower())`. Keeps `pao2/fio2`, `0.893`, `21.32` whole.
  No stemming, no stopwords (Qdrant's IDF modifier down-weights common terms).
- Index: `zlib.crc32(token.encode())` (stable across processes, unlike `hash()`).
- Document weight: BM25 TF saturation with no length normalization (b=0): `tf * (K1 + 1) / (tf + K1)`, `K1 = 1.2`.
  Chunks from `HybridChunker` are near-uniform in length, so b=0 avoids tracking corpus average length.
- Query weight (`query=True`): 1.0 per unique token.
- Empty or token-free text returns an empty `SparseVector(indices=[], values=[])`.
- If two tokens collide on a crc32 index, their weights are summed (no duplicate indices sent to Qdrant).

## Collection schema (`ingest.ensure_collection`)

- New collections: unnamed dense vector unchanged (`VectorParams(size=dim, distance=COSINE)`), plus
  `sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)}`. Payload indexes unchanged.
- Existing collection without a `bm25` sparse vector: raise `ValueError` naming the migration command. This
  check runs before any delete/upsert, so a document is never left half-written.

Constant: `SPARSE_VECTOR = "bm25"`.

## Ingest

Each point's vector becomes `{"": dense, "bm25": sparse_vector(chunk_text)}`. Point IDs, payload, the
delete-then-upsert sequence, hash-skip, and "ready only after upsert" are unchanged.

## Migration

`uv run python ingest.py --migrate-to <new_collection>` → `migrate(qdrant, src, dst) -> int`.

- Reads the dense dimension from `src`, calls `ensure_collection(dst, dim)`.
- Scrolls `src` in batches with `with_vectors=True, with_payload=True`; for each point upserts the same ID and
  payload into `dst` with `{"": dense, "bm25": sparse_vector(payload["text"])}`.
- No Docling, no OpenAI. Idempotent (same IDs overwrite). Returns the number of points copied and fails if
  `count(dst) != count(src)` afterwards.
- `src` is never modified. After migrating, the user sets `QDRANT_COLLECTION=<new>` in `.env`. Postgres
  document rows need no change (they don't store the collection name).
- Refuses `src == dst`.

## Config (`config.py`)

`hybrid_search: bool = False` (env `HYBRID_SEARCH`).

## Search (`ask.search`)

Signature gains `hybrid: bool = False` (search never reads settings itself, same as `rerank_model`). Each caller
that already passes `rerank_model` (`ask.main`, `research.main`, `jobs.py`, `evaluate.py`) adds
`hybrid=settings.hybrid_search` — a one-argument change per caller.

- `hybrid=False`: the current dense query against the unnamed dense vector. Behavior unchanged.
- `hybrid=True`:
  ```
  query_points(collection,
      prefetch=[Prefetch(query=dense, filter=F, limit=HYBRID_CANDIDATES),
                Prefetch(query=sparse_q, using="bm25", filter=F, limit=HYBRID_CANDIDATES)],
      query=FusionQuery(fusion=Fusion.RRF), query_filter=F,
      limit=max(k, RERANK_CANDIDATES) if rerank_model else k, with_payload=True)
  ```
  `HYBRID_CANDIDATES = 20`. `F` is the existing server-built `project_id` (+ optional `sha256_in`) filter, applied
  to both prefetches and the outer query.
- If `sparse_vector(question, query=True)` is empty, fall back to the dense query.
- Reranker, when set, reorders the fused candidates exactly as it reorders dense candidates today.
- `score` is the RRF score when hybrid is on. Nothing thresholds on `score` (checked: only `web.py`, which scores
  Tavily results, reads a score).

## Tests (unit, `-m "not live"`, in-memory `QdrantClient(":memory:")`, OpenAI mocked)

1. `sparse_vector`: deterministic; keeps `pao2/fio2` and `21.32` as single tokens; TF saturation values;
   query weights are 1.0; empty text → empty vector; no duplicate indices.
2. Hybrid search retrieves an exact-term chunk that a dense-only search ranks below k (mock embeddings make
   the dense ranking deliberately wrong).
3. **Project isolation with hybrid on:** a project-B chunk that exactly matches the query terms is never
   returned for project A (both prefetches filtered).
4. Hybrid with an all-punctuation question falls back to dense and still returns results.
5. `migrate`: copies all points with the same IDs and payload, adds `bm25`, leaves `src` unchanged; second run
   copies the same count and adds nothing; `src == dst` raises.
6. Ingest into a collection lacking `bm25` raises the migration-command error before writing.
7. Existing test still holds: re-ingesting the same file creates zero new points.

Live tests keep using `${QDRANT_COLLECTION}_test` and clean up.

## Evaluation (costs money; run in this order)

1. Migrate the main collection to a new one; switch `QDRANT_COLLECTION`.
2. `evaluate.py` with `HYBRID_SEARCH=false`: must reproduce recall@k 0.90 / MRR 0.71 (same dense vectors) —
   validates the migration.
3. `evaluate.py` with `HYBRID_SEARCH=true`: record recall@k, MRR, catch rate, and whether r19 is hit.
4. If 3 beats step 2: `evals.run` + `evals.compare` vs run 6 baseline; log in `RESULTS.md`.
5. Ship rule: enable in `.env` only if recall@k rises, MRR does not drop, and compare is PASS or an explained
   REVIEW. Otherwise leave it off and log the result.

## Out of scope

Stemming, stopwords, weighted fusion (dense vs sparse), per-request UI toggle, deleting the old collection.
