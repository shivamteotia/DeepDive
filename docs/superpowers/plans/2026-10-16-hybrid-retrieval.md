# Hybrid Keyword + Vector Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a BM25-style keyword (sparse) vector next to the dense embedding, fused server-side by Qdrant RRF behind `HYBRID_SEARCH`, so exact-term evidence (gold r19 / e2e a11) is retrieved.

**Architecture:** `ingest.sparse_vector()` (stdlib: regex tokens → crc32 indices → BM25 TF weights, b=0) feeds a sparse vector named `bm25` with Qdrant's IDF modifier. New collections get both vectors; old ones are copied by `ingest.migrate()` (no Docling, no OpenAI). `ask.search(..., hybrid=True)` runs a dense + sparse prefetch, both carrying the server-built project filter, fused by RRF.

**Tech Stack:** Python 3.12, uv, qdrant-client 1.19.1 (local `:memory:` mode in unit tests), pytest, ruff (line length 130).

**Spec:** `docs/superpowers/specs/2026-10-16-hybrid-retrieval-design.md`

## Global Constraints

- No new dependencies. Stdlib only (`re`, `zlib`, `collections.Counter`) on top of what is installed.
- Never hard-code a model name; config only via `config.Settings` (nothing else reads `os.environ`).
- Every Qdrant query carries the server-built `project_id` filter — on **both** prefetches and the outer query.
- Point IDs stay `uuid5(project_id, document_sha256, chunk_index)` (`ingest.point_id`); re-ingesting creates zero new points.
- Never mark a document ready before its upsert succeeded (unchanged delete-then-upsert sequence).
- Constants: `SPARSE_VECTOR = "bm25"`, `BM25_K1 = 1.2`, `HYBRID_CANDIDATES = 20`, `MIGRATE_BATCH = 256`.
- Token regex exactly: `[a-z0-9]+(?:[./,][a-z0-9]+)*` on lowercased text.
- `hybrid_search: bool = False` — off by default.
- Live tests only touch `${QDRANT_COLLECTION}_test`. Unit tests: `uv run pytest -m "not live"`.
- After every code change run the `code-checker` agent and fix what it reports (CLAUDE.md).
- Never print `.env` values.

## Review Focus

- `HYBRID_SEARCH=true` before migrating → a clear `ValueError` naming `--migrate-to`, not Qdrant's raw error (test in Task 4).
- Question with no keyword tokens (e.g. `"?"`) and hybrid on → silent fallback to dense, still returns results (test in Task 4).
- Re-running the migration after new documents were ingested into the destination → succeeds, no false count failure (test in Task 3).
- Mixed case, ratios, decimals and trailing punctuation (`PaO2/FiO2`, `21.32%,`) tokenize identically in chunk and question (test in Task 1).
- Reranker and hybrid both on → fused candidates are reranked and truncated to k (test in Task 4).

---

### Task 1: `sparse_vector`

**Files:**
- Modify: `ingest.py` (imports at top; new constants and function after `embed`, ~line 85)
- Test: `tests/test_ingest.py` (append)

**Interfaces:**
- Produces: `ingest.SPARSE_VECTOR: str = "bm25"`, `ingest.BM25_K1 = 1.2`,
  `ingest.sparse_vector(text: str, *, query: bool = False) -> models.SparseVector`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_ingest.py` (add `import zlib` to its imports):

```python
def weights(v):
    return dict(zip(v.indices, v.values, strict=True))


def test_sparse_vector_is_deterministic_and_keeps_ratios_and_decimals_whole():
    text = "PaO2/FiO2 was 21.32%, values, values."
    v = ingest.sparse_vector(text)
    assert v == ingest.sparse_vector(text)
    w = weights(v)
    for tok in (b"pao2/fio2", b"21.32", b"was", b"values"):
        assert zlib.crc32(tok) in w
    assert zlib.crc32(b"pao2") not in w and zlib.crc32(b"21") not in w
    assert weights(ingest.sparse_vector("pao2/fio2 21.32", query=True)).keys() <= w.keys()  # question matches chunk


def test_sparse_vector_doc_weights_saturate_and_query_weights_are_one():
    doc = weights(ingest.sparse_vector("missing missing values"))
    assert doc[zlib.crc32(b"missing")] == pytest.approx(2 * 2.2 / 3.2)
    assert doc[zlib.crc32(b"values")] == pytest.approx(1.0)
    assert set(weights(ingest.sparse_vector("missing missing values", query=True)).values()) == {1.0}


def test_sparse_vector_of_text_without_tokens_is_empty():
    for text in ("", "   ", "?!%"):
        v = ingest.sparse_vector(text)
        assert v.indices == [] and v.values == []


def test_sparse_vector_sums_colliding_indices(monkeypatch):
    monkeypatch.setattr(ingest.zlib, "crc32", lambda b: 7)
    v = ingest.sparse_vector("alpha beta", query=True)
    assert v.indices == [7] and v.values == [2.0]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_ingest.py -k sparse_vector -v`
Expected: FAIL — `AttributeError: module 'ingest' has no attribute 'sparse_vector'`

- [ ] **Step 3: Implement** — in `ingest.py` add to the stdlib imports `import re`, `import zlib`, `from collections import Counter`; after the existing constants add:

```python
SPARSE_VECTOR = "bm25"
BM25_K1 = 1.2
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./,][a-z0-9]+)*")
```

and after `embed`:

```python
def sparse_vector(text: str, *, query: bool = False) -> models.SparseVector:
    """Keyword vector for the IDF-weighted sparse index: BM25 term-frequency saturation with b=0 (chunks are
    near-uniform in length); Qdrant applies IDF. Query terms weigh 1.0. Colliding crc32 indices are summed."""
    weights: dict[int, float] = {}
    for token, tf in Counter(TOKEN_RE.findall(text.lower())).items():
        i = zlib.crc32(token.encode())
        weights[i] = weights.get(i, 0.0) + (1.0 if query else tf * (BM25_K1 + 1) / (tf + BM25_K1))
    return models.SparseVector(indices=list(weights), values=list(weights.values()))
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: all PASS

- [ ] **Step 5: Lint, code-checker, commit**

```bash
uv run ruff check . --fix
git add ingest.py tests/test_ingest.py
git commit -m "Add stdlib BM25 sparse_vector for keyword retrieval"
```

---

### Task 2: Collection schema + ingest writes both vectors

**Files:**
- Modify: `ingest.py` — `ensure_collection` (~line 86), point construction in `ingest` (~line 113)
- Test: `tests/test_ingest.py` (append)

**Interfaces:**
- Consumes: `sparse_vector`, `SPARSE_VECTOR` (Task 1)
- Produces: `ingest.has_sparse(qdrant: QdrantClient, collection: str) -> bool`;
  `ingest.MIGRATE_HINT: str` (error text containing `--migrate-to`, reused by Task 4);
  `ensure_collection` creates `sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)}` and raises
  `ValueError(MIGRATE_HINT...)` for an existing collection without `bm25`. Points are written with
  `vector={"": dense, "bm25": sparse}`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_ingest.py`:

```python
def test_new_collection_has_keyword_vector_and_points_carry_it(env):
    env.run("projA")
    assert ingest.has_sparse(env.q, COLL)
    for p in env.q.scroll(COLL, with_vectors=True, limit=10)[0]:
        assert set(p.vector) == {"", ingest.SPARSE_VECTOR}
        assert weights(p.vector[ingest.SPARSE_VECTOR]) == pytest.approx(weights(ingest.sparse_vector(p.payload["text"])))


def test_ingest_into_collection_without_keyword_vector_fails_before_writing(env):
    env.q.create_collection(COLL, vectors_config=ingest.models.VectorParams(size=3, distance=ingest.models.Distance.COSINE))
    with pytest.raises(ValueError, match="--migrate-to"):
        env.run("projA")
    assert count(env.q) == 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_ingest.py -k "keyword_vector" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'has_sparse'` and no `ValueError` raised.

- [ ] **Step 3: Implement** — replace `ensure_collection` in `ingest.py`:

```python
MIGRATE_HINT = "run `uv run python ingest.py --migrate-to <new_collection>` and set QDRANT_COLLECTION to it"


def has_sparse(qdrant: QdrantClient, collection: str) -> bool:
    return SPARSE_VECTOR in (qdrant.get_collection(collection).config.params.sparse_vectors or {})


def ensure_collection(qdrant: QdrantClient, collection: str, dim: int) -> None:
    if not qdrant.collection_exists(collection):
        qdrant.create_collection(
            collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
    elif not has_sparse(qdrant, collection):
        raise ValueError(f"collection {collection!r} has no {SPARSE_VECTOR!r} keyword vector; {MIGRATE_HINT}")
    for field in ("project_id", "document_sha256"):
        qdrant.create_payload_index(collection, field, models.PayloadSchemaType.KEYWORD)
```

and in `ingest()` change the point's `vector=vec,` to:

```python
            vector={"": vec, SPARSE_VECTOR: sparse_vector(c["text"])},
```

(`ensure_collection` already runs before the delete/upsert, so the schema error leaves nothing written.)

- [ ] **Step 4: Run the whole ingest + API suites** (API tests ingest through `jobs`)

Run: `uv run pytest tests/test_ingest.py tests/test_jobs.py tests/test_api_documents.py -v`
Expected: all PASS, including `test_reingest_same_file_adds_nothing` (idempotency unchanged).

- [ ] **Step 5: Lint, code-checker, commit**

```bash
uv run ruff check . --fix
git add ingest.py tests/test_ingest.py
git commit -m "Create collections with a bm25 sparse vector and write it on ingest"
```

---

### Task 3: Migration (`ingest.py --migrate-to`)

**Files:**
- Modify: `ingest.py` — new `migrate()` after `ingest()`; `main()` CLI; module docstring usage line
- Test: `tests/test_ingest.py` (append)

**Interfaces:**
- Consumes: `ensure_collection`, `sparse_vector`, `SPARSE_VECTOR` (Tasks 1–2)
- Produces: `ingest.migrate(qdrant: QdrantClient, src: str, dst: str) -> int` (points copied); `MIGRATE_BATCH = 256`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_ingest.py`:

```python
def old_collection(q, name="old", n=300):  # n > MIGRATE_BATCH exercises paging
    q.create_collection(name, vectors_config=ingest.models.VectorParams(size=3, distance=ingest.models.Distance.COSINE))
    q.upsert(name, [
        ingest.models.PointStruct(id=ingest.point_id("projA", "sha", i), vector=[1.0, float(i), 0.0], payload={
            "project_id": "projA", "document_sha256": "sha", "chunk_index": i, "text": f"missing values chunk {i}"})
        for i in range(n)
    ])


def test_migrate_copies_points_and_adds_keyword_vector():
    q = QdrantClient(":memory:")
    old_collection(q)
    assert ingest.migrate(q, "old", "new") == 300
    assert q.count("new", exact=True).count == 300
    src = {p.id: p for p in q.scroll("old", limit=500, with_vectors=True)[0]}
    for p in q.scroll("new", limit=500, with_vectors=True)[0]:
        assert p.payload == src[p.id].payload
        assert p.vector[""] == pytest.approx(src[p.id].vector)
        assert weights(p.vector[ingest.SPARSE_VECTOR]) == pytest.approx(weights(ingest.sparse_vector(p.payload["text"])))
    assert not ingest.has_sparse(q, "old") and q.count("old", exact=True).count == 300  # source untouched


def test_migrate_rerun_adds_nothing_and_tolerates_newer_points():
    q = QdrantClient(":memory:")
    old_collection(q)
    ingest.migrate(q, "old", "new")
    q.upsert("new", [ingest.models.PointStruct(  # a document ingested into the new collection after migrating
        id=ingest.point_id("projA", "sha2", 0), payload={"project_id": "projA", "text": "new doc"},
        vector={"": [0.0, 1.0, 0.0], ingest.SPARSE_VECTOR: ingest.sparse_vector("new doc")})])
    assert ingest.migrate(q, "old", "new") == 300
    assert q.count("new", exact=True).count == 301


def test_migrate_refuses_same_collection():
    with pytest.raises(ValueError, match="differ"):
        ingest.migrate(QdrantClient(":memory:"), "c", "c")
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_ingest.py -k migrate -v`
Expected: FAIL — `AttributeError: module 'ingest' has no attribute 'migrate'`

- [ ] **Step 3: Implement** — add `MIGRATE_BATCH = 256` to the constants, and after `ingest()`:

```python
def migrate(qdrant: QdrantClient, src: str, dst: str) -> int:
    """Copy every point of src into dst with the same id, payload and dense vector, adding the bm25 keyword vector.
    No Docling, no OpenAI; src is only read. Safe to re-run (same ids overwrite)."""
    if src == dst:
        raise ValueError("source and destination collections must differ")
    ensure_collection(qdrant, dst, qdrant.get_collection(src).config.params.vectors.size)
    copied, offset = 0, None
    while True:
        points, offset = qdrant.scroll(src, limit=MIGRATE_BATCH, offset=offset, with_payload=True, with_vectors=True)
        if points:
            qdrant.upsert(dst, [
                models.PointStruct(id=p.id, payload=p.payload, vector={
                    "": p.vector[""] if isinstance(p.vector, dict) else p.vector,
                    SPARSE_VECTOR: sparse_vector(p.payload.get("text", "")),
                })
                for p in points
            ], wait=True)
            copied += len(points)
        if offset is None:
            break
    src_n, dst_n = qdrant.count(src, exact=True).count, qdrant.count(dst, exact=True).count
    if dst_n < src_n:
        raise RuntimeError(f"migration incomplete: {dst} has {dst_n} points, {src} has {src_n}")
    return copied
```

Replace `main()`:

```python
def main() -> None:
    from openai import OpenAI

    from config import Settings

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("project_id", nargs="?")
    ap.add_argument("pdf", type=Path, nargs="?")
    ap.add_argument("--migrate-to", metavar="COLLECTION",
                    help="copy QDRANT_COLLECTION into COLLECTION with keyword vectors added, then exit")
    args = ap.parse_args()

    s = Settings()
    qdrant = QdrantClient(url=s.qdrant_cluster_endpoint, api_key=s.qdrant_api_key)
    if args.migrate_to:
        n = migrate(qdrant, s.qdrant_collection, args.migrate_to)
        print(f"copied {n} points {s.qdrant_collection} -> {args.migrate_to}; set QDRANT_COLLECTION={args.migrate_to} in .env")
        return
    if args.project_id is None or args.pdf is None:
        ap.error("project_id and pdf are required unless --migrate-to is given")
    n = ingest(
        args.project_id, args.pdf,
        qdrant=qdrant,
        openai_client=OpenAI(api_key=s.openai_api_key),
        collection=s.qdrant_collection,
        embed_model=s.openai_embed_model,
    )
    print(f"{args.pdf.name}: " + (f"indexed {n} chunks" if n else "already indexed, skipped"))
```

Docstring usage becomes:

```
Usage: uv run python ingest.py <project_id> <pdf>
       uv run python ingest.py --migrate-to <new_collection>   # add keyword vectors to an existing collection
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_ingest.py -v` and `uv run python ingest.py --help` (no network: argparse exits before Settings)
Expected: all PASS; help lists `--migrate-to COLLECTION`.

- [ ] **Step 5: Lint, code-checker, commit**

```bash
uv run ruff check . --fix
git add ingest.py tests/test_ingest.py
git commit -m "Add --migrate-to: copy a collection adding bm25 keyword vectors"
```

---

### Task 4: Hybrid search in `ask.search`

**Files:**
- Modify: `ask.py` — import line 13, `search()` lines 59–86, new constant next to `RERANK_CANDIDATES`
- Test: `tests/test_ask.py` (append; add `import ingest`)

**Interfaces:**
- Consumes: `ingest.sparse_vector`, `ingest.SPARSE_VECTOR`, `ingest.has_sparse`, `ingest.MIGRATE_HINT`, `ingest.ensure_collection`
- Produces: `ask.search(..., hybrid: bool = False)` (keyword-only, default False); `ask.HYBRID_CANDIDATES = 20`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_ask.py`:

```python
QUESTION = "How were missing values handled?"


@pytest.fixture
def hq():
    """Dense ranking is deliberately wrong: the question embeds to [1,0,0], the answer chunk (3) points elsewhere."""
    q = QdrantClient(":memory:")
    ingest.ensure_collection(q, COLL, 3)

    def pt(i, project, dense, text):
        return models.PointStruct(id=i, payload=payload(project, f"{project}.pdf", [i], text),
                                  vector={"": dense, ingest.SPARSE_VECTOR: ingest.sparse_vector(text)})

    q.upsert(COLL, [
        pt(1, "projA", [1, 0, 0], "background on pneumonia cohorts"),
        pt(2, "projA", [1, 0.1, 0], "baseline characteristics of patients"),
        pt(3, "projA", [0, 1, 0], "variables with missing values greater than 20% were excluded"),
        pt(4, "projB", [1, 0, 0], "secret: missing values handled by imputation"),
    ])
    return q


def hsearch(q, project, k=2, question=QUESTION, **kw):
    return ask.search(q, FakeOpenAI(), collection=COLL, embed_model="m", project_id=project, question=question, k=k, **kw)


def ids(hits):
    return [h["point_id"] for h in hits]


def test_hybrid_finds_exact_term_chunk_that_dense_ranks_below_k(hq):
    assert "3" not in ids(hsearch(hq, "projA"))
    assert "3" in ids(hsearch(hq, "projA", hybrid=True))


def test_hybrid_never_crosses_projects(hq):
    hits = hsearch(hq, "projA", k=8, hybrid=True)
    assert {h["project_id"] for h in hits} == {"projA"} and "4" not in ids(hits)
    assert ids(hsearch(hq, "projB", k=8, hybrid=True)) == ["4"]


def test_hybrid_question_without_keywords_falls_back_to_dense(hq):
    assert set(ids(hsearch(hq, "projA", question="?", hybrid=True))) == {"1", "2"}


def test_hybrid_on_unmigrated_collection_names_the_migration(q):
    with pytest.raises(ValueError, match="--migrate-to"):
        hsearch(q, "projA", hybrid=True)


def test_hybrid_candidates_are_reranked_to_k(hq, monkeypatch):
    seen = {}

    def fake_scores(model, query, texts, **kw):
        seen["n"] = len(texts)
        return [float("variables" in t) for t in texts]

    monkeypatch.setattr(ask, "rerank_scores", fake_scores)
    assert ids(hsearch(hq, "projA", k=1, hybrid=True, rerank_model="fake")) == ["3"]
    assert seen["n"] == 3  # all of projA's fused candidates, none of projB's
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_ask.py -k hybrid -v`
Expected: FAIL — `TypeError: search() got an unexpected keyword argument 'hybrid'`

- [ ] **Step 3: Implement** — in `ask.py` change `from ingest import embed` to `import ingest` plus `from ingest import embed` (keep `embed` name used elsewhere), add `HYBRID_CANDIDATES = 20` after `RERANK_CANDIDATES`, then change `search`:

Signature (add `hybrid` after `sha256_in`):

```python
def search(qdrant: QdrantClient, openai_client, *, collection: str, embed_model: str, project_id: str,
           question: str, k: int = 8, rerank_model: str = "", rerank_api_key: str = "",
           sha256_in: list[str] | None = None, hybrid: bool = False) -> list[dict]:
```

Add to the docstring: `With hybrid, fuses dense and bm25 keyword candidates by RRF (both filtered by project); a question with no keyword tokens falls back to dense.`

Replace the `vector = ...` / `hits = qdrant.query_points(...)` block with:

```python
    vector = embed(openai_client, embed_model, [question])[0]
    flt = models.Filter(must=must)
    limit = max(k, RERANK_CANDIDATES) if rerank_model else k
    sparse = ingest.sparse_vector(question, query=True) if hybrid else None
    if sparse is not None and sparse.indices:
        candidates = max(limit, HYBRID_CANDIDATES)
        try:
            hits = qdrant.query_points(
                collection, limit=limit, with_payload=True, query_filter=flt,
                prefetch=[models.Prefetch(query=vector, filter=flt, limit=candidates),
                          models.Prefetch(query=sparse, using=ingest.SPARSE_VECTOR, filter=flt, limit=candidates)],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
            ).points
        except Exception as e:
            if not ingest.has_sparse(qdrant, collection):
                raise ValueError(f"HYBRID_SEARCH is on but {collection!r} has no keyword vector; {ingest.MIGRATE_HINT}") from e
            raise
    else:
        hits = qdrant.query_points(collection, query=vector, limit=limit, with_payload=True, query_filter=flt).points
```

The rest (`out = ...`, rerank block, `E{i}` numbering) is unchanged.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_ask.py tests/test_research.py -v`
Expected: all PASS (old dense tests unchanged: `hybrid` defaults to False).

- [ ] **Step 5: Lint, code-checker, commit**

```bash
uv run ruff check . --fix
git add ask.py tests/test_ask.py
git commit -m "Add hybrid dense+bm25 RRF search behind a hybrid flag"
```

---

### Task 5: `HYBRID_SEARCH` setting wired into every caller

**Files:**
- Modify: `config.py` (after `reranker_model`), `ask.py` `main()` (~line 207), `research.py` `main()` (~line 237),
  `jobs.py` `search_fn` (~line 172), `evaluate.py` `main()` (~lines 107–114)
- Test: `tests/test_api_research.py` (append)

**Interfaces:**
- Consumes: `ask.search(..., hybrid=...)` (Task 4), `ingest.ensure_collection`, `ingest.sparse_vector`, `ingest.SPARSE_VECTOR`
- Produces: `Settings.hybrid_search: bool = False`

- [ ] **Step 1: Write the failing test** — append to `tests/test_api_research.py` (add `import ask`, `import ingest`,
  `from fastapi.testclient import TestClient`, `from qdrant_client import QdrantClient`, `from app import create_app`,
  `from conftest import FakeOpenAI, InlinePool, make_settings` to its imports):

```python
def test_api_research_uses_hybrid_search_when_enabled(tmp_path, runner, monkeypatch, fake_research):
    q = QdrantClient(":memory:")
    app = create_app(make_settings(hybrid_search=True), qdrant=q, openai_client=FakeOpenAI(), runner=runner,
                     upload_dir=tmp_path / "uploads", ingest_pool=InlinePool(), research_pool=InlinePool())
    calls = []
    real = ask.search
    monkeypatch.setattr(ask, "search", lambda *a, **kw: calls.append(kw) or real(*a, **kw))
    with TestClient(app) as api:
        api.maxrim = app.state.maxrim
        api.post("/auth/register", json={"email": "h@example.com", "password": "correct horse battery"})
        token = api.post("/auth/login", json={"email": "h@example.com", "password": "correct horse battery"}).json()["access_token"]
        h = {"Authorization": f"Bearer {token}"}
        a, b = project(api, h, "A"), project(api, h, "B")
        doc_a, doc_b = upload(api, h, a, PDF_A), upload(api, h, b, PDF_B)
        ingest.ensure_collection(q, "unit", 3)
        q.upsert("unit", [
            models.PointStruct(id=str(uuid.uuid4()), payload={
                "project_id": pid, "document_sha256": sha, "source_filename": name, "page_numbers": [1],
                "section_headings": [], "text": "missing values were imputed"},
                vector={"": [1, 0, 0], ingest.SPARSE_VECTOR: ingest.sparse_vector("missing values were imputed")})
            for pid, sha, name in [(a, doc_a["sha256"], "a.pdf"), (b, doc_b["sha256"], "b.pdf")]
        ])
        r = api.post(f"/api/v1/projects/{a}/research-runs", headers=h, json={"question": "missing values?"})
        assert api.get(f"/api/v1/research-runs/{r.json()['id']}", headers=h).json()["status"] == "completed"
    assert calls and all(kw["hybrid"] is True for kw in calls)
    assert {x["source_filename"] for x in fake_research["hits"]} == {"a.pdf"}  # isolation holds with hybrid on
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_api_research.py -k hybrid -v`
Expected: FAIL — `KeyError: 'hybrid'` (jobs doesn't pass it yet). If instead `make_settings(hybrid_search=True)` is rejected,
the setting is missing — also expected.

- [ ] **Step 3: Implement**

`config.py`, after `reranker_model`:

```python
    hybrid_search: bool = False  # dense + bm25 keyword search fused by RRF; needs a migrated collection (ingest.py --migrate-to)
```

`jobs.py` `search_fn`: add `hybrid=cfg.hybrid_search,` to the `ask.search(...)` call (after `sha256_in=ready`).

`ask.py` `main()` and `research.py` `main()`: add `hybrid=s.hybrid_search` to the `search(...)` / `partial(ask.search, ...)` call.

`evaluate.py` `main()`: add `hybrid=s.hybrid_search` to the `partial(ask.search, ...)` call and
`"hybrid": s.hybrid_search,` to the `result["config"]` dict (next to `"rerank_model"`), so every saved eval records it.

- [ ] **Step 4: Run the full unit suite**

Run: `uv run pytest -m "not live" -q`
Expected: all PASS (135 before this work + the new tests).

- [ ] **Step 5: Lint, code-checker, commit**

```bash
uv run ruff check . --fix
git add config.py ask.py research.py jobs.py evaluate.py tests/test_api_research.py
git commit -m "Wire HYBRID_SEARCH setting into ask, research, API jobs and evaluate"
```

---

### Task 6: Live migration and evaluation (costs money — confirm with the user before starting)

**Files:**
- Modify: `.env` (user edits `QDRANT_COLLECTION`; never print values), `evals/baselines/RESULTS.md`, `CLAUDE.md`

- [ ] **Step 1: Migrate** (no OpenAI cost; reads/writes Qdrant Cloud). Pick the new name `<current>_hybrid`:

```bash
uv run python ingest.py --migrate-to <current_collection>_hybrid
```
Expected: `copied N points ...`. Then the user sets `QDRANT_COLLECTION=<current_collection>_hybrid` in `.env`.

- [ ] **Step 2: Migration sanity eval, hybrid off** — must reproduce the baseline exactly:

```powershell
$env:HYBRID_SEARCH='false'; uv run python evaluate.py
```
Expected: recall@k 0.90, MRR 0.71 (same dense vectors). If not, stop: the migration is wrong.

- [ ] **Step 3: Hybrid eval**

```powershell
$env:HYBRID_SEARCH='true'; uv run python evaluate.py
```
Record recall@k, MRR, catch rate, true rejection, and whether r19 is retrieved (per-question rows in the `data/evals/` JSON).

- [ ] **Step 4: E2E eval only if Step 3 beats Step 2** (API must run with `HYBRID_SEARCH=true`)

```powershell
$env:HYBRID_SEARCH='true'; uv run uvicorn app:create_app --factory   # separate terminal
uv run python -m evals.run --label "hybrid-bm25"
uv run python -m evals.compare
```

- [ ] **Step 5: Log and decide** — add a "run 7" entry at the top of `evals/baselines/RESULTS.md` (numbers from Steps 2–4,
  a11 / r19 outcome). Ship rule: set `HYBRID_SEARCH=true` in `.env` only if recall@k rose, MRR did not drop, and compare
  is PASS or an explained REVIEW. Otherwise leave it off and log the negative result.
  Update `CLAUDE.md`: the reranker paragraph's "Next idea" line becomes the hybrid result; mention `--migrate-to` and
  `HYBRID_SEARCH` under Commands/Stack.

- [ ] **Step 6: Commit**

```bash
git add evals/baselines/RESULTS.md CLAUDE.md
git commit -m "Log hybrid retrieval eval (run 7)"
```
