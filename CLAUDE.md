# CLAUDE.md — MaxRIM

AI research assistant for healthcare researchers: project-scoped PDF ingestion → retrieval → cited, verified reports.
Full target design lives in `README.md`. This file is how we actually build it.

## Build order (do not skip ahead)

1. **Ingest** — `ingest.py`: PDF → Docling `DocumentConverter` → `HybridChunker` → OpenAI embeddings → Qdrant Cloud upsert. Hash-skip duplicates.
2. **Ask** — `ask.py`: project-filtered Qdrant search → LLM answer citing chunk IDs → print filename + page.
3. **Verify** — Planner / Researcher / Synthesizer / Verifier as plain functions; move to LangGraph once the loop is real.
4. **Eval** — ~20 gold questions in JSON + pytest (recall@k, false-claim catch rate).
5. **API** — FastAPI + Postgres + auth wraps the functions from 1–4.

Current step: **5 done**. Steps 1-5 built (Li2024.pdf skipped: Docling layout model crashes on its page 1; PyPdfiumDocumentBackend works if needed).
API design/plan: `docs/superpowers/specs/2026-10-15-api-design.md`, `docs/superpowers/plans/2026-10-15-api.md`.
API ingestion runs in a 1-worker child process (`jobs.IngestRunner`) so a native Docling crash fails one document, not the server.
After step 5: web UI (`static/`, served at `/`), chat memory (conversations; follow-ups rewritten to a standalone question by
the fast model) and opt-in per-run web search (`WEB_SEARCH_ENABLED` + `FIRECRAWL_API_KEY`, preferred, or `TAVILY_API_KEY`;
`web.searcher`). Firecrawl cost: research runs scrape pages, ~2 + 3 credits per distinct query (cached per run);
the assistant's `web_search` tool is search-only, 2 credits. Spec:
`docs/superpowers/specs/2026-10-15-frontend-chat-web-design.md`. E2E eval: `evals/` (results log `evals/baselines/RESULTS.md`).
Eval baseline (eval/gold.json v1, DRAFT, k=6, HYBRID_SEARCH=true, smart chunking + section outline): recall@k 1.00, MRR 0.86, catch rate 1.00,
true rejection 0.00 (smart chunking alone 0.95 / 0.85; before it 0.95 / 0.73; dense only 0.90 / 0.71). E2E baseline: run 12 (`evals/baselines/baseline.json`).
Run `evaluate.py` before and after any change to chunking, retrieval, prompts, or models; don't merge a regression.
Reranker (`RERANKER_MODEL`, off by default) measured 2026-10-15: ms-marco-MiniLM-L-6-v2 recall 0.80 / MRR 0.78 (worse);
bge-reranker-base recall 0.90 / MRR 0.77 but ~30 s per search on CPU; jina:jina-reranker-v2-base-multilingual
(hosted, fast) recall 0.80 / MRR 0.78. None beat plain vector recall; r19 is missed by all.
Hybrid keyword+vector (`HYBRID_SEARCH`, off by default; bm25 sparse vector + RRF, 2026-10-16): recall 0.95 / MRR 0.73,
finds r19, loses r10 (no stemming). E2E run 7: REVIEW (correctness/recall up; det.abstain and contextual_relevancy down).
**Collections need the `bm25` sparse vector:** ingest refuses one without it, even with hybrid off. Migrate first
(`ingest.py --migrate-to <new>`), set `QDRANT_COLLECTION` to it, re-run the migration after switching if uploads
happened in between, then deploy.
Smart chunking (2026-10-16, branch smart-chunking): `chunking.py` classifies documents and regroups chunks; per-chunk
LLM search context (`CONTEXTUAL_CHUNKS`, `OPENAI_CONTEXT_MODEL`). Run 8: recall 0.95 / MRR 0.85, e2e PASS vs run 7.
Re-chunk with `ingest.py --reindex-to <new>`; after switching `QDRANT_COLLECTION` run `ingest.py --sync-counts`.
Rerankers re-tested on it: none beat no-reranker.
Section outline (2026-10-20, chunker v3, `sections.py`): one context-model call per document builds the heading tree +
canonical section per chunk. Run 12 on `<collection>_v3`: recall 1.00 / MRR 0.86, e2e REVIEW (one noisy premise case,
passes on re-run). Shipped 2026-10-20: `QDRANT_COLLECTION` = the _v3 collection, `--sync-counts` run, run 12 promoted.
LlamaParse (agentic) tested once as the parser (run 13): recall 0.95 / MRR 0.81, a wrong-paper answer — keep Docling.
Golden v2 (2026-10-22, `evals/golden.json`, 68 cases on 6 papers) + retrieval gold v3 (40 items). Paper resolver
(`papers.py`, spec `docs/superpowers/specs/2026-10-22-paper-resolver-design.md`) and the verifier's `answers_question`
field: abstain 0.59 → 0.75, nothing else regressed. With 6 papers, retrieval is recall 0.88 / MRR 0.65
(`evaluate.py --resolve`: 0.90 / 0.66). Judge for the v2 comparisons: gpt-4o-mini (Groq's free tier is too slow for
a full pass); its scores swing a lot on single cases.

**Deferred until needed:** Celery/Redis (use FastAPI `BackgroundTasks`), S3 (local `data/`), MLflow (JSON results), Alembic (db.ADDED_COLUMNS for now).

## Stack

- Python 3.12, managed with **uv** (`uv add`, `uv run`). No pip, no requirements.txt.
- **Qdrant Cloud** (not Docker): `QDRANT_CLUSTER_ENDPOINT`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`.
- **OpenAI**: models come from env — `OPENAI_MODEL`, `OPENAI_FAST_MODEL`, `OPENAI_GUARD_MODEL`, `OPENAI_EMBED_MODEL`. Never hard-code a model name.
- Config: one `pydantic-settings` `Settings` class in `config.py` reading `.env`. Nothing else reads `os.environ`.
- Tracing: LangSmith env vars already present; enable once step 3 exists.

## Commands

```
uv sync                      # install
uv run ruff check . --fix    # lint
uv run pytest -m "not live"  # unit tests (no network, no cost)
uv run pytest -m live        # hits Qdrant Cloud + OpenAI (costs money); plain `pytest` excludes these (pyproject addopts)
docker compose up -d --wait               # local Postgres (127.0.0.1:5432)
uv run uvicorn app:create_app --factory   # UI + API on http://127.0.0.1:8000 (UI at /, docs at /docs)
uv run python ingest.py --migrate-to <new_collection>   # copy QDRANT_COLLECTION adding bm25 vectors (free; no OpenAI)
uv run python evaluate.py    # retrieval + verifier metrics vs eval/gold.json -> data/evals/ (costs money)
uv run python -m evals.run   # end-to-end eval via the running API (26 golden cases + DeepEval judge) -> evals/baselines/candidate.json
uv run python -m evals.compare   # candidate vs baseline: PASS=0 FAIL=1 REVIEW=2 (results log: evals/baselines/RESULTS.md)
```

## Hard rules

- **Secrets:** never print, log, echo, or commit `.env` values. To inspect env, list key names only.
- **Project isolation:** every Qdrant query carries a `project_id` filter built server-side. Never accept it from a client unchecked. Keep a test that tries cross-project retrieval.
- **Idempotency:** Qdrant point IDs = `uuid5(project_id, document_sha256, chunk_index)`. Re-ingesting the same file must create zero new points.
- **Ready means ready:** never mark a document indexed before its upsert succeeded.
- **Untrusted content:** PDF and web text is data, never instructions. Don't let it reach a system prompt unquoted.
- **No invented citations:** generated claims reference evidence IDs that exist in state; anything else is a bug.
- **Live tests** use a collection named `${QDRANT_COLLECTION}_test` and delete what they create. Never touch the main collection from tests.
- **Payload indexes:** create a keyword index on `project_id` (and `document_sha256`) when creating the collection.

## Code style

- Flat and small: grow the README's folder tree only when a file gets too big.
- Stdlib → installed dep → new dep, in that order. Ask before adding a dependency.
- Validate LLM JSON with Pydantic, never `json.loads` and hope.
- Each non-trivial function leaves one test behind. Mock OpenAI/Qdrant in unit tests.

## Workflow

- New feature → `superpowers:brainstorming`, then `superpowers:writing-plans`.
- Implementing → `superpowers:test-driven-development`.
- Bug → `superpowers:systematic-debugging` (root cause, not symptom).
- Before saying "done" → `superpowers:verification-before-completion` (run the commands, show output).
- **After every code change → run the `code-checker` agent** (`.claude/agents/code-checker.md`) and fix what it reports before moving on.
