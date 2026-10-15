# CLAUDE.md — MaxRIM

AI research assistant for healthcare researchers: project-scoped PDF ingestion → retrieval → cited, verified reports.
Full target design lives in `README.md`. This file is how we actually build it.

## Build order (do not skip ahead)

1. **Ingest** — `ingest.py`: PDF → Docling `DocumentConverter` → `HybridChunker` → OpenAI embeddings → Qdrant Cloud upsert. Hash-skip duplicates.
2. **Ask** — `ask.py`: project-filtered Qdrant search → LLM answer citing chunk IDs → print filename + page.
3. **Verify** — Planner / Researcher / Synthesizer / Verifier as plain functions; move to LangGraph once the loop is real.
4. **Eval** — ~20 gold questions in JSON + pytest (recall@k, false-claim catch rate).
5. **API** — FastAPI + Postgres + auth wraps the functions from 1–4.

Current step: **1**. Update this line when a step's exit check passes.

**Deferred until needed:** Celery/Redis (use FastAPI `BackgroundTasks`), S3 (local `data/`), MLflow (JSON results), web search, frontend.

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
uv run pytest -m live        # hits Qdrant Cloud + OpenAI (costs money)
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
