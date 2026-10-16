# Step 5 — MaxRIM API (FastAPI + Postgres + auth) — Design

Date: 2026-10-15 · Status: approved in chat, pending spec review

## Goal

Wrap the existing step 1–4 functions (`ingest.ingest`, `ask.search`, `research.run`) in an authenticated HTTP API so a
researcher can create projects, upload PDFs, watch indexing status, start verified research runs, and inspect their
evidence and verification — without using the CLI.

**Users:** single-user / local. Each account owns its projects; no sharing.

**Success:** README §10 acceptance scenario passes through the API — two projects; re-uploading the same PDF adds no
duplicate document or vectors; Project A queries never return Project B chunks; a research run completes and its
evidence and verification are inspectable; a failed ingest can be retried to a consistent state.

## Decisions

| Area | Choice | Upgrade when |
|---|---|---|
| Background work | In the API process via FastAPI `BackgroundTasks`; **ingestion runs in a 1-worker process pool** | Uploads get heavy or must survive restarts → separate worker polling a jobs table |
| Schema | SQLAlchemy 2 ORM, `create_all` on startup, no Alembic | First schema change against data that must be kept |
| Auth | Email + password; stdlib `hashlib.scrypt` hashing; HS256 JWT via PyJWT signed with `JWT_SECRET`, expiry `ACCESS_TOKEN_MINUTES` | Multiple users/orgs → external identity provider |
| Ingest isolation | Child process, so a native Docling crash (seen on `Li2024.pdf`) fails one job, not the server | — |

**Out of scope:** `/evaluate` endpoint, web search, frontend, rate limiting, document deletion,
`RESEARCH_RUN_MAX_COST_USD` enforcement, project sharing.

## Components

New top-level modules (matching the existing flat layout):

- `db.py` — SQLAlchemy models, engine/session factory, `init_db()`.
- `auth.py` — `hash_password`, `verify_password`, `create_token`, `current_user` FastAPI dependency.
- `jobs.py` — `ingest_document(doc_id)` (submits to the process pool, records outcome) and `run_research(run_id)`;
  `recover_interrupted()` for startup.
- `app.py` — FastAPI app, lifespan (init DB, create shared Qdrant/OpenAI clients, recover jobs), routes.

Also: `docker-compose.yml` (only `postgres:16`, named volume, password from `.env`), new `Settings` fields
`database_url`, `jwt_secret`, `access_token_minutes`.

New dependencies: `fastapi`, `uvicorn`, `sqlalchemy`, `psycopg[binary]`, `pyjwt`, `python-multipart`.
Column types must work on both Postgres and SQLite (`sqlalchemy.Uuid`, `sqlalchemy.JSON`) so unit tests run on SQLite.

## Data model

| Table | Columns |
|---|---|
| `users` | `id` UUID PK, `email` unique (lowercased), `password_hash`, `created_at` |
| `projects` | `id` UUID PK, `owner_id` FK users, `name`, `created_at` |
| `documents` | `id` UUID PK, `project_id` FK projects, `filename`, `sha256`, `status` ∈ {queued, processing, ready, failed}, `chunk_count` nullable, `error` nullable, `created_at`, `updated_at`; unique (`project_id`, `sha256`) |
| `research_runs` | `id` UUID PK, `project_id` FK projects, `question`, `status` ∈ {queued, running, completed, failed}, `result` JSON nullable, `error` nullable, `created_at`, `finished_at` nullable |

- Qdrant payload `project_id` = `str(projects.id)`. The CLI's `demo` data is not part of any API project.
- `research_runs.result` holds the dict returned by `research.finalize` (status, claims with verdicts, gaps, evidence,
  per-iteration log) — the audit record. Evidence entries keep `eid, point_id, source_filename, page_numbers, text`.
- Deferred tables from README §7 (`project_members`, `document_versions`, `ingestion_jobs`, `evidence_items`, `claims`,
  `verification_results`, `audit_events`) are added when something needs to query them.

## API

All `/api/v1` routes require `Authorization: Bearer <token>`; missing/invalid/expired → `401`.
A project or run not owned by the caller → `404` (never `403`; existence is not revealed).

| Method | Route | Behaviour |
|---|---|---|
| GET | `/health` | `{"status": "ok"}` |
| POST | `/auth/register` | `{email, password}`; password ≥ 12 chars; duplicate email → `409`; returns user (no hash) `201` |
| POST | `/auth/login` | `{email, password}` → `{access_token, token_type: "bearer"}`; wrong credentials → `401` (same message for unknown email and wrong password) |
| POST | `/api/v1/projects` | `{name}` → project `201` |
| GET | `/api/v1/projects` | caller's projects |
| POST | `/api/v1/projects/{id}/documents` | multipart PDF. Rejects non-`%PDF` content (`400`) and > 50 MB (`413`). Same sha256 already in project: `queued`/`processing`/`ready` → return it `200`; `failed` → reset to `queued`, re-enqueue, `202`. New → save, `queued`, enqueue, `202` |
| GET | `/api/v1/projects/{id}/documents` | documents with status, chunk_count, error |
| GET | `/api/v1/projects/{id}/documents/{doc_id}` | one document |
| POST | `/api/v1/projects/{id}/research-runs` | `{question, max_iterations=3 (1–5)}`; no `ready` document → `409`; else `queued` run, enqueue, `202` |
| GET | `/api/v1/research-runs/{id}` | run with status, question, result summary (status, claims, gaps), error |
| GET | `/api/v1/research-runs/{id}/evidence` | `result.evidence` |
| GET | `/api/v1/research-runs/{id}/verification` | per-iteration verify entries from `result.log` plus final claims with verdicts |

## Data flow

**Upload:** validate → stream to `data/uploads/{project_id}/{sha256}.pdf` → row `queued` → `BackgroundTasks`
calls `jobs.ingest_document(doc_id)` → set `processing` → submit `ingest.ingest(str(project_id), path, ...)` to the
1-worker `ProcessPoolExecutor` (child builds its own clients from `Settings`) → `ready` + `chunk_count`, or `failed` +
error. `BrokenProcessPool` (native crash) → `failed` with "PDF parser crashed", pool recreated. Retries are safe:
point IDs are deterministic and `ingest` deletes leftovers before upsert.

**Research:** `BackgroundTasks` calls `jobs.run_research(run_id)` → `running` → `research.run(...)` with
`search_fn` built server-side from the run's `project_id` (and `RERANKER_MODEL` settings) → `completed` + `result`,
or `failed` + error. Research runs execute in the API process (threads); several may run at once.

**Startup recovery:** documents in `queued`/`processing` and runs in `queued`/`running` → `failed`,
error "interrupted by server restart". Users retry by re-uploading / re-running.

**Errors:** stored as `"<ExceptionType>: <message>"` truncated to 500 chars; never include settings or keys.
Upload size enforced while streaming, not after reading the whole body into memory.

## Testing

**Unit (`pytest -m "not live"`, no network, no cost):** `TestClient` + SQLite in-memory + `QdrantClient(":memory:")` +
fake OpenAI + fake ingest function (no Docling). Covers: register/login; duplicate email; wrong password; missing,
malformed and expired token → 401; other user's project, document, run → 404; duplicate upload returns same document;
failed upload re-upload re-queues; non-PDF → 400; oversize → 413; research before any ready doc → 409; startup
recovery marks interrupted jobs failed; research search_fn only sees its project's chunks; native-crash path marks
document failed.

**Live (`pytest -m live`):** Postgres from `docker compose`, real Qdrant + OpenAI, a test collection
`${QDRANT_COLLECTION}_test` deleted afterwards: README acceptance scenario through the API. The app takes its
collection name from `Settings`, so the live test overrides `QDRANT_COLLECTION` for the app it builds; it never
writes to the main collection. Live test users/projects are created with random emails and are left in the dev DB.
