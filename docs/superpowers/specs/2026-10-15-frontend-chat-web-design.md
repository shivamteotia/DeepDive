# Frontend, chat memory and web search: design

Status: written 2026-10-15 while the user was away. They said "don't ask my permission or wait for my
permission", so the decisions below are mine and are listed for review.

## Goal

A Consensus-style web UI over the existing API: ask a question, get a cited, verified answer with source
cards. Follow-up questions in the same conversation use earlier turns (chat memory). Web search can be
switched on per question (Tavily; keys are already in `.env`).

## Decisions (and why)

| # | Decision | Why |
|---|---|---|
| D1 | No-build frontend: `static/index.html` + `app.js` + `styles.css`, plain JS, served by FastAPI at `/` | No Node toolchain; one origin, so no CORS; nothing to deploy separately |
| D2 | Chat memory = a `conversations` table; research runs get a nullable `conversation_id` | A conversation is an ordered list of runs. Small schema change |
| D3 | A follow-up is rewritten into a standalone question by the fast model, from the last 4 turns (question plus verified claims), then researched as usual | Retrieval and verification stay single-question; memory only changes what gets searched. The rewritten question is stored and shown |
| D4 | Web search is off by default, enabled per run (`web_search: true`), and only allowed when `WEB_SEARCH_ENABLED` and `TAVILY_API_KEY` are both set (else 400) | The eval's unanswerable cases assume papers only. Web results are untrusted, so users opt in |
| D5 | Web hits join the same evidence list (`source_type: "web"`, `url`, `title`; `source_filename` = page title) through the server-side `search_fn` | The same citation check, verifier and JSON-quoted evidence apply. `research.py` is unchanged |
| D6 | The new columns are added with `ALTER TABLE … ADD COLUMN IF NOT EXISTS` in `make_sessions`, Postgres only | `create_all` doesn't alter existing tables. SQLite tests start from fresh tables. Alembic stays deferred |
| D7 | JWT kept in `localStorage` | Local single-user app. Revisit before any multi-user deployment |
| D8 | `MESH_API_KEY` is unused | Its provider is unknown; Tavily is the configured web search |

## API additions

- `POST /api/v1/projects/{pid}/conversations` `{title?}` returns a conversation. `GET` lists them, newest first.
- `GET /api/v1/conversations/{cid}` returns the conversation plus its runs (oldest first). `DELETE` removes the
  conversation; its runs are kept with `conversation_id` set to NULL.
- `POST /api/v1/projects/{pid}/research-runs` takes optional `conversation_id` (must belong to the project,
  else 404) and `web_search` (bool).
- Run output adds `conversation_id`, `web_search`, `standalone_question`.
- The conversation title defaults to the first question, cut to 80 characters.

## Web search (`web.py`)

`search(query, api_key, timeout, max_results=5) -> list[dict]` POSTs to `https://api.tavily.com/search` with
Bearer auth. Each result becomes `{point_id: "web:"+sha256(url)[:32], project_id, source_type: "web", url, title,
source_filename: title or url, page_numbers: [], section_headings: [], text: content[:2000], score}`.
A failure (HTTP error or timeout) logs nothing secret and returns []: research falls back to the papers.

## Frontend

- **Login/register** screen.
- **Left sidebar:** project picker (and create), document list with upload and status, conversation list with "New chat".
- **Main pane:**
  - **Empty state:** centered search box ("Ask your papers…"), a "Web" toggle and example prompts.
  - **Thread:** each turn shows
    - the question
    - the rewritten question, if different
    - a status chip: Verified / Needs review
    - "Key findings" as numbered claims with citation chips
    - a "What the sources don't say" box for the gaps text
    - source cards (paper filename and pages, or web title, domain and link), each with a snippet
  - **Citation chips:** clicking one highlights its source card.
- **Polling:** while a run is queued or running, the UI polls every 2 s and shows "Searching… / Verifying…".

## Testing

Unit and API tests cover:
- conversation CRUD and ownership (404 across users and projects)
- the follow-up rewrite: called only when there is history; history quoted as JSON data
- the web toggle: 400 when disabled; merged evidence when enabled
- `web.search` mapping and failure handling (httpx MockTransport)
- `/` serves the UI

No browser tests. The UI was checked by hand against the running API.

## Out of scope

Streaming answers, PDF viewer, filters by year/journal (we have no metadata), sharing, multi-user hardening.
