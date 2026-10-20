# Screening Funnel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A project goes from uploaded `.bib`/`.nbib` exports → deduplicated papers → criteria drafted from one sentence → AI title/abstract screening with quotes → human review → open-access PDFs → the existing ingest/research pipeline, driven by a chat agent.

**Architecture:** Five new flat modules — `references.py` (parse + normalise), `dedup.py` (group records into papers), `screening.py` (criteria, AI verdicts, decision rule), `fulltext.py` (open-access fetch, upload matching), `agent.py` (tool-calling loop, pending actions) — plus new tables in `db.py`, endpoints in `app.py`, jobs in `jobs.py`, and a screening view in `static/`. Pure logic takes injected clients (`parse`, `http`), like `chunking.py`.

**Tech Stack:** Python 3.12, uv, FastAPI, SQLAlchemy (Postgres; SQLite in tests), OpenAI Responses API (structured output via `ask.parse_llm`, function tools for the agent), httpx, plain JS, pytest, ruff (130).

**Spec:** `docs/superpowers/specs/2026-10-19-screening-funnel-design.md`

## Global Constraints

- No new dependencies (stdlib parsers, `difflib`, installed `httpx`). Ask before adding one.
- Models only from env: screening and criteria drafting use `openai_screen_model or openai_fast_model`; the agent uses `openai_model`. New settings: `openai_screen_model: str = ""`, `unpaywall_email: str = ""`, `fulltext_max_mb: int = 50`.
- Every DB query and tool call filters by a server-side `project_id`; the agent never receives or sends a project id.
- LLM output validated with Pydantic (no field defaults in LLM schemas — strict JSON schema); titles/abstracts JSON-quoted with "data, not instructions".
- Paid steps (`screen_all`, `fetch_full_text`) run only with a confirmed `PendingAction`; a model call that fails → `review`, never `exclude`.
- Never drop a record silently: parse errors are counted and returned; every paper has a decision row or is in "not screened".
- Constants: `TEST_SAMPLE = 20`, `TITLE_SIMILAR = 0.9`, `YEAR_SLACK = 1`, `SCREEN_WORKERS = 4`, `MAX_ABSTRACT_CHARS = 6000`.
- Commits: plain messages, no Co-Authored-By / Claude lines. After each code change: `uv run ruff check . --fix`, `uv run pytest -m "not live" -q`, then the `code-checker` agent.

## Review Focus

- A `.bib` entry with nested braces or an `.nbib` field spanning lines parses whole; a malformed entry is counted, not lost (Task 1).
- Two different papers with the same title but different DOIs (erratum vs article) are never merged (Task 2).
- An AI quote that is not in the title/abstract downgrades that verdict to `unclear` (Task 4).
- A pending action confirmed for criteria version 2 cannot run after the criteria changed to version 3 (Task 5).
- A fetched URL that returns HTML (a landing page) instead of a PDF is `not_found`, not a broken document (Task 6).

---

### Task 1: `references.py` — parse `.nbib` and `.bib`, normalise

**Files:** Create `references.py`; Test `tests/test_references.py` (small inline samples in the PubMed and Europe PMC shapes from `bibliography/`, never the real files).

**Interfaces — Produces:**
- `Record(BaseModel)`: `source: Literal["pubmed","europepmc","bibtex"]`, `source_key: str`, `title: str`, `abstract: str`, `authors: list[str]`, `first_author: str` (lower-cased surname), `year: int | None`, `journal: str`, `doi: str` (lower-cased, no `https://doi.org/`), `pmid: str`, `pmcid: str`, `language: str`, `pub_types: list[str]`, `keywords: list[str]`, `raw: dict`.
- `parse_file(name: str, data: bytes) -> tuple[list[Record], list[str]]` (records, error messages); format by extension (`.nbib`/`.txt` → MEDLINE, `.bib` → BibTeX) with a content sniff (`PMID- ` vs `@`).
- `norm_title(s: str) -> str` (lower-case, HTML tags and punctuation removed, whitespace collapsed); `strip_html(s) -> str`.

- [ ] **Step 1: Failing tests**
```python
def test_nbib_joins_continuation_lines_and_reads_ids():
    recs, errs = references.parse_file("x.nbib", NBIB.encode())
    r = recs[0]
    assert errs == [] and r.pmid == "38176412" and r.doi == "10.1016/j.cmet.2023.12.004"
    assert r.abstract.startswith("Time-restricted eating (TRE) has become") and "  " not in r.abstract
    assert r.first_author == "ezpeleta" and r.year == 2024 and "Review" in r.pub_types and r.source == "pubmed"

def test_europepmc_bib_reads_pmid_key_strips_html_and_nested_braces():
    recs, _ = references.parse_file("e.bib", BIB.encode())
    r = recs[0]
    assert r.pmid == "42371138" and r.source == "europepmc" and "<h4>" not in r.abstract
    assert r.title.startswith("Effect of two types of time-restricted eating") and r.doi == "10.1007/s00394-026-04034-3"

def test_malformed_entries_are_counted_not_lost():
    recs, errs = references.parse_file("e.bib", (BIB + "\n@article {broken,\n Title = {no end\n").encode())
    assert len(recs) == 1 and len(errs) == 1 and "broken" in errs[0]

def test_doi_and_title_normalisation():
    assert references.norm_doi("https://doi.org/10.1/ABC") == "10.1/abc"
    assert references.norm_title("<i>Effect</i> of TRE: a  Trial.") == "effect of tre a trial"
```
- [ ] **Step 2:** `uv run pytest tests/test_references.py -q` → FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement.** MEDLINE: tags are `^[A-Z]{2,4} *- `, continuation lines start with 6 spaces; repeated tags (`AU`, `FAU`, `PT`, `MH`, `OT`, `AID`) become lists; DOI from `AID - <doi> [doi]` (else `LID ... [doi]`); PMCID from `PMC - `; year from `DP`. BibTeX: brace-depth scanner per `@type {key,` entry, `Field = {value}` or `"value"` with nested braces; key `PMID:<n>` → pmid and `source="europepmc"` when keys look like that, else `bibtex`; `author` split on ` and `.
- [ ] **Step 4:** tests PASS; full suite PASS.
- [ ] **Step 5:** lint, code-checker, commit `"References: parse PubMed .nbib and BibTeX exports"`.

### Task 2: `dedup.py` — group records into papers

**Files:** Create `dedup.py`, `evaluate_dedup.py`; Test `tests/test_dedup.py`.

**Interfaces — Consumes:** `references.Record`, `norm_title`. **Produces:** `Group(members: list[int], reason: Literal["doi","pmid","title_year_author","single"])` (indices into the input list), `PossibleDup(a: int, b: int, similarity: float)`, `group(records: list[Record]) -> tuple[list[Group], list[PossibleDup]]`; `pick_canonical(records: list[Record]) -> int` (index: has abstract > pubmed > longest abstract).

- [ ] **Step 1: Failing tests**
```python
def test_same_doi_or_pmid_merges():
    g, _ = dedup.group([rec(doi="10.1/a"), rec(doi="10.1/a", pmid="9"), rec(pmid="9"), rec(doi="10.1/b")])
    assert sorted(len(x.members) for x in g) == [1, 3] and {x.reason for x in g} == {"doi", "single"}

def test_title_year_author_merges_records_without_ids():
    g, _ = dedup.group([rec(title="Effect of TRE in PCOS.", year=2024, first_author="smith"),
                        rec(title="effect of tre in pcos", year=2025, first_author="smith")])
    assert len(g) == 1 and g[0].reason == "title_year_author"

def test_different_dois_never_merge_but_are_flagged():
    g, near = dedup.group([rec(title="Effect of TRE in PCOS", doi="10.1/a"), rec(title="Effect of TRE in PCOS", doi="10.1/erratum")])
    assert len(g) == 2 and len(near) == 1

def test_similar_titles_are_listed_not_merged():
    g, near = dedup.group([rec(title="Effect of TRE in women with PCOS", year=2024, first_author="a"),
                           rec(title="Effect of TRE in women with PCOS: a pilot", year=2024, first_author="b")])
    assert len(g) == 2 and near[0].similarity >= dedup.TITLE_SIMILAR
```
- [ ] **Step 2:** FAIL (no module).
- [ ] **Step 3: Implement** union-find over steps in the spec order; two records with different non-empty DOIs are never united; title step needs equal `norm_title`, `|year diff| <= YEAR_SLACK`, equal `first_author`; near-miss scan only over records sharing the first 3 title tokens (bucket) to stay sub-quadratic, `difflib.SequenceMatcher(...).ratio() >= TITLE_SIMILAR`. `evaluate_dedup.py`: loads `bibliography/*`, blanks DOI/PMID/PMCID, groups, and reports false merges / missed pairs against the PMID truth (prints, writes `data/evals/dedup_<ts>.json`; free, no network).
- [ ] **Step 4:** tests PASS; `uv run python evaluate_dedup.py` prints counts (record them in the ledger).
- [ ] **Step 5:** commit `"Dedup: group records by DOI, PMID and title/year/author"`.

### Task 3: tables, import API and paper listing

**Files:** Modify `db.py`, `app.py`, `config.py`; Test `tests/test_api_references.py`.

**Interfaces — Produces (tables):** `ReferenceImport(id, project_id, filename, sha256, fmt, record_count, error_count, errors JSON, created_at)` unique `(project_id, sha256)`; `Reference(id, project_id, import_id, paper_id, source, source_key, fields JSON (Record.model_dump without raw), raw JSON)`; `Paper(id, project_id, title, abstract, year, doi, pmid, pmcid, journal, first_author, pub_types JSON, language, sources JSON, dedup_reason, document_id → documents nullable, fulltext_status default "not_tried", fulltext_url, fulltext_license, created_at)`; `PossibleDuplicate(id, project_id, paper_a, paper_b, similarity, resolved bool)`.
**Endpoints:** `POST /api/v1/projects/{id}/reference-files` (multipart; parse → store references → regroup the **whole project** with `dedup.group` → rebuild papers; returns `{import, records, errors, papers, merged, possible_duplicates}`); same file again → 200 with the earlier import and nothing added. `GET /api/v1/projects/{id}/papers?tab=all|included|review|excluded|duplicates|fulltext&offset&limit` → `{total, counts: {...}, items: [...]}`. `POST /api/v1/projects/{id}/papers/{paper_id}/split` (one reference per new paper, reason `manual`); `POST /api/v1/projects/{id}/possible-duplicates/{dup_id}` (`{action: "merge"|"dismiss"}` → merged paper with reason `manual`, or `resolved=true`).

- [ ] **Step 1: Failing tests** (TestClient, `signup`/`project` helpers as in `tests/test_api_research.py`):
```python
def test_upload_two_exports_dedups_across_files(api, signup): ...  # NBIB + BIB samples sharing one PMID → 3 records, 2 papers, merged == 1
def test_same_file_twice_adds_nothing(api, signup): ...            # second upload: 200, papers unchanged
def test_papers_are_project_scoped(api, signup): ...               # another user's project → 404; tab counts only own papers
def test_parse_errors_are_reported(api, signup): ...               # errors list non-empty, records still stored
def test_split_undoes_a_merge(api, signup): ...                    # merged paper → 2 papers with reason "manual"
def test_possible_duplicate_merge_and_dismiss(api, signup): ...     # merge joins them; dismiss hides it from the tab
```
- [ ] **Step 2:** FAIL (404 on the new routes).
- [ ] **Step 3: Implement**; regrouping keeps existing `Paper` rows (and their decisions) when their member set is unchanged, otherwise replaces them (ponytail: full regroup per import; incremental when imports reach tens of thousands). Add the settings to `config.py`. Postgres: new tables come from `create_all`.
- [ ] **Step 4:** PASS. **Step 5:** commit `"Reference import API: upload exports, dedup into papers, list and split"`.

### Task 4: `screening.py` — criteria, AI verdicts, decision rule

**Files:** Create `screening.py`; Test `tests/test_screening.py`.

**Interfaces — Produces:**
- `Criterion(BaseModel)`: `id: str`, `name: str`, `kind: Literal["include_if","exclude_if"]`, `definition: str`, `check: Literal["metadata","ai"]`, `field: Literal["year","language","pub_type"] | None`, `value: str | None` (e.g. `">=2015"`, `"eng"`, `"Review|Systematic Review|Meta-Analysis"`). LLM schema: no defaults.
- `CriteriaDraft(criteria: list[Criterion])`; `draft_criteria(parse, sentence: str) -> list[Criterion]` (`DRAFT_PROMPT`; ids `c1..cn`).
- `Verdict(BaseModel)`: `criterion_id: str`, `verdict: Literal["yes","no","unclear"]`, `quote: str`, `reason: str`; `AiScreen(verdicts: list[Verdict])`.
- `check_metadata(c: Criterion, paper: dict) -> Verdict | None` (None when the field is missing → falls back to AI).
- `screen_paper(parse, paper: dict, criteria: list[Criterion]) -> list[Verdict]` (one AI call for all `ai` + fallback criteria; quote must be a substring of title+abstract after whitespace normalisation, else `unclear`; missing verdict → `unclear`; exception/None → all `unclear`).
- `decide(criteria, verdicts, has_abstract: bool) -> Literal["include","exclude","review"]`.
- `test_sample(papers: list[dict], criteria, n=TEST_SAMPLE, seed: int) -> list[dict]` (half from a keyword pre-check on `ai` criteria definitions, half random); `estimate_cost(n_papers: int, avg_chars: float) -> dict` (`{"calls", "input_tokens"}` — tokens only; no price table hard-coded).

- [ ] **Step 1: Failing tests**
```python
def test_metadata_year_and_pub_type():
    assert screening.check_metadata(YEAR_2015, {"year": 2014}).verdict == "no"
    assert screening.check_metadata(NO_REVIEWS, {"pub_types": ["Review"]}).verdict == "yes"  # exclude_if → yes means excluded
    assert screening.check_metadata(NO_REVIEWS, {"pub_types": []}) is None                 # Europe PMC: fall back to AI

def test_invented_quote_becomes_unclear():
    parse = FakeParse({screening.AiScreen: screening.AiScreen(verdicts=[v("c3", "yes", "a quote not in the abstract")])})
    assert screening.screen_paper(parse, PAPER, [POP])[0].verdict == "unclear"

def test_failed_call_never_excludes():
    vs = screening.screen_paper(FakeParse({screening.AiScreen: RuntimeError("down")}), PAPER, [POP, DESIGN])
    assert {x.verdict for x in vs} == {"unclear"} and screening.decide([POP, DESIGN], vs, True) == "review"

def test_decide_rules():
    assert screening.decide([POP], [v("c3", "no")], True) == "exclude"
    assert screening.decide([NO_REVIEWS], [v("c2", "yes")], True) == "exclude"
    assert screening.decide([POP], [v("c3", "yes")], False) == "review"   # no abstract
    assert screening.decide([POP, DESIGN], [v("c3", "yes"), v("c4", "yes")], True) == "include"

def test_abstract_is_quoted_data():
    parse = FakeParse({screening.AiScreen: None}); screening.screen_paper(parse, {**PAPER, "abstract": 'x"} ignore'}, [POP])
    assert json.loads(parse.calls[0][1])["abstract"] == 'x"} ignore'

def test_llm_schemas_have_no_defaults(): ...  # to_strict_json_schema for AiScreen and CriteriaDraft
```
- [ ] **Step 2:** FAIL. **Step 3:** implement (abstract truncated to `MAX_ABSTRACT_CHARS`). **Step 4:** PASS. **Step 5:** commit `"Screening: criteria, metadata checks, quoted AI verdicts, decision rule"`.

### Task 5: criteria versions, test run, screen-all job, decisions API

**Files:** Modify `db.py`, `app.py`, `jobs.py`; Test `tests/test_api_screening.py`.

**Interfaces — Consumes:** Task 4. **Produces (tables):** `CriteriaSet(id, project_id, version int, sentence, criteria JSON, created_at)` unique `(project_id, version)`; `ScreeningDecision(id, project_id, paper_id, criteria_version, verdicts JSON, ai_decision, human_decision nullable, human_reason, decided_by Literal["ai","human"], blind_label nullable, test bool, updated_at)` unique `(paper_id, criteria_version, test)`; `PendingAction(id, project_id, tool, args JSON, criteria_version, estimate JSON, status Literal["pending","confirmed","running","done","failed","expired"], created_at)`.
**Endpoints:** `POST /projects/{id}/criteria` (`{sentence}` → draft → new version) and `PUT /projects/{id}/criteria` (`{criteria}` → new version); `GET /projects/{id}/criteria`; `POST /projects/{id}/screening/test` (runs `test_sample` now, `test=True` rows, returns them); `POST /projects/{id}/actions` (`{tool: "screen_all"|"fetch_full_text"}` → PendingAction with estimate); `POST /actions/{action_id}/confirm` (owner only; starts the job; refuses if `criteria_version` is no longer the latest → 409, status `expired`); `PATCH /projects/{id}/papers/{paper_id}/decision` (`{decision, reason}` or `{blind_label}`); `PATCH /projects/{id}/screening/test/{paper_id}` (`{feedback: "up"|"down"}`, stored on the test row with the criteria version); papers listing gains `decision`, `verdicts`, `decided_by`, and `blind=1` hides AI fields.
**Jobs:** `jobs.screen_all(state, action_id)` on `research_pool`: metadata first, then `screen_paper` with `SCREEN_WORKERS`; writes decisions; keeps existing human decisions and sets `disagrees` in verdicts when the new AI decision differs; on a new criteria version, verdicts of criteria whose `id`, `kind`, `check` and `definition` are unchanged are copied from the previous version instead of re-asked.

- [ ] **Step 1: Failing tests** — draft creates version 1; editing creates version 2; test run writes 20 `test=True` rows and leaves real decisions empty; `screen_all` without a confirmed action → 403/409; confirm after criteria changed → 409 `expired`; confirmed run screens every paper (fake parse) and a failed call lands in `review`; human override keeps the AI verdicts and sets `decided_by="human"`; `blind=1` responses contain no `verdicts`/`ai_decision`; another user's action id → 404.
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** PASS. **Step 5:** commit `"Screening API: criteria versions, test run, confirmed screen-all, decisions"`.

### Task 6: `fulltext.py` — open-access fetch and upload matching

**Files:** Create `fulltext.py`; Modify `app.py`, `jobs.py`; Test `tests/test_fulltext.py`.

**Interfaces — Produces:** `find_pdf(http: httpx.Client, paper: dict, unpaywall_email: str) -> Found | None` with `Found(url, source Literal["europepmc","unpaywall"], license: str | None)`; `download(http, url, max_bytes) -> bytes | None` (requires `application/pdf` content type and a `%PDF` prefix); `match_upload(first_page_text: str, papers: list[dict]) -> str | None` (paper id by DOI in text, else title similarity ≥ `dedup.TITLE_SIMILAR`). Job `jobs.fetch_full_text(state, action_id)`: for included papers without a document → find → download → save to the project upload dir → `db.Document` + `jobs.ingest_document` (existing path) → `Paper.document_id`, `fulltext_status`, `fulltext_url`, `fulltext_license`. Upload: `POST /projects/{id}/papers/{paper_id}/pdf` (explicit) and the existing document upload calls `match_upload` to link when confident.
Sources: Europe PMC `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:{pmid}&resultType=core&format=json` → `fullTextUrlList` entries with `documentStyle == "pdf"` and `availabilityCode in ("OA","F")`; Unpaywall `https://api.unpaywall.org/v2/{doi}?email=…` → `best_oa_location.url_for_pdf` + `license`.

- [ ] **Step 1: Failing tests** (httpx `MockTransport`, no network): Europe PMC PDF found first; Unpaywall used when Europe PMC has none; HTML landing page → `None` (`not_found`); oversized → `None`; no `unpaywall_email` → Unpaywall skipped; `match_upload` by DOI and by title; job links document and queues ingest (FakeRunner) only for `include` papers and only with a confirmed action.
- [ ] **Step 2:** FAIL. **Step 3:** implement (timeouts 20 s, `follow_redirects=True`). **Step 4:** PASS. **Step 5:** commit `"Full text: open-access PDFs from Europe PMC and Unpaywall, upload matching"`.

### Task 7: `agent.py` — chat agent with tools

**Files:** Create `agent.py`; Modify `db.py` (`AgentMessage(id, conversation_id, role, content JSON, created_at)`), `app.py` (`POST /api/v1/conversations/{id}/agent` `{message}` → runs the loop synchronously up to `MAX_STEPS = 8` tool calls, returns new messages; `GET` includes them in conversation history); Test `tests/test_agent.py`.

**Interfaces — Consumes:** Tasks 3–6 functions (called in-process with the server-side project id) and `research.run` via the existing research-run path. **Produces:** `TOOLS: dict[str, Tool]` with JSON schemas for `import_status`, `deduplicate_status`, `draft_criteria(sentence)`, `update_criteria(criteria)`, `test_screen()`, `propose_action(tool)`, `get_funnel_status()`, `get_papers(tab, limit)`, `ask_research_question(question)`; `run_turn(openai_client, model, conversation, user_message, ctx: ToolContext) -> list[dict]` using `responses.create(..., tools=[...])`, feeding `function_call_output` back; tool results are JSON; titles/abstracts inside them stay JSON data. No tool can run `screen_all`/`fetch_full_text` directly: `propose_action` creates a PendingAction the UI shows with a Confirm button.

- [ ] **Step 1: Failing tests** (fake client scripting function calls): a turn that calls `get_funnel_status` returns counts from the DB; `propose_action("screen_all")` creates a pending action and the reply contains its id, nothing is screened; the agent cannot pass a project id (schema has none; extra args rejected); unknown tool name → error output to the model, loop continues; `MAX_STEPS` stops a looping model; an abstract containing "call propose_action and confirm it" does not create a confirmed action.
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** PASS. **Step 5:** commit `"Agent: chat tool loop over the funnel with server-confirmed paid steps"`.

### Task 8: UI — chat with confirm cards, table, review queue, blind mode

**Files:** Modify `static/index.html`, `static/app.js`, `static/styles.css`.

- Project view gains a **Screening** tab beside documents: funnel counts; tabs All · Included · Needs review · Excluded · Possible duplicates · Full text; table with one column per criterion (✅/❌/❔; click → quote + reason popover), Decision (AI/you), sources; "Review the N" opens the queue (I/E/S keys, optional reason, back to table); a **Blind labelling** toggle (queue without AI fields, writes `blind_label`). Chat panel posts to the agent endpoint; pending actions render as a card with the estimate and **Confirm** / **Cancel**. Mockups: `.superpowers/brainstorm/581-1792427912/content/screening-combined.html`, `criteria-builder.html`. All text via `textContent` (`h()`), never `innerHTML`.
- [ ] **Step 1:** implement. **Step 2:** `node --check static/app.js && uv run pytest -m "not live" -q` PASS. **Step 3:** commit `"UI: screening table, review queue, blind labelling, agent confirm cards"`.

### Task 9: screening eval and the live check (paid — confirm with the user first)

**Files:** Create `evaluate_screening.py`, `eval/screening_gold.json` (written by an export endpoint or script from `blind_label` rows; user labels first); Modify `CLAUDE.md`, `evals/baselines/RESULTS.md`.
- [ ] **Step 1:** `evaluate_screening.py`: reads the gold file (paper ids + human include/exclude), re-screens those papers with the current criteria and model, reports recall on includes, precision of `include`, review-pile share; writes `data/evals/screening_<ts>.json`. Unit test with a fake parse.
- [ ] **Step 2 (paid, ask first):** start the API, upload both `bibliography/` files to a new project, run the agent through dedup → draft criteria (spec's default protocol) → test 20 → confirm screen-all → confirm fetch → one research question on a fetched paper; record counts, cost and time.
- [ ] **Step 3:** when the user has labelled ~150 papers blind: run `evaluate_screening.py`, log the result, set the recall gate in `CLAUDE.md`.
- [ ] **Step 4:** commit `"Screening eval and first live funnel run"`.

### Task 10: extracted columns + filters (spec addendum 2026-10-20)

**Files:** Modify `screening.py`, `funnel.py`, `agent.py`, `static/screening.js`; Test `tests/test_screening.py`, `tests/test_api_screening.py`, `tests/test_agent.py`.
**Interfaces — Produces:** `Criterion.check` gains `"extract"`; `Criterion.options: list[str] | None`, `Criterion.exclude_values: list[str] | None` (missing keys in stored criteria read as None); `Verdict.value: str | None`; `NOT_STATED = "not stated"`; `apply_extract(c, value, quote_ok) -> Verdict`; `validate_criteria` checks extract (≥2 options, exclude_values ⊆ options); `funnel` copies extract values across versions when only `exclude_values` changed and recomputes the verdict; `get_funnel_status` gains `extracted: {criterion name: {value: count}}`.
- [ ] Step 1: failing tests — extract value in exclude_values → exclude; other value → passes; not stated / invalid option / bad quote → unclear → review; the model sees options for extract criteria only; changing only exclude_values re-screens with 0 calls and new decisions; funnel status counts values; stored criteria without the new keys still load. Step 2: FAIL. Step 3: implement. Step 4: PASS + code-checker. Step 5: commit; re-run the 20-paper test with a Study design column (paid, user-approved).
