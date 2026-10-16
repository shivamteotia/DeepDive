# Metric questions (all values / best model / across papers) — design

Date: 2026-10-16. Status: approved in chat (parts 1–2); awaiting written-spec review.
Builds on branch `smart-chunking` (uses its `paper_title` payload).

## Goal

Researchers ask three shapes of metric question that today's run answers incompletely:

1. **All values** — "What sensitivity was reported?" → every reported value in the paper: each model, cohort and
   condition, cited.
2. **Best model** — "What is the sensitivity of the best reported model?" → only the value(s) of the model the
   **authors themselves designate** as their best/final/proposed model (decided: option A), plus the passage where
   they say so. Never the highest number by our own judgement.
3. **Across papers** — "What sensitivity was reported in all uploaded papers?" → one block per paper, and an explicit
   "not reported in X" for every paper that was searched and has none. No paper is silently dropped.

Applies to any reported metric (sensitivity, specificity, AUROC/AUC, C-index, accuracy, PPV, NPV, F1, …).

## Why today's run fails these

- Retrieval is k=6 per sub-question over the whole project: values spread over text and tables, or over several
  papers, fall outside the top 6; a lower-ranked paper can vanish from an "all papers" answer.
- The synthesizer writes free-form claims, so nothing forces one claim per (model × cohort × condition).
- The verifier checks each claim against its own citation only; a "best" claim can pass while the designation or a
  better model sits in an unretrieved chunk.

## Evidence in the current corpus (2026-10-16, `<collection>_v2`)

- EA2022: sensitivity only as an operating point — "at a sensitivity of 0.90, specificity = 0.630 (95% CI …)"; other
  points (0.85, 0.95) are in a supplement. CI-M AUROC 0.852 (gold f01).
- JPan2023: AUC for three models in two cohorts — predictive model 0.850 / 0.893, APACHE II 0.795 / 0.746,
  SOFA 0.69 / 0.742. Designation: "our predictive model … outperformed the traditional APACHE II and SOFA scoring
  systems". No sensitivity anywhere in the text.

## Part 1 — understanding and retrieval

### Planner (`research.plan`, fast model, same single call)

`Plan` gains fields (all optional in the LLM output, validated by Pydantic):

```python
class Plan(BaseModel):
    sub_questions: list[str]
    scope: Literal["standard", "all_values", "best_model", "across_papers"] = "standard"
    metric: str = ""                 # e.g. "sensitivity"; "" for non-metric questions
    metric_terms: list[str] = []     # synonyms as papers write them: "recall", "true positive rate"
    target_paper: str = ""           # paper the user names ("the SCAP study"), "" if none
```

Invalid or missing → `scope="standard"`: today's behaviour is the default. Prompt rules: `across_papers` when the
question says all/every/each paper or study; `best_model` when it asks for the best/final/proposed model;
`all_values` when it asks what values of a metric were reported; otherwise `standard`.

### Documents available to a run

`research.run(..., documents: list[Doc] | None = None)` where
`Doc = {"sha256": str, "filename": str, "title": str}`, built server-side:
- API (`jobs.run_research`): the project's ready documents from Postgres; `title` from the first Qdrant point's
  `paper_title` (fallback filename).
- CLI (`research.main`): `ask.list_documents(qdrant, collection, project_id)` — distinct `document_sha256` /
  `source_filename` / `paper_title` in the project (scroll with the project filter).

`search_fn` gains an optional keyword: `search_fn(q, document: str | None = None)`. With `document`, the search is
restricted to that sha256 **and** still carries the server-side project filter (`ask.search(..., sha256_in=[sha])`
intersected with the ready set in the API).

### Retrieval by scope

| scope | searches |
|---|---|
| `standard` | unchanged |
| `all_values`, `best_model` | target paper if resolved, else each document separately: query = `question + metric + metric_terms`, `k=10` per document |
| `across_papers` | every document separately, same query, `k=6` per document |

- Target paper resolution: case-insensitive token overlap between `target_paper` and each document's title/filename;
  best match with ≥ 50% of the target's content tokens, else none (→ each document).
- Cap: `MAX_COVERAGE_DOCS = 12`. More documents → fall back to project-wide retrieval (`standard` path with the metric
  query) and add the gap line "Searched the project as a whole: more than 12 papers."
- Coverage record per searched document: `{filename, searched: True, passages: n}` kept in the run log.

## Part 2 — answer, verification, UI

### Rows (synthesizer)

`Claim` gains an optional structured row (schema fields optional, so `standard` answers are unchanged):

```python
class MetricRow(BaseModel):
    paper: str          # filename or title of the cited document
    model: str          # as the paper names it
    condition: str      # cohort / threshold / time point, e.g. "validation cohort", "at sensitivity 0.90"
    metric: str
    value: str          # verbatim incl. CI
    best: bool = False

class Claim(BaseModel):
    text: str
    evidence_ids: list[str]
    row: MetricRow | None = None
```

For the three scopes, `ask.answer` gets scope-specific rules appended to INSTRUCTIONS:
- `all_values`: one row per reported (model × condition) value of the metric; no selection or summarising; each row's
  `text` is a full sentence ("In JPan2023, the predictive model's AUC in the validation cohort was 0.893 (95% CI
  0.809-0.949).").
- `best_model`: first a designation claim (no row) citing the authors' own words that name their best/final/proposed
  model; then only that model's rows with `best=true`. If no passage designates one, say so in `gaps` and answer as
  `all_values`. Never choose the highest value.
- `across_papers`: rows for every paper; a paper with no value gets no row.

### Verification and coverage (`research.run` / `finalize`)

- Each row's claim is verified like any claim (existing fail-closed rules on value, model, cohort, outcome).
- `best=true` survives only if a SUPPORTED designation claim cites the same document; otherwise `best` is cleared and
  the run is `needs_review`.
- Coverage gap: each searched document with no surviving row adds "Not reported in: <filename> (searched; no
  <metric> found)." (only for `all_values` with no target, and `across_papers`).
- Claim dicts in the result carry `row` (or omit it); `audit_record` keeps it, so `/research-runs/{id}` returns it.

### UI (`static/app.js`)

When any claim has a `row`, findings render as a table: Paper | Model | Condition | Value | Sources (citation chips
as today), a "★ Authors' best model" badge on `best` rows; non-row claims (the designation sentence) render above the
table as normal findings; the coverage lines appear in "What the sources don't say". Answers without rows look
exactly as today.

## Untrusted content

Unchanged rules: evidence stays JSON-quoted data; `target_paper` and `metric_terms` are LLM output used only as search
text and for title matching, never as filters on project or document identity.

## Tests (unit, fake LLM)

1. Plan parsing: each scope; missing/invalid scope → `standard`; metric fields default empty.
2. Retrieval: `all_values` without target searches each document once with `document=sha`; with a resolvable target
   only that document; `across_papers` searches every document; >12 documents → one project-wide search + gap line;
   every call still goes through the server-built filter (fake search_fn asserts `document ∈ documents`).
3. Target resolution: "the SCAP study" → JPan title; unrelated text → none.
4. Rows: answer rows verified per row; an UNSUPPORTED row removed; `best` kept only with a SUPPORTED designation in the
   same document; designation missing → `best` cleared, `needs_review`.
5. Coverage: a searched document with no surviving row gets the "Not reported in" line; `standard` adds none.
6. `standard` questions: planner/retrieval/answer/finalize behave exactly as before (existing tests stay green).
7. API: `documents` passed from ready documents; a document from another project is never searched.

## Evals

New cases in `evals/golden.json` (project `main`):
- "What AUC values were reported in the SCAP study?" → facts 0.850, 0.893, 0.795, 0.746, 0.69, 0.742.
- "What was the AUC of the best model reported in the SCAP study?" → facts 0.850, 0.893; forbidden 0.795, 0.746.
- "What sensitivity was reported in all uploaded papers?" → facts 0.90 (EA2022) and gap mentions JPan2023.
- "What AUROC was reported in each uploaded paper?" → facts 0.852 (EA2022) and 0.850/0.893 (JPan2023).

Gate: `evaluate.py` unchanged vs run 8 (retrieval code path for `evaluate.py` is untouched); `evals.run` +
`evals.compare` vs run 8 baseline: no regression on the existing 26 cases, all 4 new cases pass.

## Out of scope

Ingest-time metrics table (option 2), charts, cross-paper statistics, choosing a "best" model by value.
