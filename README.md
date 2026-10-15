# MaxRIM — Research Intelligence & Management

> **Status:** Architecture defined · Backend-first development · Implementation status to be verified

**MaxRIM** is a proposed AI-assisted research intelligence and management platform designed for healthcare researchers and research teams. It brings project-specific literature, incremental document ingestion, evidence retrieval, and citation-verified report generation into one auditable workflow.

MaxRIM is **research assistance software, not a clinical decision-making or diagnostic system**. Outputs require human review, especially for clinical, regulatory, or scientific claims.

## 1. Problem and vision

Researchers often work across multiple studies, accumulate papers over weeks or months, and repeatedly search the same literature. General-purpose chat systems can mix evidence across projects, lose page-level provenance, invent citations, or overlook newly uploaded documents.

MaxRIM aims to provide:

- **Project workspaces:** isolate documents, retrieval, reports, and execution history by research project.
- **Incremental ingestion:** upload five papers today and five more next week without unnecessarily reprocessing the first five.
- **Layout-aware PDF parsing:** retain sections, tables, captions, and source location information where available.
- **Grounded research synthesis:** answer broad research questions through planning, retrieval, synthesis, and claim-level verification.
- **Transparent evidence:** show source documents, pages, quoted passages or locations, web URLs, and verification outcomes.
- **Reproducible evaluation:** track parsing, retrieval, generation, and verification experiments.

### Initial use cases

1. Ask questions across the papers attached to a single research project.
2. Generate cited background reviews, evidence summaries, or research briefings.
3. Compare methods, cohorts, metrics, limitations, and findings across studies.
4. Update a literature corpus as new studies are uploaded.
5. Identify evidence gaps and unsupported statements before presenting a draft to a researcher.
6. Maintain a traceable research history across multiple independent projects.

Potential later extensions include structured systematic-review extraction, screening workflows, study comparison matrices, and exportable reports. These are **not** claimed as implemented features.

## 2. High-level architecture

```mermaid
flowchart TD
    U[Researcher / API client] --> API[FastAPI backend]
    API --> AUTH[Authentication and project authorization]
    AUTH --> PG[(PostgreSQL: projects, files, jobs, reports, audits)]
    AUTH --> OBJ[(Object storage: original PDFs and artifacts)]
    AUTH --> QUEUE[Background ingestion queue]
    QUEUE --> PARSE[Docling: PDF to DoclingDocument]
    PARSE --> CHUNK[Docling HybridChunker]
    CHUNK --> EMB[Embedding model]
    EMB --> Q[(Qdrant: project-scoped chunks)]
    AUTH --> GRAPH[LangGraph research workflow]
    GRAPH --> PLAN[Planner]
    PLAN --> RESEARCH[Researcher: project Qdrant + optional web]
    RESEARCH --> SYNTH[Synthesizer]
    SYNTH --> VERIFY[Verifier]
    VERIFY -->|Supported| FINAL[Final cited report]
    VERIFY -->|Unsupported claims| SYNTH
    VERIFY -->|Insufficient evidence| RESEARCH
    RESEARCH -->|Replan when necessary| PLAN
    FINAL --> PG
    GRAPH --> OBS[Tracing, evaluation, metrics and audit logs]
```

**Architectural principle:** ingestion is a separate, asynchronous workflow. The agentic graph operates on documents that have already been processed and indexed; it must not silently assume an uploaded file is searchable before indexing completes.

## 3. Technology stack (proposed)

| Layer | Proposed technology | Responsibility |
|---|---|---|
| API | Python, FastAPI, Pydantic | Project, document, job, and research endpoints |
| Metadata | PostgreSQL | Users, projects, document versions, job state, reports, citations |
| File storage | S3-compatible object storage / local development storage | Original PDFs, extracted artifacts, report exports |
| Background processing | Celery + Redis (or equivalent) | Durable ingestion and retryable tasks |
| Parsing | Docling | Layout-aware PDF conversion to structured documents |
| Chunking | Docling `HybridChunker` | Structure-aware, tokenizer-aware chunks |
| Embeddings | OpenAI `text-embedding-3-small` initially | Vector representations; provider configurable |
| Vector search | Qdrant | Project-scoped retrieval with payload metadata |
| Agent orchestration | LangGraph | Planner–Researcher–Synthesizer–Verifier workflow |
| Web search | Tavily or alternative, optional | Fresh public evidence with URL provenance |
| Evaluation | DeepEval + custom checks | Retrieval and faithfulness evaluation |
| Experiment tracking | MLflow | Runs, configurations, datasets, and metrics |
| Observability | Structured logs and traces | Job diagnostics, tool usage, latency, and audit trail |
| Deployment | Docker / Docker Compose initially | Reproducible local services |

**Cost and privacy note:** open-source software does not guarantee zero operational cost. Embedding/LLM/web-search APIs may be billed. Do not send patient-identifiable or confidential documents to external APIs without appropriate organizational approval, contracts, controls, and de-identification where required.

## 4. Document ingestion: Docling → Qdrant

### 4.1 Ingestion pipeline

```mermaid
flowchart LR
    A[Upload PDF] --> B[Validate file, project access and hash]
    B --> C{Existing same version?}
    C -->|Yes| D[Skip / reuse indexed version]
    C -->|No| E[Persist original + queue job]
    E --> F[Docling conversion]
    F --> G[Native HybridChunker]
    G --> H[Normalize chunk metadata]
    H --> I[Batch embeddings]
    I --> J[Qdrant upsert]
    J --> K[Commit indexed version + job status]
```

**Parsing:** Use Docling's supported `DocumentConverter` / PDF pipeline APIs to obtain a `DoclingDocument`. `DoclingPdfParser` is a conceptual name in the initial design, **not a class assumed to exist in every Docling version**. Confirm the installed Docling API and pin its version.

**Chunking:** Apply `HybridChunker` directly to the `DoclingDocument`; do **not** first flatten the document to Markdown and then perform generic character splitting. Retain headings and provenance, and explicitly map available Docling item metadata into Qdrant payload fields. Bounding boxes and captions may require resolving chunk provenance back to original document items; do not assume every chunk automatically exposes every location field.

**Tables:** Preserve table structure and references. Evaluate whether table-aware serialization or separate table chunks improve retrieval, especially for clinical study characteristics and outcome tables. Images and figures may require additional extraction/description and should not be treated as reliably searchable merely because the PDF was parsed.

**Embeddings:** Keep the embedding provider, model name, vector dimension, and model version configurable. A change in embedding dimensions or semantic model requires an explicit collection migration or re-embedding plan.

### 4.2 Recommended chunk payload

```json
{
  "project_id": "proj_123",
  "document_id": "doc_456",
  "document_version_id": "ver_002",
  "chunk_id": "chunk_stable_id",
  "source_filename": "risk_matrix.pdf",
  "source_sha256": "...",
  "page_numbers": [4],
  "section_headings": ["Results", "Risk factors"],
  "content_type": "text",
  "text": "Exact text used for embedding and evidence display",
  "provenance": [{"page": 4, "bbox": null, "item_ref": "..."}],
  "parser_version": "pinned-version",
  "chunker_version": "pinned-version",
  "embedding_model": "text-embedding-3-small",
  "indexed_at": "ISO-8601 timestamp"
}
```

This is a **target schema**, not a promise that Docling emits these exact keys. The ingestion adapter must construct and validate them. Use deterministic point IDs derived from project, document version, and chunk identity so retries are idempotent.

### 4.3 Incremental upserts and versioning

- Compute a SHA-256 hash of uploaded content, scoped to the relevant project/document identity.
- If an identical version is already indexed, return its status rather than duplicating vectors.
- If a document changes, create a new immutable document version; index it before marking it active.
- On successful activation, exclude old versions from retrieval and safely remove obsolete points according to retention policy.
- Use retryable jobs, batch upserts, dead-letter/error states, and reconciliation for partial failures.
- Record `uploaded → queued → parsing → chunking → embedding → indexing → ready`, with `failed` and `superseded` states.
- Never mark a document `ready` until the required vectors and metadata have been persisted and checked.
- Preserve ingestion configuration and versions for reproducibility.

### 4.4 Project isolation in Qdrant

For many research projects, the initial recommendation is **one or a small number of collections with mandatory `project_id` filtering** and suitable payload indexes. Separate collections per project are an alternative when operational or isolation requirements justify them.

**A filter alone is not a security boundary.** The backend must authorize the requesting user for the project and construct the filter server-side. Never trust a client-supplied project filter or allow cross-project retrieval by omitting it. Add integration tests that attempt cross-project access.

## 5. Multi-agent orchestration: LangGraph

The agentic workflow begins **after** indexing. The graph works with structured state and bounded transitions rather than relying on a free-form conversation alone.

### 5.1 Shared state (conceptual)

```python
from typing import TypedDict, Literal

class EvidenceItem(TypedDict):
    evidence_id: str
    sub_question_id: str
    claim_or_excerpt: str
    source_type: Literal["document", "web"]
    source_id: str
    locator: str             # e.g., PDF page or canonical URL
    retrieved_text: str

class ResearchState(TypedDict, total=False):
    project_id: str
    original_query: str
    sub_questions: list[str]
    evidence_items: list[EvidenceItem]
    draft_report: str
    verification_report: dict
    final_report: str
    iteration_count: int
    execution_log: list[dict]
```

This illustrates state shape, **not** a complete runnable graph. In implementation, use stable IDs, structured agent output schemas, reducers for parallel evidence aggregation, explicit error fields, and persistence/checkpointing.

### 5.2 Agent 1 — Planner

**Purpose:** break the research request into focused, answerable sub-questions.

**Input:** original query, project context, allowed sources, available document metadata.

**Output:** structured sub-questions with stable IDs, search priorities, and evidence requirements. Validate LLM-produced JSON with Pydantic rather than assuming valid JSON.

Example request: *What are the known risks of adopting generative AI in healthcare based on my uploaded PDFs and recent news?*

Example plan:

1. What risks are documented in the uploaded literature?
2. What relevant recent developments are reported by reliable public sources?
3. Which mitigations are supported by the retrieved evidence?

### 5.3 Agent 2 — Researcher

**Purpose:** collect source-grounded evidence per sub-question.

**Tools:**

- `qdrant_search_tool`: search only documents belonging to the authorized project, with metadata filters and optional reranking.
- `web_search_tool`: optional, user-/policy-enabled web search; store canonical URL, retrieval timestamp, publisher/date where available, and quoted supporting passage.

**Output:** structured `evidence_items`, not unsupported prose. Every item needs an immutable evidence ID and a verifiable locator, such as a document ID and page number or a web URL and captured excerpt. Deduplicate overlapping passages, and retain contradictory evidence instead of silently dropping it.

Potential later enhancements: hybrid dense + keyword search, query expansion, reciprocal rank fusion, cross-encoder reranking, and retrieval-quality thresholds. These should be evaluated rather than assumed beneficial.

### 5.4 Agent 3 — Draft Synthesizer

**Purpose:** write a coherent report using **only** evidence in the state.

**Rules:**

- Do not invent facts, citations, numerical values, or references.
- Use stable evidence IDs for each factual claim; render human-readable citations from the source registry.
- Clearly distinguish findings, limitations, conflicting studies, and missing evidence.
- If evidence is inadequate, explicitly state uncertainty instead of filling gaps with general model knowledge.
- Keep claims granular enough for verification.

**Output:** `draft_report` with claim-to-evidence references.

### 5.5 Agent 4 — Verifier (core quality-control loop)

**Purpose:** inspect each factual claim and its linked evidence, including citation correctness.

**Classification:**

| Result | Meaning | Action |
|---|---|---|
| `SUPPORTED` | Cited passage supports the claim | Keep |
| `UNSUPPORTED` | Citation is irrelevant, contradictory, missing, or insufficient for the assertion | Remove or rewrite |
| `THINLY_SUPPORTED` | Some support exists but evidence is incomplete or overstated | Qualify, retrieve more, or mark unresolved |

**Verifier output** should include claim ID, verdict, evidence IDs checked, concise rationale, recommended action, and iteration number. Verification is an LLM-assisted quality gate, **not proof of truth**: independent deterministic checks and human review remain necessary.

### 5.6 Conditional routing and stop conditions

```mermaid
flowchart TD
    START([Start]) --> P[Planner]
    P --> R[Researcher]
    R --> S[Synthesizer]
    S --> V[Verifier]
    V --> D{Verdict}
    D -->|All supported| END([Publish cited report])
    D -->|Unsupported claims| S
    D -->|Thin evidence| R
    R -->|Replan needed| P
    D -->|Budget exhausted| SAFE([Return qualified draft / human review])
```

Prevent infinite loops using configurable `max_iterations`, tool-call limits, token/cost budgets, timeout handling, and a progress check. If the verifier cannot resolve a claim, return an explicitly qualified result or route for human review; **never auto-approve because retries are exhausted**.

For parallel sub-question research, use LangGraph fan-out/fan-in with deterministic evidence merging. Store graph checkpoints so failures can be inspected or resumed.

## 6. Worked example: detecting an unsupported claim

**Question:** “What are the known risks of adopting generative AI in healthcare based on my uploaded PDFs and recent news?”

1. Planner generates sub-questions on documented risks, recent regulatory developments, and mitigations.
2. Researcher retrieves project-scoped PDF excerpts and permitted web sources, each with an evidence ID.
3. Synthesizer drafts: “Generative AI poses privacy risks [Doc: risk_matrix.pdf, p. 2] and is banned by the FDA [Web: URL].”
4. Verifier inspects the cited source. **In this illustrative scenario**, it supports guidance rather than a ban, so the second claim is `UNSUPPORTED`.
5. The draft is revised to describe only what the source actually supports.
6. Verifier rechecks the revision and approves it only if every retained claim has adequate support.
7. The user receives the final report, citation links, and an audit trail recording the rejected claim and revision.

**Important:** The “FDA ban” is an intentionally false **example claim**, not a factual statement about FDA policy.

## 7. Data model (proposed)

Core PostgreSQL entities:

| Entity | Key fields |
|---|---|
| `users` | id, identity, status |
| `projects` | id, name, description, owner_id, settings |
| `project_members` | project_id, user_id, role |
| `documents` | id, project_id, filename, active_version_id |
| `document_versions` | id, document_id, sha256, object_key, parser_config, status |
| `ingestion_jobs` | id, version_id, stage, attempts, error, timestamps |
| `research_runs` | id, project_id, query, graph_version, status, configuration |
| `evidence_items` | id, run_id, source reference, excerpt, locator |
| `claims` | id, run_id, text, cited_evidence_ids |
| `verification_results` | id, claim_id, verdict, rationale, iteration |
| `reports` | id, run_id, version, final_content, status |
| `audit_events` | id, actor, action, resource, timestamp, metadata |

Define foreign keys, uniqueness rules, access controls, deletion semantics, and retention policy before production use. Store secrets outside the database and source code.

## 8. API contract (planned)

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/v1/projects` | Create research project |
| `GET` | `/api/v1/projects` | List accessible projects |
| `POST` | `/api/v1/projects/{project_id}/documents` | Upload and enqueue document |
| `GET` | `/api/v1/projects/{project_id}/documents` | List documents and index states |
| `GET` | `/api/v1/ingestion-jobs/{job_id}` | Check ingestion status |
| `POST` | `/api/v1/projects/{project_id}/research-runs` | Start agentic research run |
| `GET` | `/api/v1/research-runs/{run_id}` | Get run state and final output |
| `GET` | `/api/v1/research-runs/{run_id}/evidence` | Inspect retrieved evidence |
| `GET` | `/api/v1/research-runs/{run_id}/verification` | Inspect claim verification and revisions |

Use asynchronous run IDs for long jobs rather than holding HTTP requests open. SSE or WebSocket progress updates can be added later.

## 9. Suggested backend repository layout

```text
maxrim/
├── README.md
├── .env.example
├── .gitignore
├── pyproject.toml
├── docker-compose.yml
├── alembic/
├── src/maxrim/
│   ├── main.py
│   ├── api/
│   │   ├── projects.py
│   │   ├── documents.py
│   │   └── research.py
│   ├── core/
│   │   ├── config.py
│   │   ├── auth.py
│   │   └── logging.py
│   ├── db/
│   │   ├── models.py
│   │   └── session.py
│   ├── ingestion/
│   │   ├── pipeline.py
│   │   ├── docling_parser.py
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   └── jobs.py
│   ├── retrieval/
│   │   ├── qdrant_store.py
│   │   ├── search.py
│   │   └── citations.py
│   ├── agents/
│   │   ├── state.py
│   │   ├── graph.py
│   │   ├── planner.py
│   │   ├── researcher.py
│   │   ├── synthesizer.py
│   │   └── verifier.py
│   ├── services/
│   │   ├── projects.py
│   │   └── research_runs.py
│   └── evaluation/
│       ├── datasets.py
│       ├── metrics.py
│       └── tracking.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── security/
│   └── evaluation/
└── docs/
    ├── architecture.md
    ├── api.md
    └── decisions/
```

This directory tree is a **recommended target**, not an assertion that these files already exist.

## 10. Development roadmap

| Phase | Deliverables | Exit criterion |
|---|---|---|
| **1. Backend foundation** | FastAPI, PostgreSQL models, project authorization, Docker, migrations | Create and list authorized projects |
| **2. Ingestion MVP** | Upload, Docling parsing, native chunking, embeddings, Qdrant, job status | PDF is searchable with correct page/source metadata |
| **3. Incremental updates** | Hashing, versions, idempotent upsert, retry/recovery | Upload 5 + 5 papers without duplicate active chunks |
| **4. Retrieval MVP** | Project filters, citations, baseline search, retrieval tests | Relevant chunks returned with zero cross-project leakage in tests |
| **5. LangGraph MVP** | Planner, Researcher, Synthesizer, Verifier, bounded loops | Unsupported claim is caught, corrected or qualified, and logged |
| **6. Evaluation** | Gold dataset, DeepEval/custom metrics, MLflow tracking | Baseline metrics and regressions recorded |
| **7. Production hardening** | Permissions, rate limits, security, observability, deployment | Reliability and security checks pass |
| **8. Frontend** | Project dashboard, upload status, research workspace, evidence inspector | Researchers can complete core workflow without direct API calls |

### First end-to-end acceptance scenario

- Create **Project A** and **Project B**.
- Upload five PDFs into Project A; wait for `ready`.
- Upload the same five again; verify no duplicate active vectors.
- Upload five additional PDFs into Project A; verify ten unique active documents are searchable.
- Add unrelated PDFs to Project B; verify Project A queries cannot retrieve Project B data.
- Ask a broad research question and inspect sub-questions, retrieved excerpts, citations, draft, verification decisions, and final report.
- Inject a deliberately unsupported statement in a controlled test and confirm that the verifier rejects or qualifies it.
- Simulate an ingestion failure and retry; verify consistent final state.

## 11. Evaluation strategy

Evaluation must measure more than whether a generated answer sounds convincing.

| Layer | Measures |
|---|---|
| Parsing | Section/table fidelity, page mapping accuracy, extraction coverage |
| Chunking | Context completeness, chunk size, provenance retention |
| Retrieval | Recall@k, Precision@k, MRR/nDCG where applicable, project leakage tests |
| Generation | Answer relevance, completeness, citation coverage |
| Verification | Unsupported-claim detection precision/recall, false approvals, revision success |
| Operations | Ingestion success, duplicate rate, p50/p95 latency, tokens/cost, retry counts |

Create a small **expert-reviewed evaluation set** with source documents, questions, expected evidence passages, and intentionally unsupported claims. Use DeepEval where suitable, custom deterministic citation checks, and manual review for clinically significant claims. Log run settings, dataset version, model IDs, prompt versions, metrics, and artifact references in MLflow. Treat judge-model scores as proxies, not ground truth.

## 12. Security, privacy, and responsible use

- Enforce authenticated access and project-level authorization on every endpoint and retrieval call.
- Encrypt transport and protect stored documents and credentials.
- Use de-identified or appropriately approved datasets; avoid exposing patient information to third-party services.
- Treat PDFs and web pages as **untrusted content**. Instructions embedded in retrieved sources must never override system policies or trigger unauthorized tool use.
- Make web access opt-in or governed by project policy; clearly label web evidence versus uploaded literature.
- Log evidence IDs and verification decisions while minimizing sensitive content in telemetry.
- Support retention/deletion policies for original files, document versions, vectors, reports, and backups.
- Require researcher review before publishing or acting on generated scientific conclusions.

## 13. Configuration and local development

The following are **proposed environment variables**, not a claim that the current repository supports them:

```dotenv
APP_ENV=development
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/maxrim
REDIS_URL=redis://localhost:6379/0
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=maxrim_documents
OBJECT_STORAGE_ENDPOINT=http://localhost:9000
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=replace_me
TAVILY_API_KEY=replace_me_optional
WEB_SEARCH_ENABLED=false
MAX_GRAPH_ITERATIONS=3
MLFLOW_TRACKING_URI=http://localhost:5000
```

Never commit real `.env` files or API keys. Once the backend scaffold exists, the intended developer workflow is to create the environment, start Docker services, run database migrations, start the FastAPI service and background worker, then execute integration tests. **Exact commands should be added only after the repository's dependencies and entry points are implemented and verified.**

## 14. Key engineering decisions and open questions

**Agreed direction**

- Backend before frontend.
- Docling-first parsing, with a later controlled parser comparison.
- Native structure-aware chunking with retained provenance.
- Qdrant retrieval scoped to each authorized project.
- Incremental, versioned ingestion rather than rebuilding the whole index.
- Evidence-first LangGraph orchestration and citation verification.
- Evaluation and auditability as core features, not afterthoughts.

**To decide through implementation and benchmarks**

- Shared Qdrant collection vs. separate collections for special isolation needs.
- Embedding model, chunk size, tokenizer, and reranker choice.
- PDF table and figure extraction strategy.
- Whether and when external web search is allowed.
- Authentication provider and deployment environment.
- How to define acceptance thresholds for claim verification.
- Appropriate handling of copyrighted PDFs, sensitive datasets, and web-source licensing.

## 15. Current status

This README describes the **target design and development plan**. It does not imply that the APIs, agents, database tables, integrations, or tests are already implemented. Track implementation status against the roadmap and update this section after each verified milestone.

---

**MaxRIM's guiding principle:** *Every research claim should be connected to inspectable evidence, every document should belong to the right project, and every automated decision should be traceable.*
