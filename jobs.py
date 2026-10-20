"""Background work for the API: PDF ingestion in a child process, research runs in threads, restart recovery.

Jobs run on the API's own executors (1 ingest thread, RESEARCH_WORKERS research threads), never on the threadpool that
serves requests, so a batch of uploads queues instead of freezing the API."""

import hashlib
import logging
import multiprocessing
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

import ask
import db
import extraction
import fulltext
import funnel
import pubmed
import rails
import research
import screening
import web
from config import Settings

log = logging.getLogger(__name__)
RESTART_ERROR = "interrupted by server restart"
RESEARCH_K = 6
RESEARCH_WORKERS = 2
WEB_RESULTS = 3  # per sub-question; papers stay the main evidence (k=6)
INGEST_TIMEOUT_SECONDS = 30 * 60  # ponytail: fixed cap per PDF; make it a setting if large scanned PDFs need longer


@dataclass
class State:
    settings: Settings
    sessions: sessionmaker
    qdrant: object
    openai: object
    runner: object  # IngestRunner, or a test fake with .run(*args) and .shutdown()
    upload_dir: Path
    ingest_pool: object = None  # executor with .submit(fn, *args) / .shutdown(...)
    research_pool: object = None
    http_transport: object = None  # tests: an httpx MockTransport for full-text fetching


def ingest_in_child(project_id: str, pdf_path: str, filename: str, collection: str) -> dict:
    """Runs inside the worker process with its own clients. Returns the document's chunk count in Qdrant."""
    from openai import OpenAI
    from qdrant_client import QdrantClient

    import ingest

    s = Settings()
    qdrant = QdrantClient(url=s.qdrant_cluster_endpoint, api_key=s.qdrant_api_key)
    path = Path(pdf_path)
    ingest.ingest(project_id, path, qdrant=qdrant, openai_client=OpenAI(api_key=s.openai_api_key), collection=collection,
                  embed_model=s.openai_embed_model, source_filename=filename,
                  context_model=s.openai_context_model or s.openai_fast_model, contextual=s.contextual_chunks)
    f = ingest.doc_filter(project_id, ingest.sha256_file(path))
    first, _ = qdrant.scroll(collection, scroll_filter=f, limit=1, with_payload=["doc_type"])
    return {"chunk_count": qdrant.count(collection, count_filter=f, exact=True).count,
            "doc_type": first[0].payload.get("doc_type") if first else None, "first_page": fulltext.first_page_text(pdf_path)}


class IngestFailed(RuntimeError):
    """A Python error inside the child; the message is already "<ExceptionType>: <message>"."""


def _child_main(fn, args, conn) -> None:
    try:
        conn.send(("ok", fn(*args)))
    except BaseException as exc:  # noqa: BLE001 — report every child failure to the parent as text
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


class IngestRunner:
    """Runs each ingest in a fresh child process, one at a time. Docling is memory-heavy, can crash natively
    (Li2024.pdf did) or hang; in a child that fails one document instead of the API, and a hang is killed on timeout.
    ponytail: one PDF at a time; move to a separate worker polling the documents table when uploads get heavy."""

    def __init__(self, fn=ingest_in_child, timeout: float = INGEST_TIMEOUT_SECONDS):
        self.fn = fn
        self.timeout = timeout
        self._lock = threading.Lock()
        self._proc = None
        self._closed = False

    def run(self, *args) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("ingest runner is shut down")
            ctx = multiprocessing.get_context("spawn")
            recv, send = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_child_main, args=(self.fn, args, send), daemon=True)
            proc.start()
            send.close()
            self._proc = proc
            try:
                deadline = time.monotonic() + self.timeout
                # Short polls plus a liveness check: on Windows a child killed before it has taken over its end of the
                # pipe leaves the parent holding a duplicate write handle, so EOF alone never arrives.
                while not recv.poll(0.5):
                    if not proc.is_alive() and not recv.poll(0):
                        raise self._died()
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"PDF ingestion timed out after {self.timeout:.0f} s")
                try:
                    status, value = recv.recv()
                except EOFError:  # child died without reporting: native crash, or killed by shutdown()
                    raise self._died() from None
            finally:
                if proc.is_alive():
                    proc.kill()
                proc.join()
                recv.close()
                self._proc = None
        if status == "error":
            raise IngestFailed(value)
        return value

    def _died(self) -> RuntimeError:
        if self._closed:
            return RuntimeError("ingest runner is shut down")
        return RuntimeError("PDF parser crashed (native error); this PDF may be unsupported")

    def shutdown(self) -> None:
        """Refuse new work and kill the running child now (no lock: run() holds it for the whole job)."""
        self._closed = True
        proc = self._proc
        if proc is not None and proc.is_alive():
            proc.kill()


def short_error(exc: Exception, settings: Settings) -> str:
    msg = str(exc) if isinstance(exc, IngestFailed) else f"{type(exc).__name__}: {exc}"
    for secret in (settings.openai_api_key, settings.qdrant_api_key, settings.jena_ai_api_key, settings.jwt_secret,
                   settings.tavily_api_key, settings.firecrawl_api_key, settings.pubmed_api_key):
        if secret:
            msg = msg.replace(secret, "***")
    return msg[:500]


def claim(s: Session, model, row_id: uuid.UUID, from_status: str, to_status: str) -> bool:
    """Atomically move a row between statuses; False if it wasn't in from_status (someone else has it)."""
    n = s.execute(update(model).where(model.id == row_id, model.status == from_status).values(status=to_status)).rowcount
    s.commit()
    return n == 1


def ingest_document(state: State, document_id: uuid.UUID) -> None:
    with state.sessions() as s:  # short sessions: never hold a transaction open while Docling runs for minutes
        if not claim(s, db.Document, document_id, "queued", "processing"):
            return
        doc = s.get(db.Document, document_id)
        project_id = doc.project_id
        args = (str(doc.project_id), str(state.upload_dir / str(doc.project_id) / f"{doc.sha256}.pdf"), doc.filename,
                state.settings.qdrant_collection)
    try:
        out = state.runner.run(*args)
        values = {"status": "ready", "chunk_count": out["chunk_count"], "doc_type": out.get("doc_type"), "error": None}
    except Exception as exc:  # noqa: BLE001 — job boundary: any failure is recorded on the document
        values = {"status": "failed", "error": short_error(exc, state.settings)}
    with state.sessions() as s:
        s.execute(update(db.Document).where(db.Document.id == document_id).values(**values))
        s.commit()
        if values["status"] == "ready":  # an upload that names one of the project's papers gets linked to it
            funnel.match_document(s, project_id, document_id, out.get("first_page") or "")


def run_research(state: State, run_id: uuid.UUID) -> None:
    with state.sessions() as s:
        if not claim(s, db.ResearchRun, run_id, "queued", "running"):
            return
        run = s.get(db.ResearchRun, run_id)
        project_id, question, max_iterations = str(run.project_id), run.question, run.max_iterations
        use_web, use_pubmed = run.web_search, run.pubmed
        ready = list(s.scalars(select(db.Document.sha256).where(db.Document.project_id == run.project_id,
                                                                  db.Document.status == "ready")))
        history = conversation_history(s, run)
    cfg = state.settings

    web_cache: dict = {}
    topic = {"q": question}  # the follow-up-resolved question, set before research.run

    def search_fn(q: str) -> list[dict]:  # project and ready-document filters are fixed here, server-side
        if use_pubmed:  # one PubMed search for the topic (the 20 latest papers), whatever sub-question asks: the
            if "pubmed" not in web_cache:  # planner's generic sub-questions ("AUROC values?") match unrelated papers
                web_cache["pubmed"] = pubmed.search(topic["q"], api_key=cfg.pubmed_api_key, project_id=project_id)
            return list(web_cache["pubmed"])
        hits = ask.search(state.qdrant, state.openai, collection=cfg.qdrant_collection, embed_model=cfg.openai_embed_model,
                          project_id=project_id, question=q, k=RESEARCH_K, rerank_model=cfg.reranker_model,
                          rerank_api_key=cfg.jena_ai_api_key, sha256_in=ready, hybrid=cfg.hybrid_search)
        if use_web and (find := web.searcher(cfg)):  # opt-in per run; untrusted, quoted exactly like PDF chunks
            if q not in web_cache:  # a run repeats queries (sub-questions, re-checks): pay once per query
                web_cache[q] = find(q, project_id=project_id, max_results=WEB_RESULTS)
            hits += web_cache[q]
        return hits

    def blocked(message: str) -> dict:  # a refused run: no claims, no evidence, the refusal as its gap
        return {"question": question, "status": "blocked", "claims": [], "gaps": message, "evidence": [],
                "log": [{"step": "guardrail", "message": message}], "removed_claims": []}

    try:
        if refusal := rails.check_input(state.openai, cfg.openai_guard_model, question):
            values = {"status": "completed", "result": blocked(refusal), "error": None, "standalone_question": None}
        else:
            standalone = research.rewrite_followup(state.openai, cfg.openai_fast_model, history, question)
            topic["q"] = standalone
            result = research.run(state.openai, search_fn=search_fn, project_id=project_id, model=cfg.openai_model,
                                  fast_model=cfg.openai_fast_model, guard_model=cfg.openai_guard_model,
                                  question=pubmed.review_question(standalone) if use_pubmed else standalone,
                                  max_iterations=max_iterations)
            answer = "\n".join([c["text"] for c in result.get("claims") or []] + [result.get("gaps") or ""])
            if replaced := rails.check_output(state.openai, cfg.openai_guard_model, question, answer):
                result = blocked(replaced)
            values = {"status": "completed", "result": research.audit_record(result), "error": None,
                      "standalone_question": standalone if standalone != question else None}
    except Exception as exc:  # noqa: BLE001 — job boundary: any failure is recorded on the run
        values = {"status": "failed", "error": short_error(exc, cfg)}
    with state.sessions() as s:
        s.execute(update(db.ResearchRun).where(db.ResearchRun.id == run_id).values(**values, finished_at=db.utcnow()))
        s.commit()


def screen_parse(state: State):
    """screening.Parse on the screening model (OPENAI_SCREEN_MODEL, default the fast model)."""
    model = state.settings.openai_screen_model or state.settings.openai_fast_model
    return lambda instructions, input, fmt: ask.parse_llm(state.openai, model, instructions, input, fmt)


def screen_all(state: State, action_id: uuid.UUID) -> None:
    """Runs a confirmed screen_all PendingAction."""
    with state.sessions() as s:
        if not claim(s, db.PendingAction, action_id, "confirmed", "running"):
            return
        a = s.get(db.PendingAction, action_id)
        project_id, version = a.project_id, a.criteria_version
        if version != funnel.latest_version(s, project_id):  # criteria edited while this run waited in the queue
            a.status = "expired"
            s.commit()
            return
    try:
        values = {"status": "done", "error": None,
                  "result": funnel.screen_all(state.sessions, project_id, version, screen_parse(state))}
    except Exception as exc:  # noqa: BLE001 — job boundary: any failure is recorded on the action
        values = {"status": "failed", "error": short_error(exc, state.settings)}
    with state.sessions() as s:
        s.execute(update(db.PendingAction).where(db.PendingAction.id == action_id).values(**values))
        s.commit()


EXTRACT_K = 4  # passages of the paper's own PDF read per extraction


def passages_for(state: State, project_id: uuid.UUID, sha: str, query: str) -> list[str]:
    """The most relevant passages of one ingested PDF (project- and document-filtered, server-side)."""
    cfg = state.settings
    hits = ask.search(state.qdrant, state.openai, collection=cfg.qdrant_collection, embed_model=cfg.openai_embed_model,
                      project_id=str(project_id), question=query, k=EXTRACT_K, sha256_in=[sha], hybrid=cfg.hybrid_search)
    return [h.get("text") or "" for h in hits if h.get("text")]


def extract_column(state: State, action_id: uuid.UUID) -> None:
    """Runs a confirmed extract_column PendingAction: one value per target paper, from its PDF when ingested."""
    with state.sessions() as s:
        if not claim(s, db.PendingAction, action_id, "confirmed", "running"):
            return
        a = s.get(db.PendingAction, action_id)
        project_id = a.project_id
        col = funnel.project_column(s, project_id, a.args.get("column_id"))
        if col is None:
            a.status, a.error = "failed", "the column was deleted"
            s.commit()
            return
        column = extraction.Column(name=col.name, instruction=col.instruction, kind=col.kind, options=col.options)
        column_id = col.id
        targets = funnel.extract_targets(s, project_id, column_id, a.args.get("tab") or "included",
                                         only=set(a.args.get("paper_ids") or []))
    parse, query = screen_parse(state), f"{column.name}: {column.instruction}"
    counts = {"filled": 0, "fulltext": 0, "not_stated": 0, "errors": 0}

    def one(t):
        paper, sha = t
        passages = []
        if sha:
            try:
                passages = passages_for(state, project_id, sha, query)
            except Exception as e:  # noqa: BLE001 — search down: no value now (retried), never a silent downgrade
                log.warning("passages for extraction failed: %s", type(e).__name__)
                return paper, None
        return paper, extraction.extract_one(parse, column, paper, passages)

    try:
        with ThreadPoolExecutor(max_workers=screening.SCREEN_WORKERS) as pool:
            for i in range(0, len(targets), funnel.SCREEN_BATCH):
                done = list(pool.map(one, targets[i:i + funnel.SCREEN_BATCH]))
                with state.sessions() as s:
                    alive = set(s.scalars(select(db.Paper.id).where(db.Paper.id.in_([uuid.UUID(p["id"]) for p, _ in done]))))
                    for paper, x in done:
                        if x is None:  # no answer: nothing stored, the next run asks again
                            counts["errors"] += 1
                            continue
                        if uuid.UUID(paper["id"]) not in alive:
                            continue
                        try:
                            with s.begin_nested():  # a paper regrouped away meanwhile fails alone, not the batch
                                v = funnel.save_value(s, project_id, column_id, uuid.UUID(paper["id"]), x)
                        except IntegrityError:
                            continue
                        counts["filled"] += 1  # what is stored, not what the model said
                        counts["fulltext"] += v.source == "fulltext"
                        counts["not_stated"] += v.value == extraction.NOT_STATED
                    s.commit()
        values = {"status": "done", "error": None, "result": counts}
    except Exception as exc:  # noqa: BLE001 — job boundary: any failure is recorded on the action
        values = {"status": "failed", "error": short_error(exc, state.settings), "result": counts}
    with state.sessions() as s:
        s.execute(update(db.PendingAction).where(db.PendingAction.id == action_id).values(**values))
        s.commit()


def save_fetched_pdf(state: State, project_id: uuid.UUID, data: bytes, filename: str) -> tuple[uuid.UUID, bool]:
    """Store a downloaded PDF as the project's document (once per file). (document id, queued for ingest): a new
    document, or a re-queued one whose earlier ingest failed."""
    sha = hashlib.sha256(data).hexdigest()
    folder = state.upload_dir / str(project_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{sha}.pdf"
    tmp = folder / f".fetch-{uuid.uuid4().hex}"
    tmp.write_bytes(data)
    tmp.replace(path)  # atomic: a crash never leaves a truncated {sha}.pdf behind
    for _ in range(2):  # a concurrent upload of the same bytes may win the insert; then use its row
        with state.sessions() as s:
            doc = s.scalar(select(db.Document).where(db.Document.project_id == project_id, db.Document.sha256 == sha))
            if doc is not None:
                if doc.status != "failed":
                    return doc.id, False
                doc.status, doc.error, doc.chunk_count = "queued", None, None
                s.commit()
                return doc.id, True
            doc = db.Document(project_id=project_id, filename=filename[:255], sha256=sha)
            s.add(doc)
            try:
                s.commit()
                return doc.id, True
            except IntegrityError:
                s.rollback()
    raise RuntimeError("could not store the fetched PDF")


def _fetch_one(state: State, http, project_id: uuid.UUID, p: dict) -> tuple[str, uuid.UUID | None]:
    """("found" | "not_found" | "skipped", document id to ingest). A candidate PDF already standing for another
    paper (a repository's generic file) is passed over."""
    cfg = state.settings
    for f in fulltext.find_pdfs(http, p, cfg.unpaywall_email):
        data = fulltext.download(http, f.url, cfg.fulltext_max_mb * 1024 * 1024)
        if not data:
            continue
        sha = hashlib.sha256(data).hexdigest()
        with state.sessions() as s:
            doc_id = s.scalar(select(db.Document.id).where(db.Document.project_id == project_id, db.Document.sha256 == sha))
            if doc_id is not None and funnel.linked_elsewhere(s, doc_id, uuid.UUID(p["id"])):
                continue
        doc_id, queued = save_fetched_pdf(state, project_id, data, f"{p['title'][:200]}.pdf")
        with state.sessions() as s:
            paper = s.get(db.Paper, uuid.UUID(p["id"]), with_for_update=True, populate_existing=True)
            if paper is None or (paper.document_id not in (None, doc_id) and paper.fulltext_status == "uploaded"):
                return "skipped", queued and doc_id  # regrouped away, or the user uploaded one meanwhile
            if funnel.linked_elsewhere(s, doc_id, paper.id):
                if queued:  # re-queued for the paper it belongs to: still ingest it
                    state.ingest_pool.submit(ingest_document, state, doc_id)
                continue
            funnel.link_document(paper, doc_id, "open_access_found", f.url, f.license)
            s.commit()
        return "found", doc_id if queued else None
    with state.sessions() as s:
        paper = s.get(db.Paper, uuid.UUID(p["id"]), with_for_update=True)
        if paper is not None and paper.document_id is None:
            paper.fulltext_status = "not_found"
            s.commit()
    return "not_found", None


def fetch_full_text(state: State, action_id: uuid.UUID) -> None:
    """Runs a confirmed fetch_full_text PendingAction: a free PDF for each included paper that has none; the PDF
    goes through the normal ingest job. ponytail: one paper at a time, polite to the free APIs."""
    with state.sessions() as s:
        if not claim(s, db.PendingAction, action_id, "confirmed", "running"):
            return
        project_id = s.get(db.PendingAction, action_id).project_id
        papers = [funnel.paper_out(p) for p in funnel.included_without_pdf(s, project_id)]
    counts = {"found": 0, "not_found": 0, "skipped": 0, "errors": 0}
    try:
        with fulltext.client(state.http_transport) as http:
            for p in papers:
                try:
                    outcome, queue = _fetch_one(state, http, project_id, p)
                except Exception as exc:  # noqa: BLE001 — one paper's failure (disk, database) must not stop the rest
                    log.warning("full-text fetch failed for one paper: %s", type(exc).__name__)
                    outcome, queue = "errors", None
                counts[outcome] += 1
                if queue:
                    state.ingest_pool.submit(ingest_document, state, queue)
        values = {"status": "done", "error": None, "result": counts}
    except Exception as exc:  # noqa: BLE001 — job boundary: any failure is recorded on the action
        values = {"status": "failed", "error": short_error(exc, state.settings), "result": counts}
    with state.sessions() as s:
        s.execute(update(db.PendingAction).where(db.PendingAction.id == action_id).values(**values))
        s.commit()


def conversation_history(s: Session, run: db.ResearchRun) -> list[dict]:
    """Earlier completed turns of the run's conversation: their questions and verified claim texts."""
    if run.conversation_id is None:
        return []
    earlier = s.scalars(select(db.ResearchRun).where(db.ResearchRun.conversation_id == run.conversation_id,
                                                     db.ResearchRun.status == "completed",
                                                     db.ResearchRun.id != run.id)  # not created_at <: clocks tie
                        .order_by(db.ResearchRun.created_at))
    return [{"question": r.standalone_question or r.question,
             "answer": [c["text"] for c in (r.result or {}).get("claims", [])]} for r in earlier
            if (r.result or {}).get("status") != "blocked"]  # a refused question never reaches the model later


def recover_interrupted(sessions: sessionmaker) -> None:
    """On startup: work that was in flight when the server stopped will never finish; mark it failed (retryable)."""
    with sessions() as s:
        s.execute(update(db.Document).where(db.Document.status.in_(("queued", "processing")))
                  .values(status="failed", error=RESTART_ERROR))
        s.execute(update(db.ResearchRun).where(db.ResearchRun.status.in_(("queued", "running")))
                  .values(status="failed", error=RESTART_ERROR, finished_at=db.utcnow()))
        s.execute(update(db.PendingAction).where(db.PendingAction.status.in_(("confirmed", "running")))
                  .values(status="failed", error=RESTART_ERROR))  # re-confirming a new action continues the run
        s.commit()
