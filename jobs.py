"""Background work for the API: PDF ingestion in a child process, research runs in threads, restart recovery.

Jobs run on the API's own executors (1 ingest thread, RESEARCH_WORKERS research threads), never on the threadpool that
serves requests, so a batch of uploads queues instead of freezing the API."""

import multiprocessing
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

import ask
import db
import research
import web
from config import Settings

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


def ingest_in_child(project_id: str, pdf_path: str, filename: str, collection: str) -> int:
    """Runs inside the worker process with its own clients. Returns the document's chunk count in Qdrant."""
    from openai import OpenAI
    from qdrant_client import QdrantClient

    import ingest

    s = Settings()
    qdrant = QdrantClient(url=s.qdrant_cluster_endpoint, api_key=s.qdrant_api_key)
    path = Path(pdf_path)
    ingest.ingest(project_id, path, qdrant=qdrant, openai_client=OpenAI(api_key=s.openai_api_key), collection=collection,
                  embed_model=s.openai_embed_model, source_filename=filename)
    return qdrant.count(collection, count_filter=ingest.doc_filter(project_id, ingest.sha256_file(path)), exact=True).count


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
                   settings.tavily_api_key):
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
        args = (str(doc.project_id), str(state.upload_dir / str(doc.project_id) / f"{doc.sha256}.pdf"), doc.filename,
                state.settings.qdrant_collection)
    try:
        values = {"status": "ready", "chunk_count": state.runner.run(*args), "error": None}
    except Exception as exc:  # noqa: BLE001 — job boundary: any failure is recorded on the document
        values = {"status": "failed", "error": short_error(exc, state.settings)}
    with state.sessions() as s:
        s.execute(update(db.Document).where(db.Document.id == document_id).values(**values))
        s.commit()


def run_research(state: State, run_id: uuid.UUID) -> None:
    with state.sessions() as s:
        if not claim(s, db.ResearchRun, run_id, "queued", "running"):
            return
        run = s.get(db.ResearchRun, run_id)
        project_id, question, max_iterations = str(run.project_id), run.question, run.max_iterations
        use_web = run.web_search
        ready = list(s.scalars(select(db.Document.sha256).where(db.Document.project_id == run.project_id,
                                                                  db.Document.status == "ready")))
        history = conversation_history(s, run)
    cfg = state.settings

    def search_fn(q: str) -> list[dict]:  # project and ready-document filters are fixed here, server-side
        hits = ask.search(state.qdrant, state.openai, collection=cfg.qdrant_collection, embed_model=cfg.openai_embed_model,
                          project_id=project_id, question=q, k=RESEARCH_K, rerank_model=cfg.reranker_model,
                          rerank_api_key=cfg.jena_ai_api_key, sha256_in=ready)
        if use_web:  # opt-in per run; web text is untrusted and quoted exactly like PDF chunks
            hits += web.search(q, api_key=cfg.tavily_api_key, project_id=project_id,
                               timeout=cfg.web_search_timeout_seconds, max_results=WEB_RESULTS)
        return hits

    try:
        standalone = research.rewrite_followup(state.openai, cfg.openai_fast_model, history, question)
        result = research.run(state.openai, search_fn=search_fn, project_id=project_id, model=cfg.openai_model,
                              fast_model=cfg.openai_fast_model, guard_model=cfg.openai_guard_model, question=standalone,
                              max_iterations=max_iterations)
        values = {"status": "completed", "result": research.audit_record(result), "error": None,
                  "standalone_question": standalone if standalone != question else None}
    except Exception as exc:  # noqa: BLE001 — job boundary: any failure is recorded on the run
        values = {"status": "failed", "error": short_error(exc, cfg)}
    with state.sessions() as s:
        s.execute(update(db.ResearchRun).where(db.ResearchRun.id == run_id).values(**values, finished_at=db.utcnow()))
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
             "answer": [c["text"] for c in (r.result or {}).get("claims", [])]} for r in earlier]


def recover_interrupted(sessions: sessionmaker) -> None:
    """On startup: work that was in flight when the server stopped will never finish; mark it failed (retryable)."""
    with sessions() as s:
        s.execute(update(db.Document).where(db.Document.status.in_(("queued", "processing")))
                  .values(status="failed", error=RESTART_ERROR))
        s.execute(update(db.ResearchRun).where(db.ResearchRun.status.in_(("queued", "running")))
                  .values(status="failed", error=RESTART_ERROR, finished_at=db.utcnow()))
        s.commit()
