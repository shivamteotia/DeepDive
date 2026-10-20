"""MaxRIM HTTP API.

Run: docker compose up -d   then   uv run uvicorn app:create_app --factory
Docs: http://127.0.0.1:8000/docs
"""

import hashlib
import json
import re
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import agent
import auth
import db
import funnel
import jobs
import screening
from config import Settings

ROOT = Path(__file__).parent
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def save_pdf(file: UploadFile, dest: Path) -> str:
    """Copy the upload to dest in 1 MB chunks, enforcing the PDF signature and size limit. Returns sha256 hex.
    ponytail: Starlette has already spooled the body to a temp file; a hard body limit belongs in a reverse proxy."""
    digest, size = hashlib.sha256(), 0
    with dest.open("wb") as out:
        while chunk := file.file.read(1 << 20):
            if size == 0 and not chunk.startswith(b"%PDF-"):
                raise HTTPException(400, "File is not a PDF")
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"File larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
            digest.update(chunk)
            out.write(chunk)
    if size == 0:
        raise HTTPException(400, "File is not a PDF")
    return digest.hexdigest()


def display_name(filename: str | None) -> str:
    return re.split(r"[\\/]", filename or "")[-1][:255] or "upload.pdf"


class RegisterIn(BaseModel):
    email: str = Field(pattern=EMAIL_PATTERN, max_length=320)
    password: str = Field(min_length=12, max_length=1024)


class LoginIn(BaseModel):
    email: str = Field(max_length=320)
    password: str = Field(max_length=1024)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    email: str


Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class ProjectIn(BaseModel):
    name: Name


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    name: str
    created_at: datetime


Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=2000)]


class RunIn(BaseModel):
    question: Question
    max_iterations: int = Field(3, ge=1, le=5)
    conversation_id: uuid.UUID | None = None
    web_search: bool = False


class DuplicateAction(BaseModel):
    action: Literal["merge", "dismiss", "keep_article"]


class SentenceIn(BaseModel):
    sentence: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=2000)]


class CriteriaIn(BaseModel):
    criteria: list[screening.Criterion] = Field(max_length=30)


class ActionIn(BaseModel):
    tool: Literal["screen_all", "fetch_full_text"]


class FeedbackIn(BaseModel):
    feedback: Literal["up", "down"]


class DecisionIn(BaseModel):
    """Send decision (+ reason) and/or blind_label; a field left out is left unchanged, null clears it."""
    decision: Literal["include", "exclude"] | None = None
    reason: Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)] = ""
    blind_label: Literal["include", "exclude"] | None = None


class AgentIn(BaseModel):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class ConversationIn(BaseModel):
    title: Annotated[str, StringConstraints(strip_whitespace=True, max_length=200)] = ""


TITLE_CHARS = 80


def public_iteration(entry: dict) -> dict:
    """A verify-log entry for clients: rejected claims keep only their verdict. Their text (and the rationale that
    may quote it) stays in the stored audit record; it is exactly what we decided not to present."""
    claims = [c if c["verdict"] != "UNSUPPORTED" else {"verdict": c["verdict"], "evidence_ids": c["evidence_ids"]}
              for c in entry.get("claims", [])]
    return {**entry, "claims": claims}


def run_out(run: db.ResearchRun) -> dict:
    r = run.result
    return {"id": run.id, "project_id": run.project_id, "question": run.question, "max_iterations": run.max_iterations,
            "status": run.status, "error": run.error, "created_at": run.created_at, "finished_at": run.finished_at,
            "conversation_id": run.conversation_id, "web_search": bool(run.web_search),
            "standalone_question": run.standalone_question,
            "result": {k: r[k] for k in ("status", "claims", "gaps")} if r else None}


def conversation_out(c: db.Conversation) -> dict:
    return {"id": c.id, "project_id": c.project_id, "title": c.title, "created_at": c.created_at,
            "updated_at": c.updated_at}


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    project_id: uuid.UUID
    filename: str
    sha256: str
    status: str
    chunk_count: int | None
    doc_type: str | None = None
    error: str | None
    created_at: datetime
    updated_at: datetime


def create_app(settings: Settings | None = None, *, qdrant=None, openai_client=None, runner=None,
               upload_dir: Path | None = None, ingest_pool=None, research_pool=None) -> FastAPI:
    settings = settings or Settings()
    if len(settings.jwt_secret) < 32:
        raise ValueError("JWT_SECRET must be at least 32 characters")
    if qdrant is None:
        from qdrant_client import QdrantClient
        qdrant = QdrantClient(url=settings.qdrant_cluster_endpoint, api_key=settings.qdrant_api_key)
    if openai_client is None:
        from openai import OpenAI
        openai_client = OpenAI(api_key=settings.openai_api_key)
    state = jobs.State(settings=settings, sessions=db.make_sessions(settings.database_url), qdrant=qdrant,
                       openai=openai_client, runner=runner or jobs.IngestRunner(),
                       upload_dir=upload_dir or ROOT / "data" / "uploads",
                       ingest_pool=ingest_pool or ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest"),
                       research_pool=research_pool or ThreadPoolExecutor(max_workers=jobs.RESEARCH_WORKERS,
                                                                         thread_name_prefix="research"))
    jobs.recover_interrupted(state.sessions)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        state.runner.shutdown()  # kills a running Docling child now instead of waiting minutes
        state.ingest_pool.shutdown(wait=False, cancel_futures=True)
        state.research_pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="MaxRIM", lifespan=lifespan)
    app.state.maxrim = state

    def get_db() -> Iterator[Session]:
        with state.sessions() as s:
            yield s

    DB = Annotated[Session, Depends(get_db)]
    bearer = HTTPBearer(auto_error=False)

    def current_user(creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)], s: DB) -> db.User:
        uid = auth.decode_token(creds.credentials, settings.jwt_secret) if creds else None
        user = s.get(db.User, uid) if uid else None
        if user is None:
            raise HTTPException(401, "Not authenticated", headers={"WWW-Authenticate": "Bearer"})
        return user

    CurrentUser = Annotated[db.User, Depends(current_user)]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/auth/register", status_code=201, response_model=UserOut)
    def register(body: RegisterIn, s: DB):
        user = db.User(email=body.email.lower(), password_hash=auth.hash_password(body.password))
        s.add(user)
        try:
            s.commit()
        except IntegrityError:
            s.rollback()
            raise HTTPException(409, "Email already registered") from None
        return user

    @app.post("/auth/login")
    def login(body: LoginIn, s: DB):
        user = s.scalar(select(db.User).where(db.User.email == body.email.lower()))
        ok = auth.verify_password(body.password, user.password_hash if user else auth.DUMMY_HASH)
        if user is None or not ok:
            raise HTTPException(401, "Invalid email or password")
        return {"access_token": auth.create_token(user.id, settings.jwt_secret, settings.access_token_minutes),
                "token_type": "bearer"}

    def owned_project(project_id: uuid.UUID, user: CurrentUser, s: DB) -> db.Project:
        project = s.get(db.Project, project_id)
        if project is None or project.owner_id != user.id:
            raise HTTPException(404, "Project not found")
        return project

    OwnedProject = Annotated[db.Project, Depends(owned_project)]

    @app.post("/api/v1/projects", status_code=201, response_model=ProjectOut)
    def create_project(body: ProjectIn, user: CurrentUser, s: DB):
        project = db.Project(owner_id=user.id, name=body.name)
        s.add(project)
        s.commit()
        return project

    @app.get("/api/v1/projects", response_model=list[ProjectOut])
    def list_projects(user: CurrentUser, s: DB):
        return s.scalars(select(db.Project).where(db.Project.owner_id == user.id).order_by(db.Project.created_at)).all()

    @app.get("/api/v1/projects/{project_id}/documents", response_model=list[DocumentOut])
    def list_documents(project: OwnedProject, s: DB):
        return s.scalars(select(db.Document).where(db.Document.project_id == project.id)
                         .order_by(db.Document.created_at)).all()

    def find_document(s: Session, project_id: uuid.UUID, sha: str) -> db.Document | None:
        return s.scalar(select(db.Document).where(db.Document.project_id == project_id, db.Document.sha256 == sha))

    def store_pdf(file: UploadFile, project: db.Project, s: Session) -> tuple[db.Document, bool]:
        """Save an uploaded PDF as the project's document. (document, queued): queued=False when the same file is
        already queued, processing or ready (no duplicate); the caller submits the ingest job when queued."""
        folder = state.upload_dir / str(project.id)
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f".upload-{uuid.uuid4().hex}"
        try:
            sha = save_pdf(file, tmp)
            doc = find_document(s, project.id, sha)
            if doc is not None and doc.status != "failed":
                return doc, False
            tmp.replace(folder / f"{sha}.pdf")
        finally:
            tmp.unlink(missing_ok=True)
        if doc is None:
            doc = db.Document(project_id=project.id, filename=display_name(file.filename), sha256=sha)
            s.add(doc)
        else:  # retry of a failed ingest
            doc.status, doc.error, doc.chunk_count = "queued", None, None
        try:
            s.commit()
        except IntegrityError:  # the same file was uploaded concurrently; return the row that won
            s.rollback()
            return find_document(s, project.id, sha), False
        return doc, True

    @app.post("/api/v1/projects/{project_id}/documents", status_code=202, response_model=DocumentOut)
    def upload_document(file: UploadFile, response: Response, project: OwnedProject, s: DB):
        doc, queued = store_pdf(file, project, s)
        if queued:
            state.ingest_pool.submit(jobs.ingest_document, state, doc.id)
        else:
            response.status_code = 200
        return doc

    @app.post("/api/v1/projects/{project_id}/papers/{paper_id}/pdf", status_code=202, response_model=DocumentOut)
    def upload_paper_pdf(paper_id: uuid.UUID, file: UploadFile, response: Response, project: OwnedProject, s: DB):
        """The user's own copy of a paper's PDF (e.g. one with no free copy): linked to that paper, then ingested."""
        paper = s.get(db.Paper, paper_id)
        if paper is None or paper.project_id != project.id:
            raise HTTPException(404, "Paper not found")
        current = s.get(db.Document, paper.document_id) if paper.document_id else None
        if current is not None and current.status != "failed":  # never silently replace a paper's PDF
            raise HTTPException(409, "This paper already has a PDF")
        doc, queued = store_pdf(file, project, s)
        if funnel.linked_elsewhere(s, doc.id, paper.id):
            if queued:  # a failed copy re-queued by this upload: ingest it for the paper it belongs to
                state.ingest_pool.submit(jobs.ingest_document, state, doc.id)
            raise HTTPException(409, "This PDF is already linked to another paper")
        funnel.link_document(paper, doc.id, "uploaded")
        s.commit()
        if queued:
            state.ingest_pool.submit(jobs.ingest_document, state, doc.id)
        else:
            response.status_code = 200
        return doc

    @app.get("/api/v1/projects/{project_id}/documents/{document_id}", response_model=DocumentOut)
    def get_document(document_id: uuid.UUID, project: OwnedProject, s: DB):
        doc = s.get(db.Document, document_id)
        if doc is None or doc.project_id != project.id:
            raise HTTPException(404, "Document not found")
        return doc

    def owned_run(run_id: uuid.UUID, user: CurrentUser, s: DB) -> db.ResearchRun:
        run = s.get(db.ResearchRun, run_id)
        if run is None or s.get(db.Project, run.project_id).owner_id != user.id:
            raise HTTPException(404, "Research run not found")
        return run

    OwnedRun = Annotated[db.ResearchRun, Depends(owned_run)]

    @app.post("/api/v1/projects/{project_id}/research-runs", status_code=202)
    def start_research_run(body: RunIn, project: OwnedProject, s: DB):
        convo = None
        if body.conversation_id is not None:
            convo = s.get(db.Conversation, body.conversation_id)
            if convo is None or convo.project_id != project.id:  # also covers other users' conversations
                raise HTTPException(404, "Conversation not found")
            busy = s.scalar(select(db.ResearchRun.id).where(db.ResearchRun.conversation_id == convo.id,
                                                            db.ResearchRun.status.in_(("queued", "running"))).limit(1))
            if busy is not None:  # its answer isn't history yet; a follow-up now would lose its context
                raise HTTPException(409, "The previous question in this chat is still running")
        if body.web_search and not (settings.web_search_enabled and settings.tavily_api_key):
            raise HTTPException(400, "Web search is not enabled on this server")
        ready = s.scalar(select(db.Document.id).where(db.Document.project_id == project.id,
                                                       db.Document.status == "ready").limit(1))
        if ready is None and not body.web_search:  # web-only chat needs no papers; papers added later join later turns
            raise HTTPException(409, "No indexed documents yet — upload a paper or turn on web search")
        run = db.ResearchRun(project_id=project.id, question=body.question, max_iterations=body.max_iterations,
                             conversation_id=convo.id if convo else None, web_search=body.web_search)
        s.add(run)
        if convo is not None:
            if convo.title == db.Conversation.title.default.arg:  # first question names the chat
                convo.title = body.question[:TITLE_CHARS]
            convo.updated_at = db.utcnow()
        s.commit()
        state.research_pool.submit(jobs.run_research, state, run.id)
        return run_out(run)

    @app.get("/api/v1/research-runs/{run_id}")
    def get_research_run(run: OwnedRun):
        return run_out(run)

    def owned_conversation(conversation_id: uuid.UUID, user: CurrentUser, s: DB) -> db.Conversation:
        c = s.get(db.Conversation, conversation_id)
        if c is None or s.get(db.Project, c.project_id).owner_id != user.id:
            raise HTTPException(404, "Conversation not found")
        return c

    OwnedConversation = Annotated[db.Conversation, Depends(owned_conversation)]

    @app.get("/api/v1/config")
    def client_config(user: CurrentUser):
        return {"web_search": bool(settings.web_search_enabled and settings.tavily_api_key)}

    @app.post("/api/v1/projects/{project_id}/conversations", status_code=201)
    def create_conversation(body: ConversationIn, project: OwnedProject, s: DB):
        c = db.Conversation(project_id=project.id, **({"title": body.title} if body.title else {}))
        s.add(c)
        s.commit()
        return conversation_out(c)

    @app.get("/api/v1/projects/{project_id}/conversations")
    def list_conversations(project: OwnedProject, s: DB):
        rows = s.scalars(select(db.Conversation).where(db.Conversation.project_id == project.id)
                         .order_by(db.Conversation.updated_at.desc()))
        return [conversation_out(c) for c in rows]

    @app.get("/api/v1/conversations/{conversation_id}")
    def get_conversation(c: OwnedConversation, s: DB):
        runs = s.scalars(select(db.ResearchRun).where(db.ResearchRun.conversation_id == c.id)
                         .order_by(db.ResearchRun.created_at))
        return {**conversation_out(c), "runs": [run_out(r) for r in runs], "messages": agent_messages(s, c.id)}

    def agent_messages(s: Session, conversation_id: uuid.UUID) -> list[dict]:
        rows = s.scalars(select(db.AgentMessage).where(db.AgentMessage.conversation_id == conversation_id)
                         .order_by(db.AgentMessage.created_at, db.AgentMessage.id))
        return [{"id": str(m.id), "role": m.role, "content": m.content, "created_at": m.created_at} for m in rows]

    @app.post("/api/v1/conversations/{conversation_id}/agent")
    def agent_turn(body: AgentIn, c: OwnedConversation, s: DB):
        """One message to the screening agent; runs its tool loop now and returns the new messages."""
        project = s.get(db.Project, c.project_id)
        # ponytail: loads the whole chat and holds a DB connection for the turn (model calls included); fine for a
        # few users, move the loop to a background job + polling if many chat at once.
        history = [{"role": m["role"], "content": m["content"]["text"]} for m in agent_messages(s, c.id)
                   if m["role"] in ("user", "assistant") and m["content"].get("text")]

        def ask_research(question: str) -> dict:
            try:
                return start_research_run(RunIn(question=question, conversation_id=c.id), project, s)
            except HTTPException as e:
                return {"error": e.detail}
            except ValueError as e:  # RunIn validation (e.g. a too-short question)
                return {"error": str(e)[:300]}

        ctx = agent.ToolContext(s=s, project_id=c.project_id, parse=jobs.screen_parse(state), ask_research=ask_research)
        new = agent.run_turn(state.openai, settings.openai_model, history, body.message, ctx)
        rows = [db.AgentMessage(conversation_id=c.id, role="user", content={"text": body.message})]
        rows += [db.AgentMessage(conversation_id=c.id, role=m["role"],
                                 content=json.loads(json.dumps(m["content"], default=str))) for m in new]
        for n, r in enumerate(rows):  # one timestamp per turn would tie; keep the order explicit
            r.created_at = db.utcnow() + timedelta(microseconds=n)
        s.add_all(rows)
        c.updated_at = db.utcnow()
        s.commit()
        return {"messages": agent_messages(s, c.id)[-len(rows):]}

    @app.delete("/api/v1/conversations/{conversation_id}", status_code=204)
    def delete_conversation(c: OwnedConversation, s: DB):
        s.execute(update(db.ResearchRun).where(db.ResearchRun.conversation_id == c.id).values(conversation_id=None))
        s.delete(c)
        s.commit()
        return Response(status_code=204)

    @app.get("/api/v1/research-runs/{run_id}/evidence")
    def get_research_evidence(run: OwnedRun):
        return (run.result or {}).get("evidence", [])

    @app.get("/api/v1/research-runs/{run_id}/verification")
    def get_research_verification(run: OwnedRun):
        r = run.result or {}
        return {"status": r.get("status"), "claims": r.get("claims", []), "gaps": r.get("gaps", ""),
                "iterations": [public_iteration(e) for e in r.get("log", []) if e.get("step") == "verify"]}

    # --- screening funnel: reference imports and papers ---

    @app.post("/api/v1/projects/{project_id}/reference-files", status_code=201)
    def upload_reference_file(file: UploadFile, response: Response, project: OwnedProject, s: DB):
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large")
        try:
            summary, created = funnel.import_file(s, project.id, display_name(file.filename), data)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        except IntegrityError:  # the same file imported concurrently: the other request won
            s.rollback()
            summary, created = funnel.import_file(s, project.id, display_name(file.filename), data)
        if not created:
            response.status_code = 200
        return summary

    @app.get("/api/v1/projects/{project_id}/papers")
    def list_papers(project: OwnedProject, s: DB, tab: Literal[funnel.TABS] = "all",
                    offset: Annotated[int, Field(ge=0)] = 0, limit: Annotated[int, Field(ge=1, le=500)] = 50,
                    blind: bool = False):
        if blind and tab in ("included", "review", "excluded", "fulltext"):
            raise HTTPException(422, "Blind mode lists papers without the AI's decisions; use tab=all")
        return funnel.list_papers(s, project.id, tab, offset, limit, blind)

    @app.post("/api/v1/projects/{project_id}/papers/{paper_id}/split")
    def split_paper(paper_id: uuid.UUID, project: OwnedProject, s: DB):
        out = funnel.split(s, project.id, paper_id)
        if out is None:
            raise HTTPException(404, "Paper not found")
        if not out:
            raise HTTPException(409, "This paper has only one reference")
        return {"papers": out}

    @app.post("/api/v1/projects/{project_id}/possible-duplicates/{dup_id}")
    def resolve_duplicate(dup_id: uuid.UUID, body: DuplicateAction, project: OwnedProject, s: DB):
        try:
            out = funnel.resolve_duplicate(s, project.id, dup_id, body.action)
        except funnel.Conflict as e:
            raise HTTPException(409, str(e)) from None
        if out is None:
            raise HTTPException(404, "Possible duplicate not found")
        return out

    # --- screening: criteria, test run, decisions, confirmed actions ---

    def new_criteria(s: Session, project: db.Project, criteria: list[screening.Criterion], sentence: str | None):
        problem = screening.validate_criteria(criteria)
        if problem:
            raise HTTPException(422, problem)
        return funnel.save_criteria(s, project.id, criteria, sentence)

    @app.post("/api/v1/projects/{project_id}/criteria", status_code=201)
    def draft_criteria(body: SentenceIn, project: OwnedProject, s: DB):
        criteria = screening.draft_criteria(jobs.screen_parse(state), body.sentence)
        if not criteria:
            raise HTTPException(502, "Could not draft criteria from that sentence; try again or write them yourself")
        return new_criteria(s, project, criteria, body.sentence)

    @app.put("/api/v1/projects/{project_id}/criteria", status_code=201)
    def edit_criteria(body: CriteriaIn, project: OwnedProject, s: DB):
        return new_criteria(s, project, body.criteria, None)

    @app.get("/api/v1/projects/{project_id}/criteria")
    def get_criteria(project: OwnedProject, s: DB):
        cs = funnel.criteria_set(s, project.id)
        if cs is None:
            raise HTTPException(404, "No screening criteria yet")
        return funnel.criteria_out(cs)

    @app.post("/api/v1/projects/{project_id}/screening/test")
    def test_screen(project: OwnedProject, s: DB):
        out = funnel.run_test(s, project.id, jobs.screen_parse(state))
        if out is None:
            raise HTTPException(409, "No screening criteria yet")
        return out

    @app.patch("/api/v1/projects/{project_id}/screening/test/{paper_id}")
    def test_feedback(paper_id: uuid.UUID, body: FeedbackIn, project: OwnedProject, s: DB):
        out = funnel.set_feedback(s, project.id, paper_id, body.feedback)
        if out is None:
            raise HTTPException(404, "That paper is not in the test run of the current criteria")
        return out

    @app.patch("/api/v1/projects/{project_id}/papers/{paper_id}/decision")
    def set_decision(paper_id: uuid.UUID, body: DecisionIn, project: OwnedProject, s: DB):
        changes = body.model_dump(include=body.model_fields_set)
        if not {"decision", "blind_label"} & changes.keys():
            raise HTTPException(422, "Send decision and/or blind_label")
        try:
            out = funnel.set_decision(s, project.id, paper_id, changes)
        except funnel.Conflict as e:
            raise HTTPException(409, str(e)) from None
        if out is None:
            raise HTTPException(404, "Paper not found")
        return out

    @app.post("/api/v1/projects/{project_id}/actions", status_code=201)
    def propose_action(body: ActionIn, project: OwnedProject, s: DB):
        try:
            return funnel.propose(s, project.id, body.tool)
        except funnel.Conflict as e:
            raise HTTPException(409, str(e)) from None

    def owned_action(action_id: uuid.UUID, user: CurrentUser, s: DB) -> db.PendingAction:
        a = s.get(db.PendingAction, action_id)
        if a is None or s.get(db.Project, a.project_id).owner_id != user.id:
            raise HTTPException(404, "Action not found")
        return a

    OwnedAction = Annotated[db.PendingAction, Depends(owned_action)]

    @app.get("/api/v1/actions/{action_id}")
    def get_action(a: OwnedAction):
        return funnel.action_out(a)

    @app.post("/api/v1/actions/{action_id}/cancel")
    def cancel_action(a: OwnedAction, s: DB):
        if not jobs.claim(s, db.PendingAction, a.id, "pending", "cancelled"):
            raise HTTPException(409, f"This action is {a.status}")
        s.refresh(a)
        return funnel.action_out(a)

    @app.post("/api/v1/actions/{action_id}/confirm")
    def confirm_action(a: OwnedAction, s: DB):
        """The only way a paid step starts: its owner confirms this server-issued action (never a chat message)."""
        if a.status != "pending":
            raise HTTPException(409, f"This action is {a.status}")
        if a.criteria_version != funnel.latest_version(s, a.project_id):
            a.status = "expired"
            s.commit()
            raise HTTPException(409, "expired: the criteria changed after this estimate; propose the run again")
        funnel.lock(s, a.project_id)  # two confirms at once: the second sees the first as busy
        busy = s.scalar(select(db.PendingAction.id).where(db.PendingAction.project_id == a.project_id,
                                                          db.PendingAction.status.in_(("confirmed", "running"))).limit(1))
        if busy is not None:
            raise HTTPException(409, "Another screening run is in progress")
        if not jobs.claim(s, db.PendingAction, a.id, "pending", "confirmed"):
            raise HTTPException(409, "This action was already confirmed")
        state.research_pool.submit(jobs.screen_all if a.tool == "screen_all" else jobs.fetch_full_text, state, a.id)
        s.refresh(a)
        return funnel.action_out(a)

    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(ROOT / "static" / "index.html")

    return app
