"""Postgres (SQLite in unit tests) models for users, projects, documents and research runs."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, Uuid, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


def utcnow() -> datetime:
    return datetime.now(UTC)


def as_utc(t: datetime) -> datetime:
    """A stored time as aware UTC: Postgres returns it in the session's time zone, SQLite naive (already UTC)."""
    return t.astimezone(UTC) if t.tzinfo else t.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    password_hash: Mapped[str] = mapped_column(String(300))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("project_id", "sha256"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20), default="queued")
    chunk_count: Mapped[int | None] = mapped_column(Integer)
    doc_type: Mapped[str | None] = mapped_column(String(20))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Conversation(Base):
    """Chat memory: an ordered list of research runs in one project."""
    __tablename__ = "conversations"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True)
    title: Mapped[str] = mapped_column(String(200), default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ResearchRun(Base):
    __tablename__ = "research_runs"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    max_iterations: Mapped[int] = mapped_column(Integer, default=3)
    status: Mapped[str] = mapped_column(String(20), default="queued")
    result: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id", ondelete="SET NULL"),
                                                               index=True)
    web_search: Mapped[bool] = mapped_column(Boolean, default=False)
    pubmed: Mapped[bool] = mapped_column(Boolean, default=False)  # evidence = latest PubMed abstracts, not the PDFs
    standalone_question: Mapped[str | None] = mapped_column(Text)


class AgentMessage(Base):
    """One screening-agent chat message: role user / assistant / tool (a tool call and its JSON result)."""
    __tablename__ = "agent_messages"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(10))
    content: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ReferenceImport(Base):
    """One uploaded reference export (.bib / .nbib). The same file twice in a project imports nothing."""
    __tablename__ = "reference_imports"
    __table_args__ = (UniqueConstraint("project_id", "sha256"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64))
    record_count: Mapped[int] = mapped_column(Integer, default=0)
    errors: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Paper(Base):
    """A deduplicated paper: canonical fields from its best reference; its references point here."""
    __tablename__ = "papers"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True)
    title: Mapped[str] = mapped_column(Text)
    abstract: Mapped[str] = mapped_column(Text, default="")
    year: Mapped[int | None] = mapped_column(Integer)
    doi: Mapped[str] = mapped_column(String(255), default="")
    pmid: Mapped[str] = mapped_column(String(20), default="")
    pmcid: Mapped[str] = mapped_column(String(20), default="")
    journal: Mapped[str] = mapped_column(Text, default="")
    first_author: Mapped[str] = mapped_column(String(200), default="")
    authors: Mapped[list] = mapped_column(JSON, default=list)
    language: Mapped[str] = mapped_column(String(100), default="")  # "fre|eng|spa"
    pub_types: Mapped[list] = mapped_column(JSON, default=list)
    sources: Mapped[list] = mapped_column(JSON, default=list)
    dedup_reason: Mapped[str] = mapped_column(String(30), default="single")
    document_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"))
    fulltext_status: Mapped[str] = mapped_column(String(30), default="not_tried")
    fulltext_url: Mapped[str | None] = mapped_column(Text)
    fulltext_license: Mapped[str | None] = mapped_column(String(100))
    # A preprint the user replaced by its journal article ("keep the journal article"): kept on record, not screened.
    superseded_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("papers.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Reference(Base):
    """One source record from an import. `pinned` = placed by the user (split/merge): dedup never moves it."""
    __tablename__ = "references"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True)
    import_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("reference_imports.id"), index=True)
    paper_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("papers.id", ondelete="SET NULL"), index=True)
    source: Mapped[str] = mapped_column(String(20))
    source_key: Mapped[str] = mapped_column(String(255))
    fields: Mapped[dict] = mapped_column(JSON)  # references.Record without raw
    raw: Mapped[dict] = mapped_column(JSON)
    pinned: Mapped[bool] = mapped_column(Boolean, default=False)


class PossibleDuplicate(Base):
    __tablename__ = "possible_duplicates"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True)
    paper_a: Mapped[uuid.UUID] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"))
    paper_b: Mapped[uuid.UUID] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"))
    similarity: Mapped[float] = mapped_column(default=0.0)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)


class CriteriaSet(Base):
    """One version of a project's screening criteria; editing makes a new version."""
    __tablename__ = "criteria_sets"
    __table_args__ = (UniqueConstraint("project_id", "version"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    sentence: Mapped[str] = mapped_column(Text, default="")
    criteria: Mapped[list] = mapped_column(JSON)  # screening.Criterion dicts
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ScreeningDecision(Base):
    """A paper screened under one criteria version. test=True rows come from the 20-paper test run and are not the
    paper's decision. Final decision = human_decision if set, else ai_decision."""
    __tablename__ = "screening_decisions"
    __table_args__ = (UniqueConstraint("paper_id", "criteria_version", "test"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True)
    paper_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    criteria_version: Mapped[int] = mapped_column(Integer)
    verdicts: Mapped[list] = mapped_column(JSON, default=list)  # screening.Verdict dicts
    ai_decision: Mapped[str | None] = mapped_column(String(10))
    human_decision: Mapped[str | None] = mapped_column(String(10))
    human_reason: Mapped[str] = mapped_column(Text, default="")
    decided_by: Mapped[str] = mapped_column(String(10), default="ai")
    disagrees: Mapped[bool] = mapped_column(Boolean, default=False)  # human decision differs from the AI's
    blind_label: Mapped[str | None] = mapped_column(String(10))
    feedback: Mapped[str | None] = mapped_column(String(10))  # test rows: thumbs "up" / "down"
    test: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class ExtractColumn(Base):
    """A data extraction column (study design, sample size, ...) filled per paper by extraction.extract_one."""
    __tablename__ = "extract_columns"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    instruction: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(String(10))
    options: Mapped[list | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ExtractValue(Base):
    __tablename__ = "extract_values"
    __table_args__ = (UniqueConstraint("column_id", "paper_id"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True)
    column_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("extract_columns.id", ondelete="CASCADE"), index=True)
    paper_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    value: Mapped[str] = mapped_column(Text)
    quote: Mapped[str] = mapped_column(Text, default="")
    reason: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(10))  # fulltext | abstract
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class LabelSample(Base):
    """The fixed, stratified sample a researcher blind-labels for the screening recall eval: one row per paper, in a
    shuffled order, with the AI pile it was drawn from and that pile's size when the sample was made (the weights)."""
    __tablename__ = "label_samples"
    __table_args__ = (UniqueConstraint("project_id", "paper_id"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True)
    paper_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer)
    stratum: Mapped[str] = mapped_column(String(10))  # the AI's decision when sampled: include | review | exclude
    stratum_size: Mapped[int] = mapped_column(Integer)  # papers in that pile when sampled
    stratum_sampled: Mapped[int] = mapped_column(Integer)  # papers drawn from it
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PendingAction(Base):
    """A paid step proposed by the agent or the UI; it runs only once its owner confirms it through the API."""
    __tablename__ = "pending_actions"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"), index=True)
    tool: Mapped[str] = mapped_column(String(30))
    args: Mapped[dict] = mapped_column(JSON, default=dict)
    criteria_version: Mapped[int | None] = mapped_column(Integer)
    estimate: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    result: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# Columns added after research_runs existed. create_all never alters a table, so add them in place (Postgres only:
# SQLite is used by tests and always starts from fresh tables).
ADDED_COLUMNS = [
    "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL",
    "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS web_search BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS standalone_question TEXT",
    "CREATE INDEX IF NOT EXISTS ix_research_runs_conversation_id ON research_runs (conversation_id)",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_type VARCHAR(20)",
    "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS pubmed BOOLEAN NOT NULL DEFAULT FALSE",
]


def make_sessions(database_url: str) -> sessionmaker:
    """Create missing tables and return a session factory.
    ponytail: create_all, no migrations; add Alembic at the first schema change on data that must be kept."""
    kw = {}
    if database_url.startswith("sqlite"):
        kw = {"connect_args": {"check_same_thread": False}}
        if database_url in ("sqlite://", "sqlite:///:memory:"):  # in-memory: one shared connection or the tables vanish
            kw["poolclass"] = StaticPool  # (a file database gets a connection per thread: shared, sqlite3 can crash)
    engine = create_engine(database_url, **kw)
    Base.metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            for stmt in ADDED_COLUMNS:
                conn.execute(text(stmt))
    return sessionmaker(engine, expire_on_commit=False)
