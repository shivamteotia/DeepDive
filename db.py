"""Postgres (SQLite in unit tests) models for users, projects, documents and research runs."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, Uuid, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


def utcnow() -> datetime:
    return datetime.now(UTC)


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
    standalone_question: Mapped[str | None] = mapped_column(Text)


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


# Columns added after research_runs existed. create_all never alters a table, so add them in place (Postgres only:
# SQLite is used by tests and always starts from fresh tables).
ADDED_COLUMNS = [
    "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL",
    "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS web_search BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS standalone_question TEXT",
    "CREATE INDEX IF NOT EXISTS ix_research_runs_conversation_id ON research_runs (conversation_id)",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_type VARCHAR(20)",
]


def make_sessions(database_url: str) -> sessionmaker:
    """Create missing tables and return a session factory.
    ponytail: create_all, no migrations; add Alembic at the first schema change on data that must be kept."""
    kw = {"connect_args": {"check_same_thread": False}, "poolclass": StaticPool} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, **kw)
    Base.metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            for stmt in ADDED_COLUMNS:
                conn.execute(text(stmt))
    return sessionmaker(engine, expire_on_commit=False)
