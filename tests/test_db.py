import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

import db


def seed(s):
    u = db.User(email="a@example.com", password_hash="h")
    s.add(u)
    s.flush()
    p = db.Project(owner_id=u.id, name="P")
    s.add(p)
    s.flush()
    return u, p


def test_document_defaults_and_unique_sha_per_project():
    sessions = db.make_sessions("sqlite://")
    with sessions() as s:
        _, p = seed(s)
        s.add(db.Document(project_id=p.id, filename="f.pdf", sha256="a" * 64))
        s.commit()
        d = s.scalar(select(db.Document))
        assert d.status == "queued" and d.chunk_count is None and d.error is None and d.created_at and d.updated_at
        s.add(db.Document(project_id=p.id, filename="g.pdf", sha256="a" * 64))
        with pytest.raises(IntegrityError):
            s.commit()


def test_research_run_result_roundtrips_json():
    sessions = db.make_sessions("sqlite://")
    with sessions() as s:
        _, p = seed(s)
        s.add(db.ResearchRun(project_id=p.id, question="q", result={"claims": [{"text": "c"}], "log": []}))
        s.commit()
    with sessions() as s:
        run = s.scalar(select(db.ResearchRun))
        assert run.status == "queued" and run.max_iterations == 3 and run.result["claims"][0]["text"] == "c"


def test_each_sqlite_factory_is_isolated():
    a, b = db.make_sessions("sqlite://"), db.make_sessions("sqlite://")
    with a() as s:
        seed(s)
        s.commit()
    with b() as s:
        assert s.scalar(select(db.User)) is None
