import os

import pytest
from conftest import FakeRunner, make_settings
from sqlalchemy import select

import db
import jobs


def crash(*args):
    os._exit(1)  # simulates Docling's native access violation: the worker dies without a Python exception


def add(a, b):
    return a + b


def test_runner_survives_native_crash():
    r = jobs.IngestRunner(fn=crash)
    try:
        with pytest.raises(RuntimeError, match="crashed"):
            r.run(1)
        r.fn = add
        assert r.run(2, 3) == 5  # a fresh worker replaces the dead one
    finally:
        r.shutdown()


def make_state(tmp_path, runner=None):
    return jobs.State(settings=make_settings(), sessions=db.make_sessions("sqlite://"), qdrant=None, openai=None,
                      runner=runner or FakeRunner(), upload_dir=tmp_path)


def add_doc(state, status):
    with state.sessions() as s:
        u = db.User(email=f"{status}@x.io", password_hash="h")
        s.add(u)
        s.flush()
        p = db.Project(owner_id=u.id, name="P")
        s.add(p)
        s.flush()
        d = db.Document(project_id=p.id, filename="f.pdf", sha256=status.ljust(64, "0"), status=status)
        run_status = {"processing": "running", "ready": "completed"}.get(status, status)
        r = db.ResearchRun(project_id=p.id, question="q", status=run_status)
        s.add_all([d, r])
        s.commit()
        return d.id, r.id


def test_recover_interrupted_fails_only_unfinished_work(tmp_path):
    state = make_state(tmp_path)
    ids = {st: add_doc(state, st) for st in ("queued", "processing", "ready", "failed")}
    jobs.recover_interrupted(state.sessions)
    with state.sessions() as s:
        docs = {st: s.get(db.Document, ids[st][0]) for st in ids}
        runs = {st: s.get(db.ResearchRun, ids[st][1]) for st in ids}
    assert docs["queued"].status == docs["processing"].status == "failed"
    assert docs["queued"].error == "interrupted by server restart"
    assert docs["ready"].status == "ready" and docs["ready"].error is None
    assert runs["queued"].status == runs["processing"].status == "failed" and runs["processing"].finished_at
    assert runs["ready"].status == "completed"


def test_ingest_document_only_claims_queued(tmp_path):
    runner = FakeRunner()
    state = make_state(tmp_path, runner)
    doc_id, _ = add_doc(state, "ready")
    jobs.ingest_document(state, doc_id)
    assert runner.calls == []


def test_ingest_document_records_success_and_failure(tmp_path):
    runner = FakeRunner()
    state = make_state(tmp_path, runner)
    ok_id, _ = add_doc(state, "queued")
    jobs.ingest_document(state, ok_id)
    runner.result = RuntimeError("boom")
    with state.sessions() as s:
        doc = s.scalar(select(db.Document).where(db.Document.id == ok_id))
        assert doc.status == "ready" and doc.chunk_count == 3
        doc.status = "queued"  # re-queue the same document for the failure case
        s.commit()
    jobs.ingest_document(state, ok_id)
    with state.sessions() as s:
        d = s.get(db.Document, ok_id)
        assert d.status == "failed" and d.error == "RuntimeError: boom"
    (args, _) = runner.calls
    assert args[2] == "f.pdf" and args[3] == "unit" and args[1].endswith(f"{d.sha256}.pdf")


def test_short_error_redacts_secrets_and_truncates():
    s = make_settings()
    msg = jobs.short_error(ValueError(f"bad key sk-test-secret and qd-secret {'x' * 600}"), s)
    assert "sk-test-secret" not in msg and "qd-secret" not in msg and "***" in msg and len(msg) == 500


def sleepy(*args):
    import time
    time.sleep(60)


def fail(*args):
    raise ValueError("no text extracted")


def test_runner_times_out_hung_parse_and_recovers():
    import time
    r = jobs.IngestRunner(fn=sleepy, timeout=2)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        r.run()
    assert time.monotonic() - start < 30
    r.fn, r.timeout = add, 60  # recovery is the point; a cold child spawn alone can take >2 s on a loaded machine
    assert r.run(2, 3) == 5


def test_child_python_errors_are_reported_without_double_prefix():
    with pytest.raises(jobs.IngestFailed) as exc_info:
        jobs.IngestRunner(fn=fail).run()
    assert jobs.short_error(exc_info.value, make_settings()) == "ValueError: no text extracted"


def test_shutdown_kills_running_child_and_refuses_new_work():
    import threading
    import time
    r = jobs.IngestRunner(fn=sleepy, timeout=60)
    out = {}

    def go():
        try:
            r.run()
        except RuntimeError as e:
            out["err"] = str(e)

    th = threading.Thread(target=go)
    th.start()
    deadline = time.monotonic() + 15
    while r._proc is None and time.monotonic() < deadline:  # wait until the child has been started
        time.sleep(0.05)
    r.shutdown()
    th.join(15)
    assert not th.is_alive() and "shut down" in out["err"]
    with pytest.raises(RuntimeError, match="shut down"):
        r.run()
