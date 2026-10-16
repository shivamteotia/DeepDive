import uuid
from pathlib import Path

import app as app_module
import db
import jobs

PDF = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n"


def project(api, h, name="P"):
    return api.post("/api/v1/projects", headers=h, json={"name": name}).json()["id"]


def upload(api, h, pid, data=PDF, name="paper.pdf"):
    return api.post(f"/api/v1/projects/{pid}/documents", headers=h, files={"file": (name, data, "application/pdf")})


def get_doc(api, h, pid, did):
    return api.get(f"/api/v1/projects/{pid}/documents/{did}", headers=h).json()


def test_upload_indexes_and_reports_ready(api, signup, runner):
    h = signup()
    pid = project(api, h)
    r = upload(api, h, pid)
    assert r.status_code == 202 and r.json()["status"] == "queued"
    doc = get_doc(api, h, pid, r.json()["id"])
    assert doc["status"] == "ready" and doc["chunk_count"] == 3 and doc["filename"] == "paper.pdf"
    assert doc["doc_type"] == "research_article"
    (args,) = runner.calls
    assert args[0] == pid and args[2] == "paper.pdf" and args[3] == "unit"
    assert Path(args[1]).read_bytes() == PDF and Path(args[1]).name == f"{doc['sha256']}.pdf"


def test_duplicate_upload_returns_same_document(api, signup, runner):
    h = signup()
    pid = project(api, h)
    first = upload(api, h, pid).json()
    again = upload(api, h, pid, name="renamed.pdf")
    assert again.status_code == 200 and again.json()["id"] == first["id"]
    assert len(runner.calls) == 1 and len(api.get(f"/api/v1/projects/{pid}/documents", headers=h).json()) == 1


def test_same_file_in_two_projects_is_two_documents(api, signup):
    h = signup()
    a, b = project(api, h, "A"), project(api, h, "B")
    assert upload(api, h, a).json()["id"] != upload(api, h, b).json()["id"]


def test_failed_ingest_is_recorded_redacted_and_retryable(api, signup, runner):
    h = signup()
    pid = project(api, h)
    runner.result = RuntimeError("boom with sk-test-secret inside")
    first = upload(api, h, pid).json()
    doc = get_doc(api, h, pid, first["id"])
    assert doc["status"] == "failed" and "boom" in doc["error"] and "sk-test-secret" not in doc["error"]
    runner.result = {"chunk_count": 5, "doc_type": "generic"}
    retry = upload(api, h, pid)
    assert retry.status_code == 202 and retry.json()["id"] == first["id"]
    doc = get_doc(api, h, pid, first["id"])
    assert doc["status"] == "ready" and doc["chunk_count"] == 5 and doc["error"] is None and doc["doc_type"] == "generic"


def test_restart_mid_processing_then_reupload_recovers(api, signup, runner):
    h = signup()
    pid = project(api, h)
    did = upload(api, h, pid).json()["id"]
    with api.maxrim.sessions() as s:  # simulate a crash while processing
        s.get(db.Document, uuid.UUID(did)).status = "processing"
        s.commit()
    jobs.recover_interrupted(api.maxrim.sessions)
    assert get_doc(api, h, pid, did)["error"] == "interrupted by server restart"
    assert upload(api, h, pid).status_code == 202
    assert get_doc(api, h, pid, did)["status"] == "ready"


def test_rejects_non_pdf_empty_and_oversize(api, signup, runner, monkeypatch):
    h = signup()
    pid = project(api, h)
    assert upload(api, h, pid, data=b"hello world").status_code == 400
    assert upload(api, h, pid, data=b"").status_code == 400
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 10)
    assert upload(api, h, pid).status_code == 413
    assert runner.calls == []
    assert list(api.maxrim.upload_dir.rglob(".upload-*")) == []  # no temp files left behind


def test_filename_path_parts_are_stripped(api, signup, runner):
    h = signup()
    pid = project(api, h)
    doc = upload(api, h, pid, name="..\\..\\evil.pdf").json()
    assert doc["filename"] == "evil.pdf"
    assert Path(runner.calls[0][1]).parent == api.maxrim.upload_dir / pid


def test_documents_are_private(api, signup):
    alice, bob = signup("alice@example.com"), signup("bob@example.com")
    pid = project(api, alice)
    did = upload(api, alice, pid).json()["id"]
    assert upload(api, bob, pid).status_code == 404
    assert api.get(f"/api/v1/projects/{pid}/documents/{did}", headers=bob).status_code == 404
    other = project(api, alice, "Other")
    assert api.get(f"/api/v1/projects/{other}/documents/{did}", headers=alice).status_code == 404
