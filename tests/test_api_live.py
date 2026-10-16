"""README §10 acceptance scenario through the API. Costs money; takes several minutes (Docling on CPU)."""

import time
import uuid
from pathlib import Path

import pytest
from qdrant_client import models

pytestmark = pytest.mark.live
PAPERS = Path(__file__).parent.parent / "data" / "Papers"


def test_acceptance_scenario_through_api(tmp_path):
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import OperationalError

    from app import create_app
    from config import Settings

    base = Settings()
    try:
        with create_engine(base.database_url).connect() as c:
            c.execute(text("select 1"))
    except OperationalError:
        pytest.skip("Postgres not reachable: run `docker compose up -d --wait`")
    pdf_a, pdf_b = PAPERS / "EA2022.pdf", PAPERS / "JPan2023.pdf"
    if not (pdf_a.exists() and pdf_b.exists()):
        pytest.skip("needs data/Papers/EA2022.pdf and JPan2023.pdf")

    coll = f"{base.qdrant_collection}_test"
    app = create_app(base.model_copy(update={"qdrant_collection": coll}), upload_dir=tmp_path / "uploads")
    q = app.state.maxrim.qdrant

    def count(pid):
        f = models.Filter(must=[models.FieldCondition(key="project_id", match=models.MatchValue(value=pid))])
        return q.count(coll, count_filter=f, exact=True).count

    try:
        with TestClient(app) as api:
            def signup():
                email, pw = f"live-{uuid.uuid4().hex[:8]}@example.com", "live-test-password"
                api.post("/auth/register", json={"email": email, "password": pw})
                token = api.post("/auth/login", json={"email": email, "password": pw}).json()["access_token"]
                return {"Authorization": f"Bearer {token}"}

            def put(h, pid, pdf):
                return api.post(f"/api/v1/projects/{pid}/documents", headers=h,
                                files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")})

            def wait(url, h, busy, minutes=20):  # jobs run on the API's executors; poll until they finish
                deadline = time.monotonic() + minutes * 60
                while (body := api.get(url, headers=h).json())["status"] in busy and time.monotonic() < deadline:
                    time.sleep(2)
                return body

            h, other = signup(), signup()
            a = api.post("/api/v1/projects", headers=h, json={"name": "A"}).json()["id"]
            b = api.post("/api/v1/projects", headers=h, json={"name": "B"}).json()["id"]

            doc_a = put(h, a, pdf_a).json()
            doc_a = wait(f"/api/v1/projects/{a}/documents/{doc_a['id']}", h, ("queued", "processing"))
            assert doc_a["status"] == "ready", doc_a["error"]
            assert count(a) == doc_a["chunk_count"] > 0

            again = put(h, a, pdf_a)
            assert again.status_code == 200 and again.json()["id"] == doc_a["id"] and count(a) == doc_a["chunk_count"]

            doc_b = put(h, b, pdf_b).json()
            assert wait(f"/api/v1/projects/{b}/documents/{doc_b['id']}", h, ("queued", "processing"))["status"] == "ready"

            run = api.post(f"/api/v1/projects/{a}/research-runs", headers=h,
                           json={"question": "What predicts mortality and how well does the model perform?",
                                 "max_iterations": 2}).json()
            got = wait(f"/api/v1/research-runs/{run['id']}", h, ("queued", "running"))
            assert got["status"] == "completed", got["error"]
            evidence = api.get(f"/api/v1/research-runs/{run['id']}/evidence", headers=h).json()
            assert evidence and {e["source_filename"] for e in evidence} == {"EA2022.pdf"}  # never project B's paper
            assert api.get(f"/api/v1/research-runs/{run['id']}/verification", headers=h).json()["iterations"]
            assert api.get(f"/api/v1/research-runs/{run['id']}", headers=other).status_code == 404
    finally:
        if q.collection_exists(coll):
            q.delete_collection(coll)
