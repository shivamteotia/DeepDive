"""Final-review finding #1: queued ingests must not tie up the threads that serve requests."""

import threading
import time

from conftest import FakeOpenAI, FakeRunner, make_settings
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app import create_app

PDF = b"%PDF-1.4\n%%EOF\n"


def wait_for(predicate, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_uploads_queue_without_blocking_requests(tmp_path):
    gate = threading.Event()

    class BlockingRunner(FakeRunner):
        def run(self, *args):
            gate.wait(10)
            return super().run(*args)

    app = create_app(make_settings(), qdrant=QdrantClient(":memory:"), openai_client=FakeOpenAI(),
                     runner=BlockingRunner(), upload_dir=tmp_path)
    with TestClient(app) as api:
        api.post("/auth/register", json={"email": "a@example.com", "password": "correct horse battery"})
        token = api.post("/auth/login", json={"email": "a@example.com", "password": "correct horse battery"}).json()
        h = {"Authorization": f"Bearer {token['access_token']}"}
        pid = api.post("/api/v1/projects", headers=h, json={"name": "P"}).json()["id"]

        def statuses():
            return sorted(d["status"] for d in api.get(f"/api/v1/projects/{pid}/documents", headers=h).json())

        start = time.monotonic()
        for i in range(3):
            r = api.post(f"/api/v1/projects/{pid}/documents", headers=h,
                         files={"file": (f"p{i}.pdf", PDF + bytes([i]), "application/pdf")})
            assert r.status_code == 202
        assert time.monotonic() - start < 5  # uploads return while ingestion is still blocked
        assert api.get("/health").status_code == 200
        assert wait_for(lambda: statuses() == ["processing", "queued", "queued"])  # one at a time; rest truly queued
        gate.set()
        assert wait_for(lambda: statuses() == ["ready", "ready", "ready"])
