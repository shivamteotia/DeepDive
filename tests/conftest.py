from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app import create_app
from config import Settings


def make_settings(**overrides) -> Settings:
    base = {
        "openai_api_key": "sk-test-secret", "openai_embed_model": "emb", "openai_model": "m", "openai_fast_model": "f",
        "openai_guard_model": "g", "qdrant_cluster_endpoint": "http://unused", "qdrant_api_key": "qd-secret",
        "qdrant_collection": "unit", "database_url": "sqlite://", "jwt_secret": "j" * 32, "access_token_minutes": 5,
    }
    return Settings(_env_file=None, **{**base, **overrides})


class FakeRunner:
    """Stands in for jobs.IngestRunner: records calls, returns a chunk count or raises."""

    def __init__(self):
        self.calls = []
        self.result = 3

    def run(self, *args):
        self.calls.append(args)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def shutdown(self):
        pass


class FakeOpenAI:
    """Embeds every text to the same vector. responses.parse is unused: research.run is faked in API tests."""

    def __init__(self):
        self.embeddings = SimpleNamespace(create=lambda model, input: SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0, 0.0, 0.0]) for _ in input]))


class InlinePool:
    """Runs submitted jobs immediately, so API tests see finished work as soon as the request returns.
    tests/test_api_concurrency.py covers the real executors."""

    def submit(self, fn, *args):
        fn(*args)

    def shutdown(self, **kw):
        pass


@pytest.fixture
def runner():
    return FakeRunner()


@pytest.fixture
def api(tmp_path, runner):
    app = create_app(make_settings(), qdrant=QdrantClient(":memory:"), openai_client=FakeOpenAI(), runner=runner,
                     upload_dir=tmp_path / "uploads", ingest_pool=InlinePool(), research_pool=InlinePool())
    with TestClient(app) as client:
        client.maxrim = app.state.maxrim
        yield client


@pytest.fixture
def signup(api):
    def _signup(email="a@example.com", password="correct horse battery"):
        api.post("/auth/register", json={"email": email, "password": password})
        token = api.post("/auth/login", json={"email": email, "password": password}).json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    return _signup
