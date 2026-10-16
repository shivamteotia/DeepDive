import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from conftest import make_settings

from app import create_app

SECRET = "j" * 32  # matches make_settings()


def test_health(api):
    assert api.get("/health").json() == {"status": "ok"}


def test_register_login_then_access(api, signup):
    assert api.get("/api/v1/projects", headers=signup()).status_code == 200


def test_register_validation_and_duplicates(api):
    assert api.post("/auth/register", json={"email": "a@example.com", "password": "short"}).status_code == 422
    assert api.post("/auth/register", json={"email": "not-an-email", "password": "x" * 12}).status_code == 422
    r = api.post("/auth/register", json={"email": "A@Example.com", "password": "x" * 12})
    assert r.status_code == 201 and r.json()["email"] == "a@example.com" and "password_hash" not in r.json()
    assert api.post("/auth/register", json={"email": "a@example.com", "password": "y" * 12}).status_code == 409


def test_login_failures_are_indistinguishable(api, signup):
    signup()
    wrong_pw = api.post("/auth/login", json={"email": "a@example.com", "password": "wrong password!"})
    no_user = api.post("/auth/login", json={"email": "nobody@example.com", "password": "wrong password!"})
    assert wrong_pw.status_code == no_user.status_code == 401
    assert wrong_pw.json() == no_user.json()


def test_bad_tokens_get_401(api, signup):
    headers = signup()
    uid = jwt.decode(headers["Authorization"].split()[1], options={"verify_signature": False})["sub"]
    future = datetime.now(UTC) + timedelta(minutes=5)
    tokens = [
        None,
        "garbage",
        jwt.encode({"sub": uid, "exp": future}, "other" * 8, algorithm="HS256"),  # wrong secret
        jwt.encode({"sub": uid, "exp": datetime.now(UTC) - timedelta(minutes=1)}, SECRET, algorithm="HS256"),  # expired
        jwt.encode({"sub": uid, "exp": future}, "", algorithm="none"),  # unsigned
        jwt.encode({"sub": str(uuid.uuid4()), "exp": future}, SECRET, algorithm="HS256"),  # user does not exist
    ]
    for t in tokens:
        h = {"Authorization": f"Bearer {t}"} if t else {}
        r = api.get("/api/v1/projects", headers=h)
        assert r.status_code == 401, t


def test_short_jwt_secret_refuses_to_start():
    with pytest.raises(ValueError, match="JWT_SECRET"):
        create_app(make_settings(jwt_secret="short"), qdrant=object(), openai_client=object(), runner=object())
