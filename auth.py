"""Password hashing (stdlib scrypt) and HS256 access tokens."""

import hashlib
import hmac
import os
import uuid
from datetime import UTC, datetime, timedelta

import jwt

_SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}


def _derive(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode("utf-8", "surrogatepass"), salt=salt, **_SCRYPT)


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    return f"scrypt${salt.hex()}${_derive(password, salt).hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, salt_hex, digest_hex = stored.split("$")
        salt, digest = bytes.fromhex(salt_hex), bytes.fromhex(digest_hex)
    except ValueError:
        return False
    if algo != "scrypt" or not salt:
        return False
    return hmac.compare_digest(_derive(password, salt), digest)


# Checked against when the email is unknown, so response time doesn't reveal which accounts exist.
DUMMY_HASH = hash_password("timing-equaliser")


def create_token(user_id: uuid.UUID, secret: str, minutes: int) -> str:
    now = datetime.now(UTC)
    return jwt.encode({"sub": str(user_id), "iat": now, "exp": now + timedelta(minutes=minutes)}, secret, algorithm="HS256")


def decode_token(token: str, secret: str) -> uuid.UUID | None:
    """User id from a valid, unexpired HS256 token; None for anything else."""
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"], options={"require": ["exp", "sub"]})
        return uuid.UUID(payload["sub"])
    except (jwt.PyJWTError, ValueError):
        return None
