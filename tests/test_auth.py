import uuid
from datetime import UTC, datetime, timedelta

import jwt

import auth

SECRET = "s" * 32


def test_hash_roundtrip_and_salting():
    h1, h2 = auth.hash_password("correct horse"), auth.hash_password("correct horse")
    assert h1 != h2 and h1.startswith("scrypt$")
    assert auth.verify_password("correct horse", h1) and not auth.verify_password("wrong horse", h1)


def test_malformed_hash_is_rejected_not_raised():
    for bad in ("", "scrypt$zz$zz", "bcrypt$00$00", "a$b"):
        assert auth.verify_password("x", bad) is False


def test_token_roundtrip_and_rejections():
    uid = uuid.uuid4()
    assert auth.decode_token(auth.create_token(uid, SECRET, 5), SECRET) == uid
    assert auth.decode_token(auth.create_token(uid, SECRET, 5), "t" * 32) is None
    assert auth.decode_token(auth.create_token(uid, SECRET, -1), SECRET) is None  # expired
    assert auth.decode_token("garbage", SECRET) is None
    unsigned = jwt.encode({"sub": str(uid), "exp": datetime.now(UTC) + timedelta(minutes=5)}, "", algorithm="none")
    assert auth.decode_token(unsigned, SECRET) is None
    no_exp = jwt.encode({"sub": str(uid)}, SECRET, algorithm="HS256")
    assert auth.decode_token(no_exp, SECRET) is None
