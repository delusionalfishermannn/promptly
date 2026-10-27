"""Password hashing, opaque token issuance and signature helpers.

Design notes
------------
* Passwords: ``hashlib.scrypt`` (stdlib, memory-hard) with a per-password salt.
* Session / API tokens: opaque high-entropy secrets. Only a SHA-256 digest is
  persisted, so a database leak does not leak usable credentials. Lookup uses the
  digest prefix for indexing (see ``db.models.AuthSession.token_hash``).
* Artifact signing: HMAC-SHA256 over the canonical manifest. This is an MVP
  integrity/authenticity control (tamper-evident, operator-held key). Upgrade path:
  Ed25519 / KMS-backed asymmetric signing, documented in DESIGN.md §7.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from typing import Any

SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32


def hash_password(password: str) -> str:
    if len(password) < 10:
        raise ValueError("password must be at least 10 characters")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=SCRYPT_DKLEN,
    )
    return (
        f"scrypt${base64.b64encode(salt).decode()}${SCRYPT_N}${base64.b64encode(digest).decode()}"
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, salt_b64, n_raw, digest_b64 = encoded.split("$", 3)
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    try:
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        computed = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n_raw),
            r=SCRYPT_R,
            p=SCRYPT_P,
            dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(computed, expected)


def new_token(prefix: str = "pf") -> str:
    """URL-safe opaque token, e.g. ``pf_sk_<43 chars>``."""
    return f"{prefix}_{secrets.token_urlsafe(32)}"


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def sign_payload(payload: Any, key: str) -> str:
    mac = hmac.new(key.encode("utf-8"), canonical_json(payload).encode("utf-8"), hashlib.sha256)
    return "hmac-sha256:" + mac.hexdigest()


def verify_signature(payload: Any, signature: str, key: str) -> bool:
    expected = sign_payload(payload, key)
    return hmac.compare_digest(expected, signature)
