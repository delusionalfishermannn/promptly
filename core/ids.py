"""Sortable-ish, prefixed identifiers and small deterministic helpers."""

from __future__ import annotations

import hashlib
import re
import secrets
import uuid

_SAFE = re.compile(r"[^a-z0-9]+")


def new_id(prefix: str) -> str:
    """e.g. ``new_id("opp") -> "opp_1f9c...32hex"`` (URL- and log-friendly)."""
    return f"{prefix}_{uuid.uuid4().hex}"


def short_id(prefix: str, length: int = 10) -> str:
    return f"{prefix}_{secrets.token_hex(length // 2)}"


def slugify(value: str, *, max_length: int = 60) -> str:
    slug = _SAFE.sub("-", value.lower()).strip("-")
    return (slug or "topic")[:max_length].strip("-")


def content_hash(*parts: str) -> str:
    """Stable SHA-256 over normalised parts (used for dedupe + artifact integrity)."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def prompt_hash(text: str) -> str:
    return content_hash(text.strip())


def stable_float(key: str, salt: str = "") -> float:
    """Deterministic pseudo-random float in [0,1) — keeps mock providers reproducible."""
    digest = hashlib.blake2b(f"{salt}|{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)
