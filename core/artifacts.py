"""Content-addressed object store (local filesystem for the MVP; S3/GCS via adapter).

Layout (relative to ``ARTIFACTS_DIR``)::

    <kind>/<YYYYMM>/<sha256[:2]>/<sha256><ext>        # immutable content
    manifests/<kind>/<subject_id>/<YYYYMM>/<sha256>.json   # human-navigable pointers

Every write returns an :class:`ArtifactRef` with the pointer URI (``local://...`` or
``s3://bucket/key``), size and digest. The ``artifacts`` DB table stores that ref;
nothing large is ever written into Postgres. The store is append-only by
convention — the API never overwrites or deletes blobs.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from core.clock import utcnow
from core.config import get_settings

JSON = Any


@dataclass(frozen=True)
class ArtifactRef:
    uri: str
    sha256: str
    size_bytes: int
    content_type: str
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
            "kind": self.kind,
        }


class ObjectStore(Protocol):
    def put_bytes(self, data: bytes, *, kind: str, ext: str, content_type: str) -> ArtifactRef: ...
    def read_bytes(self, uri: str) -> bytes: ...
    def exists(self, uri: str) -> bool: ...


class LocalObjectStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or get_settings().artifacts_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- internals
    def _dest(self, kind: str, digest: str, ext: str) -> Path:
        month = utcnow().strftime("%Y%m")
        suffix = ext if ext.startswith(".") or not ext else f".{ext}"
        return self.root / kind / month / digest[:2] / f"{digest}{suffix}"

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    # ---- API
    def put_bytes(
        self,
        data: bytes,
        *,
        kind: str,
        ext: str = "bin",
        content_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        digest = hashlib.sha256(data).hexdigest()
        path = self._dest(kind, digest, ext)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)  # atomic publish
        return ArtifactRef(
            uri=f"local://{self._relative(path)}",
            sha256=digest,
            size_bytes=len(data),
            content_type=content_type,
            kind=kind,
        )

    def put_json(self, payload: JSON, *, kind: str, subject_id: str | None = None) -> ArtifactRef:
        data = json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8")
        ref = self.put_bytes(data, kind=kind, ext="json", content_type="application/json")
        if subject_id:
            self.link_manifest(kind, subject_id, ref)
        return ref

    def put_text(
        self, text: str, *, kind: str, ext: str = "md", subject_id: str | None = None
    ) -> ArtifactRef:
        ref = self.put_bytes(
            text.encode("utf-8"), kind=kind, ext=ext, content_type=_content_type(ext)
        )
        if subject_id:
            self.link_manifest(kind, subject_id, ref)
        return ref

    def link_manifest(self, kind: str, subject_id: str, ref: ArtifactRef) -> str:
        """Write a stable pointer so humans can find the latest artifact for a subject."""
        month = utcnow().strftime("%Y%m")
        target = self.root / "manifests" / kind / subject_id / month / f"{ref.sha256[:16]}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(ref.to_dict(), indent=2), encoding="utf-8")
        return f"local://{self._relative(target)}"

    def read_bytes(self, uri: str) -> bytes:
        return self._path(uri).read_bytes()

    def read_json(self, uri: str) -> JSON:
        return json.loads(self.read_bytes(uri).decode("utf-8"))

    def exists(self, uri: str) -> bool:
        try:
            return self._path(uri).exists()
        except ValueError:
            return False

    def _path(self, uri: str) -> Path:
        if not uri.startswith("local://"):
            raise ValueError(f"unsupported artifact uri: {uri}")
        rel = uri.removeprefix("local://")
        candidate = (self.root / rel).resolve()
        # Path traversal guard: blobs must live under the artifacts root.
        if self.root.resolve() not in candidate.parents and candidate != self.root.resolve():
            raise ValueError("artifact uri escapes the store root")
        return candidate

    def total_size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def export_to(self, destination: Path) -> Path:
        """Used by backups (``make backup``) — produces a portable tarball."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.make_archive(str(destination.with_suffix("")), "gztar", root_dir=self.root)
        return destination.with_suffix(".tar.gz")


def _content_type(ext: str) -> str:
    ext = ext.lstrip(".")
    return {
        "json": "application/json",
        "md": "text/markdown",
        "txt": "text/plain",
        "csv": "text/csv",
        "yaml": "application/yaml",
        "zip": "application/zip",
        "html": "text/html",
    }.get(ext, "application/octet-stream")


_STORE: ObjectStore | None = None


def get_store() -> ObjectStore:
    global _STORE
    if _STORE is None:
        backend = os.environ.get("PF_OBJECT_STORE", "local")
        if backend == "local":
            _STORE = LocalObjectStore()
        else:  # pragma: no cover - documented extension point
            raise RuntimeError(
                f"object store '{backend}' not implemented; implement ObjectStore and register it here"
            )
    return _STORE


def reset_store() -> None:
    global _STORE
    _STORE = None
