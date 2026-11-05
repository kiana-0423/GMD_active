"""Deterministic identifiers and content hashes.

Identifiers are derived from content and provenance, never from random UUIDs or clocks,
so that resubmitting the same work after a crash maps onto the same record (idempotency).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.dtype.str, "shape": list(value.shape), "data": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    return value


def content_hash(payload: Any, *, length: int = 64) -> str:
    """SHA-256 of a canonical JSON encoding of ``payload`` (arrays included bit-exactly)."""

    encoded = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def stable_id(prefix: str, *parts: Any, length: int = 16) -> str:
    """``<prefix>-<hash>``: same inputs always give the same identifier."""

    return f"{prefix}-{content_hash(list(parts), length=length)}"


def file_sha256(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: str | Path) -> str:
    """Hash of every file (relative path + content hash) below ``root``, order-independent."""

    root = Path(root)
    entries = sorted(
        (path.relative_to(root).as_posix(), file_sha256(path)) for path in root.rglob("*") if path.is_file()
    )
    return content_hash(entries)
