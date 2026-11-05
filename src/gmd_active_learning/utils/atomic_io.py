# Ported from PHAL (hotspot_al.utils.atomic_io) at 96263b6; MIT License, see NOTICE.
"""Crash-safe JSON persistence for small bookkeeping files.

Writing directly to the destination leaves a truncated, unparseable file if the
process dies mid-write.  For state that is read back on every run, that turns a
crash into a permanent failure, so replacement goes through a temporary file in
the same directory followed by an atomic rename.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_atomic(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    """Replace ``path`` with ``payload`` as JSON, atomically."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=indent)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            # A failure while writing must not leave the temporary file behind (UTIL-004).
            handle.close()
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return target


def read_json_or_default(path: str | Path, default: Any, *, logger: Any | None = None) -> Any:
    """Read JSON, falling back to ``default`` when the file is absent or corrupt.

    Bookkeeping state can be rebuilt, so a damaged file left behind by an
    interrupted write must not permanently wedge the caller.
    """

    source = Path(path)
    if not source.exists():
        return default
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        if logger is not None:
            logger.warning("discarding unreadable state file %s (%s); continuing from defaults", source, exc)
        return default


__all__ = ["read_json_or_default", "write_json_atomic"]
