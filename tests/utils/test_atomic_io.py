"""Atomic JSON writes (UTIL-004): a failed write never leaves a truncated target or a stray temp file."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import gmd_active_learning.utils.atomic_io as atomic_io
from gmd_active_learning.utils.atomic_io import read_json_or_default, write_json_atomic


def _leftovers(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir() if path.name.endswith(".tmp"))


def test_round_trip_and_replace(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "state.json"
    write_json_atomic(target, {"a": 1})
    write_json_atomic(target, {"a": 2, "b": [1.5]})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 2, "b": [1.5]}
    assert _leftovers(target.parent) == []


def test_failed_replace_keeps_the_old_file_and_removes_the_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "state.json"
    write_json_atomic(target, {"version": 1})

    def crash(*_args: object) -> None:
        raise OSError("killed during rename")

    monkeypatch.setattr(atomic_io.os, "replace", crash)
    with pytest.raises(OSError, match="killed"):
        write_json_atomic(target, {"version": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"version": 1}
    assert _leftovers(tmp_path) == []


def test_crash_after_writing_the_temp_file_leaves_no_debris(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: a failure inside the temp-file block (here fsync) used to leave the temp file behind."""

    target = tmp_path / "state.json"
    write_json_atomic(target, {"version": 1})

    def crash(_fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(atomic_io.os, "fsync", crash)
    with pytest.raises(OSError, match="disk full"):
        write_json_atomic(target, {"version": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"version": 1}
    assert _leftovers(tmp_path) == []


def test_unserialisable_payload_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        write_json_atomic(tmp_path / "state.json", {"bad": object()})
    assert list(tmp_path.iterdir()) == []


def test_corrupt_or_missing_json_returns_default(tmp_path: Path) -> None:
    corrupt = tmp_path / "state.json"
    corrupt.write_text('{"truncated": ', encoding="utf-8")
    assert read_json_or_default(corrupt, {"fresh": True}) == {"fresh": True}
    assert read_json_or_default(tmp_path / "missing.json", []) == []
    assert os.path.exists(corrupt)
