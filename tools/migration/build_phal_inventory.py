"""Freeze an inventory of the PHAL reference tree (ref/lammps_monitor).

The inventory records every tracked file with its size, line count and SHA-256 so
that the feature ledger can be checked for completeness *after* the reference
tree has been removed, and so that drift in the reference tree is detectable
while it still exists.

Usage (from the GMD_active repository root)::

    python tools/migration/build_phal_inventory.py \
        --ref ref/lammps_monitor --out docs/migration/phal_inventory.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Ordered: first matching prefix wins.
CATEGORY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("src/", "source"),
    ("tests/fixtures/", "fixture"),
    ("tests/", "test"),
    ("scripts/", "script"),
    ("containers/", "container"),
    ("configs/", "config"),
    ("examples/", "example"),
    ("docs/", "doc"),
    ("data/", "data"),
    ("runtime/", "runtime-skeleton"),
    (".github/", "ci"),
)

TEXT_SUFFIXES = {
    ".py", ".md", ".yaml", ".yml", ".json", ".toml", ".txt", ".sh", ".pbs", ".inp",
    ".template", ".xyz", ".extxyz", ".lammpstrj", ".out", ".cif", ".dat", "",
}


@dataclass(frozen=True, slots=True)
class InventoryEntry:
    path: str
    category: str
    bytes: int
    lines: int | None
    sha256: str


def categorize(relative_path: str) -> str:
    """Return the inventory category for a repository-relative POSIX path."""
    for prefix, category in CATEGORY_PREFIXES:
        if relative_path.startswith(prefix):
            return category
    return "meta"


def list_files(ref_root: Path) -> list[str]:
    """List tracked files via ``git ls-files``; fall back to a directory walk."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(ref_root), "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
        names = [name for name in completed.stdout.decode("utf-8").split("\0") if name]
        if names:
            return sorted(names)
    except (OSError, subprocess.CalledProcessError):
        pass
    return sorted(
        path.relative_to(ref_root).as_posix()
        for path in ref_root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(ref_root).parts
    )


def _count_lines(data: bytes, suffix: str) -> int | None:
    if suffix not in TEXT_SUFFIXES:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text.count("\n") + (0 if not text or text.endswith("\n") else 1)


def build_inventory(ref_root: Path) -> list[InventoryEntry]:
    entries: list[InventoryEntry] = []
    for relative in list_files(ref_root):
        data = (ref_root / relative).read_bytes()
        entries.append(
            InventoryEntry(
                path=relative,
                category=categorize(relative),
                bytes=len(data),
                lines=_count_lines(data, Path(relative).suffix),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )
    return entries


def _git_value(ref_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(ref_root), *args], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def build_document(ref_root: Path) -> dict[str, Any]:
    entries = build_inventory(ref_root)
    summary: dict[str, dict[str, int]] = {}
    for entry in entries:
        bucket = summary.setdefault(entry.category, {"files": 0, "lines": 0})
        bucket["files"] += 1
        bucket["lines"] += entry.lines or 0
    return {
        "source": "PHAL (hotspot_al) reference tree",
        "commit": _git_value(ref_root, "rev-parse", "HEAD"),
        "remote": _git_value(ref_root, "remote", "get-url", "origin"),
        "file_count": len(entries),
        "summary": dict(sorted(summary.items())),
        "files": [asdict(entry) for entry in entries],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", type=Path, default=Path("ref/lammps_monitor"))
    parser.add_argument("--out", type=Path, default=Path("docs/migration/phal_inventory.json"))
    args = parser.parse_args(argv)
    if not args.ref.is_dir():
        parser.error(f"reference tree not found: {args.ref}")
    document = build_document(args.ref)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {document['file_count']} entries to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
