"""Validate the PHAL -> GMD_active feature ledger against the frozen PHAL inventory.

Checks performed by :func:`validate`:

* every entry has the required fields with allowed values, and IDs are unique;
* every ``phal_sources`` / ``phal_tests`` reference resolves to at least one
  inventoried file (glob patterns such as ``runtime/**`` are allowed;
  a ``:Symbol`` suffix on a source path is ignored for resolution);
* every inventoried file is covered by at least one entry (test files and
  fixtures may be covered through ``phal_tests``);
* destinations live under the agreed GMD_active roots, and a
  ``drop-with-justification`` disposition carries a non-empty justification.

The inventory, not the live reference tree, is the source of truth, so the
check keeps working after ``ref/lammps_monitor`` has been removed.

Usage (from the GMD_active repository root)::

    python tools/migration/check_ledger.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Iterable

import yaml

REQUIRED_FIELDS: tuple[str, ...] = (
    "id", "area", "feature", "behavior", "phal_sources", "symbols", "config_keys",
    "phal_tests", "phal_status", "external_deps", "gmd_active_equivalent",
    "destination", "disposition", "acceptance", "phase", "notes", "migration_status",
)
AREAS = frozenset({
    "monitoring", "events", "hotspots", "extraction", "candidates", "datasets", "training",
    "labeling", "scheduling", "backends", "md", "config", "cli", "io", "utils", "scf",
    "tooling", "containers", "docs", "tests",
})
PHAL_STATUSES = frozenset({"implemented", "partial", "stub", "experimental", "documented-only"})
DISPOSITIONS = frozenset({
    "port", "merge-with-existing", "replace-with-gmd", "replace-with-gmd_sgt",
    "port-as-optional-tooling", "drop-with-justification",
})
MIGRATION_STATUSES = frozenset({"not-started", "in-progress", "migrated", "verified", "replaced", "dropped"})
PHASES = frozenset(range(1, 9))
DESTINATION_ROOTS: tuple[str, ...] = (
    "src/gmd_active_learning/", "tools/", "tests/", "docs/", "configs/", "examples/",
    "containers/", "pyproject.toml", "README.md", "LICENSE", "NOTICE", ".github/", ".gitignore",
    ".pre-commit-config.yaml", "mkdocs.yml",
)
# Capabilities that move into a sibling project are written "<PROJECT>:<path in that repo>".
UPSTREAM_PREFIXES: tuple[str, ...] = ("GMD:", "GMD_SGT:")
ID_PATTERN = re.compile(r"^[A-Z]{2,5}-\d{3}$")
# Categories whose files may be covered via ``phal_tests`` instead of ``phal_sources``.
TEST_CATEGORIES = frozenset({"test", "fixture"})


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _strip_symbol(reference: str) -> str:
    """``src/x.py:Foo`` -> ``src/x.py`` (paths never contain ``:`` in this repo)."""
    return reference.split(":", 1)[0].strip()


def _is_glob(pattern: str) -> bool:
    return any(char in pattern for char in "*?[")


def resolve(reference: str, inventory_paths: Iterable[str]) -> list[str]:
    """Return the inventoried paths matched by a ledger path reference."""
    target = _strip_symbol(reference).rstrip("/")
    paths = list(inventory_paths)
    if _is_glob(target):
        return [path for path in paths if fnmatchcase(path, target)]
    if target in paths:
        return [target]
    # A bare directory reference covers everything below it.
    prefix = target + "/"
    return [path for path in paths if path.startswith(prefix)]


def validate(ledger: list[dict[str, Any]], inventory: dict[str, Any]) -> list[str]:
    """Return a list of human-readable problems; an empty list means the ledger is valid."""
    errors: list[str] = []
    files = inventory.get("files", [])
    categories = {item["path"]: item["category"] for item in files}
    all_paths = list(categories)
    covered: set[str] = set()
    seen_ids: set[str] = set()

    if not isinstance(ledger, list) or not ledger:
        return ["ledger must be a non-empty list of entries"]

    for index, entry in enumerate(ledger):
        label = str(entry.get("id", f"<entry #{index}>"))
        missing = [field for field in REQUIRED_FIELDS if field not in entry]
        if missing:
            errors.append(f"{label}: missing fields {missing}")
            continue
        if not ID_PATTERN.match(label):
            errors.append(f"{label}: id does not match {ID_PATTERN.pattern}")
        if label in seen_ids:
            errors.append(f"{label}: duplicate id")
        seen_ids.add(label)
        for field, allowed in (
            ("area", AREAS),
            ("phal_status", PHAL_STATUSES),
            ("disposition", DISPOSITIONS),
            ("migration_status", MIGRATION_STATUSES),
        ):
            if entry[field] not in allowed:
                errors.append(f"{label}: {field}={entry[field]!r} not in {sorted(allowed)}")
        if entry["phase"] not in PHASES:
            errors.append(f"{label}: phase={entry['phase']!r} not in 1..8")
        for field in ("feature", "behavior", "acceptance"):
            if not str(entry[field]).strip():
                errors.append(f"{label}: empty {field}")

        sources = _as_list(entry["phal_sources"])
        if not sources:
            errors.append(f"{label}: phal_sources is empty")
        for reference in sources:
            matched = resolve(reference, all_paths)
            if not matched:
                errors.append(f"{label}: phal_sources reference {reference!r} matches no inventoried file")
            covered.update(matched)
        for reference in _as_list(entry["phal_tests"]):
            matched = resolve(reference, all_paths)
            if not matched:
                errors.append(f"{label}: phal_tests reference {reference!r} matches no inventoried file")
            covered.update(path for path in matched if categories.get(path) in TEST_CATEGORIES)

        destination = str(entry["destination"]).strip()
        if entry["disposition"] == "drop-with-justification":
            if not str(entry["notes"]).strip():
                errors.append(f"{label}: drop-with-justification requires a justification in notes")
        else:
            for item in [part.strip() for part in destination.split(",") if part.strip()] or [""]:
                if item.startswith(UPSTREAM_PREFIXES):
                    if not entry["disposition"].startswith("replace-with-"):
                        errors.append(f"{label}: upstream destination {item!r} requires a replace-with-* disposition")
                elif not item.startswith(DESTINATION_ROOTS):
                    errors.append(f"{label}: destination {item!r} is outside {DESTINATION_ROOTS}")

    uncovered = sorted(set(all_paths) - covered)
    for path in uncovered:
        errors.append(f"uncovered inventory file: {path} ({categories[path]})")
    return errors


def load(ledger_path: Path, inventory_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ledger = yaml.safe_load(ledger_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    return ledger, inventory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the PHAL migration feature ledger.")
    parser.add_argument("--ledger", type=Path, default=Path("docs/migration/feature_ledger.yaml"))
    parser.add_argument("--inventory", type=Path, default=Path("docs/migration/phal_inventory.json"))
    args = parser.parse_args(argv)
    ledger, inventory = load(args.ledger, args.inventory)
    errors = validate(ledger, inventory)
    for error in errors:
        print(error)
    print(f"{len(ledger)} entries, {inventory['file_count']} inventoried files, {len(errors)} problems")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
