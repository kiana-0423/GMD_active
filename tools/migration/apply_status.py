"""Merge per-area status records (docs/migration/status/*.yaml) into the feature ledger.

Each record: ``{id, status, implementation: [...], evidence: [...], findings: [...], notes}``.
``status`` maps onto the ledger's ``migration_status``. Evidence is checked: a ``verified`` or
``replaced`` record must cite at least one existing test file (``path`` or ``path::test``), and every
implementation path must exist. Records are stored on the ledger entry under ``evidence``.

Usage (from the GMD_active root)::

    python tools/migration/apply_status.py            # update the ledger, print unresolved entries
    python tools/migration/apply_status.py --check    # fail if the ledger is out of date or evidence is missing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

STATUS_MAP = {
    "verified": "verified",
    "migrated": "migrated",
    "replaced": "replaced",
    "dropped": "dropped",
    "in-progress": "in-progress",
    "not-started": "not-started",
}
RESOLVED = {"verified", "replaced", "dropped"}


def load_records(status_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(status_dir.glob("*.yaml")):
        for record in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
            ledger_id = str(record["id"])
            if ledger_id in records:
                raise SystemExit(f"{ledger_id} has status records in more than one file ({path.name})")
            records[ledger_id] = {**record, "_source": path.name}
    return records


def check_record(record: dict[str, Any], root: Path) -> list[str]:
    problems: list[str] = []
    status = record.get("status")
    if status not in STATUS_MAP:
        problems.append(f"{record['id']}: unknown status {status!r}")
    for path in record.get("implementation") or []:
        if not (root / str(path).split("::")[0].split(":")[0]).exists():
            problems.append(f"{record['id']}: implementation path {path} does not exist")
    tests = [str(item).split("::")[0] for item in record.get("evidence") or []]
    for test in tests:
        if not (root / test).exists():
            problems.append(f"{record['id']}: evidence {test} does not exist")
    if status in {"verified", "replaced"} and not any(test.startswith("tests/") for test in tests):
        problems.append(f"{record['id']}: status {status} needs at least one test as evidence")
    if status == "dropped" and not str(record.get("notes") or "").strip():
        problems.append(f"{record['id']}: dropped needs a justification in notes")
    return problems


def apply(ledger: list[dict[str, Any]], records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    known = {entry["id"] for entry in ledger}
    unknown = sorted(set(records) - known)
    if unknown:
        raise SystemExit(f"status records for unknown ledger ids: {unknown}")
    for entry in ledger:
        record = records.get(entry["id"])
        if record is None:
            continue
        entry["migration_status"] = STATUS_MAP[record["status"]]
        entry["evidence"] = {
            "implementation": list(record.get("implementation") or []),
            "tests": list(record.get("evidence") or []),
            "findings": list(record.get("findings") or []),
            "notes": " ".join(str(record.get("notes") or "").split()),
            "source": record["_source"],
        }
    return ledger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    ledger_path = root / "docs/migration/feature_ledger.yaml"
    text = ledger_path.read_text(encoding="utf-8")
    header = "".join(line for line in text.splitlines(keepends=True) if line.startswith("#"))
    ledger = yaml.safe_load(text)
    records = load_records(root / "docs/migration/status")
    problems = [problem for record in records.values() for problem in check_record(record, root)]
    updated = apply([dict(entry) for entry in ledger], records)
    body = yaml.safe_dump(updated, sort_keys=False, allow_unicode=True, width=110, default_flow_style=False)
    new_text = header + "\n" + body
    unresolved = [entry for entry in updated if entry["migration_status"] not in RESOLVED]
    for problem in problems:
        print(f"EVIDENCE: {problem}")
    print(f"{len(updated)} entries; {len(updated) - len(unresolved)} resolved; {len(unresolved)} unresolved")
    for entry in unresolved:
        print(f"  {entry['id']:8s} {entry['migration_status']:12s} {entry['feature'][:80]}")
    if args.check:
        stale = new_text != text
        if stale:
            print("ledger is out of date; run tools/migration/apply_status.py")
        return 1 if problems or stale else 0
    ledger_path.write_text(new_text, encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
