"""Acceptance checks for the PHAL -> GMD_active feature ledger (docs/migration/).

These tests use only files inside GMD_active, so they keep running after the
PHAL reference tree (ref/lammps_monitor) is removed. The drift check runs only
while the reference tree is still present.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
MIGRATION_DOCS = ROOT / "docs" / "migration"
LEDGER_PATH = MIGRATION_DOCS / "feature_ledger.yaml"
INVENTORY_PATH = MIGRATION_DOCS / "phal_inventory.json"
REFERENCE_TREE = ROOT / "ref" / "lammps_monitor"


def _load_tool(name: str) -> ModuleType:
    path = ROOT / "tools" / "migration" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"migration_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ledger_is_valid_and_covers_the_whole_inventory() -> None:
    check = _load_tool("check_ledger")
    ledger = yaml.safe_load(LEDGER_PATH.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    errors = check.validate(ledger, inventory)
    assert errors == [], "\n".join(errors[:50])


def test_rendered_ledger_is_up_to_date() -> None:
    render = _load_tool("render_ledger")
    ledger = yaml.safe_load(LEDGER_PATH.read_text(encoding="utf-8"))
    rendered = (MIGRATION_DOCS / "feature_ledger.md").read_text(encoding="utf-8")
    assert rendered == render.render(ledger), "run: python tools/migration/render_ledger.py"


@pytest.mark.skipif(not REFERENCE_TREE.is_dir(), reason="PHAL reference tree has been removed")
def test_reference_tree_matches_frozen_inventory() -> None:
    """Changes to ref/lammps_monitor must be reflected in the inventory (and ledger)."""
    inventory_tool = _load_tool("build_phal_inventory")
    frozen = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    current = inventory_tool.build_document(REFERENCE_TREE)
    assert current["commit"] == frozen["commit"]
    frozen_hashes = {item["path"]: item["sha256"] for item in frozen["files"]}
    current_hashes = {item["path"]: item["sha256"] for item in current["files"]}
    assert current_hashes == frozen_hashes, "run: python tools/migration/build_phal_inventory.py"
