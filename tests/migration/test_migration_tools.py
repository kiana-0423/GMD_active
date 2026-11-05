"""Unit tests for the PHAL migration tooling in tools/migration/."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools" / "migration"


def _load_tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"migration_{name}", TOOLS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


inventory_tool = _load_tool("build_phal_inventory")
check_tool = _load_tool("check_ledger")
render_tool = _load_tool("render_ledger")


def _entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "id": "MON-001",
        "area": "monitoring",
        "feature": "Force monitor",
        "behavior": "Per-atom force magnitude.",
        "phal_sources": ["src/pkg/force.py:ForceMonitor"],
        "symbols": ["ForceMonitor"],
        "config_keys": [],
        "phal_tests": ["tests/unit/test_force.py"],
        "phal_status": "implemented",
        "external_deps": [],
        "gmd_active_equivalent": "none",
        "destination": "src/gmd_active_learning/monitors/force.py",
        "disposition": "port",
        "acceptance": "Identical per-atom scores on fixture frames.",
        "phase": 2,
        "notes": "",
        "migration_status": "not-started",
    }
    entry.update(overrides)
    return entry


@pytest.fixture()
def inventory() -> dict[str, object]:
    files = [
        {"path": "src/pkg/force.py", "category": "source"},
        {"path": "tests/unit/test_force.py", "category": "test"},
        {"path": "tests/fixtures/frames.xyz", "category": "fixture"},
        {"path": "runtime/logs/.gitkeep", "category": "runtime-skeleton"},
    ]
    return {"file_count": len(files), "files": files}


def _full_ledger() -> list[dict[str, object]]:
    return [
        _entry(),
        _entry(
            id="TST-001",
            area="tests",
            feature="Fixtures",
            phal_sources=["tests/fixtures/**"],
            phal_tests=[],
            destination="tests/fixtures/phal",
        ),
        _entry(
            id="OPS-001",
            area="tooling",
            feature="Runtime skeleton",
            phal_sources=["runtime/"],
            phal_tests=[],
            disposition="drop-with-justification",
            destination="n/a",
            notes="Replaced by GMD_active's configured run directories.",
        ),
    ]


class TestInventory:
    def test_categories_follow_prefix_order(self) -> None:
        assert inventory_tool.categorize("src/hotspot_al/models.py") == "source"
        assert inventory_tool.categorize("tests/fixtures/cp2k/toy.out") == "fixture"
        assert inventory_tool.categorize("tests/unit/test_x.py") == "test"
        assert inventory_tool.categorize("README.md") == "meta"

    def test_inventory_hashes_and_counts_lines(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
        (tmp_path / "blob.npy").write_bytes(b"\x93NUMPY\x00\xff")
        entries = {entry.path: entry for entry in inventory_tool.build_inventory(tmp_path)}
        assert set(entries) == {"src/a.py", "blob.npy"}
        assert entries["src/a.py"].lines == 2
        assert entries["src/a.py"].category == "source"
        assert entries["blob.npy"].lines is None
        assert len(entries["src/a.py"].sha256) == 64


class TestCheckLedger:
    def test_complete_ledger_is_valid(self, inventory: dict[str, object]) -> None:
        assert check_tool.validate(_full_ledger(), inventory) == []

    def test_uncovered_file_is_reported(self, inventory: dict[str, object]) -> None:
        ledger = _full_ledger()[:2]
        errors = check_tool.validate(ledger, inventory)
        assert errors == ["uncovered inventory file: runtime/logs/.gitkeep (runtime-skeleton)"]

    def test_phantom_source_is_reported(self, inventory: dict[str, object]) -> None:
        ledger = _full_ledger()
        ledger[0]["phal_sources"] = ["src/pkg/force.py", "src/pkg/typo.py"]
        errors = check_tool.validate(ledger, inventory)
        assert any("src/pkg/typo.py" in error for error in errors)

    def test_test_files_cannot_cover_sources(self, inventory: dict[str, object]) -> None:
        ledger = _full_ledger()
        ledger[0]["phal_sources"] = ["tests/unit/test_force.py"]
        ledger[0]["phal_tests"] = ["src/pkg/force.py"]
        errors = check_tool.validate(ledger, inventory)
        assert "uncovered inventory file: src/pkg/force.py (source)" in errors

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("area", "telemetry"),
            ("phal_status", "done"),
            ("disposition", "rewrite"),
            ("migration_status", "finished"),
            ("phase", 0),
            ("id", "mon-1"),
            ("destination", "/abs/path.py"),
            ("acceptance", "  "),
        ],
    )
    def test_invalid_values_are_rejected(self, inventory: dict[str, object], field: str, value: object) -> None:
        ledger = _full_ledger()
        ledger[0][field] = value
        assert check_tool.validate(ledger, inventory) != []

    def test_duplicate_ids_and_missing_fields(self, inventory: dict[str, object]) -> None:
        ledger = _full_ledger()
        ledger.append(copy.deepcopy(ledger[0]))
        broken = copy.deepcopy(ledger[0])
        broken["id"] = "MON-002"
        del broken["acceptance"]
        ledger.append(broken)
        errors = check_tool.validate(ledger, inventory)
        assert "MON-001: duplicate id" in errors
        assert any(error.startswith("MON-002: missing fields") for error in errors)

    def test_upstream_destinations_require_replace_disposition(self, inventory: dict[str, object]) -> None:
        ledger = _full_ledger()
        ledger[0]["destination"] = "GMD_SGT:gmd_sgt/training/loss.py, tests/reference/masked_loss.py"
        ledger[0]["disposition"] = "replace-with-gmd_sgt"
        assert check_tool.validate(ledger, inventory) == []
        ledger[0]["disposition"] = "port"
        errors = check_tool.validate(ledger, inventory)
        assert errors == [
            "MON-001: upstream destination 'GMD_SGT:gmd_sgt/training/loss.py' requires a replace-with-* disposition"
        ]

    def test_drop_requires_justification(self, inventory: dict[str, object]) -> None:
        ledger = _full_ledger()
        ledger[2]["notes"] = ""
        errors = check_tool.validate(ledger, inventory)
        assert errors == ["OPS-001: drop-with-justification requires a justification in notes"]


def test_render_groups_by_area_and_summarises() -> None:
    text = render_tool.render(_full_ledger())
    assert text.startswith("# PHAL → GMD_active feature ledger")
    assert "| not-started | 3 |" in text
    assert "## Monitoring" in text and "## Tests" in text and "## Tooling" in text
    assert "| MON-001 | Force monitor | implemented |" in text
