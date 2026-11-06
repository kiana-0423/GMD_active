"""DOC-002: docs/config-reference.md is generated from the section models and lists every key."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("render_config_reference", ROOT / "tools/docs/render_config_reference.py")
assert spec is not None and spec.loader is not None
RENDER = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RENDER)
REFERENCE = ROOT / "docs" / "config-reference.md"


def test_reference_is_up_to_date() -> None:
    assert REFERENCE.read_text(encoding="utf-8") == RENDER.render(), "run: PYTHONPATH=src python tools/docs/render_config_reference.py"
    assert RENDER.main(["--check"]) == 0


def test_every_schema_leaf_is_documented() -> None:
    text = REFERENCE.read_text(encoding="utf-8")
    missing = []
    for name, model in RENDER.section_models().items():
        assert f"\n## {name}\n" in text
        for key, _field, _nested in RENDER.leaves(model, name):
            if f"| `{key}` |" not in text:
                missing.append(key)
    assert not missing, f"undocumented config keys: {missing}"


def test_every_key_of_the_packaged_default_appears_in_the_reference() -> None:
    """Cross-check against configs/default.yaml (generated from the same models by another tool)."""

    text = REFERENCE.read_text(encoding="utf-8")
    default = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    sections = set(RENDER.section_models())

    def walk(mapping: dict, prefix: str) -> list[str]:
        keys = []
        for key, value in mapping.items():
            dotted = f"{prefix}.{key}"
            keys.append(dotted)
            if isinstance(value, dict) and f"| `{dotted}` | section" in text:
                keys += walk(value, dotted)
        return keys

    missing = [key for name, values in default.items() if name in sections for key in walk(values or {}, name) if f"| `{key}` |" not in text]
    assert not missing, missing


def test_check_mode_detects_a_stale_reference(tmp_path: Path) -> None:
    stale = tmp_path / "ref.md"
    stale.write_text("# stale\n", encoding="utf-8")
    assert RENDER.main(["--check", "--output", str(stale)]) == 1
    assert RENDER.main(["--output", str(stale)]) == 0 and RENDER.main(["--check", "--output", str(stale)]) == 0


def test_yaml_comments_fill_missing_descriptions() -> None:
    comments = RENDER.yaml_comments()
    assert comments["monitor.metrics.lj_residual.lazy_threshold"].startswith("fit atoms")
    row = next(line for line in REFERENCE.read_text(encoding="utf-8").splitlines() if line.startswith("| `monitor.engine` |"))
    assert "global_lj" in row
