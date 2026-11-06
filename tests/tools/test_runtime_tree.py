"""OPS-008 runtime tree, CFG-004 runtime overlay, CFG-006 ML-SCF overlay, CFG-005 drop check."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from gmd_active_learning.config import build_config
from gmd_active_learning.utils.yaml import load_yaml, merge_dicts, merge_yaml_files

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("init_runtime_tree", ROOT / "tools/runtime/init_runtime_tree.py")
assert spec is not None and spec.loader is not None
TREE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TREE)

RUNTIME_OVERLAY = ROOT / "containers/config/runtime.yaml"
ML_OVERLAY = ROOT / "containers/config/cp2k_ml_scf.yaml"
EXAMPLE = ROOT / "configs/example_run.yaml"
PHAL_DIRS = {  # PHAL runtime/** skeleton, re-scoped: md/traj -> md/runs (GMD outputs), cp2k/inputs -> cp2k/tasks
    "datasets", "models/latest", "models/archive", "checkpoints", "md/restart", "cp2k/outputs", "cp2k/scf",
    "cp2k/scf_dataset/train", "cp2k/scf_dataset/validation", "cp2k/scf_dataset/test", "cp2k_ml/models",
    "cp2k_ml/checkpoints", "cp2k_ml/validation", "cp2k_ml/logs", "cp2k_ml/cache", "logs", "cache",
}


def _paths(mapping: dict, prefix: str = "") -> dict[str, str]:
    found: dict[str, str] = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            found.update(_paths(value, f"{prefix}{key}."))
        elif isinstance(value, str) and "/" in value:
            found[f"{prefix}{key}"] = value
    return found


def test_tree_is_created_under_a_configurable_root(tmp_path: Path) -> None:
    root = tmp_path / "rt"
    assert TREE.main(["--root", str(root)]) == 0
    assert TREE.missing(root) == []
    assert PHAL_DIRS <= set(TREE.RUNTIME_TREE)
    assert "cp2k_ml/models" in (root / "README.md").read_text()
    assert TREE.main(["--root", str(root), "--check"]) == 0
    assert TREE.main(["--root", str(tmp_path / "absent"), "--check"]) == 1


def test_committed_runtime_overlay_is_the_generated_one() -> None:
    assert RUNTIME_OVERLAY.read_text(encoding="utf-8") == TREE.overlay_text("/runtime")


@pytest.mark.parametrize("root", ["/runtime", "/scratch/user/gmd_rt"])
def test_overlay_validates_and_every_path_is_under_the_root(root: str, tmp_path: Path) -> None:
    raw = merge_dicts(load_yaml(EXAMPLE), TREE.overlay(root))
    config = build_config(raw, base_dir=EXAMPLE.parent)
    assert config["cp2k"].task_dir == f"{root}/cp2k/tasks"  # type: ignore[attr-defined]
    assert config["scf"].recording.enabled is True  # type: ignore[attr-defined]
    for key, value in _paths(TREE.overlay(root)).items():
        assert value.startswith(root + "/"), key
        relative = value[len(root) + 1:]
        assert any(relative == entry or relative.startswith(entry + "/") or entry.startswith(relative) for entry in TREE.RUNTIME_TREE), key


def test_relative_root_is_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert TREE.overlay("rt")["cp2k"]["task_dir"] == str(tmp_path.resolve() / "rt" / "cp2k" / "tasks")


def test_ml_scf_overlay_validates_over_runtime_overlay() -> None:
    raw = merge_yaml_files(EXAMPLE, RUNTIME_OVERLAY, ML_OVERLAY)
    config = build_config(raw, base_dir=EXAMPLE.parent)
    ml = config["scf"].ml_scf  # type: ignore[attr-defined]
    assert ml.enabled and ml.fallback_to_standard_scf and ml.require_final_scf_convergence
    assert ml.predictor.type == "dummy" and ml.mixing.method == "pulay" and ml.mixing.history_length == 8
    assert ml.prediction_dir.startswith("/runtime/cp2k_ml/")
    assert config["backends"].dft.engine == "cp2k"  # type: ignore[attr-defined]  # conventional labeling stays the default


def test_runtime_contract_doc_lists_every_directory() -> None:
    text = (ROOT / "docs/runtime-data-contract.md").read_text(encoding="utf-8")
    for relative in TREE.RUNTIME_TREE:
        assert f"`{relative}/`" in text, relative


def test_dropped_milestone1_config_templates_are_not_referenced() -> None:
    """CFG-005: PHAL's empty configs/{default,lammps,train}.yaml have no GMD counterpart or consumer."""

    for name in ("configs/lammps.yaml", "configs/train.yaml"):
        assert not (ROOT / name).exists()
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "lammps.yaml" in text or "train.yaml" in text:
            offenders.append(str(path))
    assert offenders == []
