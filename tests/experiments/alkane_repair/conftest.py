"""Fixtures of the repair-benchmark tests: the smoke profile copied into a temporary directory with its
artifacts root redirected there (offline: no CP2K, GMD or training unless a test says otherwise)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

CONFIGS = ROOT / "configs" / "demos" / "alkane_repair"
PROTOCOL = ROOT / "docs" / "demos" / "alkane_repair" / "protocol.md"


def copy_profile(tmp_path: Path, name: str = "repair_smoke", **overrides) -> Path:
    """Copy ``<name>{,_alpha,_model}.yaml`` (+ protocol) into ``tmp_path``; artifacts go to tmp_path/art."""

    (tmp_path / "protocol.md").write_text(PROTOCOL.read_text())
    alpha = yaml.safe_load((CONFIGS / f"{name}_alpha.yaml").read_text())
    alpha["artifacts_root"] = str(tmp_path / "art")
    (tmp_path / f"{name}_alpha.yaml").write_text(yaml.safe_dump(alpha))
    shutil.copyfile(CONFIGS / f"{name}_model.yaml", tmp_path / f"{name}_model.yaml")
    if (CONFIGS / f"{name}_authorization.yaml").is_file():
        shutil.copyfile(CONFIGS / f"{name}_authorization.yaml", tmp_path / f"{name}_authorization.yaml")
    bench = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
    bench["protocol_document"] = "protocol.md"
    for key, value in overrides.items():
        bench[key] = value
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(bench))
    return path


@pytest.fixture()
def smoke(tmp_path: Path):
    from tools.experiments.alkane_repair.config import load_benchmark

    return load_benchmark(copy_profile(tmp_path))
