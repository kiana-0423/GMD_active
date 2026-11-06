"""Shared helpers for the alkane hotspot tests (offline; real CP2K tests are marked ``cp2k``)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:  # tools/ is imported from the repository root
    sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "tests" / "fixtures" / "alkane_hotspots"
CONFIGS = ROOT / "configs" / "demos" / "alkane_hotspots"


def tiny_config(tmp_path: Path, **overrides):
    """The tiny fixture config with artifacts under ``tmp_path`` and optional top-level overrides."""

    import yaml

    from tools.experiments.alkane_hotspots.config import AlkaneConfig, LoadedConfig

    data = yaml.safe_load((FIXTURES / "tiny.yaml").read_text())
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(data))
    return LoadedConfig(AlkaneConfig.model_validate(data), path, artifacts_override=tmp_path / "artifacts")
