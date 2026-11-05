"""Config sections (defaults == YAML, validation, PHAL mapping) and package boundaries (MON-017, CORE-007/008 slice)."""

from __future__ import annotations

import ast
import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from gmd_active_learning.core.exceptions import ConfigurationError
from gmd_active_learning.events.config import EventsConfig, events_config_from_phal, load_events_config
from gmd_active_learning.hotspots.config import HotspotsConfig, load_hotspots_config
from gmd_active_learning.monitors.config import MonitorConfig, load_monitor_config, monitor_config_from_phal

ROOT = Path(__file__).resolve().parents[2]
SECTIONS = ROOT / "configs" / "sections"
PACKAGES = ("monitors", "events", "hotspots")


@pytest.mark.parametrize(
    ("name", "model"), [("monitor", MonitorConfig), ("events", EventsConfig), ("hotspots", HotspotsConfig)]
)
def test_section_yaml_equals_model_defaults(name: str, model: type) -> None:
    data = yaml.safe_load((SECTIONS / f"{name}.yaml").read_text(encoding="utf-8"))
    assert set(data) == {name}
    assert model.model_validate(data[name]).model_dump() == model().model_dump()


def test_unknown_keys_and_inconsistent_settings_are_rejected() -> None:
    with pytest.raises(ConfigurationError):
        load_monitor_config({"monitor": {}})
    with pytest.raises(ConfigurationError):
        load_monitor_config({"metrics": {"rmin": {"threshold": 7.0}}})  # >= neighbour cutoff
    with pytest.raises(ConfigurationError):
        load_monitor_config({"metrics": {"rmin": {"enabled": False}}})  # stop_min_distance needs rmin
    assert load_monitor_config({"metrics": {"rmin": {"enabled": False}}, "actions": {"stop_min_distance": None}})
    with pytest.raises(ConfigurationError):
        load_monitor_config({"metrics": {"force": {"enabled": False}}, "actions": {"stop_max_force": 10.0}})
    with pytest.raises(ConfigurationError):
        load_events_config({"pre_trigger_frames": -1})
    with pytest.raises(ConfigurationError):
        load_hotspots_config({"threshold": 6.0})  # the second hotspot threshold no longer exists (F-16)


def test_phal_configuration_is_translated() -> None:
    import json

    phal = json.loads((ROOT / "tests" / "fixtures" / "phal_reference" / "ethane_box" / "config.json").read_text(encoding="utf-8"))
    config = monitor_config_from_phal(phal)
    assert config.neighbors.lj_cutoff == 5.0 and config.stages.scheduled_full_interval == 5
    assert config.metrics.rmin.threshold is None and config.metrics.delta_force.weight == 1.5
    assert config.metrics.delta_q.scale == 1.15 and config.metrics.delta_q.power == 6.0
    assert config.running_stats.warmup_policy == "raw" and config.running_stats.first_frame_deltas == "zero"
    assert monitor_config_from_phal(phal, legacy_scoring=False).running_stats.warmup_policy == "suppress"
    events = events_config_from_phal(phal)
    assert (events.pre_trigger_frames, events.post_trigger_frames, events.maxlen, events.cooldown_steps) == (3, 2, 16, 0)


def _modules() -> list[str]:
    names = []
    for package in PACKAGES:
        module = importlib.import_module(f"gmd_active_learning.{package}")
        names.append(module.__name__)
        names.extend(info.name for info in pkgutil.walk_packages(module.__path__, prefix=f"{module.__name__}."))
    names.append("gmd_active_learning.utils.neighbors")
    return names


def test_every_module_imports() -> None:
    for name in _modules():
        importlib.import_module(name)


FORBIDDEN = ("torch", "subprocess", "gmd_sgt", "lammps", "nequip", "allegro", "hotspot_al", "gmd_active_learning.adapters")


def test_core_packages_do_not_import_infrastructure() -> None:
    offenders = []
    for name in _modules():
        path = Path(importlib.import_module(name).__file__ or "")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    offenders.append(f"{name}: relative import")  # relative imports would evade this guard
                targets = [node.module or ""]
            for target in targets:
                if any(target == bad or target.startswith(bad + ".") for bad in FORBIDDEN):
                    offenders.append(f"{name}: {target}")
    assert offenders == []


def test_importing_monitors_pulls_in_no_torch_or_adapter() -> None:
    code = (
        "import sys, gmd_active_learning.monitors, gmd_active_learning.events, gmd_active_learning.hotspots;"
        "bad=[m for m in sys.modules if m=='torch' or m.startswith('gmd_active_learning.adapters')];"
        "print(','.join(bad))"
    )
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT,
                            env={"PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"})
    assert output.stdout.strip() == ""
