from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from gmd_active_learning.cli import build_parser, main
from gmd_active_learning.config import build_config, load_config
from gmd_active_learning.core.exceptions import ConfigurationError

REPO = Path(__file__).resolve().parents[1]


def _minimal(tmp_path: Path, **extra: object) -> Path:
    executable = tmp_path / "gmd"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    (tmp_path / "box.xyz").write_text("1\n5 5 5\nAr 0 0 0\n")
    raw = {
        "workflow": {"run_id": "t"},
        "md": {"executable": "gmd", "initial_structure": "box.xyz"},
        "mlip": {},
        "cp2k": {"executable": "cp2k.psmp"},
        **extra,
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_parser_lists_all_commands() -> None:
    commands = build_parser()._subparsers._group_actions[0].choices  # noqa: SLF001
    assert {"validate-config", "doctor", "init", "run", "status", "stop", "retrain", "rollback", "resolve-unknown", "scf"} <= set(commands)


def test_example_config_is_valid() -> None:
    loaded = load_config(REPO / "configs" / "example_run.yaml")
    assert {"workflow", "md", "mlip", "cp2k", "monitor", "extraction"} <= set(loaded.sections)
    # relative paths resolve against the config file, not the CWD
    assert Path(loaded["md"].initial_structure).is_absolute()  # type: ignore[attr-defined]


def test_config_errors_are_explicit(tmp_path: Path) -> None:
    path = _minimal(tmp_path)
    raw = yaml.safe_load(path.read_text())
    with pytest.raises(ConfigurationError, match="unknown config section"):
        build_config({**raw, "lammps": {}}, base_dir=tmp_path)
    with pytest.raises(ConfigurationError, match="missing required"):
        build_config({key: value for key, value in raw.items() if key != "cp2k"}, base_dir=tmp_path)
    with pytest.raises(ConfigurationError, match="invalid config section 'workflow'"):
        build_config({**raw, "workflow": {"run_id": "t", "max_cycles": 0}}, base_dir=tmp_path)
    with pytest.raises(ConfigurationError, match="invalid config section 'monitor'"):
        build_config({**raw, "monitor": {"no_such_key": 1}}, base_dir=tmp_path)


def test_config_hash_is_stable_and_sensitive(tmp_path: Path) -> None:
    first = load_config(_minimal(tmp_path)).config_hash
    assert load_config(tmp_path / "run.yaml").config_hash == first
    changed = _minimal(tmp_path, workflow={"run_id": "t", "max_cycles": 2})
    assert load_config(changed).config_hash != first


def test_validate_config_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate-config", "--config", str(_minimal(tmp_path))]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True
    bad = tmp_path / "bad.yaml"
    bad.write_text("workflow: {run_id: x}\nlammps: {}\n")
    assert main(["validate-config", "--config", str(bad)]) == 2


def test_doctor_reports_unavailable_backends(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _minimal(tmp_path, cp2k={"executable": str(tmp_path / "missing-cp2k")}, mlip={"source_path": str(tmp_path / "nowhere")})
    (tmp_path / "gmd").unlink()
    code = main(["doctor", "--config", str(path)])
    rows = {row["backend"]: row for row in json.loads(capsys.readouterr().out)}
    assert code == 1
    assert rows["md:gmd"]["available"] is False
    assert rows["dft:cp2k"]["available"] is False


def test_status_requires_existing_state(tmp_path: Path) -> None:
    assert main(["status", "--work-dir", str(tmp_path)]) == 2
