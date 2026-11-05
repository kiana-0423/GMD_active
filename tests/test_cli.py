"""gmd-al CLI (ledger CLI-001/002/003); doctor cases ported from PHAL tests/unit/test_cli.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from gmd_active_learning import cli
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
        "md": {"executable": "./gmd", "initial_structure": "box.xyz"},
        "mlip": {},
        "cp2k": {"executable": "cp2k.psmp"},
        **extra,
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_parser_lists_all_commands() -> None:
    commands = build_parser()._subparsers._group_actions[0].choices  # noqa: SLF001
    assert {"validate-config", "doctor", "init", "run", "status", "stop", "retrain", "rollback", "resolve-unknown", "scf",
            "convert-phal-config"} <= set(commands)


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


def _no_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GMD_EXECUTABLE", "CP2K_EXECUTABLE", "GMD_SGT_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda _name: None)
    monkeypatch.setattr(cli, "_check_gmd_sgt", lambda _source: (cli.CheckResult("mlip:gmd_sgt", False, "not importable"), None))
    monkeypatch.setattr(cli, "_check_torch", lambda require_cuda: [cli.CheckResult("torch", True, "torch x"),
                                                                    cli.CheckResult("cuda", False, "cuda unavailable", require_cuda)])


def test_collect_runtime_checks_reports_missing_executables(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_runtime(monkeypatch)
    results, paths = cli.collect_runtime_checks()
    assert paths == {}
    rows = {row.name: row for row in results}
    assert not rows["md:gmd"].ok and "GMD_EXECUTABLE" in rows["md:gmd"].detail
    assert not rows["dft:cp2k"].ok and "CP2K_EXECUTABLE" in rows["dft:cp2k"].detail
    assert not rows["cuda"].required  # GMD runs TorchScript on the CPU; CUDA only matters for mlip.device=cuda


def test_environment_variables_take_precedence_over_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _no_runtime(monkeypatch)
    gmd, cp2k = tmp_path / "gmd-bin", tmp_path / "cp2k-bin"
    for path in (gmd, cp2k):
        path.write_text("#!/bin/sh\n")
        path.chmod(0o755)
    monkeypatch.setenv("GMD_EXECUTABLE", str(gmd))
    monkeypatch.setenv("CP2K_EXECUTABLE", str(cp2k))
    monkeypatch.setattr(cli.shutil, "which", lambda _name: "/usr/bin/ignored")
    _results, paths = cli.collect_runtime_checks()
    assert paths == {"gmd": str(gmd.resolve()), "cp2k": str(cp2k.resolve())}


def test_configured_executables_win_and_cuda_is_required_for_cuda_training(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _no_runtime(monkeypatch)
    monkeypatch.setenv("GMD_EXECUTABLE", "/nonexistent/gmd")
    path = _minimal(tmp_path, mlip={"device": "cuda:0"})
    results, paths = cli.collect_runtime_checks(load_config(path))
    rows = {row.name: row for row in results}
    assert rows["md:gmd"].ok and paths["gmd"] == str((tmp_path / "gmd").resolve())
    assert rows["cuda"].required and not rows["cuda"].ok


def test_write_runtime_config_emits_a_loadable_valid_config(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "runtime.local.yaml"
    written = cli.write_runtime_config(target, {"gmd": "/opt/gmd/bin/gmd", "cp2k": "/opt/cp2k/bin/cp2k.psmp"})
    assert written == target
    loaded = load_config(target)
    assert loaded["md"].executable == "/opt/gmd/bin/gmd"  # type: ignore[attr-defined]
    assert loaded["cp2k"].executable == "/opt/cp2k/bin/cp2k.psmp"  # type: ignore[attr-defined]
    empty = cli.write_runtime_config(tmp_path / "empty.yaml", {})
    assert load_config(empty)["md"].executable == "gmd"  # type: ignore[attr-defined]  # placeholder, PATH lookup


def test_doctor_returns_zero_without_strict(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _no_runtime(monkeypatch)
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "[MISS] md:gmd" in out and "[OK] torch: torch x" in out and "(optional)" in out


def test_doctor_strict_fails_when_a_required_check_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_runtime(monkeypatch)
    assert main(["doctor", "--strict"]) == 1


def test_doctor_strict_passes_when_everything_required_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "collect_runtime_checks", lambda _config=None: (
        [cli.CheckResult("md:gmd", True, "/opt/gmd"), cli.CheckResult("cuda", False, "cpu only", required=False)], {"gmd": "/opt/gmd"}))
    assert main(["doctor", "--strict"]) == 0


def test_doctor_writes_config_when_requested(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "runtime.local.yaml"
    monkeypatch.setattr(cli, "collect_runtime_checks", lambda _config=None: ([cli.CheckResult("md:gmd", True, "/opt/gmd")], {"gmd": "/opt/gmd"}))
    assert main(["doctor", "--write-config", str(target)]) == 0
    assert "wrote runtime config" in capsys.readouterr().out
    assert load_config(target)["md"].executable == "/opt/gmd"  # type: ignore[attr-defined]


def test_write_config_flag_defaults_to_documented_path() -> None:
    args = build_parser().parse_args(["doctor", "--write-config"])
    assert args.write_config == str(cli.DEFAULT_RUNTIME_CONFIG)


def test_doctor_with_config_reports_backend_runtime_checks_as_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _minimal(tmp_path, cp2k={"executable": str(tmp_path / "missing-cp2k")}, mlip={"source_path": str(tmp_path / "nowhere")})
    (tmp_path / "gmd").unlink()
    assert main(["doctor", "--config", str(path), "--json"]) == 0
    rows = {row["backend"]: row for row in json.loads(capsys.readouterr().out)}
    assert rows["md:gmd"]["available"] is False
    assert rows["dft:cp2k"]["available"] is False
    assert main(["doctor", "--config", str(path), "--json", "--strict"]) == 1


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_format_result_marks_status() -> None:
    assert cli._format_result(cli.CheckResult("dft:cp2k", True, "/opt/cp2k")) == "[OK] dft:cp2k: /opt/cp2k"  # noqa: SLF001
    assert cli._format_result(cli.CheckResult("dft:cp2k", False, "missing")) == "[MISS] dft:cp2k: missing"  # noqa: SLF001
    assert cli._format_result(cli.CheckResult("cuda", False, "none", required=False)) == "[MISS] cuda: none (optional)"  # noqa: SLF001


_HEAVY = ("torch", "gmd_sgt", "gmd_active_learning.adapters.dft.scf", "gmd_active_learning.adapters.sgt")


@pytest.mark.parametrize("argv", [["--help"], ["doctor", "--help"], ["scf"], ["validate-config", "--config", str(REPO / "configs" / "default.yaml")]])
def test_startup_does_not_import_torch_gmd_sgt_or_scf_tooling(argv: list[str]) -> None:
    code = (
        "import json, sys\n"
        "from gmd_active_learning.cli import main\n"
        "try:\n    main(json.loads(sys.argv[1]))\nexcept SystemExit:\n    pass\n"
        f"heavy = [m for m in sys.modules if m.split('.')[0] in {{'torch', 'gmd_sgt'}} or m.startswith({_HEAVY[2:]!r})]\n"
        "sys.stderr.write('HEAVY=' + json.dumps(sorted(heavy)))\n"
    )
    if argv == ["scf"]:  # the placeholder parser alone (parsing without dispatch)
        code = code.replace("main(json.loads(sys.argv[1]))", "from gmd_active_learning.cli import build_parser; build_parser().parse_known_args(['scf'])")
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, "-c", code, json.dumps(argv)], capture_output=True, text=True, env=env, timeout=120)
    heavy = json.loads(done.stderr.rsplit("HEAVY=", 1)[1])
    if argv[0] == "validate-config":  # validating the scf/mlip sections loads their pydantic models, nothing else
        heavy = [name for name in heavy if name.split(".")[0] in {"torch", "gmd_sgt"}]
    assert heavy == [], heavy


def test_scf_subcommand_is_delegated_lazily(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["scf", "--help"])
    assert exc.value.code == 0
    assert "dataset" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["doctor", "--no-such-flag"])


def test_status_requires_existing_state(tmp_path: Path) -> None:
    assert main(["status", "--work-dir", str(tmp_path)]) == 2
