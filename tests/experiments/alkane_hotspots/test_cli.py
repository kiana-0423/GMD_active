"""CLI: command registry/extension hook, offline commands, and the real-CP2K smoke (marker ``cp2k``)."""

from __future__ import annotations

import importlib.machinery
import json
import os
import shutil
import sys
import types
from pathlib import Path

import pytest
import yaml

from ._helpers import CONFIGS, FIXTURES
from tools.experiments.alkane_hotspots import cli


def test_registry_refuses_duplicates_and_extensions_register(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "COMMANDS", dict(cli.COMMANDS))
    with pytest.raises(ValueError, match="already registered"):
        cli.register_command("generate", "again", None, lambda args, loaded: 0)
    name = "tools.experiments.alkane_hotspots._test_extension"
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)

    def register_cli(register):
        register("beta-demo", "registered by an extension", lambda parser: parser.add_argument("--flag", action="store_true"),
                 lambda args, loaded: 7 if args.flag else 3)

    module.register_cli = register_cli
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(cli, "_LOADED_EXTENSIONS", set())
    assert cli.load_extensions(("_test_extension", "module_that_does_not_exist")) == ["_test_extension"]
    assert cli.load_extensions(("_test_extension",)) == ["_test_extension"]  # idempotent: no duplicate registration
    parser = cli.build_parser()
    args = parser.parse_args(["beta-demo", "--config", "x.yaml", "--flag"])
    assert cli.COMMANDS["beta-demo"].handler(args, None) == 7
    for command in ("generate", "label-full", "fidelity-prepare", "fidelity-run", "fidelity-report", "status"):
        assert command in cli.COMMANDS


def _tiny_yaml(tmp_path: Path) -> Path:
    data = yaml.safe_load((FIXTURES / "tiny.yaml").read_text())
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_generate_status_and_prepare_offline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _tiny_yaml(tmp_path)
    root = tmp_path / "artifacts"
    assert cli.main(["generate", "--config", str(config), "--artifacts-root", str(root)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["n_parents"] == 16 and (root / "manifests" / "generate.json").exists()
    assert cli.main(["fidelity-prepare", "--config", str(config), "--artifacts-root", str(root)]) == 0
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["n_specs"] > 0 and prepared["n_fragment_calculations"] <= prepared["n_specs"]
    assert cli.main(["status", "--config", str(config), "--artifacts-root", str(root)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["remaining"]["full"] == len([1 for line in (root / "parents.jsonl").read_text().splitlines() if '"calibration"' in line])
    assert status["remaining"]["fragment"] == prepared["n_fragment_calculations"]
    assert status["remaining"]["estimated_cpu_core_hours"] > 0 and status["budgets"]["spent_cpu_core_hours"] == 0.0


def test_pilot_generate_is_deterministic(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = CONFIGS / "pilot.yaml"
    for name in ("a", "b"):
        assert cli.main(["generate", "--config", str(config), "--artifacts-root", str(tmp_path / name)]) == 0
    capsys.readouterr()
    assert (tmp_path / "a" / "parents.jsonl").read_bytes() == (tmp_path / "b" / "parents.jsonl").read_bytes()
    parents = [json.loads(line) for line in (tmp_path / "a" / "parents.jsonl").read_text().splitlines()]
    assert len(parents) == 16 and {p["chain_length"] for p in parents} == {16}
    calibration = {p["chain_length"] for p in parents if p["split"] == "calibration"}
    assert calibration == {16}


@pytest.mark.cp2k
def test_real_cp2k_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """C8H18: two full parents and one capped graph fragment (cap relaxation + single point) at 200 Ry."""

    executable = os.environ.get("CP2K_BIN") or shutil.which("cp2k.psmp")
    if not executable:
        pytest.skip("CP2K binary not found (set CP2K_BIN or put cp2k.psmp on PATH)")
    data = yaml.safe_load((CONFIGS / "smoke.yaml").read_text())
    data["cp2k"]["executable"] = executable
    config = tmp_path / "smoke.yaml"
    config.write_text(yaml.safe_dump(data))
    root = tmp_path / "artifacts"
    common = ["--config", str(config), "--artifacts-root", str(root)]
    for command in (["generate"], ["fidelity-prepare"], ["fidelity-run"], ["fidelity-report"], ["status"]):
        assert cli.main([*command, *common]) == 0
    capsys.readouterr()
    records = [json.loads(line) for line in (root / "labels.jsonl").read_text().splitlines()]
    assert sorted(r["kind"] for r in records) == ["fragment", "full", "full"]
    assert all(r["status"] == "valid" and r["cp2k"]["smoke_only"] for r in records)
    assert len({r["cp2k"]["settings_hash"] for r in records}) == 1 and records[0]["cp2k"]["version"]
    fragment = next(r for r in records if r["kind"] == "fragment")
    assert fragment["caps_relaxed"] and fragment["cost"]["attempts"] == 2 and set(fragment["cost"]["stage_wall_s"]) == {"hopt", "sp"}
    assert all(r["cost"]["wall_s"] > 0 and r["cost"]["timing_sources"] == ["local_job_record"] for r in records)
    report = json.loads((root / "fidelity" / "report_calibration.json").read_text())
    (summary,) = report["settings"]
    assert summary["n_evaluated"] == 1 and summary["pooled_core_force_rmse"] is not None
    assert report["decision"]["status"] in {"selected", "failed"}
    # Re-running is idempotent: no new CP2K job.
    tasks = sorted(p.name for p in (root / "cp2k_tasks").iterdir())
    assert cli.main(["fidelity-run", *common]) == 0
    assert json.loads(capsys.readouterr().out)["reused"] and sorted(p.name for p in (root / "cp2k_tasks").iterdir()) == tasks


def _load_example(name: str):
    import importlib.util

    from ._helpers import ROOT

    path = ROOT / "examples" / "alkane_hotspot_demo" / name
    spec = importlib.util.spec_from_file_location(f"alkane_example_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_scripts_offline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _tiny_yaml(tmp_path)
    root = tmp_path / "artifacts"
    assert _load_example("generate_parents.py").main(["--config", str(config), "--artifacts-root", str(root)]) == 0
    assert (root / "parents_summary.json").exists()
    assert _load_example("run_fidelity_gate.py").main(["--config", str(config), "--artifacts-root", str(root), "--prepare-only"]) == 0
    assert '"remaining"' in capsys.readouterr().out and (root / "fidelity" / "fragments.jsonl").exists()
