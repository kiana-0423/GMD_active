"""OPS-002 (GMD_SGT runtime smoke) and OPS-003 (tiny H2O train/export/infer chain)."""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SGT_PATH = os.environ.get("GMD_SGT_PATH", str(ROOT.parent / "GMD_SGT"))


def _load(relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(relative.replace("/", "_").removesuffix(".py"), ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SMOKE = _load("tools/smoke/sgt_runtime_smoke.py")
CHAIN = _load("tools/smoke/sgt_tiny_h2o/run_chain.py")


def _sgt_available() -> bool:
    return (Path(SGT_PATH) / "gmd_sgt").is_dir() or importlib.util.find_spec("gmd_sgt") is not None


needs_sgt = pytest.mark.skipif(not _sgt_available(), reason=f"GMD_SGT not found ($GMD_SGT_PATH or {SGT_PATH})")


# ------------------------------------------------------------------------------- runtime smoke
@needs_sgt
def test_runtime_smoke_passes_on_cpu(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("GMD_SGT_SMOKE_DEVICE", "cpu")
    assert SMOKE.main(["--json", "--gmd-sgt", SGT_PATH]) == SMOKE.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["device"] == "cpu" and report["tensor_check"] is True and report["gmd_sgt_import"] is True
    assert tuple(int(p) for p in report["gmd_sgt"].split(".")[:2]) >= (0, 4)
    assert {"torch", "cuda_available", "mps_available", "python", "exit_code"} <= set(report)


def test_unsupported_or_unavailable_device_exits_3(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setenv("GMD_SGT_SMOKE_DEVICE", "tpu")
    assert SMOKE.run(None)[0] == SMOKE.EXIT_DEVICE_UNAVAILABLE
    if not torch.cuda.is_available():
        monkeypatch.setenv("GMD_SGT_SMOKE_DEVICE", "cuda")
        code, report = SMOKE.run(None)
        assert code == SMOKE.EXIT_DEVICE_UNAVAILABLE and "CUDA is unavailable" in report["error"]


def test_conda_environment_guard_exits_4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GMD_SGT_REQUIRED_CONDA_ENV", "definitely-not-this-env")
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "other")
    assert SMOKE.run(None)[0] == SMOKE.EXIT_ENV_ERROR


def test_missing_torch_or_gmd_sgt_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    real = SMOKE.importlib.import_module

    def fake(name: str, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        if name.split(".")[0] in {"torch", "gmd_sgt"}:
            raise ImportError(f"no module {name}")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(SMOKE.importlib, "import_module", fake)
    code, report = SMOKE.run(None)
    assert code == SMOKE.EXIT_IMPORT_ERROR and "torch import failed" in report["error"]


# ------------------------------------------------------------------------------- tiny chain
@needs_sgt
def test_tiny_h2o_chain_trains_exports_and_infers_on_cpu(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    started = time.monotonic()
    code = CHAIN.main(["--work-dir", str(tmp_path), "--gmd-sgt", SGT_PATH, "--report", str(tmp_path / "report.json")])
    elapsed = time.monotonic() - started
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert code == CHAIN.EXIT_OK, report
    assert elapsed < 90.0  # OPS-003 acceptance
    for key in ("export_inference", "checkpoint_inference"):
        assert report[key]["finite"] is True and np.isfinite(report[key]["energy_ev"])
        assert np.asarray(report[key]["forces_ev_per_a"]).shape == (3, 3)
    assert report["export_inference"]["kind"] == "export" and report["export_vs_checkpoint_max_force_deviation"] < 1e-3
    assert Path(report["export"]).is_file()
    # Distinct exit codes on the produced model.
    assert CHAIN.main(["--infer-only", "--model", report["export"], "--symbols", "COH", "--gmd-sgt", SGT_PATH]) == CHAIN.EXIT_ELEMENT_MISMATCH
    assert CHAIN.main(["--infer-only", "--model", str(tmp_path / "missing.pt"), "--gmd-sgt", SGT_PATH]) == CHAIN.EXIT_MODEL_PATH_ERROR
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"not a model")
    assert CHAIN.main(["--infer-only", "--model", str(garbage), "--gmd-sgt", SGT_PATH]) == CHAIN.EXIT_MODEL_LOAD_FAILED

