"""Monitor benchmark tool: quick run, CSV/JSON columns, regression gate, timing outside tracemalloc."""

from __future__ import annotations

import csv
import json
import sys
import tracemalloc
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = PROJECT_ROOT / "tools" / "benchmarks" / "benchmark_monitor.py"
PHAL_COLUMNS = ["metric", "n_atoms", "direct_s", "fast_s", "speedup", "direct_peak_mb", "fast_peak_mb"]


def _load_tool() -> ModuleType:
    spec = spec_from_file_location("gmd_benchmark_monitor", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_tool()


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, int]:
    out = tmp_path_factory.mktemp("bench")
    csv_path, json_path = out / "bench.csv", out / "bench.json"
    code = bench.main(["--quick", "--output", str(csv_path), "--json-output", str(json_path)])
    return csv_path, json_path, code


def test_quick_run_writes_phal_csv_columns(quick_run: tuple[Path, Path, int]) -> None:
    csv_path, _json_path, code = quick_run
    assert code == 0
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        header = reader.fieldnames
    assert header is not None and header[: len(PHAL_COLUMNS)] == PHAL_COLUMNS
    assert list(header) == list(bench.CSV_COLUMNS)
    assert {row["metric"] for row in rows} == {"neighbor_list", "rmin", "coordination", "lj_residual"}
    assert {int(row["n_atoms"]) for row in rows} == {32, 64}
    for row in rows:
        assert float(row["fast_s"]) >= 0.0 and float(row["fast_peak_mb"]) >= 0.0
        if row["metric"] in {"rmin", "coordination"}:  # two implementations -> a real speedup
            assert float(row["direct_s"]) > 0.0 and float(row["speedup"]) > 0.0 and row["direct_impl"]
        else:  # single path: no fake "direct" column
            assert row["direct_s"] == row["speedup"] == row["direct_peak_mb"] == row["direct_impl"] == ""


def test_quick_run_json_is_a_usable_baseline(quick_run: tuple[Path, Path, int]) -> None:
    _csv_path, json_path, _code = quick_run
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["threshold"] == pytest.approx(0.2)
    assert payload["seed"] == bench.DEFAULT_SEED and payload["quick"] is True
    assert payload["regressions"] == []
    assert all(set(bench.CSV_COLUMNS) <= set(row) for row in payload["results"])
    lj = [row for row in payload["results"] if row["metric"] == "lj_residual"]
    assert lj and all(row["direct_s"] is None and row["speedup"] is None for row in lj)
    assert set(bench.load_baseline(json_path)) == {f"{row['metric']}:{row['n_atoms']}" for row in payload["results"]}


def _baseline(path: Path, source: Path, scale: float) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    for row in payload["results"]:
        row["fast_s"] = row["fast_s"] * scale if row["fast_s"] else 0.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_regression_gate_passes_fails_and_warns(quick_run: tuple[Path, Path, int], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _csv_path, json_path, _code = quick_run
    generous = _baseline(tmp_path / "slow_baseline.json", json_path, 1.0e6)
    impossible = tmp_path / "fast_baseline.json"
    impossible.write_text(
        json.dumps({"results": [{"metric": "rmin", "n_atoms": 32, "fast_s": 1.0e-12, "direct_s": 1.0}]}), encoding="utf-8"
    )
    common = ["--quick", "--output", str(tmp_path / "out.csv")]

    assert bench.main([*common, "--baseline", str(generous)]) == 0
    assert bench.main([*common, "--baseline", str(impossible)]) == 1
    assert "PERF REGRESSION: rmin:32" in capsys.readouterr().err
    assert bench.main([*common, "--baseline", str(impossible), "--warn-only", "--json-output", str(tmp_path / "w.json")]) == 0
    assert "PERF REGRESSION" in capsys.readouterr().err
    assert json.loads((tmp_path / "w.json").read_text(encoding="utf-8"))["regressions"]


def test_default_threshold_is_twenty_percent() -> None:
    baseline = {"rmin:100": 1.0}

    assert bench.DEFAULT_THRESHOLD == pytest.approx(0.2)
    assert bench.regressions([{"metric": "rmin", "n_atoms": 100, "fast_s": 1.19}], baseline, threshold=0.2) == []
    assert len(bench.regressions([{"metric": "rmin", "n_atoms": 100, "fast_s": 1.21}], baseline, threshold=0.2)) == 1
    assert bench.regressions([{"metric": "rmin", "n_atoms": 200, "fast_s": 9.0}], baseline, threshold=0.2) == []  # no baseline row
    assert bench.regressions([{"metric": "rmin", "n_atoms": 100, "fast_s": 9.0}], {"rmin:100": 0.0}, threshold=0.2) == []


def test_phal_style_list_baseline_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text(json.dumps([{"metric": "rmin", "n_atoms": 10, "fast_s": "0.5", "direct_s": "1"}]), encoding="utf-8")

    assert bench.load_baseline(path) == {"rmin:10": 0.5}


def test_timing_runs_without_tracemalloc_and_memory_in_its_own_pass() -> None:
    """PHAL ran the timed loop under tracemalloc, inflating every time (UTIL-008 notes)."""

    seen: list[bool] = []

    def probe() -> None:
        seen.append(tracemalloc.is_tracing())

    bench.time_call(probe, repeats=3)
    assert seen == [False] * 4  # warm-up + 3 timed calls
    seen.clear()
    bench.peak_memory_mb(probe)
    assert seen == [True]
    assert not tracemalloc.is_tracing()


def test_systems_are_deterministic_and_forces_physical() -> None:
    from gmd_active_learning.monitors.lj_local import LocalLJEstimator
    from gmd_active_learning.utils.neighbors import build_neighbor_pairs

    first, second = bench.build_system(64, seed=3), bench.build_system(64, seed=3)
    other = bench.build_system(64, seed=4)

    np.testing.assert_array_equal(first.positions, second.positions)
    np.testing.assert_array_equal(first.forces, second.forces)
    assert not np.array_equal(first.positions, other.positions)
    assert np.all(np.isfinite(first.forces)) and not np.allclose(first.forces, 1.0)
    pairs = build_neighbor_pairs(first.positions, first.cell, first.pbc, bench.CUTOFF)
    assert pairs.distances.min() > 1.0  # jittered lattice: no overlapping atoms
    residuals, fits = LocalLJEstimator(cutoff=bench.CUTOFF)(pairs, first.forces)
    valid = np.array([fit.valid for fit in fits])
    # LJ-like forces: the linear fit recovers them for most atoms (only the 1 meV/A noise remains).
    assert valid.mean() > 0.8 and np.median(residuals[valid]) < 0.05


def test_include_nonlinear_adds_a_separate_single_path_row(tmp_path: Path) -> None:
    out = tmp_path / "nl.csv"
    assert bench.main(["--atoms", "16", "--include-nonlinear", "--output", str(out)]) == 0
    with out.open(encoding="utf-8", newline="") as handle:
        rows = {row["metric"]: row for row in csv.DictReader(handle)}
    assert rows["lj_residual_nonlinear"]["direct_s"] == "" and rows["lj_residual"]["direct_s"] == ""


@pytest.mark.parametrize("argv", [["--atoms", "1"], ["--atoms", ""], ["--threshold", "-0.1"], ["--density", "0"]])
def test_invalid_arguments_are_rejected(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        bench.main(["--quick", *argv])
