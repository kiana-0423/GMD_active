# Ported from PHAL (tests.scf.test_scf_capabilities_and_benchmark) at 96263b6; MIT License, see NOTICE.
"""Offline tests for Stage 4 capability detection and Stage 7 benchmarking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gmd_active_learning.core.exceptions import CapabilityUnavailableError

from tools.scf.benchmark.modes import (
    BENCHMARK_MODES,
    MODE_ML_FULL,
    MODE_ML_GUESS,
    MODE_NATIVE,
    MODE_RESTART,
    BenchmarkError,
    load_runs,
    run_benchmark,
    runs_from_scf_histories,
    summarize_mode,
    write_report,
)
from gmd_active_learning.adapters.dft.scf.capabilities import (
    CP2KCapabilities,
    detect_cp2k_capabilities,
    parse_version_banner,
    require_ml_scf,
)
from tools.scf.benchmark.evaluate import SCFRun

NATIVE_BANNER = """\
 CP2K version 2024.1
 Source code revision git:0b9d1a2f
 cp2kflags: omp libint fftw3 libxc parallel mpi3 scalapack
"""

FORK_BANNER = """\
 CP2K version 2024.1 (phal-ml-scf)
 Source code revision git:deadbeef1234
 cp2kflags: omp libint fftw3 libxc parallel phal_ml_scf phal_ml_mixing libtorch
"""


# -- capability parsing -------------------------------------------------


def test_native_banner_reports_no_ml_support() -> None:
    parsed = parse_version_banner(NATIVE_BANNER)

    assert parsed["version"] == "2024.1"
    assert parsed["git_commit"] == "0b9d1a2f"
    assert not parsed["ml_scf_supported"]
    assert not parsed["ml_mixing_supported"]
    assert not parsed["torch_supported"]
    assert "libint" in parsed["flags"]


def test_fork_banner_reports_full_ml_support() -> None:
    parsed = parse_version_banner(FORK_BANNER)

    assert parsed["version"] == "2024.1"
    assert parsed["git_commit"] == "deadbeef1234"
    assert parsed["ml_scf_supported"]
    assert parsed["ml_mixing_supported"]
    assert parsed["torch_supported"]


def test_banner_without_a_version_yields_none() -> None:
    parsed = parse_version_banner("some unrelated output")

    assert parsed["version"] is None
    assert parsed["git_commit"] is None
    assert parsed["flags"] == ()


def test_revision_is_read_with_and_without_the_git_prefix() -> None:
    """v2026.2 dropped the ``git:`` prefix that v2024.1 printed.

    This is the real v2026.2 banner. Requiring the prefix made the commit parse
    as None after the migration, losing build provenance silently.
    """
    migrated = parse_version_banner(
        " CP2K version 2026.2\n Source code revision 67b5da8\n cp2kflags: omp libint libxc elpa\n"
    )
    assert migrated["version"] == "2026.2"
    assert migrated["git_commit"] == "67b5da8"

    legacy = parse_version_banner(" CP2K version 2024.1\n Source code revision git:b4a17a5\n")
    assert legacy["git_commit"] == "b4a17a5"


def test_version_string_form_is_also_recognised() -> None:
    parsed = parse_version_banner(" CP2K| version string: CP2K version TEST-2026.1")

    # Deliberate deviation from PHAL ("CP2K version TEST-2026.1"): both banner
    # forms normalise to the bare version, so provenance from a probe and from a
    # parsed output agree (cp2k-ml-scf RECONSTRUCTION.md 4f, finding 3).
    assert parsed["version"] == "TEST-2026.1"


def test_detect_uses_the_injected_runner(tmp_path: Path) -> None:
    executable = tmp_path / "cp2k.psmp"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)

    capabilities = detect_cp2k_capabilities(
        executable=executable, runner=lambda command: FORK_BANNER
    )

    assert capabilities.available
    assert capabilities.version == "2024.1"
    assert capabilities.git_commit == "deadbeef1234"
    assert capabilities.ml_scf_supported


def test_detect_reports_a_missing_executable_without_raising() -> None:
    capabilities = detect_cp2k_capabilities(executable="definitely-not-a-real-cp2k-binary")

    assert not capabilities.available
    assert capabilities.version is None
    assert not capabilities.ml_scf_supported
    assert any("not found" in detail for detail in capabilities.details)


def test_detect_survives_a_probe_that_raises(tmp_path: Path) -> None:
    executable = tmp_path / "cp2k.psmp"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)

    def exploding_runner(command):
        raise OSError("boom")

    capabilities = detect_cp2k_capabilities(executable=executable, runner=exploding_runner)

    assert capabilities.available
    assert capabilities.version is None
    assert any("could not run" in detail for detail in capabilities.details)


def test_detect_prefers_the_ml_scf_executable_from_config(tmp_path: Path) -> None:
    ml_executable = tmp_path / "cp2k.ml.psmp"
    ml_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    ml_executable.chmod(0o755)
    config = {
        "cp2k_ml_scf": {"executable": str(ml_executable)},
        "cp2k": {"executable": "cp2k.popt"},
    }

    capabilities = detect_cp2k_capabilities(config=config, runner=lambda command: FORK_BANNER)

    assert capabilities.executable == str(ml_executable)


def test_provenance_block_has_the_documented_shape(tmp_path: Path) -> None:
    executable = tmp_path / "cp2k.psmp"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    capabilities = detect_cp2k_capabilities(executable=executable, runner=lambda command: FORK_BANNER)

    provenance = capabilities.provenance(ml_model_digest="sha256:abc", ml_scf_enabled=True)

    assert provenance["cp2k_version"] == "2024.1"
    assert provenance["cp2k_commit"] == "deadbeef1234"
    assert provenance["ml_model_digest"] == "sha256:abc"
    assert provenance["ml_scf_enabled"] is True


def test_provenance_does_not_claim_ml_on_a_native_build(tmp_path: Path) -> None:
    executable = tmp_path / "cp2k.psmp"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    capabilities = detect_cp2k_capabilities(executable=executable, runner=lambda command: NATIVE_BANNER)

    provenance = capabilities.provenance(ml_model_digest="sha256:abc", ml_scf_enabled=True)

    assert provenance["ml_scf_enabled"] is False


def test_require_ml_scf_raises_for_a_native_build() -> None:
    native = CP2KCapabilities(
        executable="cp2k.psmp",
        resolved_path="/usr/bin/cp2k.psmp",
        version="2024.1",
        git_commit="0b9d1a2f",
        ml_scf_supported=False,
        ml_mixing_supported=False,
        torch_supported=False,
        python_torch_available=True,
    )

    with pytest.raises(CapabilityUnavailableError, match="does not support ML-SCF"):
        require_ml_scf(native)


def test_require_ml_scf_raises_when_cp2k_is_absent() -> None:
    missing = CP2KCapabilities(
        executable=None,
        resolved_path=None,
        version=None,
        git_commit=None,
        ml_scf_supported=False,
        ml_mixing_supported=False,
        torch_supported=False,
        python_torch_available=True,
    )

    with pytest.raises(CapabilityUnavailableError, match="no working CP2K executable"):
        require_ml_scf(missing)


# -- benchmark ----------------------------------------------------------


def mode_runs(label: str, iterations: dict[str, int], *, energy: float = -1234.5, **kwargs) -> list[SCFRun]:
    return [
        SCFRun(
            label=label,
            system_id=system_id,
            scf_iterations=count,
            converged=True,
            wall_time_seconds=float(count) * 2.0,
            final_energy=energy,
            **kwargs,
        )
        for system_id, count in iterations.items()
    ]


def four_mode_runs() -> list[SCFRun]:
    return [
        *mode_runs(MODE_NATIVE, {"a": 24, "b": 30}),
        *mode_runs(MODE_RESTART, {"a": 18, "b": 22}),
        *mode_runs(MODE_ML_GUESS, {"a": 15, "b": 18}),
        *mode_runs(MODE_ML_FULL, {"a": 11, "b": 13}),
    ]


def test_benchmark_compares_every_mode_against_native() -> None:
    report = run_benchmark(four_mode_runs())

    assert set(report.comparisons) == {MODE_RESTART, MODE_ML_GUESS, MODE_ML_FULL}
    assert report.comparisons[MODE_ML_FULL].baseline_iterations == pytest.approx(27.0)
    assert report.comparisons[MODE_ML_FULL].ml_iterations == pytest.approx(12.0)
    assert [summary.mode for summary in report.modes] == list(BENCHMARK_MODES)


def test_benchmark_headline_has_the_contract_keys() -> None:
    payload = run_benchmark(four_mode_runs()).to_dict()

    for key in ("baseline_iterations", "ml_iterations", "speedup", "energy_error", "force_error", "converged"):
        assert key in payload
    assert payload["headline_mode"] == MODE_ML_FULL
    assert payload["speedup"] == pytest.approx(27.0 / 12.0)
    assert payload["energy_error"] == pytest.approx(0.0)
    assert payload["converged"] is True


def test_headline_falls_back_to_ml_guess_when_full_is_absent() -> None:
    runs = [*mode_runs(MODE_NATIVE, {"a": 24}), *mode_runs(MODE_ML_GUESS, {"a": 15})]

    payload = run_benchmark(runs).to_dict()

    assert payload["headline_mode"] == MODE_ML_GUESS
    assert any("ml_full" in warning for warning in payload["warnings"])


def test_an_energy_shift_is_flagged_and_fails_convergence() -> None:
    runs = [
        *mode_runs(MODE_NATIVE, {"a": 24}, energy=-1234.5),
        *mode_runs(MODE_ML_FULL, {"a": 12}, energy=-1234.4),
    ]

    payload = run_benchmark(runs).to_dict()

    assert payload["energy_error"] == pytest.approx(0.1)
    assert payload["converged"] is False
    assert any("correctness problem" in warning for warning in payload["warnings"])


def test_a_convergence_regression_is_flagged() -> None:
    runs = [
        *mode_runs(MODE_NATIVE, {"a": 24, "b": 24}),
        SCFRun(label=MODE_ML_FULL, system_id="a", scf_iterations=12, converged=True, final_energy=-1234.5),
        SCFRun(label=MODE_ML_FULL, system_id="b", scf_iterations=99, converged=False),
    ]

    report = run_benchmark(runs)

    assert any("success rate dropped" in warning for warning in report.warnings)
    assert report.comparisons[MODE_ML_FULL].candidate_success_rate == pytest.approx(0.5)


def test_fallback_rate_is_reported_per_mode() -> None:
    runs = [
        *mode_runs(MODE_NATIVE, {"a": 24, "b": 24}),
        *mode_runs(MODE_ML_FULL, {"a": 12}, fallback_used=True),
        *mode_runs(MODE_ML_FULL, {"b": 13}),
    ]

    report = run_benchmark(runs)
    ml_summary = next(summary for summary in report.modes if summary.mode == MODE_ML_FULL)

    assert ml_summary.fallback_rate == pytest.approx(0.5)
    assert report.comparisons[MODE_ML_FULL].fallback_rate == pytest.approx(0.5)


def test_benchmark_requires_a_baseline() -> None:
    with pytest.raises(BenchmarkError, match="no runs labelled 'native'"):
        run_benchmark(mode_runs(MODE_ML_FULL, {"a": 12}))


def test_summarize_mode_reports_failure_rate() -> None:
    runs = [
        SCFRun(label=MODE_ML_FULL, system_id="a", scf_iterations=12, converged=True),
        SCFRun(label=MODE_ML_FULL, system_id="b", scf_iterations=99, converged=False),
    ]

    summary = summarize_mode(MODE_ML_FULL, runs)

    assert summary.failure_rate == pytest.approx(0.5)
    assert summary.mean_iterations == pytest.approx(12.0)


def test_load_runs_accepts_both_json_shapes(tmp_path: Path) -> None:
    entries = [run.to_dict() for run in mode_runs(MODE_NATIVE, {"a": 24})]
    flat = tmp_path / "flat.json"
    wrapped = tmp_path / "wrapped.json"
    flat.write_text(json.dumps(entries), encoding="utf-8")
    wrapped.write_text(json.dumps({"runs": entries}), encoding="utf-8")

    assert load_runs(flat) == load_runs(wrapped)


def test_load_runs_reports_a_malformed_entry(tmp_path: Path) -> None:
    path = tmp_path / "runs.json"
    path.write_text(json.dumps([{"label": "native"}]), encoding="utf-8")

    with pytest.raises(BenchmarkError, match="malformed"):
        load_runs(path)


def test_load_runs_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "runs.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(BenchmarkError, match="invalid JSON"):
        load_runs(path)


def test_write_report_produces_readable_json(tmp_path: Path) -> None:
    report = run_benchmark(four_mode_runs())

    path = write_report(report, tmp_path / "benchmark.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["benchmark_version"].startswith("phal-scf-benchmark")
    assert payload["comparisons"][MODE_ML_FULL]["speedup"] > 1.0


def test_runs_can_be_built_from_recorded_scf_histories(tmp_path: Path) -> None:
    from tests.scf.test_scf_dataset import converged_trajectory

    scf_root = tmp_path / "scf"
    for task_id in ("task-0001", "task-0002"):
        directory = scf_root / task_id
        directory.mkdir(parents=True)
        (directory / "scf_history.json").write_text(
            json.dumps(converged_trajectory().to_dict()), encoding="utf-8"
        )

    runs = runs_from_scf_histories(scf_root, label=MODE_NATIVE)

    assert [run.system_id for run in runs] == ["task-0001", "task-0002"]
    assert all(run.scf_iterations == 3 for run in runs)
    assert all(run.converged for run in runs)
    assert runs[0].metadata["initial_guess"] == "ATOMIC"


def test_runs_from_scf_histories_requires_recordings(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkError, match="no scf_history.json"):
        runs_from_scf_histories(tmp_path, label=MODE_NATIVE)
