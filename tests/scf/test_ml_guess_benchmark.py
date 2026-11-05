# Ported from PHAL (tests.scf.test_ml_guess_benchmark) at 96263b6; MIT License, see NOTICE.
"""Tests for the ML initial-guess benchmark.

What these protect is the honesty of the comparison: errors are measured
against the same system's reference run, a failed run is data rather than an
exception, and a missing arm cannot masquerade as a result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.scf.benchmark.guess import (
    BENCHMARK_FORMAT,
    GUESS_ATOMIC,
    GUESS_ML,
    GUESS_SOURCE_ARTIFACT,
    GUESS_SOURCE_FALLBACK,
    GUESS_SOURCE_TORCHSCRIPT,
    GuessRun,
    compare,
    parse_forces,
    parse_run,
    write_report,
)

TORCHSCRIPT_LOG = """
 ML-SCF| ml_guess_source: torchscript
 ML-SCF| model_digest: in-process
 ML-SCF| prediction_success: T
 ML-SCF| accelerator enabled: ML initial guess accepted
  *** SCF run converged in    15 steps ***
"""

ARTIFACT_LOG = """
 ML-SCF| ml_guess_source: artifact
 ML-SCF| model_digest: sha256:abc123
 ML-SCF| prediction_success: T
 ML-SCF| accelerator enabled: ML initial guess accepted
 ML-SCF| density file: /runtime/h2o.ml_density
  *** SCF run converged in     6 steps ***
"""

FALLBACK_LOG = """
 ML-SCF| ml_guess_source: fallback
 ML-SCF| model_digest: unknown
 ML-SCF| prediction_success: F
 ML-SCF| fallback used: no ML density file was found
 ML-SCF| continuing with SCF_GUESS ATOMIC
  *** SCF run converged in    14 steps ***
"""

OUTPUT = """
 SCF PARAMETERS         Density guess:                                    ATOMIC
  *** SCF run converged in    14 steps ***

 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]         -17.146073021320363

 ATOMIC FORCES in [a.u.]

 # Atom   Kind   Element          X              Y              Z
      1      1      O          -0.00000000     0.00000000    -0.00397535
      2      2      H           0.00000000    -0.00295024     0.00198767
      3      2      H           0.00000000     0.00295024     0.00198767
 SUM OF ATOMIC FORCES          -0.00000000    -0.00000000     0.00000000     0.00000000
"""

ABORTED = """
 * [ABORT]                                                                     *
 *  \\___/    Bad condition number R_COND = 7.103E-17 (smaller than the machine *
"""


def run(system: str, guess: str, **kwargs) -> GuessRun:
    defaults = dict(
        converged=True,
        scf_steps=10,
        runtime_s=1.0,
        energy=-17.0,
        forces=np.zeros((3, 3)),
    )
    if guess == GUESS_ML:
        defaults.update(ml_guess_accepted=True, guess_source=GUESS_SOURCE_TORCHSCRIPT)
    defaults.update(kwargs)
    return GuessRun(system=system, initial_guess=guess, **defaults)


# -- parsing ------------------------------------------------------------


def test_a_converged_run_is_read_completely():
    result = parse_run(OUTPUT, system="h2o", initial_guess=GUESS_ATOMIC, runtime_s=12.5)

    assert result.converged is True
    assert result.scf_steps == 14
    assert result.energy == pytest.approx(-17.146073021320363)
    assert result.forces is not None and result.forces.shape == (3, 3)
    assert result.runtime_s == 12.5


def test_the_force_sum_line_is_not_an_atom():
    forces = parse_forces(OUTPUT)

    assert forces is not None
    assert forces.shape == (3, 3)
    assert forces[0][2] == pytest.approx(-0.00397535)


def test_an_output_without_forces_gives_none():
    assert parse_forces("no forces here") is None


def test_a_failed_run_is_a_record_not_an_exception():
    """Convergence rate is a metric; it needs failures to be data."""

    result = parse_run(ABORTED, system="mgo", initial_guess=GUESS_ATOMIC, runtime_s=3.0)

    assert result.converged is False
    assert result.scf_steps is None
    assert "Bad condition number" in (result.failure or "")


# -- comparison ---------------------------------------------------------


def test_errors_are_measured_against_the_same_systems_reference():
    forces = np.zeros((3, 3))
    shifted = forces.copy()
    shifted[0, 0] = 1e-5

    report = compare(
        [
            run("h2o", GUESS_ATOMIC, energy=-17.0, forces=forces, scf_steps=14),
            run("h2o", GUESS_ML, energy=-17.0 + 1e-7, forces=shifted, scf_steps=8),
        ]
    )

    ml = next(item for item in report["runs"] if item["initial_guess"] == GUESS_ML)
    assert ml["energy_error"] == pytest.approx(1e-7)
    assert ml["force_error"] == pytest.approx(1e-5)
    assert ml["steps_vs_reference"] == -6
    assert ml["speedup"] == pytest.approx(14 / 8)


def test_the_reference_compares_against_itself_as_zero():
    report = compare([run("si", GUESS_ATOMIC)])

    entry = report["runs"][0]
    assert entry["energy_error"] == 0.0
    assert entry["force_error"] == 0.0
    assert entry["speedup"] == pytest.approx(1.0)


def test_a_system_the_reference_failed_on_has_no_error_to_report():
    """There is no baseline to be accurate against; saying 0 would be a lie."""

    report = compare(
        [
            run("mgo", GUESS_ATOMIC, converged=False, scf_steps=None, energy=None, forces=None),
            run("mgo", GUESS_ML, scf_steps=20),
        ]
    )

    ml = next(item for item in report["runs"] if item["initial_guess"] == GUESS_ML)
    assert ml["energy_error"] is None
    assert ml["force_error"] is None
    assert ml["speedup"] is None


def test_the_summary_reports_convergence_rate_per_guess():
    report = compare(
        [
            run("a", GUESS_ATOMIC, scf_steps=10),
            run("b", GUESS_ATOMIC, converged=False, scf_steps=None),
            run("a", GUESS_ML, scf_steps=6),
            run("b", GUESS_ML, scf_steps=30),
        ]
    )

    summary = report["summary"]
    assert summary[GUESS_ATOMIC]["convergence_rate"] == pytest.approx(0.5)
    assert summary[GUESS_ML]["convergence_rate"] == pytest.approx(1.0)
    assert summary[GUESS_ML]["total_scf_steps"] == 36
    assert summary[GUESS_ML]["total_steps_vs_reference"] == 26


def test_force_error_is_skipped_when_the_shapes_disagree():
    """Different atom counts mean it is not the same system; do not average it."""

    report = compare(
        [
            run("x", GUESS_ATOMIC, forces=np.zeros((3, 3))),
            run("x", GUESS_ML, forces=np.zeros((4, 3))),
        ]
    )

    ml = next(item for item in report["runs"] if item["initial_guess"] == GUESS_ML)
    assert ml["force_error"] is None


def test_the_report_round_trips(tmp_path: Path):
    report = compare([run("h2o", GUESS_ATOMIC)])

    path = write_report(report, tmp_path / "benchmark_ml_guess.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["benchmark_format"] == BENCHMARK_FORMAT
    assert payload["runs"][0]["system"] == "h2o"
    assert "changed the answer" in payload["note"]


# -- provenance ---------------------------------------------------------


def test_the_torchscript_path_is_recognised():
    result = parse_run(TORCHSCRIPT_LOG, system="h2o", initial_guess=GUESS_ML, runtime_s=1.0)

    assert result.guess_source == GUESS_SOURCE_TORCHSCRIPT
    assert result.ml_guess_accepted is True
    assert result.guess_model_digest == "in-process"
    assert result.guess_fallback_reason is None


def test_the_artifact_path_is_recognised():
    result = parse_run(ARTIFACT_LOG, system="h2o", initial_guess=GUESS_ML, runtime_s=1.0)

    assert result.guess_source == GUESS_SOURCE_ARTIFACT
    assert result.ml_guess_accepted is True
    assert result.guess_model_digest == "sha256:abc123"


def test_a_fallback_is_recognised_with_its_reason():
    result = parse_run(FALLBACK_LOG, system="h2o", initial_guess=GUESS_ML, runtime_s=1.0)

    assert result.guess_source == GUESS_SOURCE_FALLBACK
    assert result.ml_guess_accepted is False
    assert result.guess_fallback_reason == "no ML density file was found"


def test_acceptance_comes_from_cp2k_not_from_the_run_looking_normal():
    """A fallback run converges perfectly well; that is the whole problem."""

    result = parse_run(FALLBACK_LOG, system="h2o", initial_guess=GUESS_ML, runtime_s=1.0)

    assert result.converged is True
    assert result.ml_guess_accepted is False


def test_a_run_with_no_ml_lines_has_no_source():
    """Absent is not the same as fallback: this run never asked for ML."""

    result = parse_run(OUTPUT, system="h2o", initial_guess=GUESS_ATOMIC, runtime_s=1.0)

    assert result.guess_source is None


# -- the benchmark refuses to report a fallback as a result -------------


def test_a_benchmark_whose_ml_arm_ran_is_valid():
    report = compare(
        [run("h2o", GUESS_ATOMIC, scf_steps=14), run("h2o", GUESS_ML, scf_steps=15)]
    )

    assert report["valid"] is True
    assert report["n_fallback_runs"] == 0
    assert report["guess_sources"] == {GUESS_SOURCE_TORCHSCRIPT: 1}
    assert report["warnings"] == []


def test_a_benchmark_whose_ml_arm_fell_back_is_invalid():
    """Otherwise a fallback becomes a reported speedup of exactly 1.0."""

    report = compare(
        [
            run("h2o", GUESS_ATOMIC, scf_steps=14),
            run(
                "h2o",
                GUESS_ML,
                scf_steps=14,
                ml_guess_accepted=False,
                guess_source=GUESS_SOURCE_FALLBACK,
                guess_fallback_reason="no ML density file was found",
            ),
        ]
    )

    assert report["valid"] is False
    assert report["n_fallback_runs"] == 1
    assert "no ML density file was found" in report["warnings"][0]


def test_one_fallback_among_many_still_invalidates():
    """A partial fallback is a partial measurement, and averages over both."""

    report = compare(
        [
            run("a", GUESS_ATOMIC), run("a", GUESS_ML),
            run("b", GUESS_ATOMIC),
            run("b", GUESS_ML, ml_guess_accepted=False, guess_source=GUESS_SOURCE_FALLBACK),
        ]
    )

    assert report["valid"] is False
    assert report["n_ml_runs"] == 2
    assert report["n_fallback_runs"] == 1


def test_a_baseline_only_benchmark_is_not_marked_valid():
    """No ML arm means nothing was measured about a model, either way."""

    report = compare([run("h2o", GUESS_ATOMIC)])

    assert report["valid"] is False
    assert report["n_ml_runs"] == 0
