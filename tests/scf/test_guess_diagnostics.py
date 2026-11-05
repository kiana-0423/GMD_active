# Ported from PHAL (tests.scf.test_guess_diagnostics) at 96263b6; MIT License, see NOTICE.
"""Tests for the ML initial-guess failure diagnosis.

The diagnosis exists to answer "which of four causes", and its main risk is
answering confidently from runs that carry no information: a fallback run, or
one that never converged. Those cases are checked first.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.scf.diagnostics.guess import (
    CAUSE_INITIALISATION,
    CAUSE_NONE,
    CAUSE_PREDICTION,
    NOT_APPLICABLE,
    DiagnosticsError,
    GuessDistance,
    SystemDiagnosis,
    classify,
    density_distance,
    diagnose_set,
    trajectory_metrics,
    write_report,
)


def distance(metric: float) -> GuessDistance:
    return GuessDistance(frobenius=metric, metric=metric, electron_error=0.0)


def descent(first: float, rate: float, n: int = 6, energy_error: float = 0.3):
    residuals = [first * rate**index for index in range(n)]
    energies = [-10.0 + energy_error * rate**index for index in range(n)]
    return trajectory_metrics(residuals, energies)


# -- distance -----------------------------------------------------------


def test_distance_is_zero_against_itself():
    density = np.array([[[2.0, 0.1], [0.1, 1.0]]])
    overlap = np.eye(2)

    result = density_distance(density, density, overlap)

    assert result.frobenius == pytest.approx(0.0)
    assert result.metric == pytest.approx(0.0)


def test_the_metric_norm_uses_the_overlap():
    """In a non-orthogonal basis, Frobenius and the S-metric differ."""

    guess = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    final = np.zeros((1, 2, 2))
    overlap = np.array([[1.0, 0.5], [0.5, 1.0]])

    result = density_distance(guess, final, overlap)

    assert result.frobenius == pytest.approx(np.sqrt(2.0))
    # Tr(dP S dP S) with dP = I is Tr(S S) = 2*(1 + 0.25)
    assert result.metric == pytest.approx(np.sqrt(2.5))


def test_the_electron_error_of_the_guess_is_reported():
    guess = np.array([[[2.0, 0.0], [0.0, 2.0]]])
    final = np.array([[[1.5, 0.0], [0.0, 1.5]]])

    result = density_distance(guess, final, np.eye(2), n_electrons=3.0)

    assert result.electron_error == pytest.approx(1.0)


def test_mismatched_shapes_are_refused():
    with pytest.raises(DiagnosticsError, match="shapes differ"):
        density_distance(np.zeros((1, 2, 2)), np.zeros((1, 3, 3)), np.eye(2))


def test_a_mismatched_overlap_is_refused():
    with pytest.raises(DiagnosticsError, match="does not match"):
        density_distance(np.zeros((1, 2, 2)), np.zeros((1, 2, 2)), np.eye(3))


# -- trajectory ---------------------------------------------------------


def test_the_reduction_rate_is_the_geometric_mean_ratio():
    metrics = trajectory_metrics([1.0, 0.5, 0.25, 0.125], [-1.0, -1.5, -1.9, -2.0])

    assert metrics.first_residual == 1.0
    assert metrics.reduction_rate == pytest.approx(0.5)
    assert metrics.n_iterations == 4


def test_a_single_excursion_does_not_dominate_the_rate():
    """Residuals fall multiplicatively; an arithmetic mean would mislead."""

    steady = trajectory_metrics([1.0, 0.1, 0.01, 0.001], [])
    bumpy = trajectory_metrics([1.0, 2.0, 0.2, 0.001], [])

    assert steady.reduction_rate == pytest.approx(0.1)
    assert bumpy.reduction_rate < 0.5


def test_the_initial_energy_error_is_against_the_converged_energy():
    metrics = trajectory_metrics([1.0, 0.1], [-9.0, -10.0])

    assert metrics.initial_energy_error == pytest.approx(1.0)


def test_missing_residuals_are_skipped():
    metrics = trajectory_metrics([1.0, None, 0.1, float("nan")], [])

    assert metrics.n_iterations == 2


# -- classification -----------------------------------------------------


def test_a_farther_guess_is_a_prediction_problem():
    cause, evidence = classify(
        atomic_distance=distance(1.0),
        ml_distance=distance(2.0),
        atomic_trajectory=descent(1.0, 0.4),
        ml_trajectory=descent(2.0, 0.5),
        scf_steps_atomic=14,
        scf_steps_ml=20,
    )

    assert cause == CAUSE_PREDICTION
    assert any("+100" in item for item in evidence)


def test_a_closer_guess_that_is_still_slower_is_an_initialisation_problem():
    """The finding that matters: closer in norm, worse to start from."""

    cause, evidence = classify(
        atomic_distance=distance(4.0),
        ml_distance=distance(1.2),
        atomic_trajectory=descent(1.7, 0.1, energy_error=1.1),
        ml_trajectory=descent(0.25, 0.2, energy_error=0.8),
        scf_steps_atomic=8,
        scf_steps_ml=10,
    )

    assert cause == CAUSE_INITIALISATION
    assert any("reduction rate" in item for item in evidence)


def test_a_faster_run_is_not_a_failure_at_all():
    cause, _evidence = classify(
        atomic_distance=distance(2.0),
        ml_distance=distance(1.0),
        atomic_trajectory=descent(1.0, 0.4),
        ml_trajectory=descent(0.5, 0.3),
        scf_steps_atomic=14,
        scf_steps_ml=9,
    )

    assert cause == CAUSE_NONE


def test_a_fallback_run_says_nothing_about_the_model():
    """Both arms are the same run; classifying it would invent evidence."""

    cause, evidence = classify(
        atomic_distance=distance(1.0),
        ml_distance=distance(1.0),
        atomic_trajectory=None,
        ml_trajectory=None,
        scf_steps_atomic=14,
        scf_steps_ml=14,
        ml_guess_used=False,
    )

    assert cause == NOT_APPLICABLE
    assert "fell back" in evidence[0]


def test_an_unconverged_run_has_no_reference_to_measure_against():
    cause, _evidence = classify(
        atomic_distance=distance(1.0),
        ml_distance=distance(2.0),
        atomic_trajectory=None,
        ml_trajectory=None,
        scf_steps_atomic=None,
        scf_steps_ml=None,
        both_converged=False,
    )

    assert cause == NOT_APPLICABLE


# -- set-level ----------------------------------------------------------


def system(name, atomic, ml, steps_atomic, steps_ml, *, fitted=None, difficulty="easy",
           cause=CAUSE_PREDICTION, energy_ml=3.0, energy_atomic=0.4):
    return SystemDiagnosis(
        system=name,
        difficulty=difficulty,
        in_training_set=fitted,
        atomic_distance=distance(atomic),
        ml_distance=distance(ml),
        scf_steps_atomic=steps_atomic,
        scf_steps_ml=steps_ml,
        atomic_trajectory=descent(1.0, 0.4, energy_error=energy_atomic),
        ml_trajectory=descent(1.0, 0.4, energy_error=energy_ml),
        classification=cause,
    )


def test_a_generalisation_gap_is_detected():
    """Good where fitted, bad where not: that is C and not simply a bad model."""

    diagnoses = [
        system("seen-a", 1.0, 0.95, 14, 15, fitted=True),
        system("seen-b", 1.0, 1.0, 14, 15, fitted=True),
        system("unseen-a", 1.0, 2.0, 14, 18, fitted=False),
        system("unseen-b", 1.0, 2.2, 14, 19, fitted=False),
    ]

    summary = diagnose_set(diagnoses)

    assert summary["verdict"] == "C_generalisation_failure"
    assert summary["generalisation"]["gap"] > 0.5


def test_a_uniformly_bad_model_is_not_called_a_generalisation_failure():
    """Equally bad on both sides is cause A, not C."""

    diagnoses = [
        system("seen", 1.0, 2.0, 14, 18, fitted=True),
        system("unseen", 1.0, 2.1, 14, 18, fitted=False),
    ]

    summary = diagnose_set(diagnoses)

    assert summary["verdict"] == CAUSE_PREDICTION
    assert summary["generalisation"]["indicated"] is False


def test_not_applicable_systems_are_excluded_from_the_verdict():
    diagnoses = [
        system("real", 1.0, 2.0, 14, 18, fitted=False),
        system("fellback", 1.0, 1.0, 14, 14, cause=NOT_APPLICABLE),
        system("alsofellback", 1.0, 1.0, 14, 14, cause=NOT_APPLICABLE),
    ]

    summary = diagnose_set(diagnoses)

    assert summary["verdict"] == CAUSE_PREDICTION
    assert summary["n_usable_systems"] == 1


def test_a_weak_distance_cost_correlation_is_reported_as_a_secondary_finding():
    """If distance does not predict cost, a better fit is not the fix."""

    # Excess distance ranks 1,2,3,4 against step-delta ranks 3,1,4,2: Spearman
    # is exactly zero, so distance carries no information about cost here.
    diagnoses = [
        system("a", 1.0, 1.5, 14, 19, fitted=False),
        system("b", 1.0, 2.0, 14, 15, fitted=False),
        system("c", 1.0, 2.5, 14, 20, fitted=False),
        system("d", 1.0, 3.0, 14, 16, fitted=False),
    ]

    summary = diagnose_set(diagnoses)
    findings = " ".join(item["finding"] for item in summary["secondary_findings"])

    assert summary["distance_vs_cost"]["spearman_excess_distance_vs_step_delta"] == pytest.approx(
        0.0, abs=1e-9
    )
    assert "Spearman" in findings


def test_closer_but_slower_systems_are_surfaced_even_when_a_is_dominant():
    diagnoses = [
        system("far", 1.0, 2.0, 14, 18, fitted=False),
        system("far2", 1.0, 2.2, 14, 19, fitted=False),
        system("close", 4.0, 1.2, 8, 10, fitted=True, cause=CAUSE_INITIALISATION),
    ]

    summary = diagnose_set(diagnoses)
    initialisation = [
        item for item in summary["secondary_findings"] if item["cause"] == CAUSE_INITIALISATION
    ]

    assert any("close" in item.get("systems", []) for item in initialisation)


def test_the_report_round_trips(tmp_path: Path):
    diagnoses = [system("h2o", 1.4, 1.0, 14, 15, fitted=True)]

    path = write_report(diagnoses, tmp_path / "ml_density_failure_report.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    entry = payload["systems"][0]
    assert entry["system"] == "h2o"
    assert entry["scf_steps_atomic"] == 14
    assert entry["scf_steps_ml"] == 15
    assert entry["atomic_distance"]["metric"] == pytest.approx(1.4)
    assert entry["classification"]


# -- against the real diagnosis -----------------------------------------


def test_the_recorded_diagnosis_is_internally_consistent():
    path = Path("runtime/cp2k_ml/validation/ml_density_failure_report.json")
    if not path.is_file():
        pytest.skip("the diagnosis has not been run")

    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]

    # Every system that was classified must carry the evidence for it.
    for entry in payload["systems"]:
        assert entry["evidence"], entry["system"]
        if entry["classification"] == NOT_APPLICABLE:
            continue
        assert entry["atomic_distance"] and entry["ml_distance"]
        assert entry["scf_steps_atomic"] and entry["scf_steps_ml"]

    assert summary["n_usable_systems"] >= 4
    assert summary["verdict"] != "undetermined"
