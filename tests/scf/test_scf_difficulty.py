# Ported from PHAL (tests.scf.test_scf_difficulty) at 96263b6; MIT License, see NOTICE.
"""Tests for difficulty and numerical-risk classification of SCF samples.

The classification exists so a dataset can say which systems are hard, and the
tests below pin the two distinctions that make it useful: a failed run is hard
(not absent), and a fast run on a near-singular ``B`` is easy-but-risky (not
hard).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.scf.difficulty import (
    DIFFICULTY_EASY,
    DIFFICULTY_HARD,
    DIFFICULTY_MEDIUM,
    RISK_LOW,
    RISK_SEVERE,
    assess_dataset,
    assess_sample,
    difficulty_summary,
    parse_abort,
    parse_scf_settings,
    residual_signals,
)

CONVERGED = """
 SCF PARAMETERS         Density guess:                                    ATOMIC
                        max_scf:                                              60
                        eps_scf:                                        1.00E-06
                        Mixing method:                              PULAY_MIXING
  *** SCF run converged in    12 steps ***
"""

ABORTED = """
 SCF PARAMETERS         Density guess:                                    ATOMIC
                        max_scf:                                              60
                        eps_scf:                                        1.00E-06
                        Mixing method:                              PULAY_MIXING
 *******************************************************************************
 *   ___                                                                       *
 *  /   \\                                                                      *
 * [ABORT]                                                                     *
 *  \\___/    Bad condition number R_COND = 7.103E-17 (smaller than the machine *
 *    |                             working precision)                         *
 *  O/|                                                                        *
 * /| |                                                                        *
 * / \\                                                    common/mathlib.F:739 *
 *******************************************************************************
"""


def write_sample(
    root: Path,
    name: str,
    output: str,
    *,
    residuals: list[float] | None = None,
    conditions: list[float] | None = None,
) -> Path:
    directory = root / name
    (directory / "trajectory").mkdir(parents=True)
    (directory / "cp2k.out").write_text(output, encoding="utf-8")

    if residuals is None:
        return directory

    frames = []
    for index, residual in enumerate(residuals):
        label = f"iter_{index:04d}"
        (directory / "trajectory" / label).mkdir()
        detail = None
        if conditions is not None and index < len(conditions):
            detail = {
                "finite": True,
                "depth": 4,
                "rank": 4,
                "condition_number": conditions[index],
                "coefficient_l1": 1.5,
                "regularization_required": conditions[index] > 1e8,
            }
        (directory / "trajectory" / label / "metadata.json").write_text(
            json.dumps({"index": index, "residual": residual, "pulay_diagnostics": detail}),
            encoding="utf-8",
        )
        frames.append({"index": index, "residual": residual, "is_final": False})

    (directory / "trajectory" / "metadata.json").write_text(
        json.dumps({"frames": frames}), encoding="utf-8"
    )
    return directory


# -- parsing ------------------------------------------------------------


def test_settings_come_from_what_cp2k_reported():
    settings = parse_scf_settings(CONVERGED)

    assert settings.initial_guess == "ATOMIC"
    assert settings.mixing_method == "PULAY_MIXING"
    assert settings.eps_scf == pytest.approx(1e-6)
    assert settings.max_scf == 60


def test_the_abort_message_survives_cp2ks_ascii_art():
    """The reason is the difference between a hard system and a bad input."""

    message = parse_abort(ABORTED)

    assert message is not None
    assert "Bad condition number" in message
    assert "R_COND = 7.103E-17" in message


def test_no_abort_means_none():
    assert parse_abort(CONVERGED) is None


def test_residual_signals_count_where_the_scf_went_backwards():
    detail = residual_signals([1.0, 0.5, 0.8, 0.1, None])

    assert detail["n_increases"] == 1
    assert detail["monotonic"] is False
    assert detail["first_increase_at"] == 1
    assert detail["max_increase_ratio"] == pytest.approx(1.6)


def test_a_descending_residual_is_monotonic():
    assert residual_signals([1.0, 0.1, 0.01])["monotonic"] is True


# -- classification -----------------------------------------------------


def test_a_quick_smooth_run_is_easy(tmp_path: Path):
    directory = write_sample(tmp_path, "easy", CONVERGED, residuals=[1.0, 0.1, 0.01])

    result = assess_sample(directory)

    assert result.difficulty == DIFFICULTY_EASY
    assert result.converged is True
    assert result.scf_steps == 12
    assert result.numerical_risk == RISK_LOW


def test_a_failed_run_is_hard_and_keeps_its_reason(tmp_path: Path):
    """The failures are the class the whole exercise is aimed at."""

    directory = write_sample(tmp_path, "aborted", ABORTED)

    result = assess_sample(directory)

    assert result.difficulty == DIFFICULTY_HARD
    assert result.converged is False
    assert result.scf_steps is None
    assert "Bad condition number" in (result.failure or "")


def test_an_abort_inside_the_diis_solve_is_severe_risk(tmp_path: Path):
    """It recorded no diagnostics because it never got to; that is not low risk."""

    directory = write_sample(tmp_path, "aborted", ABORTED)

    assert assess_sample(directory).numerical_risk == RISK_SEVERE


def test_an_oscillating_residual_escalates(tmp_path: Path):
    # Six excursions: an SCF fighting its mixer rather than descending.
    residuals = [1.0, 2.0, 0.5, 1.5, 0.4, 1.2, 0.3, 0.9, 0.2, 0.7, 0.1, 0.6, 0.05]
    directory = write_sample(tmp_path, "osc", CONVERGED, residuals=residuals)

    result = assess_sample(directory)

    assert result.difficulty == DIFFICULTY_HARD
    assert any("rose" in reason for reason in result.reasons)


def test_a_slow_run_is_medium(tmp_path: Path):
    output = CONVERGED.replace("converged in    12 steps", "converged in    22 steps")
    directory = write_sample(tmp_path, "slow", output, residuals=[1.0, 0.5, 0.1])

    assert assess_sample(directory).difficulty == DIFFICULTY_MEDIUM


def test_conditioning_does_not_make_a_fast_run_hard(tmp_path: Path):
    """Si converges in eight monotonic steps on a B with cond ~ 1e14.

    Folding that into the difficulty label would call an easy SCF hard and lose
    the more interesting fact, which is that it was one bad matrix away from
    the MgO abort.
    """

    directory = write_sample(
        tmp_path, "risky", CONVERGED, residuals=[1.0, 0.1, 0.001], conditions=[1e5, 1e14, 1e14]
    )

    result = assess_sample(directory)

    assert result.difficulty == DIFFICULTY_EASY
    assert result.numerical_risk == RISK_SEVERE
    assert any("cond(B)" in reason for reason in result.risk_reasons)


# -- aggregation --------------------------------------------------------


def test_the_summary_counts_failures_rather_than_dropping_them(tmp_path: Path):
    write_sample(tmp_path, "good", CONVERGED, residuals=[1.0, 0.1, 0.01])
    write_sample(tmp_path, "bad", ABORTED)

    summary = difficulty_summary(assess_dataset(tmp_path))

    assert summary["n_samples"] == 2
    assert summary["counts"] == {"easy": 1, "medium": 0, "hard": 1}
    assert summary["n_converged"] == 1
    assert summary["initial_guess"] == {"ATOMIC": 2}
    assert summary["mixing_method"] == {"PULAY_MIXING": 2}
    assert list(summary["failures"])[0].startswith("Bad condition number")


def test_thresholds_are_absolute_so_labels_are_stable(tmp_path: Path):
    """Adding a sample must not relabel the ones already there."""

    write_sample(tmp_path, "a", CONVERGED, residuals=[1.0, 0.1])
    before = {item.sample_id: item.difficulty for item in assess_dataset(tmp_path)}

    slow = CONVERGED.replace("converged in    12 steps", "converged in    48 steps")
    write_sample(tmp_path, "b", slow, residuals=[1.0, 0.1])
    after = {item.sample_id: item.difficulty for item in assess_dataset(tmp_path)}

    assert after["a"] == before["a"]
    assert after["b"] == DIFFICULTY_HARD


def test_the_real_dataset_spans_all_three_classes():
    """A dataset with only one difficulty class cannot teach robustness."""

    root = Path("runtime/cp2k/pulay_dataset")
    if not root.is_dir():
        pytest.skip("the generated dataset is not present")

    summary = difficulty_summary(assess_dataset(root))

    assert all(summary["counts"][level] > 0 for level in ("easy", "medium", "hard"))
