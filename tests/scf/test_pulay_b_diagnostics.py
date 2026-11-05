# Ported from PHAL (tests.scf.test_pulay_b_diagnostics) at 96263b6; MIT License, see NOTICE.
"""Tests for the Pulay B-matrix conditioning diagnostics.

These numbers exist to answer one question before training: how much of the
dataset sits in the near-singular regime where the DIIS coefficients are
extremely sensitive to B. The tests below pin the cases where that answer would
otherwise be quietly wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from gmd_active_learning.adapters.dft.scf.pulay_matrix import (
    DEFAULT_CONDITION_LIMIT,
    PulayRecordError,
    b_matrix_diagnostics,
)


def test_identity_is_perfectly_conditioned():
    detail = b_matrix_diagnostics(np.eye(4))

    assert detail["finite"] is True
    assert detail["depth"] == 4
    assert detail["rank"] == 4
    assert detail["rank_deficient"] is False
    assert detail["condition_number"] == pytest.approx(1.0)
    assert detail["positive_definite"] is True
    assert detail["regularization_required"] is False
    assert detail["regularization_shift"] == 0.0


def test_a_near_singular_matrix_asks_for_regularization():
    matrix = np.diag([1.0, 1.0e-12])

    detail = b_matrix_diagnostics(matrix)

    assert detail["condition_number"] == pytest.approx(1.0e12, rel=1e-6)
    assert detail["regularization_required"] is True
    assert detail["regularization_shift"] > 0.0


def test_the_reported_shift_actually_reaches_the_limit():
    """The shift is only useful if applying it does what it claims."""

    matrix = np.diag([1.0, 1.0e-10, 0.5])
    detail = b_matrix_diagnostics(matrix, condition_limit=1.0e6)

    shifted = matrix + detail["regularization_shift"] * np.eye(3)
    eigenvalues = np.linalg.eigvalsh(shifted)

    assert eigenvalues.max() / eigenvalues.min() == pytest.approx(1.0e6, rel=1e-6)


def test_a_singular_matrix_reports_an_infinite_condition_number():
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]])

    detail = b_matrix_diagnostics(matrix)

    assert detail["rank"] == 1
    assert detail["rank_deficient"] is True
    assert np.isinf(detail["condition_number"])
    assert detail["regularization_required"] is True


def test_an_indefinite_matrix_is_reported_as_such():
    """B should be PSD; noise that makes it indefinite is worth seeing."""

    detail = b_matrix_diagnostics(np.diag([1.0, -1.0e-14]))

    assert detail["positive_definite"] is False
    assert detail["eigenvalue_min"] < 0.0


def test_asymmetry_is_measured_rather_than_discarded():
    """eigvalsh reads one triangle, so an asymmetric B must not pass silently."""

    matrix = np.array([[1.0, 0.3], [0.1, 1.0]])

    detail = b_matrix_diagnostics(matrix)

    assert detail["asymmetry"] == pytest.approx(0.2)


def test_non_finite_values_short_circuit():
    detail = b_matrix_diagnostics(np.array([[1.0, np.nan], [np.nan, 1.0]]))

    assert detail["finite"] is False
    assert "eigenvalues" not in detail


def test_a_non_square_matrix_is_refused():
    with pytest.raises(PulayRecordError, match="square"):
        b_matrix_diagnostics(np.zeros((2, 3)))


def test_coefficient_l1_is_one_when_no_cancellation_is_needed():
    """sum(c) = 1 with all-positive c means ||c||_1 == 1."""

    detail = b_matrix_diagnostics(np.eye(3))

    assert detail["coefficient_l1"] == pytest.approx(1.0)


def test_coefficient_l1_grows_when_the_solve_relies_on_cancellation():
    # Strongly correlated residuals: the DIIS solution takes a large positive
    # step against a large negative one, which is exactly the regime where a
    # model's error in c does not stay small.
    matrix = np.array([[1.0, 0.999], [0.999, 1.0]])

    detail = b_matrix_diagnostics(matrix)

    assert detail["coefficient_l1"] == pytest.approx(1.0)

    # c = (1.75, -0.75) here: an extrapolation, not an interpolation.
    skewed = np.array([[1.0, 1.3], [1.3, 2.0]])
    assert b_matrix_diagnostics(skewed)["coefficient_l1"] == pytest.approx(2.5)


def test_the_default_limit_is_the_documented_one():
    detail = b_matrix_diagnostics(np.eye(2))

    assert detail["condition_limit"] == DEFAULT_CONDITION_LIMIT
