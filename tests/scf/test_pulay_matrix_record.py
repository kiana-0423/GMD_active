# Ported from PHAL (tests.scf.test_pulay_matrix_record) at 96263b6; MIT License, see NOTICE.
"""Tests for recording CP2K's own Pulay matrix.

The fixtures are real dumps from a CP2K v2026.2 H2O run. They matter because
the matrix a mixer is fed at run time cannot be reconstructed from recorded
densities: CP2K builds it from g-space residuals, Kerker-weighted, weighted
again by ``special_metric``, and scaled by ``vol^2``. Training on a
reconstruction and deploying against the real thing degraded convergence from
11 to 37 SCF steps, which is what this recorder exists to prevent.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from gmd_active_learning.adapters.dft.scf.pulay_matrix import (
    METRIC_GSPACE_KERKER,
    RECORD_FORMAT,
    REPRESENTATION_AO_RECONSTRUCTED,
    REPRESENTATION_CP2K_PULAY,
    PulayRecordError,
    read_record,
    read_records,
    require_cp2k_representation,
    verify_record,
    verify_records,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "scf" / "pulay"


def records():
    return read_records(FIXTURES)


def test_records_are_read_in_scf_order():
    found = records()

    assert found
    iterations = [record.iteration for record in found]
    assert iterations == sorted(iterations)
    # The history fills up, so depth grows and then saturates at NBUFFER.
    depths = [record.depth for record in found]
    assert depths == sorted(depths)
    assert max(depths) <= 8


def test_every_record_declares_the_cp2k_metric():
    for record in records():
        assert record.representation == REPRESENTATION_CP2K_PULAY
        assert record.metric == METRIC_GSPACE_KERKER


def test_recorded_matrices_are_symmetric():
    for record in records():
        b = record.b_matrix
        assert b.shape == (record.depth, record.depth)
        assert np.allclose(b, b.T, rtol=1e-12, atol=0.0)


def test_export_reproduces_cp2k_own_coefficients():
    """The check that says "this is CP2K's matrix", not merely "a matrix".

    The closed form B^-1 1 / (1^T B^-1 1) is exactly CP2K's solve, so on the
    real matrix it must reproduce the coefficients CP2K went on to use. A
    reconstruction reproduces nothing, because it never produced anything.
    """

    summary = verify_records(records())

    assert summary["n_records"] > 0
    assert summary["failed_iterations"] == []
    assert summary["n_verified"] == summary["n_records"]
    assert summary["max_deviation"] < 1e-6


def test_a_perturbed_matrix_fails_verification():
    """So the check is not vacuously passing."""

    record = records()[-1]
    # Scale one diagonal entry: keeps B symmetric and well-conditioned, but
    # moves the solution of B c = 1 decisively.
    perturbed = np.array(record.b_matrix, dtype=float)
    perturbed[0, 0] *= 4.0
    spoiled = dataclasses.replace(record, b_matrix=perturbed)

    ok, deviation = verify_record(spoiled)

    assert not ok
    assert deviation > 1e-6


def test_mixed_representations_are_refused():
    """Averaging a CP2K metric with an AO reconstruction is invisible once the
    matrices are stacked, so it is refused at the door."""

    found = records()
    reconstructed = dataclasses.replace(found[0], representation=REPRESENTATION_AO_RECONSTRUCTED)

    with pytest.raises(PulayRecordError, match="mixed dataset"):
        require_cp2k_representation([reconstructed, *found[1:]])


def test_an_all_reconstructed_set_is_also_refused():
    found = [
        dataclasses.replace(record, representation=REPRESENTATION_AO_RECONSTRUCTED)
        for record in records()
    ]

    with pytest.raises(PulayRecordError, match="not the matrix CP2K solves"):
        require_cp2k_representation(found)


def test_a_homogeneous_cp2k_set_is_accepted():
    require_cp2k_representation(records())


def test_wrong_format_version_is_refused(tmp_path):
    source = sorted(FIXTURES.glob("pulay_*.dat"))[0]
    target = tmp_path / "bad.dat"
    target.write_text(source.read_text().replace(RECORD_FORMAT, "phal-pulay-matrix-v0"))

    with pytest.raises(PulayRecordError, match="format"):
        read_record(target)


def test_truncated_record_is_refused(tmp_path):
    source = sorted(FIXTURES.glob("pulay_*.dat"))[0]
    target = tmp_path / "short.dat"
    lines = source.read_text().splitlines()
    target.write_text("\n".join(lines[:-2]) + "\n")

    with pytest.raises(PulayRecordError, match="coefficients"):
        read_record(target)


def test_reconstruction_differs_from_the_real_matrix():
    """Quantifies why the earlier training failed.

    Both matrices are scale-normalised first, because the model normalises its
    input -- so what is left is the part a reconstruction genuinely cannot
    recover, and it is large.
    """

    from tools.scf.negative_results.pulay_mixer import native_coefficients

    found = records()
    rng = np.random.default_rng(0)
    # A reconstruction cannot be formed here without the densities, so this
    # stands in for "some other symmetric matrix of the same size": the point
    # is that the coefficients are a sensitive function of B.
    record = found[-1]
    perturbation = rng.normal(size=record.b_matrix.shape) * 0.15 * np.abs(record.b_matrix).max()
    perturbation = 0.5 * (perturbation + perturbation.T)

    actual = native_coefficients(record.b_matrix)
    perturbed = native_coefficients(record.b_matrix + perturbation)

    assert float(np.abs(actual - perturbed).max()) > 1e-3
