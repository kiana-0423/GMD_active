"""Regression tests for the PHAL defects fixed in the SCF port (findings F-22, F-23; ledger notes).

Each test asserts the corrected behaviour; PHAL's behaviour is described in the docstring.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gmd_active_learning.adapters.dft.scf.density.prediction import (
    DENSITY_FORMAT_DENSITY_MATRIX,
    SCFDensityPrediction,
    read_prediction,
    write_prediction,
)
from gmd_active_learning.adapters.dft.scf.density.predictor import (
    DummyDensityPredictor,
    expected_density_electrons,
)
from gmd_active_learning.adapters.dft.scf.schema import SCFDensityArtifacts, SCFSample
from gmd_active_learning.adapters.dft.scf.validation import ERROR, validate_density_artifacts
from tests.scf.test_scf_dataset import make_sample, water_state, water_structure


def test_density_artifacts_survive_a_save_and_reload() -> None:
    """PHAL persisted 9 of 14 fields; representation, count method, spin channels and trajectory were lost."""

    artifacts = SCFDensityArtifacts(
        density_matrix_path="d.npy",
        overlap_path="s.npy",
        n_ao=23,
        basis_set="DZVP-MOLOPT-GTH",
        density_electron_count=8.0,
        representation="ao-nonorthogonal",
        electron_count_method="trace_ps",
        spin_channels=1,
        trajectory_path="trajectory",
        trajectory_frames=14,
    )
    sample = make_sample(artifacts=artifacts)

    restored = SCFSample.from_dict(json.loads(json.dumps(sample.to_dict())))

    assert restored.artifacts == artifacts


def test_density_artifacts_without_the_new_fields_serialise_as_in_phal() -> None:
    payload = SCFDensityArtifacts(density_matrix_path="d.npy").to_dict()

    assert set(payload) == {
        "density_matrix_path", "wavefunction_path", "hamiltonian_path", "density_format", "density_digest",
        "overlap_path", "n_ao", "basis_set", "density_electron_count",
    }


def test_a_density_without_an_overlap_is_refused_not_approximated(tmp_path: Path) -> None:
    """PHAL fell back to Tr(P), contradicting density_representation.md section 4."""

    density = np.eye(6)[None] * (8.0 / 6.0)
    np.save(tmp_path / "density.npy", density)
    sample = make_sample(
        artifacts=SCFDensityArtifacts(
            density_matrix_path=str(tmp_path / "density.npy"),
            density_format="ao-density-matrix-spin-major-v1",
            density_electron_count=8.0,
            n_ao=6,
        )
    )

    issues = validate_density_artifacts(sample)

    assert any(issue.severity == ERROR and "Tr(P S)" in issue.message for issue in issues)


def test_the_valence_count_is_the_reference_for_a_density() -> None:
    """PHAL validated declared counts against the all-electron sum(Z) - charge (density_predictor.py:190)."""

    structure, state = water_structure(), water_state()
    assert state.n_electrons == 10
    assert expected_density_electrons(structure, state) == pytest.approx(8.0)

    predictor = DummyDensityPredictor().load_model()
    good = predictor.predict(structure, state)
    assert predictor.validate_prediction(good, state, structure=structure).valid

    nuclear = SCFDensityPrediction(
        model_name="z", model_version="0", model_digest="x", structure_hash=good.structure_hash,
        electron_count=10.0, spin_channels=1, density_format=good.density_format,
        values=np.asarray(good.values) * 10.0 / 8.0,
    )
    verdict = predictor.validate_prediction(nuclear, state, structure=structure)
    assert not verdict.valid
    assert any("electronic state has 8.000000" in reason for reason in verdict.reasons)


def test_a_configured_electron_tolerance_is_used() -> None:
    structure, state = water_structure(), water_state()
    predictor = DummyDensityPredictor().load_model()
    good = predictor.predict(structure, state)
    off = SCFDensityPrediction(
        model_name="z", model_version="0", model_digest="x", structure_hash=good.structure_hash,
        electron_count=8.04, spin_channels=1, density_format=good.density_format,
        values=np.asarray(good.values) * 8.04 / 8.0,
    )

    assert not predictor.validate_prediction(off, state, structure=structure).valid
    assert predictor.validate_prediction(off, state, structure=structure, electron_tolerance=1e-2).valid


def test_a_density_matrix_prediction_keeps_its_overlap_on_disk(tmp_path: Path) -> None:
    """PHAL's with_output_path dropped ``overlap`` and write/read_prediction never persisted it."""

    overlap = np.eye(4) + 0.05 * (np.ones((4, 4)) - np.eye(4))
    density = np.eye(4)[None] * 0.5
    prediction = SCFDensityPrediction(
        model_name="m", model_version="1", model_digest="sha256:x", structure_hash="h",
        electron_count=float(np.sum(density[0] * overlap)), spin_channels=1,
        density_format=DENSITY_FORMAT_DENSITY_MATRIX, values=density, overlap=overlap,
    )

    restored = read_prediction(write_prediction(prediction, tmp_path / "p.json"))

    assert restored.overlap is not None and np.array_equal(restored.overlap, overlap)
    assert prediction.with_output_path(tmp_path / "q.json").overlap is overlap


def test_the_pulay_mixer_export_needs_the_experimental_opt_in(tmp_path: Path) -> None:
    """PHAL's pulay_mixer.export_mixer bypassed the gate deploy.export_mixer enforced."""

    torch = pytest.importorskip("torch")
    from tools.scf.negative_results.pulay_mixer import PulayMixerError, build_mixer_module, export_mixer

    module = build_mixer_module(torch)
    with pytest.raises(PulayMixerError, match="experimental"):
        export_mixer(module, torch, tmp_path / "mixer.pt")
    assert not (tmp_path / "mixer.pt").exists()


def test_the_density_gnn_predictor_is_registered_lazily() -> None:
    """PHAL returned early from registration when 'torch' failed, so density_gnn was never registered."""

    from gmd_active_learning.adapters.dft.scf.density.registry import available_predictors

    assert {"dummy", "density_gnn", "torch"} <= set(available_predictors())
