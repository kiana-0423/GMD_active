# Ported from PHAL (tests.scf.test_pulay_dataset) at 96263b6; MIT License, see NOTICE.
"""Tests for Pulay dataset generation, validation and statistics.

The fixture is one real accepted sample from a CP2K v2026.2 H2 run, kept whole
so the validator is exercised against the layout it will actually meet.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tools.scf.pulay_dataset import (
    DATASET_FORMAT,
    PulayDatasetError,
    PulayTask,
    dataset_report,
    load_task_list,
    validate_dataset,
    validate_sample,
    write_dataset_report,
    write_manifest,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "scf" / "pulay_sample"

#: A second real sample, from a periodic Si run generated through the structure
#: collection path. It carries the structure descriptors and the per-frame B
#: diagnostics, neither of which the older H2 fixture has.
PERIODIC_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "scf" / "pulay_sample_periodic"


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    """A real accepted sample, copied so a test may corrupt it."""

    target = tmp_path / "dataset" / "h2-r0.74"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(FIXTURE, target)
    return target


def metadata_of(sample: Path) -> dict:
    return json.loads((sample / "trajectory" / "metadata.json").read_text(encoding="utf-8"))


def rewrite_metadata(sample: Path, **changes) -> None:
    payload = metadata_of(sample)
    payload.update(changes)
    (sample / "trajectory" / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")


# -- acceptance ---------------------------------------------------------


def test_a_real_sample_is_accepted(sample: Path):
    result = validate_sample(sample / "trajectory", converged=True, scf_steps=10)

    assert result.accepted, result.reasons
    assert result.n_frames > 3
    assert result.n_pulay > 0
    assert result.n_ao > 0
    assert result.max_pulay_deviation < 1e-6


def test_the_fixture_carries_cp2k_matrices_not_a_reconstruction(sample: Path):
    payload = metadata_of(sample)

    assert payload["pulay_representation"] == "cp2k_pulay_metric"
    assert payload["pulay_metric"] == "gspace_kerker"
    assert payload["pulay_verification"]["failed_iterations"] == []


# -- every rejection reason ---------------------------------------------


def test_non_converged_trajectory_is_rejected(sample: Path):
    """An unconverged run has no reliable final density, so no usable label."""

    result = validate_sample(sample / "trajectory", converged=False)

    assert not result.accepted
    assert any("did not converge" in reason for reason in result.reasons)


def test_missing_pulay_matrix_is_rejected(sample: Path):
    rewrite_metadata(sample, n_pulay_matrices=0)

    result = validate_sample(sample / "trajectory", converged=True)

    assert not result.accepted
    assert any("no CP2K Pulay matrices" in reason for reason in result.reasons)


def test_wrong_representation_is_rejected(sample: Path):
    rewrite_metadata(sample, pulay_representation="ao_frobenius_reconstructed")

    result = validate_sample(sample / "trajectory", converged=True)

    assert not result.accepted
    assert any("wrong representation" in reason for reason in result.reasons)


def test_wrong_metric_is_rejected(sample: Path):
    rewrite_metadata(sample, pulay_metric="ao_frobenius")

    result = validate_sample(sample / "trajectory", converged=True)

    assert not result.accepted
    assert any("wrong metric" in reason for reason in result.reasons)


def test_missing_overlap_is_rejected(sample: Path):
    (sample / "trajectory" / "overlap.npy").unlink()

    result = validate_sample(sample / "trajectory", converged=True)

    assert not result.accepted
    assert any("overlap" in reason for reason in result.reasons)


def test_non_finite_density_is_rejected(sample: Path):
    path = sample / "trajectory" / "iter_0000" / "density.npy"
    values = np.load(path)
    values[0, 0, 0] = np.nan
    np.save(path, values)

    result = validate_sample(sample / "trajectory", converged=True)

    assert not result.accepted
    assert any("non-finite" in reason for reason in result.reasons)


def test_non_hermitian_density_is_rejected(sample: Path):
    path = sample / "trajectory" / "iter_0001" / "density.npy"
    values = np.load(path)
    values[0, 0, 1] += 0.5
    np.save(path, values)

    result = validate_sample(sample / "trajectory", converged=True)

    assert not result.accepted
    assert any("Hermitian" in reason for reason in result.reasons)


def test_an_unfaithful_pulay_export_is_rejected(sample: Path):
    """The matrix must reproduce the coefficients CP2K derived from it."""

    rewrite_metadata(
        sample, pulay_verification={"max_deviation": 1.0, "failed_iterations": [5]}
    )

    result = validate_sample(sample / "trajectory", converged=True)

    assert not result.accepted
    assert any("do not reproduce" in reason for reason in result.reasons)


# -- statistics ---------------------------------------------------------


def test_report_summarises_an_accepted_dataset(sample: Path):
    root = sample.parent
    results = validate_dataset(root)
    report = dataset_report(root, results)

    assert report["dataset_format"] == DATASET_FORMAT
    assert report["representation"] == "cp2k_pulay_metric"
    assert report["systems"]["n_accepted"] == 1
    assert report["systems"]["acceptance_rate"] == 1.0
    assert report["iterations"]["n_pulay_matrices"] > 0
    # History depth grows as the buffer fills, so a real run spans a range.
    depth = report["matrix_dimensions"]["history_depth"]
    assert depth["min"] >= 2 and depth["max"] <= 8
    # B spans decades; the report says so rather than leaving it to be found.
    spread = report["element_distribution"]["log10_abs"]
    assert spread["n"] > 0
    assert spread["max"] - spread["min"] > 1.0
    assert report["verification"]["max_pulay_deviation"] < 1e-6


def test_report_counts_rejections_by_reason(sample: Path):
    rewrite_metadata(sample, n_pulay_matrices=0)
    root = sample.parent

    report = dataset_report(root, validate_dataset(root))

    assert report["systems"]["n_accepted"] == 0
    assert report["systems"]["acceptance_rate"] == 0.0
    assert report["rejection_reasons"]


def test_report_describes_the_chemistry_it_was_told_about(sample: Path):
    payload = json.loads((sample / "sample.json").read_text(encoding="utf-8"))
    payload["structure"] = {
        "elements": ["H"],
        "formula": "H2",
        "n_atoms": 2,
        "periodic": False,
        "group": "small_molecules",
        "min_distance_A": 0.74,
    }
    (sample / "sample.json").write_text(json.dumps(payload), encoding="utf-8")
    root = sample.parent

    chemistry = dataset_report(root, validate_dataset(root))["chemistry"]

    assert chemistry["elements"] == {"H": 1}
    assert chemistry["n_distinct_elements"] == 1
    assert chemistry["families"] == {"small_molecules": 1}
    assert chemistry["n_periodic"] == 0
    assert chemistry["geometry"]["n_atoms"]["max"] == 2


def test_a_sample_without_a_structure_is_counted_as_undescribed(sample: Path):
    """A task list need not describe its structures; the report must not pretend."""

    payload = json.loads((sample / "sample.json").read_text(encoding="utf-8"))
    payload["structure"] = {}
    (sample / "sample.json").write_text(json.dumps(payload), encoding="utf-8")
    root = sample.parent

    chemistry = dataset_report(root, validate_dataset(root))["chemistry"]

    assert chemistry["n_described"] == 0
    assert chemistry["n_undescribed"] == 1


def test_report_measures_how_hard_the_solves_are(sample: Path):
    root = sample.parent

    conditioning = dataset_report(root, validate_dataset(root))["conditioning"]

    assert conditioning["n_matrices"] > 0
    assert conditioning["condition_number_log10"]["n"] > 0
    # A real SCF ends with a near-singular B, so some frame must ask for a
    # shift. A dataset where none did would mean the diagnostics are inert.
    assert 0.0 < conditioning["regularization"]["fraction_required"] <= 1.0
    assert conditioning["coefficient_l1"]["max"] >= 1.0


def test_conditioning_is_read_from_the_recorded_metadata(sample: Path):
    """Recorded at generation time, not recomputed -- and provably so."""

    frame = sample / "trajectory" / "iter_0004"
    metadata = json.loads((frame / "metadata.json").read_text(encoding="utf-8"))
    metadata["pulay_diagnostics"] = {
        "finite": True,
        "depth": 3,
        "rank": 1,
        "condition_number": 1.0e30,
        "regularization_required": True,
        "regularization_relative": 0.5,
        "coefficient_l1": 42.0,
        "eigenvalue_min": 1.0,
        "eigenvalue_max": 1.0,
    }
    (frame / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    root = sample.parent

    conditioning = dataset_report(root, validate_dataset(root))["conditioning"]

    assert conditioning["coefficient_l1"]["max"] == 42.0


def test_a_directory_that_was_never_a_run_is_not_a_rejected_sample(sample: Path):
    """Rendered inputs may sit beside samples; they are not failed samples."""

    (sample.parent / "inputs").mkdir()
    (sample.parent / "inputs" / "a.inp").write_text("&GLOBAL\n&END GLOBAL\n", encoding="utf-8")

    results = validate_dataset(sample.parent)

    assert [result.sample_id for result in results] == [sample.name]


def test_a_real_periodic_sample_carries_its_chemistry_and_conditioning(tmp_path: Path):
    """End to end on a real Si run generated from a structure collection."""

    root = tmp_path / "dataset"
    root.mkdir()
    shutil.copytree(PERIODIC_FIXTURE, root / "si-diamond")

    results = validate_dataset(root)
    report = dataset_report(root, results)

    assert [result.accepted for result in results] == [True]
    chemistry = report["chemistry"]
    assert chemistry["elements"] == {"Si": 1}
    assert chemistry["n_periodic"] == 1
    assert chemistry["families"] == {"periodic_main_group": 1}
    assert chemistry["geometry"]["cell_volume_A3"]["n"] == 1

    # Recorded at generation time by scf_trajectory, not recomputed here.
    frame = json.loads(
        (root / "si-diamond" / "trajectory" / "iter_0005" / "metadata.json").read_text()
    )
    assert frame["pulay_diagnostics"]["rank"] == frame["pulay_depth"]
    assert report["conditioning"]["n_matrices"] == 6
    # Even a two-atom cell drives B to the edge: this is the modelling
    # constraint the diagnostics exist to surface.
    assert report["conditioning"]["condition_number_log10"]["max"] > 8.0


def test_report_and_manifest_are_written(sample: Path, tmp_path: Path):
    root = sample.parent
    results = validate_dataset(root)

    manifest = write_manifest(root, results)
    report = write_dataset_report(root, results)

    assert json.loads(manifest.read_text())["dataset_format"] == DATASET_FORMAT
    assert json.loads(report.read_text())["systems"]["n_total"] == 1


# -- task list ----------------------------------------------------------


def test_task_list_round_trips(tmp_path: Path):
    (tmp_path / "a.inp").write_text("&GLOBAL\n&END GLOBAL\n")
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps([{"task_id": "a", "input": "a.inp", "structure": {"system": "H2"}}]))

    tasks = load_task_list(path)

    assert len(tasks) == 1
    assert isinstance(tasks[0], PulayTask)
    assert tasks[0].input_path == (tmp_path / "a.inp").resolve()
    assert tasks[0].structure["system"] == "H2"


def test_duplicate_task_ids_are_refused(tmp_path: Path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps([{"task_id": "a", "input": "x.inp"}, {"task_id": "a", "input": "y.inp"}]))

    with pytest.raises(PulayDatasetError, match="duplicate"):
        load_task_list(path)


def test_task_without_an_input_is_refused(tmp_path: Path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps([{"task_id": "a"}]))

    with pytest.raises(PulayDatasetError, match="no 'input'"):
        load_task_list(path)
