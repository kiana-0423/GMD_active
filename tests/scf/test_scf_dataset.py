# Ported from PHAL (tests.scf.test_scf_dataset) at 96263b6; MIT License, see NOTICE.
"""Offline tests for the Stage 1 SCF dataset infrastructure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gmd_active_learning.adapters.dft.scf.dataset import (
    SPLITS,
    DatasetError,
    SCFDataset,
    assign_split,
    ingest_scf_root,
    sanitize_sample_id,
)
from gmd_active_learning.adapters.dft.scf.export import export_dataset, load_features_npz, read_jsonl
from gmd_active_learning.adapters.dft.scf.features import feature_metadata, featurize_sample
from gmd_active_learning.adapters.dft.scf.schema import (
    SCFDensityArtifacts,
    SCFElectronicState,
    SCFIterationRecord,
    SCFSample,
    SCFStructure,
    SCFTrajectory,
    SchemaError,
)
from gmd_active_learning.adapters.dft.scf.validation import validate_payload, validate_sample


def water_structure() -> SCFStructure:
    return SCFStructure(
        atomic_numbers=(8, 1, 1),
        positions=((0.0, 0.0, 0.0), (0.9572, 0.0, 0.0), (-0.2400, 0.9266, 0.0)),
        cell=((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
        pbc=(True, True, True),
    )


def water_state() -> SCFElectronicState:
    # 8 + 1 + 1 = 10 electrons at neutral charge; closed shell.
    return SCFElectronicState(
        charge=0,
        multiplicity=1,
        n_electrons=10,
        spin_polarized=False,
        spin_channels=1,
        spin_source="derived",
    )


def converged_trajectory() -> SCFTrajectory:
    return SCFTrajectory(
        iterations=(
            SCFIterationRecord(step=1, residual=1.0e-2, energy=-17.0),
            SCFIterationRecord(step=2, residual=2.0e-4, energy=-17.16),
            SCFIterationRecord(step=3, residual=8.0e-7, energy=-17.1642),
        ),
        eps_scf=1.0e-6,
        max_scf=100,
        converged=True,
        initial_guess="ATOMIC",
        cp2k_version="CP2K version TEST-2026.1",
    )


def make_sample(sample_id: str = "task-0001", **overrides: Any) -> SCFSample:
    fields: dict[str, Any] = {
        "sample_id": sample_id,
        "structure": water_structure(),
        "electronic_state": water_state(),
        "trajectory": converged_trajectory(),
    }
    fields.update(overrides)
    return SCFSample(**fields)


# -- schema -------------------------------------------------------------


def test_sample_round_trips_through_json() -> None:
    sample = make_sample(artifacts=SCFDensityArtifacts(density_matrix_path="p.dm", density_format="cp2k-dm"))

    restored = SCFSample.from_dict(json.loads(json.dumps(sample.to_dict())))

    assert restored == sample
    assert restored.structure_hash == sample.structure_hash
    assert restored.trajectory.residual_history == (1.0e-2, 2.0e-4, 8.0e-7)
    assert restored.trajectory.final_energy == -17.1642


def test_structure_hash_ignores_float_noise_but_not_geometry() -> None:
    sample = make_sample()
    jittered = make_sample(
        structure=SCFStructure(
            atomic_numbers=sample.structure.atomic_numbers,
            positions=tuple(
                tuple(value + 1e-12 for value in row) for row in sample.structure.positions
            ),
            cell=sample.structure.cell,
            pbc=sample.structure.pbc,
        )
    )
    moved = make_sample(
        structure=SCFStructure(
            atomic_numbers=sample.structure.atomic_numbers,
            positions=((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (-0.24, 0.93, 0.0)),
            cell=sample.structure.cell,
            pbc=sample.structure.pbc,
        )
    )

    assert jittered.structure_hash == sample.structure_hash
    assert moved.structure_hash != sample.structure_hash


def test_structure_hash_separates_electronic_states() -> None:
    neutral = make_sample()
    cation = make_sample(
        electronic_state=SCFElectronicState(
            charge=1, multiplicity=2, n_electrons=9, spin_polarized=True, spin_channels=2
        )
    )

    assert cation.structure_hash != neutral.structure_hash


def test_missing_required_field_is_a_schema_error() -> None:
    payload = make_sample().to_dict()
    del payload["trajectory"]["converged"]

    with pytest.raises(SchemaError, match="converged"):
        SCFSample.from_dict(payload)


def test_unsupported_schema_version_is_rejected() -> None:
    payload = make_sample().to_dict()
    payload["schema_version"] = "phal-scf-sample-v0"

    with pytest.raises(SchemaError, match="unsupported schema_version"):
        SCFSample.from_dict(payload)


def test_mismatched_positions_and_numbers_are_rejected() -> None:
    with pytest.raises(SchemaError, match="same length"):
        SCFStructure(
            atomic_numbers=(8, 1),
            positions=((0.0, 0.0, 0.0),),
            cell=((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
        )


# -- validation ---------------------------------------------------------


def test_valid_sample_reports_no_issues() -> None:
    report = validate_sample(make_sample())

    assert report.ok
    assert report.issues == ()


def test_corrupted_trajectory_is_rejected() -> None:
    payload = make_sample().to_dict()
    payload["trajectory"]["iterations"][1]["residual"] = -1.0
    payload["trajectory"]["iterations"] = list(reversed(payload["trajectory"]["iterations"]))

    report = validate_payload(payload, subject="corrupt")

    assert not report.ok
    messages = " ".join(issue.message for issue in report.errors)
    assert "not ordered" in messages
    assert "negative" in messages


def test_non_finite_energy_is_rejected() -> None:
    payload = make_sample().to_dict()
    payload["trajectory"]["iterations"][0]["energy"] = "nan"

    report = validate_payload(payload, subject="nan-energy")

    assert not report.ok


def test_electron_count_inconsistency_is_rejected() -> None:
    sample = make_sample(
        electronic_state=SCFElectronicState(
            charge=0, multiplicity=1, n_electrons=12, spin_polarized=False, spin_channels=1
        )
    )

    report = validate_sample(sample)

    assert not report.ok
    assert any("disagrees with sum(Z)" in issue.message for issue in report.errors)


def test_impossible_multiplicity_is_rejected() -> None:
    sample = make_sample(
        electronic_state=SCFElectronicState(
            charge=0, multiplicity=2, n_electrons=10, spin_polarized=True, spin_channels=2
        )
    )

    report = validate_sample(sample)

    assert not report.ok
    assert any("impossible" in issue.message for issue in report.errors)


def test_iterations_exceeding_max_scf_are_rejected() -> None:
    sample = make_sample(
        trajectory=SCFTrajectory(
            iterations=tuple(
                SCFIterationRecord(step=index, residual=1.0e-3, energy=-17.0) for index in range(1, 5)
            ),
            eps_scf=1.0e-6,
            max_scf=3,
            converged=False,
            initial_guess="ATOMIC",
        )
    )

    report = validate_sample(sample)

    assert not report.ok
    assert any("exceed MAX_SCF" in issue.message for issue in report.errors)


def test_non_converged_trajectory_is_a_warning_not_an_error() -> None:
    sample = make_sample(
        trajectory=SCFTrajectory(
            iterations=(SCFIterationRecord(step=1, residual=1.0e-2, energy=-17.0),),
            eps_scf=1.0e-6,
            max_scf=100,
            converged=False,
            initial_guess="ATOMIC",
        )
    )

    report = validate_sample(sample)

    assert report.ok
    assert any("did not converge" in issue.message for issue in report.warnings)


def test_empty_trajectory_is_rejected() -> None:
    sample = make_sample(
        trajectory=SCFTrajectory(
            iterations=(), eps_scf=1.0e-6, max_scf=100, converged=False, initial_guess="ATOMIC"
        )
    )

    report = validate_sample(sample)

    assert not report.ok
    assert any("no iterations" in issue.message for issue in report.errors)


# -- dataset ------------------------------------------------------------


def test_create_builds_the_documented_layout(tmp_path: Path) -> None:
    dataset = SCFDataset.create(tmp_path / "scf_dataset")

    assert dataset.metadata_path.is_file()
    for split in SPLITS:
        assert (dataset.root / split).is_dir()
    assert dataset.metadata["splits"] == list(SPLITS)


def test_open_rejects_a_directory_without_metadata(tmp_path: Path) -> None:
    (tmp_path / "not_a_dataset").mkdir()

    with pytest.raises(DatasetError, match="not an SCF dataset"):
        SCFDataset.open(tmp_path / "not_a_dataset")


def test_add_sample_assigns_a_deterministic_split(tmp_path: Path) -> None:
    dataset = SCFDataset.create(tmp_path / "ds")
    sample = make_sample()

    first = dataset.assign_split(sample)
    dataset.add_sample(sample)

    assert dataset.assign_split(sample) == first
    assert dataset.sample_ids() == ["task-0001"]
    assert dataset.samples(first)[0] == sample


def test_add_sample_refuses_to_silently_overwrite(tmp_path: Path) -> None:
    dataset = SCFDataset.create(tmp_path / "ds")
    dataset.add_sample(make_sample())

    with pytest.raises(DatasetError, match="already exists"):
        dataset.add_sample(make_sample())

    dataset.add_sample(make_sample(), overwrite=True)
    assert len(dataset.sample_ids()) == 1


def test_assign_split_respects_the_configured_ratios() -> None:
    ratios = {"train": 1.0, "validation": 0.0, "test": 0.0}
    hashes = [f"sha256:{index:064x}" for index in range(50)]

    assert {assign_split(value, ratios) for value in hashes} == {"train"}


def test_assign_split_spreads_hashes_across_splits() -> None:
    import hashlib

    hashes = [
        "sha256:" + hashlib.sha256(str(index).encode("utf-8")).hexdigest() for index in range(400)
    ]

    assigned = {assign_split(value) for value in hashes}

    assert assigned == set(SPLITS)


def test_sanitize_sample_id_rejects_unusable_ids() -> None:
    assert sanitize_sample_id("task/0001 a") == "task_0001_a"

    with pytest.raises(DatasetError):
        sanitize_sample_id("///")


def test_validate_reports_a_corrupt_file_without_aborting(tmp_path: Path) -> None:
    dataset = SCFDataset.create(tmp_path / "ds")
    dataset.add_sample(make_sample("good"))
    (dataset.root / "train" / "broken.json").write_text("{not json", encoding="utf-8")

    reports = dataset.validate()

    assert len(reports) == 2
    failed = [report for report in reports if not report.ok]
    assert len(failed) == 1
    assert "invalid JSON" in failed[0].errors[0].message


def test_inspect_summarises_contents(tmp_path: Path) -> None:
    dataset = SCFDataset.create(tmp_path / "ds")
    dataset.add_sample(make_sample("a"))
    dataset.add_sample(make_sample("b"))

    summary = dataset.inspect()

    assert summary["n_total"] == 2
    assert summary["n_decoded"] == 2
    assert summary["atoms"]["min"] == 3
    assert summary["scf_iterations"]["mean"] == 3.0
    assert summary["converged_fraction"] == 1.0
    assert summary["initial_guess_counts"] == {"ATOMIC": 2}
    assert summary["element_counts"] == {1: 4, 8: 2}


def test_refresh_metadata_recounts_splits(tmp_path: Path) -> None:
    dataset = SCFDataset.create(tmp_path / "ds")
    dataset.add_sample(make_sample("a"))

    metadata = dataset.refresh_metadata()

    assert sum(metadata["n_samples"].values()) == 1


# -- ingestion ----------------------------------------------------------


def test_ingest_scf_root_reads_recorded_runs(tmp_path: Path) -> None:
    ase_io = pytest.importorskip("ase.io")
    from ase import Atoms

    scf_root = tmp_path / "scf"
    task_dir = scf_root / "task-0001"
    task_dir.mkdir(parents=True)
    ase_io.write(
        str(task_dir / "region.extxyz"),
        Atoms("OH2", positions=water_structure().positions, cell=[10.0, 10.0, 10.0], pbc=True),
        format="extxyz",
    )
    (task_dir / "scf_history.json").write_text(
        json.dumps(converged_trajectory().to_dict()), encoding="utf-8"
    )
    dataset = SCFDataset.create(tmp_path / "ds")

    summary = ingest_scf_root(scf_root, dataset, config={"cp2k": {"charge": 0}})

    assert summary["n_ingested"] == 1
    assert summary["n_skipped"] == 0
    sample = dataset.samples()[0]
    assert sample.sample_id == "task-0001"
    assert sample.electronic_state.n_electrons == 10
    assert validate_sample(sample).ok


def test_ingest_skips_runs_without_a_structure(tmp_path: Path) -> None:
    scf_root = tmp_path / "scf"
    (scf_root / "task-0002").mkdir(parents=True)
    (scf_root / "task-0002" / "scf_history.json").write_text(
        json.dumps(converged_trajectory().to_dict()), encoding="utf-8"
    )
    dataset = SCFDataset.create(tmp_path / "ds")

    summary = ingest_scf_root(scf_root, dataset)

    assert summary["n_ingested"] == 0
    assert summary["skipped"][0]["task_id"] == "task-0002"


# -- features -----------------------------------------------------------


def test_featurize_produces_the_declared_layout() -> None:
    features = featurize_sample(make_sample())
    metadata = feature_metadata()

    assert features.flat().shape == (metadata["n_system_features"],)
    assert features.per_atom.shape == (3, metadata["n_per_atom_features"])
    assert features.structure_hash == make_sample().structure_hash


def test_features_are_deterministic() -> None:
    first = featurize_sample(make_sample()).flat()
    second = featurize_sample(make_sample()).flat()

    assert (first == second).all()


def test_composition_counts_unsupported_elements_separately() -> None:
    from gmd_active_learning.adapters.dft.scf.features import SUPPORTED_ELEMENTS, composition_vector

    vector = composition_vector([1, 1, 94])

    assert vector[len(SUPPORTED_ELEMENTS)] == 1.0
    assert vector[len(SUPPORTED_ELEMENTS) + 1] == pytest.approx(1 / 3)


# -- export -------------------------------------------------------------


def test_export_writes_jsonl_features_and_a_manifest(tmp_path: Path) -> None:
    dataset = SCFDataset.create(tmp_path / "ds")
    for index in range(6):
        dataset.add_sample(make_sample(f"task-{index:04d}"), split="train")

    result = export_dataset(dataset, tmp_path / "export")

    assert result.counts["train"] == 6
    assert result.n_rejected == 0
    restored = read_jsonl(result.files["train_jsonl"])
    assert [sample.sample_id for sample in restored] == [f"task-{i:04d}" for i in range(6)]

    archive = load_features_npz(result.files["train_features"])
    assert archive["system_features"].shape[0] == 6
    assert list(archive["n_iterations"]) == [3.0] * 6

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["train"] == 6
    assert manifest["artifacts"]["train_jsonl"]["sha256"].startswith("sha256:")


def test_export_skips_invalid_samples_and_reports_them(tmp_path: Path) -> None:
    dataset = SCFDataset.create(tmp_path / "ds")
    dataset.add_sample(make_sample("good"), split="train")
    bad = make_sample("bad").to_dict()
    bad["electronic_state"]["n_electrons"] = 99
    (dataset.root / "train" / "bad.json").write_text(json.dumps(bad), encoding="utf-8")

    result = export_dataset(dataset, tmp_path / "export", splits=("train",))

    assert result.counts["train"] == 1
    assert result.n_rejected == 1
    assert "bad.json" in result.rejected[0]


def test_export_can_require_convergence(tmp_path: Path) -> None:
    dataset = SCFDataset.create(tmp_path / "ds")
    dataset.add_sample(make_sample("converged"), split="train")
    dataset.add_sample(
        make_sample(
            "stalled",
            trajectory=SCFTrajectory(
                iterations=(SCFIterationRecord(step=1, residual=1.0, energy=-1.0),),
                eps_scf=1.0e-6,
                max_scf=100,
                converged=False,
                initial_guess="ATOMIC",
            ),
        ),
        split="train",
    )

    result = export_dataset(dataset, tmp_path / "export", splits=("train",), require_converged=True)

    assert result.counts["train"] == 1
    assert result.n_rejected == 1


def test_export_of_an_empty_split_still_produces_artifacts(tmp_path: Path) -> None:
    dataset = SCFDataset.create(tmp_path / "ds")

    result = export_dataset(dataset, tmp_path / "export")

    assert result.counts == {"train": 0, "validation": 0, "test": 0}
    assert read_jsonl(result.files["train_jsonl"]) == []
    assert load_features_npz(result.files["train_features"])["system_features"].shape[0] == 0
