# Ported from PHAL (tests.scf.test_scf_density_recording) at 96263b6; MIT License, see NOTICE.
"""Stage 9 tests: real CP2K electronic density collection.

No CP2K executable is required. The AO matrices are rendered in CP2K's blocked
print layout by ``tests.scf.fakes.fake_density`` and parsed back.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gmd_active_learning.adapters.dft.scf.ao_matrices import (
    AOMatrixError,
    electron_count,
    parse_density_matrix,
    parse_overlap_matrix,
)
from gmd_active_learning.adapters.dft.scf.recording.density import (
    DENSITY_MATRIX_NAME,
    DENSITY_METADATA_NAME,
    MATRIX_FORMAT_AO_SPIN,
    OVERLAP_MATRIX_NAME,
    DensityRecordingError,
    density_recording_enabled,
    load_density,
    record_density,
    write_density_artifacts,
)
from gmd_active_learning.adapters.dft.scf.scf_history import record_scf_history_text
from gmd_active_learning.adapters.dft.scf.dataset import SCFDataset, discover_density_artifacts, ingest_scf_root
from gmd_active_learning.adapters.dft.scf.schema import SCFDensityArtifacts, SCFElectronicState
from tools.scf.training.dataset_loader import (
    TARGET_DENSITY_MATRIX,
    DataError,
    build_examples,
    load_density_matrix,
)
from gmd_active_learning.adapters.dft.scf.validation import validate_density_artifacts, validate_sample

from tests.scf.fakes.fake_density import (
    fake_cp2k_density_output,
    format_ao_matrix,
    spin_polarized_density,
    symmetric_density,
    write_fake_density_artifacts,
)
from tests.scf.test_scf_dataset import make_sample

CP2K_INPUT = """&FORCE_EVAL
  &DFT
    &SCF
      MAX_SCF 100
      EPS_SCF 1.0E-6
      SCF_GUESS ATOMIC
    &END SCF
  &END DFT
&END FORCE_EVAL
"""


def recording_config(tmp_path: Path, *, record_density_flag: bool = True) -> dict:
    return {
        "cp2k": {
            "functional": "PBE",
            "basis": "DZVP-MOLOPT-SR-GTH",
            "potential": "GTH-PBE",
            "scf_recording": {
                "enabled": True,
                "output_dir": str(tmp_path / "scf"),
                "record_density": record_density_flag,
            },
        }
    }


# -- AO matrix parsing --------------------------------------------------


def test_blocked_columns_are_stitched_back_together() -> None:
    matrix = np.arange(49, dtype=float).reshape(7, 7)
    matrix = matrix + matrix.T  # symmetric, and wider than one print block

    text = format_ao_matrix(matrix, "DENSITY MATRIX")
    parsed = parse_density_matrix(text)

    assert parsed.shape == (1, 7, 7)
    assert np.allclose(parsed[0], matrix)


def test_single_block_matrix_round_trips() -> None:
    matrix = symmetric_density(3, electrons=8.0)[0]

    parsed = parse_density_matrix(format_ao_matrix(matrix, "DENSITY MATRIX"))

    assert np.allclose(parsed[0], matrix)


def test_alpha_and_beta_are_returned_as_two_channels() -> None:
    density = spin_polarized_density(4, alpha=5.0, beta=4.0)

    parsed = parse_density_matrix(fake_cp2k_density_output(density))

    assert parsed.shape == (2, 4, 4)
    assert np.trace(parsed[0]) == pytest.approx(5.0)
    assert np.trace(parsed[1]) == pytest.approx(4.0)


def test_the_last_printed_density_wins() -> None:
    """CP2K reprints the density each SCF step; the converged one is last."""

    early = symmetric_density(3, electrons=4.0, seed=1)[0]
    late = symmetric_density(3, electrons=8.0, seed=2)[0]
    text = format_ao_matrix(early, "DENSITY MATRIX") + format_ao_matrix(late, "DENSITY MATRIX")

    parsed = parse_density_matrix(text)

    assert np.trace(parsed[0]) == pytest.approx(8.0)


def test_overlap_is_parsed_separately_from_the_density() -> None:
    density = symmetric_density(4, electrons=8.0, seed=3)
    overlap = np.eye(4) + 0.05 * (np.ones((4, 4)) - np.eye(4))

    text = fake_cp2k_density_output(density, overlap=overlap)

    assert np.allclose(parse_overlap_matrix(text), overlap)
    assert np.allclose(parse_density_matrix(text)[0], density[0])


def test_output_without_a_density_raises() -> None:
    with pytest.raises(AOMatrixError, match="no density matrix"):
        parse_density_matrix(" SCF run converged in 2 steps\n")


def test_output_without_an_overlap_returns_none() -> None:
    assert parse_overlap_matrix(fake_cp2k_density_output(symmetric_density(3, electrons=6.0))) is None


def test_an_incomplete_matrix_is_rejected_not_padded() -> None:
    matrix = symmetric_density(6, electrons=10.0)[0]
    lines = format_ao_matrix(matrix, "DENSITY MATRIX").splitlines()
    # Drop one data row: a padded zero here would be a silently wrong density.
    del lines[4]

    with pytest.raises(AOMatrixError, match="incomplete"):
        parse_density_matrix("\n".join(lines))


def test_electron_count_uses_the_overlap_when_present() -> None:
    density = np.asarray([[[2.0, 0.0], [0.0, 2.0]]])
    overlap = np.asarray([[1.0, 0.5], [0.5, 1.0]])

    assert electron_count(density, None) == pytest.approx(4.0)
    assert electron_count(density, overlap) == pytest.approx(4.0)

    correlated = np.asarray([[[2.0, 1.0], [1.0, 2.0]]])
    assert electron_count(correlated, overlap) == pytest.approx(5.0)


def test_mismatched_overlap_shape_raises() -> None:
    with pytest.raises(AOMatrixError, match="does not match"):
        electron_count(np.zeros((1, 3, 3)), np.zeros((2, 2)))


# -- recording ----------------------------------------------------------


def test_density_recording_is_off_by_default() -> None:
    assert not density_recording_enabled({})
    assert not density_recording_enabled({"cp2k": {"scf_recording": {"enabled": True}}})
    assert density_recording_enabled(
        {"cp2k": {"scf_recording": {"enabled": True, "record_density": True}}}
    )


def test_recording_needs_both_switches() -> None:
    # record_density alone does nothing: it extends SCF recording, not replaces it.
    assert not density_recording_enabled(
        {"cp2k": {"scf_recording": {"enabled": False, "record_density": True}}}
    )


def test_record_density_writes_the_documented_layout(tmp_path: Path) -> None:
    density = symmetric_density(4, electrons=8.0, seed=5)
    overlap = np.eye(4)
    text = fake_cp2k_density_output(density, overlap=overlap)

    artifacts = record_density(
        text,
        output_root=tmp_path / "scf",
        task_id="task-0001",
        config=recording_config(tmp_path),
        electronic_state={"charge": 0, "multiplicity": 1},
        cp2k_version="CP2K version TEST-2026.1",
        scf_converged=True,
    )

    assert artifacts is not None
    directory = tmp_path / "scf" / "task-0001" / "density"
    assert (directory / DENSITY_MATRIX_NAME).is_file()
    assert (directory / OVERLAP_MATRIX_NAME).is_file()
    assert (directory / DENSITY_METADATA_NAME).is_file()
    assert np.allclose(np.load(directory / DENSITY_MATRIX_NAME), density)


def test_density_metadata_carries_every_required_field(tmp_path: Path) -> None:
    text = fake_cp2k_density_output(symmetric_density(3, electrons=6.0), overlap=np.eye(3))

    record_density(
        text,
        output_root=tmp_path / "scf",
        task_id="task-0001",
        config=recording_config(tmp_path),
        electronic_state={"charge": -1, "multiplicity": 2},
        cp2k_version="CP2K version TEST-2026.1",
        scf_converged=True,
    )
    payload = json.loads(
        (tmp_path / "scf" / "task-0001" / "density" / DENSITY_METADATA_NAME).read_text(encoding="utf-8")
    )

    for field in (
        "cp2k_version",
        "functional",
        "basis_set",
        "potential",
        "charge",
        "multiplicity",
        "electron_count",
        "spin_channels",
        "matrix_format",
        "scf_converged",
    ):
        assert field in payload, field
    assert payload["functional"] == "PBE"
    assert payload["basis_set"] == "DZVP-MOLOPT-SR-GTH"
    assert payload["potential"] == "GTH-PBE"
    assert payload["charge"] == -1
    assert payload["multiplicity"] == 2
    assert payload["spin_channels"] == 1
    assert payload["matrix_format"] == MATRIX_FORMAT_AO_SPIN
    assert payload["scf_converged"] is True
    assert payload["electron_count"] == pytest.approx(6.0)
    assert payload["electron_count_method"] == "trace_ps"


def test_recording_disabled_writes_nothing(tmp_path: Path) -> None:
    text = fake_cp2k_density_output(symmetric_density(3, electrons=6.0))

    artifacts = record_density(
        text,
        output_root=tmp_path / "scf",
        task_id="t",
        config=recording_config(tmp_path, record_density_flag=False),
    )

    assert artifacts is None
    assert not (tmp_path / "scf").exists()


def test_output_without_a_density_is_skipped_not_fatal(tmp_path: Path) -> None:
    artifacts = record_density(
        " SCF run converged in 2 steps\n",
        output_root=tmp_path / "scf",
        task_id="t",
        config=recording_config(tmp_path),
    )

    assert artifacts is None


def test_strict_mode_raises_when_the_density_is_missing(tmp_path: Path) -> None:
    with pytest.raises(DensityRecordingError, match="no density matrix"):
        record_density(
            " SCF run converged\n",
            output_root=tmp_path / "scf",
            task_id="t",
            config=recording_config(tmp_path),
            strict=True,
        )


def test_scf_history_recording_also_writes_the_density(tmp_path: Path) -> None:
    input_path = tmp_path / "job.inp"
    input_path.write_text(CP2K_INPUT, encoding="utf-8")
    text = fake_cp2k_density_output(symmetric_density(3, electrons=6.0), overlap=np.eye(3))

    history_path = record_scf_history_text(
        text,
        input_path=input_path,
        config=recording_config(tmp_path),
        task_id="task-0001",
        electronic_state={"charge": 0, "multiplicity": 1},
    )

    assert history_path is not None
    assert history_path.is_file()
    assert (tmp_path / "scf" / "task-0001" / "density" / DENSITY_MATRIX_NAME).is_file()


def test_scf_history_recording_without_density_flag_writes_only_json(tmp_path: Path) -> None:
    input_path = tmp_path / "job.inp"
    input_path.write_text(CP2K_INPUT, encoding="utf-8")
    text = fake_cp2k_density_output(symmetric_density(3, electrons=6.0))

    record_scf_history_text(
        text,
        input_path=input_path,
        config=recording_config(tmp_path, record_density_flag=False),
        task_id="task-0001",
    )

    assert (tmp_path / "scf" / "task-0001" / "scf_history.json").is_file()
    assert not (tmp_path / "scf" / "task-0001" / "density").exists()


def test_load_density_round_trips(tmp_path: Path) -> None:
    density = spin_polarized_density(3, alpha=4.0, beta=3.0)
    overlap = np.eye(3)
    write_density_artifacts(tmp_path / "density", density, overlap=overlap)

    loaded, loaded_overlap, metadata = load_density(tmp_path)

    assert np.allclose(loaded, density)
    assert np.allclose(loaded_overlap, overlap)
    assert metadata["spin_channels"] == 2


def test_loading_a_foreign_matrix_format_is_refused(tmp_path: Path) -> None:
    write_density_artifacts(tmp_path / "density", symmetric_density(2, electrons=2.0))
    metadata_path = tmp_path / "density" / DENSITY_METADATA_NAME
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["matrix_format"] = "some-other-format"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DensityRecordingError, match="matrix_format"):
        load_density(tmp_path)


# -- input generation ---------------------------------------------------


H2_INPUT = """&GLOBAL
  PROJECT h2
  RUN_TYPE ENERGY
&END GLOBAL
&FORCE_EVAL
  METHOD QUICKSTEP
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    POTENTIAL_FILE_NAME POTENTIAL
    &SCF
      EPS_SCF 1.0E-6
      MAX_SCF 50
    &END SCF
    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
    &END XC
  &END DFT
  &SUBSYS
    &CELL
      ABC 10.0 10.0 10.0
    &END CELL
    &COORD
      H 5.0 5.0 4.63
      H 5.0 5.0 5.37
    &END COORD
  &END SUBSYS
&END FORCE_EVAL
"""

# PHAL tested this through its CP2K input builder (hotspot_al.cp2k.cp2k_input),
# which belongs to the CP2K labeling area in GMD_active. The block itself is
# provided by recording.input and tested on a real input here.


def test_ao_matrices_are_printed_only_when_recording_density(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.recording.input import insert_ao_matrices_print

    plain = insert_ao_matrices_print(H2_INPUT, {"scf": {}})
    config = {"scf": {"recording": {"enabled": True, "output_dir": str(tmp_path), "record_density": True}}}
    with_density = insert_ao_matrices_print(H2_INPUT, config)

    assert plain == H2_INPUT
    assert "AO_MATRICES" not in plain
    assert "AO_MATRICES" in with_density
    assert "DENSITY .TRUE." in with_density
    assert "OVERLAP .TRUE." in with_density
    assert "QS_SCF 1" not in with_density
    trajectory = insert_ao_matrices_print(
        H2_INPUT, {"scf": {"recording": {"enabled": True, "record_density": True, "record_trajectory": True}}}
    )
    assert "QS_SCF 1" in trajectory


def test_enabling_density_printing_does_not_disturb_the_rest_of_the_input(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.recording.input import insert_ao_matrices_print

    config = {"scf": {"recording": {"enabled": True, "output_dir": str(tmp_path), "record_density": True}}}
    with_density = insert_ao_matrices_print(H2_INPUT, config)

    # Everything except the inserted block must be untouched, line for line.
    inserted = [line for line in with_density.splitlines() if line not in H2_INPUT.splitlines()]
    assert all("AO_MATRICES" in line or line.strip() in {
        "&PRINT", "&END PRINT", "DENSITY .TRUE.", "OVERLAP .TRUE.", "NDIGITS 12"
    } for line in inserted), inserted
    remaining = [line for line in with_density.splitlines() if line not in inserted]
    assert remaining == H2_INPUT.splitlines()


# -- dataset integration ------------------------------------------------


def water_artifacts(tmp_path: Path, *, electrons: float = 10.0, channels: int = 1, seed: int = 0):
    if channels == 1:
        density = symmetric_density(6, electrons=electrons, seed=seed)
    else:
        density = spin_polarized_density(6, alpha=electrons / 2 + 0.5, beta=electrons / 2 - 0.5, seed=seed)
    overlap = np.eye(6)
    directory = write_fake_density_artifacts(tmp_path, density, overlap=overlap)
    return density, directory


def sample_with_density(tmp_path: Path, **kwargs):
    density, directory = water_artifacts(tmp_path, **kwargs)
    artifacts = discover_density_artifacts(tmp_path)
    return make_sample("task-0001", artifacts=artifacts), density, directory


def test_discovered_artifacts_carry_the_recorded_metadata(tmp_path: Path) -> None:
    water_artifacts(tmp_path)

    artifacts = discover_density_artifacts(tmp_path)

    assert artifacts.has_density_matrix
    assert artifacts.n_ao == 6
    assert artifacts.overlap_path is not None
    assert artifacts.density_electron_count == pytest.approx(10.0)


def test_no_density_directory_yields_empty_artifacts(tmp_path: Path) -> None:
    assert discover_density_artifacts(tmp_path).is_empty


def test_a_valid_density_passes_validation(tmp_path: Path) -> None:
    sample, _density, _directory = sample_with_density(tmp_path)

    report = validate_sample(sample, check_density=True)

    assert report.ok, [str(issue) for issue in report.errors]


def test_a_missing_density_file_is_rejected(tmp_path: Path) -> None:
    sample = make_sample(
        artifacts=SCFDensityArtifacts(
            density_matrix_path=str(tmp_path / "absent.npy"), density_format=MATRIX_FORMAT_AO_SPIN
        )
    )

    issues = validate_density_artifacts(sample)

    assert any("missing" in issue.message for issue in issues)


def test_a_sample_without_a_density_is_rejected_when_one_is_required() -> None:
    issues = validate_density_artifacts(make_sample())

    assert any("no density matrix is recorded" in issue.message for issue in issues)


def test_a_non_finite_density_is_rejected(tmp_path: Path) -> None:
    density = symmetric_density(4, electrons=8.0)
    density[0, 1, 1] = np.nan
    write_fake_density_artifacts(tmp_path, density, overlap=np.eye(4))
    sample = make_sample(artifacts=discover_density_artifacts(tmp_path))

    issues = validate_density_artifacts(sample)

    assert any("non-finite" in issue.message for issue in issues)


def test_an_asymmetric_density_is_rejected(tmp_path: Path) -> None:
    density = symmetric_density(4, electrons=8.0)
    density[0, 0, 1] += 0.5
    write_fake_density_artifacts(tmp_path, density, overlap=np.eye(4))
    sample = make_sample(artifacts=discover_density_artifacts(tmp_path))

    issues = validate_density_artifacts(sample)

    assert any("Hermitian" in issue.message for issue in issues)


def test_an_electron_count_mismatch_is_rejected(tmp_path: Path) -> None:
    density = symmetric_density(4, electrons=8.0)
    write_fake_density_artifacts(tmp_path, density, overlap=np.eye(4))
    artifacts = discover_density_artifacts(tmp_path)
    # Claim a count the matrix does not hold.
    sample = make_sample(
        artifacts=SCFDensityArtifacts(
            density_matrix_path=artifacts.density_matrix_path,
            overlap_path=artifacts.overlap_path,
            density_format=artifacts.density_format,
            n_ao=artifacts.n_ao,
            density_electron_count=12.0,
        )
    )

    issues = validate_density_artifacts(sample)

    assert any("electron count" in issue.message for issue in issues)


def test_a_spin_channel_mismatch_is_rejected(tmp_path: Path) -> None:
    density = spin_polarized_density(4, alpha=5.0, beta=4.0)
    write_fake_density_artifacts(tmp_path, density, overlap=np.eye(4))
    # Sample declares a closed shell; the density has two channels.
    sample = make_sample(artifacts=discover_density_artifacts(tmp_path))

    issues = validate_density_artifacts(sample)

    assert any("spin channel" in issue.message for issue in issues)


def test_an_n_ao_disagreement_is_rejected(tmp_path: Path) -> None:
    water_artifacts(tmp_path)
    artifacts = discover_density_artifacts(tmp_path)
    sample = make_sample(
        artifacts=SCFDensityArtifacts(
            density_matrix_path=artifacts.density_matrix_path,
            overlap_path=artifacts.overlap_path,
            n_ao=99,
            density_electron_count=artifacts.density_electron_count,
        )
    )

    issues = validate_density_artifacts(sample)

    assert any("n_ao" in issue.field_path for issue in issues)


def test_an_overlap_shape_mismatch_is_rejected(tmp_path: Path) -> None:
    water_artifacts(tmp_path)
    bad_overlap = tmp_path / "bad_overlap.npy"
    np.save(bad_overlap, np.eye(3))
    artifacts = discover_density_artifacts(tmp_path)
    sample = make_sample(
        artifacts=SCFDensityArtifacts(
            density_matrix_path=artifacts.density_matrix_path,
            overlap_path=str(bad_overlap),
            density_electron_count=artifacts.density_electron_count,
        )
    )

    issues = validate_density_artifacts(sample)

    assert any("does not match density" in issue.message for issue in issues)


# -- training loader ----------------------------------------------------


def test_density_matrix_target_now_loads_real_artifacts(tmp_path: Path) -> None:
    sample, density, _directory = sample_with_density(tmp_path)

    examples = build_examples([sample], target=TARGET_DENSITY_MATRIX)

    assert len(examples) == 1, examples.rejected
    example = examples[0]
    assert example.density_target.shape == (1, 6, 6)
    assert np.allclose(example.density_target, density)
    assert example.overlap is not None
    assert example.n_ao == 6
    # The valence count the density holds, not the all-electron count.
    assert example.electron_count == pytest.approx(10.0)


def test_open_shell_density_loads_two_channels(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.schema import SCFStructure

    density = spin_polarized_density(6, alpha=5.0, beta=4.0)
    write_fake_density_artifacts(tmp_path, density, overlap=np.eye(6))
    # OH radical: 8 + 1 = 9 electrons, doublet, two spin channels.
    sample = make_sample(
        artifacts=discover_density_artifacts(tmp_path),
        structure=SCFStructure(
            atomic_numbers=(8, 1),
            positions=((0.0, 0.0, 0.0), (0.97, 0.0, 0.0)),
            cell=((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
        ),
        electronic_state=SCFElectronicState(
            charge=0, multiplicity=2, n_electrons=9, spin_polarized=True, spin_channels=2
        ),
    )

    examples = build_examples([sample], target=TARGET_DENSITY_MATRIX)

    assert len(examples) == 1, examples.rejected
    assert examples[0].density_target.shape == (2, 6, 6)
    assert examples[0].spin_channels == 2


def test_a_sample_without_a_density_is_dropped_with_a_reason(tmp_path: Path) -> None:
    examples = build_examples([make_sample()], target=TARGET_DENSITY_MATRIX)

    assert len(examples) == 0
    assert "no density matrix is recorded" in examples.rejected[0]


def test_a_corrupted_density_file_is_dropped_not_crashed(tmp_path: Path) -> None:
    water_artifacts(tmp_path)
    (tmp_path / "density" / DENSITY_MATRIX_NAME).write_bytes(b"not a numpy file")
    sample = make_sample(artifacts=discover_density_artifacts(tmp_path))

    examples = build_examples([sample], target=TARGET_DENSITY_MATRIX)

    assert len(examples) == 0
    assert "could not read" in examples.rejected[0]


def test_load_density_matrix_raises_on_a_channel_mismatch(tmp_path: Path) -> None:
    density = spin_polarized_density(4, alpha=5.0, beta=4.0)
    write_fake_density_artifacts(tmp_path, density, overlap=np.eye(4))
    sample = make_sample(artifacts=discover_density_artifacts(tmp_path))

    with pytest.raises(DataError, match="spin channel"):
        load_density_matrix(sample)


def test_ingesting_a_run_picks_up_its_density(tmp_path: Path) -> None:
    from ase import Atoms
    from ase.io import write

    from gmd_active_learning.utils.atomic_io import write_json_atomic

    from tests.scf.test_scf_dataset import converged_trajectory, water_structure

    task_dir = tmp_path / "scf" / "task-0001"
    task_dir.mkdir(parents=True)
    write(
        str(task_dir / "region.extxyz"),
        Atoms("OH2", positions=water_structure().positions, cell=[10.0, 10.0, 10.0], pbc=True),
        format="extxyz",
    )
    write_json_atomic(task_dir / "scf_history.json", converged_trajectory().to_dict())
    write_fake_density_artifacts(task_dir, symmetric_density(6, electrons=8.0), overlap=np.eye(6))

    dataset = SCFDataset.create(tmp_path / "ds")
    summary = ingest_scf_root(tmp_path / "scf", dataset, config={"cp2k": {"charge": 0}})

    assert summary["n_ingested"] == 1
    sample = dataset.samples()[0]
    assert sample.artifacts.has_density_matrix
    assert sample.artifacts.n_ao == 6
    assert validate_sample(sample, check_density=True).ok


def test_inspect_counts_samples_with_density_artifacts(tmp_path: Path) -> None:
    water_artifacts(tmp_path)
    dataset = SCFDataset.create(tmp_path / "ds")
    dataset.add_sample(make_sample("with", artifacts=discover_density_artifacts(tmp_path)), split="train")
    dataset.add_sample(make_sample("without"), split="train")

    assert dataset.inspect()["n_with_density_artifacts"] == 1
