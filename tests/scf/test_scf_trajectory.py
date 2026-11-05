# Ported from PHAL (tests.scf.test_scf_trajectory) at 96263b6; MIT License, see NOTICE.
"""Tests for SCF density-trajectory recording.

The fixtures are real CP2K v2026.2 runs (DZVP-MOLOPT-GTH/PBE), not synthetic
data: H2 converged in 10 iterations and H2O in 13, so their trajectories hold
11 and 14 density frames respectively (one per iteration plus the post-SCF
print). Using real output is the point -- the invariants asserted here
(electron conservation at *every* iteration, exact symmetry, monotone frame
ordering) are properties of what CP2K actually emits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gmd_active_learning.adapters.dft.scf.ao_matrices import (
    AOMatrixError,
    electron_count,
    parse_density_trajectory,
)
from gmd_active_learning.adapters.dft.scf.recording.trajectory import (
    DENSITY_NAME,
    ELECTRON_TOLERANCE,
    FINAL_DIR_NAME,
    METADATA_NAME,
    OVERLAP_NAME,
    TRAJECTORY_DIR_NAME,
    SCFTrajectoryError,
    load_trajectory,
    record_scf_trajectory,
    trajectory_recording_enabled,
    validate_frame,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "scf"

#: (name, electrons, n_ao, expected frames). Frames = SCF iterations + 1.
REAL_SYSTEMS = [("h2", 2.0, 10, 11), ("h2o", 8.0, 23, 14)]


def load_fixture(name: str):
    data = np.load(FIXTURES / f"{name}_scf_trajectory.npz")
    return data["densities"], data["overlap"], float(data["n_electrons"])


def recording_config(**overrides):
    recording = {"enabled": True, "record_density": True, "record_trajectory": True}
    recording.update(overrides)
    return {
        "cp2k": {
            "scf_recording": recording,
            "functional": "PBE",
            "basis": "DZVP-MOLOPT-GTH",
            "potential": "GTH-PBE",
        }
    }


def synthetic_trajectory(tmp_path: Path, name: str) -> Path:
    """Write a fixture trajectory to disk through the public recorder.

    The recorder consumes CP2K text, so the fixture arrays are replayed by
    writing the frames directly -- this exercises the on-disk layout and the
    loader without needing the multi-megabyte original output in the repo.
    """

    densities, overlap, electrons = load_fixture(name)
    directory = tmp_path / name / TRAJECTORY_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / OVERLAP_NAME, overlap)

    frames = []
    for index, density in enumerate(densities):
        is_final = index == len(densities) - 1
        label = FINAL_DIR_NAME if is_final else f"iter_{index:04d}"
        frame_dir = directory / label
        frame_dir.mkdir(parents=True, exist_ok=True)
        np.save(frame_dir / DENSITY_NAME, density)
        entry = {
            "index": index,
            "iteration": None if is_final else index + 1,
            "residual": None if is_final else 10.0 ** (-index),
            "energy": None if is_final else -1.0 - index,
            "electron_count": electron_count(density, overlap),
            "density_path": str(frame_dir / DENSITY_NAME),
            "metadata_path": str(frame_dir / METADATA_NAME),
            "is_final": is_final,
        }
        (frame_dir / METADATA_NAME).write_text(json.dumps(entry), encoding="utf-8")
        frames.append(entry)

    (directory / METADATA_NAME).write_text(
        json.dumps(
            {
                "n_frames": len(frames),
                "n_ao": int(densities.shape[-1]),
                "spin_channels": int(densities.shape[1]),
                "n_electrons": electrons,
                "overlap_path": OVERLAP_NAME,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    return directory


# -- the physical invariants --------------------------------------------


@pytest.mark.parametrize("name,electrons,n_ao,n_frames", REAL_SYSTEMS)
def test_every_iteration_conserves_electrons(name, electrons, n_ao, n_frames):
    """Tr(P S) = Ne holds at *every* SCF step, not only at convergence.

    This is what makes the electron count usable as a per-frame gate: a frame
    that fails it is corrupt, not merely unconverged.
    """

    densities, overlap, expected = load_fixture(name)

    assert densities.shape == (n_frames, 1, n_ao, n_ao)
    for index, density in enumerate(densities):
        counted = electron_count(density, overlap)
        assert counted == pytest.approx(expected, abs=1e-6), f"frame {index}"


@pytest.mark.parametrize("name,electrons,n_ao,n_frames", REAL_SYSTEMS)
def test_trace_p_alone_would_not_conserve(name, electrons, n_ao, n_frames):
    """Tr(P) drifts across the trajectory, so it cannot gate any frame.

    It is not merely offset by a constant that a consumer could correct for:
    it changes from iteration to iteration.
    """

    densities, _overlap, expected = load_fixture(name)

    traces = [float(np.trace(density, axis1=1, axis2=2).sum()) for density in densities]
    # Every frame fails the electron-count gate on Tr(P), by orders of
    # magnitude more than the 0.1% tolerance the gate allows.
    tolerance = ELECTRON_TOLERANCE * max(1.0, abs(expected))
    assert all(abs(trace - expected) > tolerance for trace in traces)
    # And it drifts during the SCF, so no single correction factor would work
    # even if one were willing to apply one.
    assert max(traces) - min(traces) > 1e-3


@pytest.mark.parametrize("name,electrons,n_ao,n_frames", REAL_SYSTEMS)
def test_every_iteration_is_hermitian_and_finite(name, electrons, n_ao, n_frames):
    densities, overlap, _expected = load_fixture(name)

    for index, density in enumerate(densities):
        assert np.isfinite(density).all(), f"frame {index}"
        asymmetry = float(np.abs(density - np.swapaxes(density, 1, 2)).max())
        assert asymmetry < 1e-8, f"frame {index}: asymmetry {asymmetry}"


# -- frame validation ----------------------------------------------------


def test_validate_frame_returns_the_overlap_corrected_count():
    densities, overlap, expected = load_fixture("h2")

    counted = validate_frame(densities[0], overlap, expected_electrons=expected, label="f")

    assert counted == pytest.approx(expected, abs=1e-6)


def test_validate_frame_rejects_a_non_hermitian_density():
    densities, overlap, expected = load_fixture("h2")
    broken = np.array(densities[0])
    broken[0, 0, 1] += 0.5

    with pytest.raises(SCFTrajectoryError, match="Hermitian"):
        validate_frame(broken, overlap, expected_electrons=expected, label="f")


def test_validate_frame_rejects_a_non_finite_density():
    densities, overlap, expected = load_fixture("h2")
    broken = np.array(densities[0])
    broken[0, 0, 0] = np.nan

    with pytest.raises(SCFTrajectoryError, match="non-finite"):
        validate_frame(broken, overlap, expected_electrons=expected, label="f")


def test_validate_frame_rejects_a_wrong_electron_count():
    densities, overlap, expected = load_fixture("h2")

    with pytest.raises(SCFTrajectoryError, match="integrates to"):
        validate_frame(densities[0], overlap, expected_electrons=expected + 1.0, label="f")


def test_validate_frame_uses_the_overlap_not_the_trace():
    """Scaling S changes Tr(P S) and must change the verdict."""

    densities, overlap, expected = load_fixture("h2")

    with pytest.raises(SCFTrajectoryError, match="integrates to"):
        validate_frame(densities[0], overlap * 1.5, expected_electrons=expected, label="f")


# -- loading and ordering ------------------------------------------------


@pytest.mark.parametrize("name,electrons,n_ao,n_frames", REAL_SYSTEMS)
def test_trajectory_round_trips_through_disk(tmp_path, name, electrons, n_ao, n_frames):
    directory = synthetic_trajectory(tmp_path, name)
    original, overlap, _expected = load_fixture(name)

    densities, loaded_overlap, metadata = load_trajectory(directory)

    assert len(densities) == n_frames == metadata["n_frames"]
    assert np.allclose(loaded_overlap, overlap)
    for index, density in enumerate(densities):
        assert np.array_equal(density, original[index]), f"frame {index}"


@pytest.mark.parametrize("name,electrons,n_ao,n_frames", REAL_SYSTEMS)
def test_loaded_trajectory_is_ordered_p0_to_pfinal(tmp_path, name, electrons, n_ao, n_frames):
    """The sequence must be [P0, P1, ..., Pfinal] in SCF order.

    Ordering comes from the recorded index, not from directory listing order:
    'final' sorts before 'iter_0000' alphabetically, so a loader that trusted
    the filesystem would put the converged density first.
    """

    directory = synthetic_trajectory(tmp_path, name)
    original, _overlap, _expected = load_fixture(name)

    densities, _loaded_overlap, metadata = load_trajectory(directory)

    indices = [int(frame["index"]) for frame in metadata["frames"]]
    assert indices == sorted(indices) == list(range(n_frames))
    assert [frame["is_final"] for frame in metadata["frames"]] == [False] * (n_frames - 1) + [True]
    # The last loaded density is the converged one.
    assert np.array_equal(densities[-1], original[-1])
    # And the alphabetical trap is real: 'final' < 'iter_0000'.
    assert sorted([FINAL_DIR_NAME, "iter_0000"])[0] == FINAL_DIR_NAME


@pytest.mark.parametrize("name,electrons,n_ao,n_frames", REAL_SYSTEMS)
def test_loaded_trajectory_conserves_electrons_end_to_end(tmp_path, name, electrons, n_ao, n_frames):
    directory = synthetic_trajectory(tmp_path, name)

    densities, overlap, metadata = load_trajectory(directory)

    for index, density in enumerate(densities):
        assert electron_count(density, overlap) == pytest.approx(
            metadata["n_electrons"], abs=1e-6
        ), f"frame {index}"


def test_load_trajectory_accepts_the_task_directory(tmp_path):
    synthetic_trajectory(tmp_path, "h2")

    densities, _overlap, _metadata = load_trajectory(tmp_path / "h2")

    assert len(densities) == 11


def test_load_trajectory_reports_a_missing_directory(tmp_path):
    with pytest.raises(SCFTrajectoryError, match="metadata"):
        load_trajectory(tmp_path / "absent")


# -- configuration -------------------------------------------------------


def test_recording_requires_all_three_switches():
    assert trajectory_recording_enabled(recording_config())
    assert not trajectory_recording_enabled(recording_config(record_trajectory=False))
    assert not trajectory_recording_enabled(recording_config(record_density=False))
    assert not trajectory_recording_enabled(recording_config(enabled=False))
    assert not trajectory_recording_enabled({})


def test_recording_is_skipped_when_disabled(tmp_path):
    result = record_scf_trajectory(
        "irrelevant",
        output_root=tmp_path,
        task_id="t",
        config=recording_config(record_trajectory=False),
    )

    assert result is None


def test_output_without_densities_is_reported_in_strict_mode(tmp_path):
    with pytest.raises(SCFTrajectoryError, match="no density matrix"):
        record_scf_trajectory(
            "nothing here",
            output_root=tmp_path,
            task_id="t",
            config=recording_config(),
            strict=True,
        )


def test_output_without_densities_is_tolerated_by_default(tmp_path):
    assert (
        record_scf_trajectory(
            "nothing here", output_root=tmp_path, task_id="t", config=recording_config()
        )
        is None
    )


def test_parse_density_trajectory_requires_printed_matrices():
    with pytest.raises(AOMatrixError, match="AO_MATRICES"):
        parse_density_trajectory("no matrices here")


# -- end to end, from CP2K text -----------------------------------------


def cp2k_output(n_iterations: int, nao: int, electrons: float) -> str:
    """Synthesise CP2K output with an overlap and one density per iteration.

    Densities are built as c * S^-1 scaled to Tr(P S) = Ne, so every frame is
    symmetric and electron-conserving the way a real one is, while the block
    layout matches what ``cp_dbcsr_write_sparse_matrix`` prints.
    """

    rng = np.random.default_rng(0)
    raw = rng.normal(size=(nao, nao))
    overlap = raw @ raw.T + nao * np.eye(nao)
    overlap /= np.diag(overlap).mean()  # unit-ish diagonal, real off-diagonals

    def block(title: str, matrix: np.ndarray) -> str:
        """Reproduce cp_dbcsr_write_sparse_matrix's layout.

        A column-index header, then one line per row carrying that row's value
        in every column of the group. All columns go in one group here; CP2K
        splits them to fit the page, which the parser handles either way.
        """

        rows, columns = matrix.shape
        lines = [f" {title}", ""]
        lines.append("     " + "".join(f"{column + 1:>19}" for column in range(columns)))
        for row in range(rows):
            values = "".join(f"{matrix[row, column]:>19.12f}" for column in range(columns))
            lines.append(f"{row + 1:>6}{1:>5}  H     2s   {values}")
        # Two blank lines terminate a matrix; CP2K emits exactly this before
        # the SCF table row, and without it the row is absorbed as matrix data.
        lines.extend(["", ""])
        return "\n".join(lines)

    parts = [block("OVERLAP MATRIX", overlap)]
    parts.append("  Step     Update method      Time    Convergence         Total energy    Change")
    parts.append("")
    inverse = np.linalg.inv(overlap)
    for step in range(1, n_iterations + 2):
        density = inverse * (1.0 + 0.05 * step)
        density = 0.5 * (density + density.T)
        density *= electrons / float((density * overlap).sum())
        parts.append(block("DENSITY MATRIX", density))
        if step <= n_iterations:
            residual = 10.0 ** (-step)
            energy = -1.0 - 0.01 * step
            parts.append(
                f"{step:>6} P_Mix/Diag. 0.40E+00    0.2  {residual:>14.8f} {energy:>20.10f} -1.00E-02"
            )
    parts.append(f"  *** SCF run converged in {n_iterations} steps ***")
    parts.append(f" ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]  {-1.0 - 0.01 * n_iterations:.12f}")
    return "\n".join(parts)


def test_record_from_cp2k_text_writes_the_documented_layout(tmp_path):
    text = cp2k_output(n_iterations=4, nao=5, electrons=6.0)

    artifacts = record_scf_trajectory(
        text,
        output_root=tmp_path,
        task_id="task-1",
        config=recording_config(),
        cp2k_version="2026.2",
        scf_converged=True,
        strict=True,
    )

    assert artifacts is not None
    assert artifacts.n_frames == 5  # 4 iterations + the final print
    directory = tmp_path / "task-1" / TRAJECTORY_DIR_NAME
    assert artifacts.directory == directory
    assert (directory / OVERLAP_NAME).is_file()
    assert (directory / METADATA_NAME).is_file()
    for index in range(4):
        assert (directory / f"iter_{index:04d}" / DENSITY_NAME).is_file()
        assert (directory / f"iter_{index:04d}" / METADATA_NAME).is_file()
    assert (directory / FINAL_DIR_NAME / DENSITY_NAME).is_file()

    # Iteration, residual and energy are carried per frame.
    assert [frame.iteration for frame in artifacts.frames] == [1, 2, 3, 4, None]
    assert artifacts.frames[0].residual == pytest.approx(0.1)
    assert artifacts.frames[3].energy == pytest.approx(-1.04)
    assert artifacts.frames[-1].is_final

    densities, overlap, metadata = load_trajectory(directory)
    assert len(densities) == 5
    for density in densities:
        assert electron_count(density, overlap) == pytest.approx(6.0, abs=1e-6)
    assert metadata["representation"] == "ao-nonorthogonal"
    assert metadata["electron_count_method"] == "trace_ps"


def test_recording_rejects_a_corrupt_frame_in_strict_mode(tmp_path):
    lines = cp2k_output(n_iterations=3, nao=4, electrons=4.0).splitlines()
    # Corrupt one element of the first density: enough to break both the
    # symmetry and the electron count, which is what a truncated or garbled
    # output looks like in practice.
    start = next(i for i, line in enumerate(lines) if "DENSITY MATRIX" in line)
    target = next(
        i for i in range(start, len(lines)) if lines[i].lstrip().startswith("1 ") and "2s" in lines[i]
    )
    # The parser reads the *last* N numbers on a row, so the trailing field is
    # the one that has to change.
    lines[target] = lines[target][:-19] + f"{99.0:>19.12f}"

    with pytest.raises(SCFTrajectoryError):
        record_scf_trajectory(
            "\n".join(lines),
            output_root=tmp_path,
            task_id="task-2",
            config=recording_config(),
            strict=True,
        )


def test_scf_table_parsing_ignores_printed_matrix_rows():
    """An AO matrix row also starts with an integer.

    Without a discriminator every printed matrix row is counted as an SCF
    iteration -- and that only shows up once AO_MATRICES printing is on, which
    is exactly the configuration trajectory recording requires.
    """

    from gmd_active_learning.adapters.dft.scf.scf_history import parse_scf_history

    history = parse_scf_history(cp2k_output(n_iterations=6, nao=8, electrons=4.0))

    assert [iteration.step for iteration in history.iterations] == [1, 2, 3, 4, 5, 6]
