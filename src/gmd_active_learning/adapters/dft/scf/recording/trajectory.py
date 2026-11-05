# Ported from PHAL (hotspot_al.cp2k.scf_trajectory) at 96263b6; MIT License, see NOTICE.
"""Persist the density at every SCF iteration, for neural-mixing training.

A mixer learns the map from (density history, residual history) to the next
density, so it needs the *path* the SCF took, not just where it ended up. This
module transcribes that path out of a CP2K output into::

    runtime/cp2k/scf/<task>/trajectory/
        overlap.npy          the AO overlap S
        metadata.json        run-level: n_ao, spin channels, electrons, ...
        iter_0000/
            density.npy      (spin_channels, nao, nao)
            metadata.json    iteration, residual, energy, electron count
        iter_0001/
            ...
        final/
            density.npy      the converged density
            metadata.json

Nothing here computes electronic structure. Every number is CP2K's; this module
copies it out and refuses anything that fails the checks in
``docs/density_representation.md``.

**The overlap is stored once.** ``S`` depends only on the geometry and the basis
set, both of which are fixed for the whole SCF, so every iteration shares one
file rather than storing N identical copies of an ``nao x nao`` matrix. Each
iteration's metadata names it, so a frame is still self-describing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from gmd_active_learning.adapters.dft.scf.ao_matrices import (
    AOMatrixError,
    electron_count,
    parse_density_trajectory,
    parse_overlap_matrix,
)
from gmd_active_learning.adapters.dft.scf.recording.density import (
    MATRIX_FORMAT_AO_SPIN,
    REPRESENTATION_AO_NONORTHOGONAL,
    _cp2k_config,
    _recording_config,
)
from gmd_active_learning.adapters.dft.scf.pulay_matrix import (
    METRIC_GSPACE_KERKER,
    REPRESENTATION_CP2K_PULAY,
    PulayRecordError,
    b_matrix_diagnostics,
    read_records,
    verify_records,
)
from gmd_active_learning.adapters.dft.scf.scf_history import parse_scf_history

TRAJECTORY_DIR_NAME = "trajectory"
DENSITY_NAME = "density.npy"
OVERLAP_NAME = "overlap.npy"
METADATA_NAME = "metadata.json"
FINAL_DIR_NAME = "final"
PULAY_MATRIX_NAME = "pulay_matrix.npy"

#: What ``trajectory/`` holds. Consumers check this before trusting the layout.
TRAJECTORY_FORMAT = "ao-density-trajectory-v1"

#: Tolerances for the per-frame checks. These match the exchange contract in
#: ``ml_scf_types.F`` so a frame that trains a model is a frame CP2K would also
#: have accepted.
HERMITIAN_TOLERANCE = 1.0e-8
ELECTRON_TOLERANCE = 1.0e-3


class SCFTrajectoryError(RuntimeError):
    """Raised when a trajectory was requested but cannot be recorded."""


@dataclass(frozen=True, slots=True)
class TrajectoryFrame:
    """One recorded SCF iteration."""

    index: int
    iteration: int | None
    residual: float | None
    energy: float | None
    electron_count: float
    density_path: Path
    metadata_path: Path
    is_final: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "iteration": self.iteration,
            "residual": self.residual,
            "energy": self.energy,
            "electron_count": self.electron_count,
            "density_path": str(self.density_path),
            "metadata_path": str(self.metadata_path),
            "is_final": self.is_final,
        }


@dataclass(frozen=True, slots=True)
class SCFTrajectoryArtifacts:
    """Where a recorded trajectory landed, and what it says about itself."""

    directory: Path
    overlap_path: Path
    metadata_path: Path
    frames: tuple[TrajectoryFrame, ...]
    n_ao: int
    spin_channels: int
    n_electrons: float

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "overlap_path": str(self.overlap_path),
            "metadata_path": str(self.metadata_path),
            "n_frames": self.n_frames,
            "n_ao": self.n_ao,
            "spin_channels": self.spin_channels,
            "n_electrons": self.n_electrons,
            "trajectory_format": TRAJECTORY_FORMAT,
            "representation": REPRESENTATION_AO_NONORTHOGONAL,
            "frames": [frame.to_dict() for frame in self.frames],
        }


def trajectory_recording_enabled(config: Mapping[str, Any]) -> bool:
    """Whether this configuration asks for per-iteration densities.

    Three switches, because a trajectory is an extension of density recording,
    which is itself an extension of SCF recording. It is the most expensive
    artifact the pipeline produces: one ``nao x nao`` matrix per iteration, and
    the printed text that carries them is O(nao^2) per iteration too.
    """

    recording = _recording_config(config)
    return (
        bool(recording.get("enabled", False))
        and bool(recording.get("record_density", False))
        and bool(recording.get("record_trajectory", False))
    )


def validate_frame(
    density: np.ndarray,
    overlap: np.ndarray,
    *,
    expected_electrons: float,
    label: str,
) -> float:
    """Check one frame and return its electron count ``Tr(P S)``.

    The three checks are the ones in ``docs/density_representation.md``:
    finite, ``P = P†``, and ``Tr(P S) = Ne``. Order matters -- finiteness is
    first, because every later comparison against a NaN evaluates false and
    would silently pass.
    """

    matrices = np.asarray(density, dtype=float)
    if matrices.ndim != 3:
        raise SCFTrajectoryError(f"{label}: expected (spin, nao, nao), got shape {matrices.shape}")

    if not np.isfinite(matrices).all():
        raise SCFTrajectoryError(f"{label}: density contains non-finite values")
    if not np.isfinite(overlap).all():
        raise SCFTrajectoryError(f"{label}: overlap contains non-finite values")

    asymmetry = float(np.abs(matrices - np.swapaxes(matrices, 1, 2)).max())
    if asymmetry > HERMITIAN_TOLERANCE:
        raise SCFTrajectoryError(f"{label}: density is not Hermitian (max asymmetry {asymmetry:g})")

    if overlap.shape != matrices.shape[1:]:
        raise SCFTrajectoryError(
            f"{label}: overlap shape {overlap.shape} does not match density {matrices.shape[1:]}"
        )

    counted = electron_count(matrices, overlap)
    tolerance = ELECTRON_TOLERANCE * max(1.0, abs(expected_electrons))
    if abs(counted - expected_electrons) > tolerance:
        raise SCFTrajectoryError(
            f"{label}: density integrates to {counted:.6f} electrons, expected "
            f"{expected_electrons:.6f} (Tr(P S), not Tr(P) -- see "
            "docs/density_representation.md)"
        )
    return counted


def record_scf_trajectory(
    output_text: str,
    *,
    output_root: str | Path,
    task_id: str,
    config: Mapping[str, Any],
    electronic_state: Mapping[str, Any] | None = None,
    cp2k_version: str | None = None,
    scf_converged: bool = False,
    strict: bool = False,
    pulay_record_dir: str | Path | None = None,
) -> SCFTrajectoryArtifacts | None:
    """Extract and persist every SCF-iteration density in ``output_text``.

    ``pulay_record_dir`` is where ``ml_pulay_recorder.F`` wrote its dumps (the
    directory named by ``PHAL_ML_SCF_PULAY_RECORD_DIR``). When given, each
    frame also receives CP2K's own ``pulay_matrix.npy`` -- the matrix a mixer
    is actually fed at run time, which cannot be reconstructed from densities.

    Returns ``None`` when trajectory recording is disabled or when CP2K printed
    no per-iteration densities. With ``strict=True`` a missing or unusable
    trajectory raises instead, which is what a data-collection run wants and a
    labeling run does not.
    """

    if not trajectory_recording_enabled(config):
        return None

    try:
        frames = parse_density_trajectory(output_text)
        overlap = parse_overlap_matrix(output_text)
    except AOMatrixError as exc:
        if strict:
            raise SCFTrajectoryError(f"{task_id}: {exc}") from exc
        return None

    if overlap is None:
        message = (
            f"{task_id}: the trajectory has no overlap matrix, so no frame's electron count "
            "can be verified; enable &AO_MATRICES OVERLAP .TRUE."
        )
        if strict:
            raise SCFTrajectoryError(message)
        return None

    if not frames:
        if strict:
            raise SCFTrajectoryError(f"{task_id}: no density frames were parsed")
        return None

    state = dict(electronic_state or {})
    cp2k_cfg = _cp2k_config(config)
    history = parse_scf_history(output_text)
    iterations = list(history.iterations)

    # CP2K prints the density once per SCF iteration and once more after the
    # loop. When the counts line up that last frame is the converged density;
    # when they do not, the run was truncated and only the final frame can be
    # named with confidence.
    n_frames = len(frames)
    final_index = n_frames - 1
    if len(iterations) == n_frames - 1:
        paired: list[Any] = list(iterations) + [None]
    else:
        paired = [None] * n_frames
        for position, iteration in enumerate(iterations[: max(0, n_frames - 1)]):
            paired[position] = iteration

    # The converged density is the reference for the electron count: it is the
    # one CP2K itself normalised, and every earlier frame conserves the same
    # number.
    expected_electrons = electron_count(frames[final_index], overlap)

    directory = Path(output_root) / task_id / TRAJECTORY_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)

    overlap_path = directory / OVERLAP_NAME
    np.save(overlap_path, np.asarray(overlap, dtype=float))

    # CP2K's own Pulay matrices, keyed by the SCF iteration they belong to.
    # Verified against the coefficients CP2K derived from them before use: a
    # matrix that cannot reproduce them is not the one CP2K solved.
    pulay_by_iteration: dict[int, Any] = {}
    pulay_summary: dict[str, Any] | None = None
    if pulay_record_dir is not None:
        try:
            records = read_records(pulay_record_dir)
        except PulayRecordError as exc:
            if strict:
                raise SCFTrajectoryError(f"{task_id}: {exc}") from exc
            records = []
        if records:
            pulay_summary = verify_records(records)
            if pulay_summary["failed_iterations"] and strict:
                raise SCFTrajectoryError(
                    f"{task_id}: recorded Pulay matrices do not reproduce CP2K's own "
                    f"coefficients at iterations {pulay_summary['failed_iterations']}; "
                    "the export is not the matrix CP2K solved"
                )
            pulay_by_iteration = {record.iteration: record for record in records}

    recorded: list[TrajectoryFrame] = []
    for index, density in enumerate(frames):
        is_final = index == final_index
        label = FINAL_DIR_NAME if is_final else f"iter_{index:04d}"
        try:
            counted = validate_frame(
                density, overlap, expected_electrons=expected_electrons, label=f"{task_id}/{label}"
            )
        except SCFTrajectoryError:
            if strict:
                raise
            return None

        frame_dir = directory / label
        frame_dir.mkdir(parents=True, exist_ok=True)
        density_path = frame_dir / DENSITY_NAME
        np.save(density_path, np.asarray(density, dtype=float))

        entry = paired[index] if index < len(paired) else None
        # CP2K numbers its mixing calls from 1 and only mixes once a history
        # exists, so early frames legitimately have no matrix.
        record = None
        if entry is not None:
            record = pulay_by_iteration.get(int(entry.step))
        if record is not None:
            np.save(frame_dir / PULAY_MATRIX_NAME, np.asarray(record.b_matrix, dtype=float))

        frame_metadata = {
            "index": index,
            "iteration": None if entry is None else int(entry.step),
            "residual": None if entry is None else entry.residual,
            "energy": None if entry is None else entry.energy,
            "electron_count": counted,
            "electron_count_method": "trace_ps",
            "representation": REPRESENTATION_AO_NONORTHOGONAL,
            "matrix_format": MATRIX_FORMAT_AO_SPIN,
            "spin_channels": int(density.shape[0]),
            "n_ao": int(density.shape[1]),
            # Relative, so the directory stays movable.
            "overlap_path": f"../{OVERLAP_NAME}",
            "is_final": is_final,
            "task_id": task_id,
            # Which B this is. A consumer that finds anything other than
            # cp2k_pulay_metric is looking at a different quantity.
            "pulay_matrix_path": None if record is None else PULAY_MATRIX_NAME,
            "pulay_representation": None if record is None else REPRESENTATION_CP2K_PULAY,
            "pulay_metric": None if record is None else METRIC_GSPACE_KERKER,
            "pulay_depth": None if record is None else record.depth,
            "pulay_native_coefficients": None if record is None else record.alpha.tolist(),
            # Spectrum, rank and conditioning of this iteration's B. Recorded
            # per frame because the difficulty of the solve varies enormously
            # *within* one SCF: B is well conditioned while the history is
            # short and near-singular once the residuals become collinear.
            "pulay_diagnostics": None if record is None else b_matrix_diagnostics(record.b_matrix),
        }
        metadata_path = frame_dir / METADATA_NAME
        metadata_path.write_text(
            json.dumps(frame_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        recorded.append(
            TrajectoryFrame(
                index=index,
                iteration=frame_metadata["iteration"],
                residual=frame_metadata["residual"],
                energy=frame_metadata["energy"],
                electron_count=counted,
                density_path=density_path,
                metadata_path=metadata_path,
                is_final=is_final,
            )
        )

    metadata_path = directory / METADATA_NAME
    run_metadata = {
        "trajectory_format": TRAJECTORY_FORMAT,
        "representation": REPRESENTATION_AO_NONORTHOGONAL,
        "matrix_format": MATRIX_FORMAT_AO_SPIN,
        "electron_count_method": "trace_ps",
        "n_frames": len(recorded),
        "n_ao": int(frames[0].shape[1]),
        "spin_channels": int(frames[0].shape[0]),
        "n_electrons": expected_electrons,
        "overlap_path": OVERLAP_NAME,
        "scf_converged": bool(scf_converged),
        "cp2k_version": cp2k_version,
        "functional": cp2k_cfg.get("functional"),
        "basis_set": cp2k_cfg.get("basis"),
        "potential": cp2k_cfg.get("potential"),
        "charge": state.get("charge"),
        "multiplicity": state.get("multiplicity"),
        "task_id": task_id,
        "pulay_representation": REPRESENTATION_CP2K_PULAY if pulay_by_iteration else None,
        "pulay_metric": METRIC_GSPACE_KERKER if pulay_by_iteration else None,
        "pulay_verification": pulay_summary,
        "n_pulay_matrices": len(pulay_by_iteration),
        "frames": [frame.to_dict() for frame in recorded],
    }
    metadata_path.write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return SCFTrajectoryArtifacts(
        directory=directory,
        overlap_path=overlap_path,
        metadata_path=metadata_path,
        frames=tuple(recorded),
        n_ao=int(frames[0].shape[1]),
        spin_channels=int(frames[0].shape[0]),
        n_electrons=expected_electrons,
    )


def load_trajectory(directory: str | Path) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any]]:
    """Load a recorded trajectory as ``(densities, overlap, metadata)``.

    ``densities`` is ordered by SCF iteration, with the converged density last
    -- the ``[P0, P1, ..., Pfinal]`` sequence a mixer trains on. Ordering comes
    from the recorded frame index, not from directory listing order, so it does
    not depend on how the filesystem sorts names.
    """

    source = Path(directory)
    if source.name != TRAJECTORY_DIR_NAME and (source / TRAJECTORY_DIR_NAME).is_dir():
        source = source / TRAJECTORY_DIR_NAME

    metadata_path = source / METADATA_NAME
    if not metadata_path.is_file():
        raise SCFTrajectoryError(f"{source}: no {METADATA_NAME}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    overlap_path = source / str(metadata.get("overlap_path", OVERLAP_NAME))
    if not overlap_path.is_file():
        raise SCFTrajectoryError(f"{source}: no overlap at {overlap_path}")
    overlap = np.asarray(np.load(overlap_path), dtype=float)

    entries: Sequence[Mapping[str, Any]] = metadata.get("frames", ())
    if not entries:
        raise SCFTrajectoryError(f"{source}: metadata lists no frames")

    densities: list[np.ndarray] = []
    for entry in sorted(entries, key=lambda item: int(item["index"])):
        path = Path(str(entry["density_path"]))
        if not path.is_absolute():
            path = source / path
        if not path.is_file():
            # Recorded elsewhere and moved: fall back to the layout.
            label = FINAL_DIR_NAME if entry.get("is_final") else f"iter_{int(entry['index']):04d}"
            path = source / label / DENSITY_NAME
        if not path.is_file():
            raise SCFTrajectoryError(f"{source}: frame {entry['index']} has no density file")
        densities.append(np.asarray(np.load(path), dtype=float))

    return densities, overlap, metadata
