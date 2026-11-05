# Ported from PHAL (hotspot_al.cp2k_ml_scf.dataset) at 96263b6; MIT License, see NOTICE.
"""On-disk SCF dataset: layout, ingestion, and split management.

Layout::

    runtime/cp2k/scf_dataset/
    ├── metadata.json
    ├── train/<sample_id>.json
    ├── validation/<sample_id>.json
    └── test/<sample_id>.json

Splits are assigned from a hash of the *structure*, not from a random number
generator.  Re-ingesting the same system therefore always lands in the same
split, which is what keeps a validation set honest when the dataset grows
incrementally over an active-learning campaign.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gmd_active_learning.adapters.dft.scf.schema import (
    SCHEMA_VERSION,
    SCFDensityArtifacts,
    SCFElectronicState,
    SCFSample,
    SCFStructure,
    SCFTrajectory,
    SchemaError,
)
from gmd_active_learning.adapters.dft.scf.validation import (
    ValidationReport,
    summarize,
    validate_payload,
)
from gmd_active_learning.utils.atomic_io import write_json_atomic

DATASET_VERSION = "phal-scf-dataset-v1"
DEFAULT_DATASET_ROOT = Path("runtime/cp2k/scf_dataset")
SPLITS: tuple[str, ...] = ("train", "validation", "test")
DEFAULT_SPLIT_RATIOS: Mapping[str, float] = {"train": 0.8, "validation": 0.1, "test": 0.1}

_SAMPLE_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


class DatasetError(RuntimeError):
    """Raised when the dataset directory is unusable or misused."""


@dataclass(frozen=True, slots=True)
class DatasetEntry:
    """One sample located on disk."""

    split: str
    path: Path
    sample: SCFSample


class SCFDataset:
    """A directory-backed collection of :class:`SCFSample` records."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # -- construction ---------------------------------------------------

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        description: str = "CP2K SCF trajectories for ML-accelerated SCF.",
        split_ratios: Mapping[str, float] | None = None,
        exist_ok: bool = True,
    ) -> "SCFDataset":
        """Create the directory layout and write ``metadata.json``."""

        dataset = cls(root)
        if dataset.metadata_path.exists() and not exist_ok:
            raise DatasetError(f"dataset already exists: {dataset.root}")
        ratios = _normalized_ratios(split_ratios or DEFAULT_SPLIT_RATIOS)
        for split in SPLITS:
            (dataset.root / split).mkdir(parents=True, exist_ok=True)
        if not dataset.metadata_path.exists():
            write_json_atomic(
                dataset.metadata_path,
                {
                    "dataset_version": DATASET_VERSION,
                    "schema_version": SCHEMA_VERSION,
                    "description": description,
                    "splits": list(SPLITS),
                    "split_ratios": dict(ratios),
                    "n_samples": {split: 0 for split in SPLITS},
                },
            )
        return dataset

    @classmethod
    def open(cls, root: str | Path) -> "SCFDataset":
        """Open an existing dataset, failing loudly if it is not one."""

        dataset = cls(root)
        if not dataset.metadata_path.is_file():
            raise DatasetError(f"not an SCF dataset (no metadata.json): {dataset.root}")
        return dataset

    # -- paths ----------------------------------------------------------

    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.json"

    def split_dir(self, split: str) -> Path:
        if split not in SPLITS:
            raise DatasetError(f"unknown split {split!r}; expected one of {', '.join(SPLITS)}")
        return self.root / split

    def sample_path(self, sample_id: str, split: str) -> Path:
        return self.split_dir(split) / f"{sanitize_sample_id(sample_id)}.json"

    # -- metadata -------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        if not self.metadata_path.is_file():
            raise DatasetError(f"not an SCF dataset (no metadata.json): {self.root}")
        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"dataset metadata is corrupt: {self.metadata_path} ({exc})") from exc
        if not isinstance(payload, dict):
            raise DatasetError(f"dataset metadata must be a JSON object: {self.metadata_path}")
        return payload

    @property
    def split_ratios(self) -> Mapping[str, float]:
        ratios = self.metadata.get("split_ratios", DEFAULT_SPLIT_RATIOS)
        return _normalized_ratios(ratios if isinstance(ratios, Mapping) else DEFAULT_SPLIT_RATIOS)

    def refresh_metadata(self) -> dict[str, Any]:
        """Recount the splits on disk and persist the result."""

        payload = self.metadata
        payload["n_samples"] = {split: len(list(self._split_files(split))) for split in SPLITS}
        write_json_atomic(self.metadata_path, payload)
        return payload

    # -- writing --------------------------------------------------------

    def add_sample(self, sample: SCFSample, *, split: str | None = None, overwrite: bool = False) -> Path:
        """Write one sample, assigning a deterministic split when unspecified."""

        target_split = split or self.assign_split(sample)
        path = self.sample_path(sample.sample_id, target_split)
        if path.exists() and not overwrite:
            raise DatasetError(f"sample already exists: {path} (pass overwrite=True to replace)")
        # A sample that moved split (because ratios changed) must not be duplicated.
        for other in SPLITS:
            if other == target_split:
                continue
            stale = self.sample_path(sample.sample_id, other)
            if stale.exists():
                stale.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)
        return write_json_atomic(path, sample.to_dict())

    def assign_split(self, sample: SCFSample) -> str:
        """Assign a split deterministically from the sample's structure hash."""

        return assign_split(sample.structure_hash, self.split_ratios)

    # -- reading --------------------------------------------------------

    def iter_entries(self, split: str | None = None) -> Iterator[DatasetEntry]:
        """Yield decoded samples, raising :class:`SchemaError` on bad records."""

        for split_name, path in self.iter_files(split):
            yield DatasetEntry(split=split_name, path=path, sample=self._load(path))

    def iter_files(self, split: str | None = None) -> Iterator[tuple[str, Path]]:
        selected = SPLITS if split is None else (split,)
        for split_name in selected:
            for path in self._split_files(split_name):
                yield split_name, path

    def samples(self, split: str | None = None) -> list[SCFSample]:
        return [entry.sample for entry in self.iter_entries(split)]

    def sample_ids(self, split: str | None = None) -> list[str]:
        return [path.stem for _, path in self.iter_files(split)]

    def counts(self) -> dict[str, int]:
        return {split: len(list(self._split_files(split))) for split in SPLITS}

    def _split_files(self, split: str) -> Iterator[Path]:
        directory = self.split_dir(split)
        if not directory.is_dir():
            return
        yield from sorted(directory.glob("*.json"))

    def _load(self, path: Path) -> SCFSample:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SchemaError(f"{path}: invalid JSON ({exc})") from exc
        return SCFSample.from_dict(payload)

    # -- validation and inspection --------------------------------------

    def validate(self, split: str | None = None) -> list[ValidationReport]:
        """Validate every record without letting one bad file abort the pass."""

        reports: list[ValidationReport] = []
        for split_name, path in self.iter_files(split):
            subject = f"{split_name}/{path.name}"
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                reports.append(_decode_failure(subject, f"invalid JSON ({exc})"))
                continue
            except UnicodeDecodeError as exc:
                reports.append(_decode_failure(subject, f"unreadable file ({exc})"))
                continue
            reports.append(validate_payload(payload, subject=subject))
        return reports

    def inspect(self, split: str | None = None) -> dict[str, Any]:
        """Summarise the dataset's contents for the CLI and for provenance."""

        counts = self.counts()
        n_atoms: list[int] = []
        iterations: list[int] = []
        converged = 0
        decoded = 0
        undecodable: list[str] = []
        guesses: dict[str, int] = {}
        elements: dict[int, int] = {}
        with_artifacts = 0

        for split_name, path in self.iter_files(split):
            try:
                sample = self._load(path)
            except SchemaError as exc:
                undecodable.append(f"{split_name}/{path.name}: {exc}")
                continue
            decoded += 1
            n_atoms.append(sample.structure.n_atoms)
            iterations.append(sample.trajectory.n_iterations)
            converged += 1 if sample.trajectory.converged else 0
            guess = sample.trajectory.initial_guess or "UNKNOWN"
            guesses[guess] = guesses.get(guess, 0) + 1
            for number in sample.structure.atomic_numbers:
                elements[int(number)] = elements.get(int(number), 0) + 1
            if not sample.artifacts.is_empty:
                with_artifacts += 1

        return {
            "root": str(self.root),
            "dataset_version": self.metadata.get("dataset_version"),
            "schema_version": self.metadata.get("schema_version"),
            "split_ratios": dict(self.split_ratios),
            "counts": counts,
            "n_total": sum(counts.values()),
            "n_decoded": decoded,
            "n_undecodable": len(undecodable),
            "undecodable": undecodable,
            "atoms": _statistics(n_atoms),
            "scf_iterations": _statistics(iterations),
            "converged_fraction": (converged / decoded) if decoded else 0.0,
            "initial_guess_counts": dict(sorted(guesses.items())),
            "element_counts": dict(sorted(elements.items())),
            "n_with_density_artifacts": with_artifacts,
        }

    def validation_summary(self, split: str | None = None) -> dict[str, Any]:
        return summarize(self.validate(split))


# -- ingestion ----------------------------------------------------------


def sample_from_scf_history(
    history: Mapping[str, Any],
    *,
    sample_id: str,
    structure: SCFStructure,
    electronic_state: SCFElectronicState,
    artifacts: SCFDensityArtifacts | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SCFSample:
    """Build a sample from a ``scf_history.json`` payload plus its structure."""

    if not isinstance(history, Mapping):
        raise SchemaError(f"scf_history must be a mapping, got {type(history).__name__}")
    trajectory = SCFTrajectory.from_dict(history)
    return SCFSample(
        sample_id=sample_id,
        structure=structure,
        electronic_state=electronic_state,
        trajectory=trajectory,
        artifacts=artifacts or SCFDensityArtifacts(),
        metadata=dict(metadata or {}),
    )


def ingest_scf_run(
    scf_history_path: str | Path,
    *,
    structure_path: str | Path,
    config: Mapping[str, Any] | None = None,
    sample_id: str | None = None,
    region_metadata: Mapping[str, Any] | None = None,
) -> SCFSample:
    """Build a sample from a recorded SCF run and its ``region.extxyz``.

    ASE is imported lazily so that the schema, validation and export layers stay
    usable in environments where only the JSON side of the dataset matters.
    """

    from ase.io import read

    from gmd_active_learning.utils.electronic_state import resolve_electronic_state

    history_file = Path(scf_history_path)
    payload = json.loads(history_file.read_text(encoding="utf-8"))
    atoms = read(str(structure_path))
    dft_cfg = dict((config or {}).get("cp2k", {})) if isinstance(config, Mapping) else {}
    state = resolve_electronic_state(atoms, dft_cfg, region_metadata=dict(region_metadata or {}))

    return sample_from_scf_history(
        payload,
        sample_id=sample_id or history_file.parent.name,
        structure=SCFStructure.from_atoms(atoms),
        electronic_state=SCFElectronicState.from_electronic_state(state),
        artifacts=discover_density_artifacts(history_file.parent),
        metadata={
            "source_scf_history": str(history_file),
            "source_structure": str(structure_path),
        },
    )


def discover_density_artifacts(task_dir: str | Path) -> SCFDensityArtifacts:
    """Find a recorded density beside an SCF history, if one exists.

    Returns empty artifacts when no density was recorded -- that is the normal
    case, not an error, since density recording is opt-in.
    """

    from gmd_active_learning.adapters.dft.scf.recording.density import (
        DENSITY_DIR_NAME,
        DENSITY_MATRIX_NAME,
        DENSITY_METADATA_NAME,
        MATRIX_FORMAT_AO_SPIN,
        OVERLAP_MATRIX_NAME,
    )
    from gmd_active_learning.adapters.dft.scf.recording.trajectory import (
        METADATA_NAME as TRAJECTORY_METADATA_NAME,
        TRAJECTORY_DIR_NAME,
    )

    directory = Path(task_dir) / DENSITY_DIR_NAME
    density_path = directory / DENSITY_MATRIX_NAME
    if not density_path.is_file():
        return SCFDensityArtifacts()

    metadata: dict[str, Any] = {}
    metadata_path = directory / DENSITY_METADATA_NAME
    if metadata_path.is_file():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                metadata = dict(loaded)
        except json.JSONDecodeError:
            # A damaged metadata file leaves the artifact declared but
            # unverifiable; validation rejects it rather than guessing.
            metadata = {}

    # The SCF trajectory sits beside the density, not inside it: it is a
    # separate, much larger artifact and is recorded independently.
    trajectory_path: str | None = None
    trajectory_frames: int | None = None
    trajectory_dir = Path(task_dir) / TRAJECTORY_DIR_NAME
    trajectory_metadata = trajectory_dir / TRAJECTORY_METADATA_NAME
    if trajectory_metadata.is_file():
        try:
            loaded_trajectory = json.loads(trajectory_metadata.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded_trajectory = {}
        if isinstance(loaded_trajectory, Mapping):
            trajectory_path = str(trajectory_dir)
            trajectory_frames = _optional_int(loaded_trajectory.get("n_frames"))

    overlap_path = directory / OVERLAP_MATRIX_NAME
    return SCFDensityArtifacts(
        density_matrix_path=str(density_path),
        overlap_path=str(overlap_path) if overlap_path.is_file() else None,
        density_format=str(metadata.get("matrix_format", MATRIX_FORMAT_AO_SPIN)),
        n_ao=_optional_int(metadata.get("n_ao")),
        basis_set=_optional_text(metadata.get("basis_set")),
        density_electron_count=_optional_number(metadata.get("electron_count")),
        representation=_optional_text(metadata.get("representation")),
        electron_count_method=_optional_text(metadata.get("electron_count_method")),
        spin_channels=_optional_int(metadata.get("spin_channels")),
        trajectory_path=trajectory_path,
        trajectory_frames=trajectory_frames,
    )


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _optional_number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def discover_scf_runs(scf_root: str | Path) -> list[Path]:
    """Return every ``<task-id>/scf_history.json`` under ``scf_root``."""

    root = Path(scf_root)
    if not root.is_dir():
        return []
    return sorted(root.glob("*/scf_history.json"))


def ingest_scf_root(
    scf_root: str | Path,
    dataset: SCFDataset,
    *,
    task_root: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Ingest every recorded SCF run into ``dataset``.

    ``task_root`` is where CP2K task directories live (each holding the
    ``region.extxyz`` written at submission time).  It defaults to ``scf_root``
    so a colocated layout works without extra configuration.
    """

    tasks = Path(task_root) if task_root is not None else Path(scf_root)
    ingested: list[str] = []
    skipped: list[dict[str, str]] = []
    for history_path in discover_scf_runs(scf_root):
        task_id = history_path.parent.name
        structure_path = _find_structure(history_path.parent, tasks / task_id)
        if structure_path is None:
            skipped.append({"task_id": task_id, "reason": "no region.extxyz alongside the SCF history"})
            continue
        try:
            sample = ingest_scf_run(
                history_path,
                structure_path=structure_path,
                config=config,
                sample_id=task_id,
            )
            dataset.add_sample(sample, overwrite=overwrite)
        except (SchemaError, DatasetError, OSError, ValueError) as exc:
            skipped.append({"task_id": task_id, "reason": str(exc)})
            continue
        ingested.append(task_id)
    dataset.refresh_metadata()
    return {"n_ingested": len(ingested), "ingested": ingested, "n_skipped": len(skipped), "skipped": skipped}


def assign_split(structure_hash: str, ratios: Mapping[str, float] | None = None) -> str:
    """Map a structure hash onto a split, reproducibly."""

    normalized = _normalized_ratios(ratios or DEFAULT_SPLIT_RATIOS)
    digest = structure_hash.rsplit(":", 1)[-1]
    # 16 hex digits give ample resolution and stay inside a 64-bit integer.
    position = (int(digest[:16], 16) % 1_000_000) / 1_000_000
    cumulative = 0.0
    for split in SPLITS:
        cumulative += normalized.get(split, 0.0)
        if position < cumulative:
            return split
    return SPLITS[0]


def sanitize_sample_id(sample_id: str) -> str:
    """Return a filesystem-safe sample id, rejecting empty results."""

    cleaned = _SAMPLE_ID_SAFE.sub("_", str(sample_id)).strip("._-")
    if not cleaned:
        raise DatasetError(f"sample_id {sample_id!r} contains no usable characters")
    return cleaned


def _find_structure(*candidates: Path) -> Path | None:
    for directory in candidates:
        for name in ("region.extxyz", "structure.extxyz", "region.xyz"):
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def _decode_failure(subject: str, message: str) -> ValidationReport:
    from gmd_active_learning.adapters.dft.scf.validation import ERROR, ValidationIssue

    return ValidationReport(subject=subject, issues=(ValidationIssue(ERROR, "file", message),))


def _normalized_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    values = {split: float(ratios.get(split, 0.0)) for split in SPLITS}
    if any(value < 0.0 for value in values.values()):
        raise DatasetError(f"split ratios must be non-negative, got {ratios!r}")
    total = sum(values.values())
    if total <= 0.0:
        raise DatasetError(f"split ratios must sum to a positive value, got {ratios!r}")
    return {split: value / total for split, value in values.items()}


def _statistics(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": round(sum(values) / len(values), 4),
    }


__all__ = [
    "DATASET_VERSION",
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_SPLIT_RATIOS",
    "SPLITS",
    "DatasetEntry",
    "DatasetError",
    "SCFDataset",
    "assign_split",
    "discover_density_artifacts",
    "discover_scf_runs",
    "ingest_scf_root",
    "ingest_scf_run",
    "sample_from_scf_history",
    "sanitize_sample_id",
]
