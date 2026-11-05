# Ported from PHAL (hotspot_al.cp2k_ml_scf.export) at 96263b6; MIT License, see NOTICE.
"""Export an :class:`SCFDataset` into training-ready artifacts.

Two formats are produced side by side:

``<split>.jsonl``
    The raw samples, one JSON object per line.  Lossless, human-inspectable,
    and the format the training loader reads.

``<split>_features.npz``
    Pre-computed feature matrices and trajectory targets, so a training run
    does not have to re-featurise a large dataset on every epoch.

Every export writes ``export_manifest.json`` recording schema and feature
versions plus a digest of each artifact, so a trained model can be traced back
to the exact bytes it was trained on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from gmd_active_learning.adapters.dft.scf.dataset import SPLITS, SCFDataset
from gmd_active_learning.adapters.dft.scf.features import (
    FEATURE_VERSION,
    feature_metadata,
    featurize_sample,
    trajectory_targets,
)
from gmd_active_learning.adapters.dft.scf.schema import SCHEMA_VERSION, SCFSample
from gmd_active_learning.adapters.dft.scf.validation import validate_sample
from gmd_active_learning.utils.atomic_io import write_json_atomic

EXPORT_VERSION = "phal-scf-export-v1"


class ExportError(RuntimeError):
    """Raised when an export cannot be produced."""


@dataclass(frozen=True, slots=True)
class ExportResult:
    """What an export produced, for the CLI and for provenance."""

    output_dir: Path
    manifest_path: Path
    files: Mapping[str, Path]
    counts: Mapping[str, int]
    n_rejected: int
    rejected: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "manifest": str(self.manifest_path),
            "files": {key: str(path) for key, path in self.files.items()},
            "counts": dict(self.counts),
            "n_rejected": self.n_rejected,
            "rejected": list(self.rejected),
        }


def export_dataset(
    dataset: SCFDataset,
    output_dir: str | Path,
    *,
    splits: Sequence[str] = SPLITS,
    include_features: bool = True,
    skip_invalid: bool = True,
    require_converged: bool = False,
) -> ExportResult:
    """Export ``dataset`` into ``output_dir``.

    ``skip_invalid`` drops records that fail validation instead of aborting;
    the rejected ids are reported so a silent shrink is impossible to miss.
    ``require_converged`` additionally drops trajectories CP2K never converged,
    which is the right default for training but not for diagnosis.
    """

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}
    counts: dict[str, int] = {}
    rejected: list[str] = []

    for split in splits:
        accepted: list[SCFSample] = []
        for entry in dataset.iter_entries(split):
            report = validate_sample(entry.sample)
            if not report.ok:
                if not skip_invalid:
                    raise ExportError(
                        f"{split}/{entry.path.name} failed validation: "
                        + "; ".join(str(issue) for issue in report.errors)
                    )
                rejected.append(f"{split}/{entry.path.name}: {report.errors[0].message}")
                continue
            if require_converged and not entry.sample.trajectory.converged:
                rejected.append(f"{split}/{entry.path.name}: SCF did not converge")
                continue
            accepted.append(entry.sample)

        counts[split] = len(accepted)
        files[f"{split}_jsonl"] = write_jsonl(accepted, target / f"{split}.jsonl")
        if include_features:
            files[f"{split}_features"] = write_features_npz(accepted, target / f"{split}_features.npz")

    manifest_path = write_json_atomic(
        target / "export_manifest.json",
        {
            "export_version": EXPORT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "feature_version": FEATURE_VERSION,
            "source_dataset": str(dataset.root),
            "splits": list(splits),
            "counts": counts,
            "n_rejected": len(rejected),
            "rejected": rejected,
            "require_converged": require_converged,
            "features": feature_metadata() if include_features else None,
            "artifacts": {key: _digest(path) for key, path in sorted(files.items())},
        },
    )

    return ExportResult(
        output_dir=target,
        manifest_path=manifest_path,
        files=files,
        counts=counts,
        n_rejected=len(rejected),
        rejected=tuple(rejected),
    )


def write_jsonl(samples: Iterable[SCFSample], path: str | Path) -> Path:
    """Write samples as newline-delimited JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), sort_keys=True))
            handle.write("\n")
    return target


def read_jsonl(path: str | Path) -> list[SCFSample]:
    """Read samples back from a ``.jsonl`` export."""

    source = Path(path)
    samples: list[SCFSample] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ExportError(f"{source}:{line_number}: invalid JSON ({exc})") from exc
            samples.append(SCFSample.from_dict(payload))
    return samples


def write_features_npz(samples: Sequence[SCFSample], path: str | Path) -> Path:
    """Write feature matrices and trajectory targets to a compressed archive."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = feature_metadata()
    n_features = int(metadata["n_system_features"])

    if samples:
        system = np.vstack([featurize_sample(sample).flat() for sample in samples])
    else:
        system = np.zeros((0, n_features), dtype=float)
    targets = trajectory_targets(samples)

    np.savez_compressed(
        target,
        feature_version=np.asarray(FEATURE_VERSION),
        sample_ids=np.asarray([sample.sample_id for sample in samples], dtype=object),
        structure_hashes=np.asarray([sample.structure_hash for sample in samples], dtype=object),
        system_features=system,
        n_atoms=np.asarray([sample.structure.n_atoms for sample in samples], dtype=np.int64),
        **{key: value for key, value in targets.items()},
    )
    return target


def load_features_npz(path: str | Path) -> dict[str, Any]:
    """Load a features archive, verifying the feature version matches."""

    with np.load(path, allow_pickle=True) as archive:
        version = str(archive["feature_version"])
        if version != FEATURE_VERSION:
            raise ExportError(
                f"{path} was written with feature_version {version!r}; this build expects {FEATURE_VERSION!r}"
            )
        return {key: archive[key] for key in archive.files}


def _digest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "sha256": None, "bytes": 0}
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return {
        "path": str(path),
        "exists": True,
        "sha256": f"sha256:{hasher.hexdigest()}",
        "bytes": path.stat().st_size,
    }


__all__ = [
    "EXPORT_VERSION",
    "ExportError",
    "ExportResult",
    "export_dataset",
    "load_features_npz",
    "read_jsonl",
    "write_features_npz",
    "write_jsonl",
]
