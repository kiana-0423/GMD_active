"""Versioned, immutable dataset store.

Layout below ``root``::

    samples/<sample_id>.npz|.json      one canonical sample each (add_sample / write_sample)
    versions/<version>/                immutable, read-only once published
        samples.npz, samples.json      canonical archive (bit-exact round trip)
        all.extxyz, train.extxyz, val.extxyz   GMD_SGT-readable exports (export.* switches)
        sgt.npz                        GMD_SGT NPZ export
        split.json                     sample -> side, group -> side
    registry/<version>.json            the DatasetVersion record (sha256 over versions/<version>)
    index.json                         publish order

A version is cumulative: parent samples plus the new ones. Its identifier is derived from the
parent version, the ordered sample IDs, their split groups and the export settings, so publishing
the same samples again returns the existing version (no duplicates), and publishing only samples
the parent already holds returns the parent. Version directories are staged and renamed into
place, then made read-only; :meth:`get` re-verifies the checksum. One writer per store is assumed.

PHAL merged every labeled extxyz into one overwritten ``train.extxyz`` with no dedup, split or
version (ledger DS-002); GMD_active's previous merger wrote only a two-line manifest (F-10).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gmd_active_learning.contracts import DatasetVersion, ExtractedRegion, LabelResult, stable_id
from gmd_active_learning.contracts.ids import tree_sha256
from gmd_active_learning.core.exceptions import DataContractError
from gmd_active_learning.datasets.config import DatasetsConfig
from gmd_active_learning.datasets.extxyz_io import write_samples_extxyz
from gmd_active_learning.datasets.npz_io import read_samples_npz, sidecar_path, write_samples_npz, write_sgt_npz
from gmd_active_learning.datasets.samples import LabeledSample, build_sample
from gmd_active_learning.datasets.split import TRAIN, VAL, assign_groups
from gmd_active_learning.utils.atomic_io import read_json_or_default, write_json_atomic

LOGGER = logging.getLogger(__name__)

STORE_FORMAT_VERSION = 1


def _make_read_only(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def collect_sample_files(directory: str | Path) -> list[Path]:
    """Canonical single-sample archives (``*.npz`` with a ``.json`` sidecar) below ``directory``, sorted."""

    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*.npz")
        if sidecar_path(path).is_file() and "versions" not in path.relative_to(root).parts
    )


class VersionedDatasetStore:
    """Implements the workflow ``DatasetStore`` protocol (``add_sample`` / ``publish`` / ``get``)."""

    def __init__(self, root: str | Path, config: DatasetsConfig | None = None) -> None:
        self.root = Path(root)
        self.config = config or DatasetsConfig()
        if not self.config.enabled:
            raise DataContractError("datasets.enabled is false; the dataset store is disabled by config")
        self.samples_dir = self.root / "samples"
        self.versions_dir = self.root / "versions"
        self.registry_dir = self.root / "registry"

    # ------------------------------------------------------------------ samples
    def sample_path(self, sample_id: str) -> Path:
        return self.samples_dir / f"{sample_id}.npz"

    def write_sample(self, sample: LabeledSample) -> Path:
        """Idempotent: an existing sample with the same ID must carry the same label content."""

        path = self.sample_path(sample.sample_id)
        if path.exists():
            existing = read_samples_npz(path)[0]
            old = existing.metadata.get("label_content_hash")
            new = sample.metadata.get("label_content_hash")
            if old is not None and new is not None and old != new:
                raise DataContractError(
                    f"sample {sample.sample_id} already stored with a different label (region {sample.region.region_id}, "
                    f"task {sample.task_id}); a relabel needs a new task_id"
                )
            return path
        write_samples_npz([sample], path)
        return path

    def add_sample(self, region: ExtractedRegion, label: LabelResult, *, provenance: dict[str, Any]) -> tuple[str, Path]:
        sample = build_sample(region, label, self.config, metadata={"provenance": dict(provenance)})
        return sample.sample_id, self.write_sample(sample)

    @staticmethod
    def load_sample(path: str | Path) -> LabeledSample:
        samples = read_samples_npz(path)
        if len(samples) != 1:
            raise DataContractError(f"{path} holds {len(samples)} samples, expected one")
        return samples[0]

    # ------------------------------------------------------------------ versions
    def _registry_path(self, version: str) -> Path:
        return self.registry_dir / f"{version}.json"

    def versions(self) -> list[str]:
        return list(read_json_or_default(self.root / "index.json", {"versions": []}, logger=LOGGER)["versions"])

    def get(self, version: str, *, verify: bool = True) -> DatasetVersion:
        path = self._registry_path(version)
        if not path.is_file():
            raise DataContractError(f"dataset version {version!r} is not in store {self.root}")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["sample_ids"] = tuple(data["sample_ids"])
        record = DatasetVersion(**data)
        if verify:
            self.verify(record)
        return record

    def verify(self, record: DatasetVersion) -> None:
        actual = tree_sha256(record.path)
        if actual != record.sha256:
            raise DataContractError(f"dataset version {record.version} was modified: sha256 {actual} != {record.sha256}")

    def read_samples(self, version: str | DatasetVersion) -> list[LabeledSample]:
        record = self.get(version) if isinstance(version, str) else version
        return read_samples_npz(Path(record.path) / "samples.npz")

    def split_of(self, version: str | DatasetVersion) -> dict[str, Any]:
        record = self.get(version) if isinstance(version, str) else version
        return json.loads((Path(record.path) / "split.json").read_text(encoding="utf-8"))

    def publish(
        self,
        sample_paths: list[Path],
        *,
        parent: DatasetVersion | None,
        split_groups: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetVersion:
        """Publish ``parent`` + the samples at ``sample_paths`` as an immutable version.

        ``split_groups`` (sample_id -> group) overrides the groups recorded on new samples; samples
        already in the parent keep their group so their side never changes.
        """

        samples = [self.load_sample(path) for path in sample_paths]
        return self.publish_samples(samples, parent=parent, split_groups=split_groups, metadata=metadata)

    def publish_samples(
        self,
        samples: list[LabeledSample],
        *,
        parent: DatasetVersion | None = None,
        split_groups: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetVersion:
        split_groups = dict(split_groups or {})
        parent_samples = self.read_samples(parent) if parent is not None else []
        inherited = self.split_of(parent)["groups"] if parent is not None else {}
        known = {sample.sample_id for sample in parent_samples}
        fresh: dict[str, LabeledSample] = {}
        for sample in samples:
            if sample.sample_id in known:
                if sample.sample_id in split_groups:
                    LOGGER.info("sample %s already published; its split group is kept", sample.sample_id)
                continue
            if sample.sample_id in split_groups:
                sample.split_group = str(split_groups[sample.sample_id])
            fresh.setdefault(sample.sample_id, sample)
        if not fresh:
            if parent is None:
                raise DataContractError("nothing to publish: no samples and no parent version")
            LOGGER.info("all samples already in %s; returning it unchanged", parent.version)
            return parent
        combined = parent_samples + [fresh[key] for key in sorted(fresh)]
        groups = assign_groups([sample.split_group for sample in combined], self.config.split, inherited=inherited)
        settings = {
            "store_format": STORE_FORMAT_VERSION,
            "split": self.config.split.model_dump(mode="json"),
            "export": self.config.export.model_dump(mode="json"),
            "float_format": self.config.float_format,
        }
        version = stable_id(
            "dsv",
            None if parent is None else parent.version,
            [sample.sample_id for sample in combined],
            [sample.split_group for sample in combined],
            settings,
        )
        if self._registry_path(version).is_file():
            LOGGER.info("dataset version %s already published", version)
            return self.get(version)
        target = self.versions_dir / version
        if not target.exists():
            self._stage(version, combined, groups, settings)
        record = self._register(version, target, combined, groups, parent, metadata or {})
        return record

    def _stage(self, version: str, samples: list[LabeledSample], groups: dict[str, str], settings: dict[str, Any]) -> None:
        staging = self.root / f".staging-{version}-{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            write_samples_npz(samples, staging / "samples.npz")
            sides = {sample.sample_id: groups[sample.split_group] for sample in samples}
            fmt = self.config.float_format
            export = self.config.export
            if export.extxyz:
                write_samples_extxyz(samples, staging / "all.extxyz", float_format=fmt)
            if export.split_files:
                write_samples_extxyz([s for s in samples if sides[s.sample_id] == TRAIN], staging / "train.extxyz", float_format=fmt)
                write_samples_extxyz([s for s in samples if sides[s.sample_id] == VAL], staging / "val.extxyz", float_format=fmt)
            if export.sgt_npz:
                write_sgt_npz(samples, staging / "sgt.npz")
            write_json_atomic(staging / "split.json", {"samples": sides, "groups": groups, "settings": settings})
            _make_read_only(staging)
            self.versions_dir.mkdir(parents=True, exist_ok=True)
            os.replace(staging, self.versions_dir / version)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _register(
        self,
        version: str,
        target: Path,
        samples: list[LabeledSample],
        groups: dict[str, str],
        parent: DatasetVersion | None,
        metadata: dict[str, Any],
    ) -> DatasetVersion:
        split = json.loads((target / "split.json").read_text(encoding="utf-8"))["samples"]
        n_val = sum(1 for side in split.values() if side == VAL)
        absolute = target.resolve()
        files = {
            name: str(absolute / filename)
            for name, filename in (
                ("train_file", "train.extxyz"),
                ("val_file", "val.extxyz"),
                ("all_file", "all.extxyz"),
                ("sgt_npz", "sgt.npz"),
                ("samples_file", "samples.npz"),
                ("split_file", "split.json"),
            )
            if (absolute / filename).is_file()
        }
        record = DatasetVersion(
            version=version,
            path=str(absolute),
            sha256=tree_sha256(target),
            n_structures=len(samples),
            n_train=len(samples) - n_val,
            n_val=n_val,
            parent_version=None if parent is None else parent.version,
            sample_ids=tuple(sample.sample_id for sample in samples),
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={
                **files,
                "format": "gmd_active_learning.datasets/v1",
                "n_groups": len(groups),
                "n_val_groups": sum(1 for side in groups.values() if side == VAL),
                "energy_policy": self.config.energy.model_dump(mode="json"),
                "stress_policy": self.config.stress.model_dump(mode="json"),
                **metadata,
            },
        )
        write_json_atomic(self._registry_path(version), record.to_dict())
        index = read_json_or_default(self.root / "index.json", {"versions": []}, logger=LOGGER)
        if version not in index["versions"]:
            index["versions"].append(version)
            write_json_atomic(self.root / "index.json", index)
        return record


__all__ = ["STORE_FORMAT_VERSION", "VersionedDatasetStore", "collect_sample_files"]
