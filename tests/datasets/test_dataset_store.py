"""Versioned, immutable dataset store and the workflow DatasetStore protocol."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from gmd_active_learning.core.exceptions import DataContractError
from gmd_active_learning.datasets import DatasetsConfig, VersionedDatasetStore, assign_groups, read_samples_extxyz
from gmd_active_learning.datasets.config import SplitConfig
from gmd_active_learning.retraining.dataset_merger import merge_datasets

from datasets.helpers import capped_cluster, label_for, sample_for


def many_samples(n_events: int = 6, per_event: int = 3):
    samples = []
    for event in range(n_events):
        for k in range(per_event):
            samples.append(sample_for(step=event * 10 + k, event_id=f"evt-{event}", task=f"task-{event}-{k}"))
    return samples


def publish_samples(store: VersionedDatasetStore, samples, parent=None):
    paths = [store.write_sample(sample) for sample in samples]
    return store.publish(paths, parent=parent, split_groups={})


def test_publish_is_versioned_immutable_and_idempotent(tmp_path) -> None:
    store = VersionedDatasetStore(tmp_path / "store")
    samples = many_samples()
    version = publish_samples(store, samples)
    assert version.n_structures == len(samples) == version.n_train + version.n_val and version.n_val > 0
    assert set(version.sample_ids) == {s.sample_id for s in samples} and version.parent_version is None
    assert store.get(version.version).sha256 == version.sha256
    again = publish_samples(store, list(reversed(samples)))
    assert again.version == version.version and again.sha256 == version.sha256
    assert store.versions() == [version.version]
    files = [Path(version.path) / name for name in ("samples.npz", "all.extxyz", "train.extxyz", "val.extxyz", "sgt.npz", "split.json")]
    assert all(path.is_file() and not os.access(path, os.W_OK) for path in files)
    for key in ("train_file", "val_file"):
        assert Path(version.metadata[key]).is_absolute() and Path(version.metadata[key]).is_file()


def test_split_keeps_whole_events_together(tmp_path) -> None:
    store = VersionedDatasetStore(tmp_path / "store")
    version = publish_samples(store, many_samples(8, 4))
    sides: dict[str, set[str]] = {}
    for name, side in (("train_file", "train"), ("val_file", "val")):
        for sample in read_samples_extxyz(version.metadata[name]):
            sides.setdefault(sample.split_group, set()).add(side)
    assert all(len(value) == 1 for value in sides.values())
    assert {"train", "val"} == set().union(*sides.values())
    # Deterministic and seed-dependent.
    groups = [f"event:evt-{k}" for k in range(50)]
    assert assign_groups(groups, SplitConfig(seed=1)) == assign_groups(groups, SplitConfig(seed=1))
    assert assign_groups(groups, SplitConfig(seed=1)) != assign_groups(groups, SplitConfig(seed=2))
    assert set(assign_groups(groups, SplitConfig(enabled=False)).values()) == {"train"}


def test_child_versions_are_cumulative_and_never_move_groups(tmp_path) -> None:
    store = VersionedDatasetStore(tmp_path / "store")
    first = publish_samples(store, many_samples(6, 2))
    new = [sample_for(step=900 + k, event_id=f"evt-new-{k}", task=f"new-{k}") for k in range(4)]
    child = publish_samples(store, [*many_samples(6, 2), *new], parent=first)  # parent samples are deduplicated
    assert child.parent_version == first.version and child.n_structures == first.n_structures + 4
    assert child.sample_ids[: first.n_structures] == first.sample_ids
    parent_split = store.split_of(first)["groups"]
    child_split = store.split_of(child)["groups"]
    assert all(child_split[group] == side for group, side in parent_split.items())
    assert publish_samples(store, many_samples(6, 2), parent=first) is first  # nothing new -> parent


def test_tampering_is_detected(tmp_path) -> None:
    store = VersionedDatasetStore(tmp_path / "store")
    version = publish_samples(store, many_samples(2, 1))
    target = Path(version.path) / "all.extxyz"
    target.chmod(stat.S_IRUSR | stat.S_IWUSR)
    target.write_text(target.read_text().replace("event", "EVENT"))
    with pytest.raises(DataContractError, match="modified"):
        store.get(version.version)


def test_protocol_add_sample_is_idempotent_and_publish_honours_split_groups(tmp_path) -> None:
    store = VersionedDatasetStore(tmp_path / "store")
    _frame, region = capped_cluster()
    label = label_for(region, "task-x")
    sid, path = store.add_sample(region, label, provenance={"candidate_id": "c1"})
    assert store.add_sample(region, label, provenance={"candidate_id": "c1"}) == (sid, path)
    with pytest.raises(DataContractError, match="different label"):
        store.add_sample(region, label_for(region, "task-x", seed=99), provenance={})
    other = sample_for(step=5, event_id="evt-b", task="task-y")
    other_path = store.write_sample(other)
    version = store.publish([path, other_path], parent=None, split_groups={sid: "custom-group", other.sample_id: "custom-group"})
    samples = store.read_samples(version)
    assert {s.split_group for s in samples} == {"custom-group"}
    assert samples[[s.sample_id for s in samples].index(sid)].metadata["provenance"] == {"candidate_id": "c1"}
    assert store.get(version.version) == version


def test_merge_datasets_publishes_through_the_store(tmp_path) -> None:
    labeled = tmp_path / "labeled"
    staging = VersionedDatasetStore(labeled)
    for sample in many_samples(3, 1):
        staging.write_sample(sample)
    out = merge_datasets(None, labeled, tmp_path / "datasets")
    record = json.loads(next((tmp_path / "datasets" / "registry").glob("*.json")).read_text())
    assert out == Path(record["path"]) and record["n_structures"] == 3
    with pytest.raises(DataContractError, match="not in store"):
        merge_datasets("existing_dataset", labeled, tmp_path / "datasets")  # no placeholder parents (F-10)
    with pytest.raises(DataContractError, match="no labeled samples"):
        merge_datasets(None, tmp_path / "empty", tmp_path / "datasets2")


def test_store_disabled_and_empty_publish(tmp_path) -> None:
    with pytest.raises(DataContractError, match="disabled"):
        VersionedDatasetStore(tmp_path, DatasetsConfig(enabled=False))
    with pytest.raises(DataContractError, match="nothing to publish"):
        VersionedDatasetStore(tmp_path).publish([], parent=None, split_groups={})


def test_export_switches(tmp_path) -> None:
    store = VersionedDatasetStore(tmp_path / "s", DatasetsConfig(export={"sgt_npz": False, "split_files": False}))
    version = publish_samples(store, many_samples(2, 1))
    assert not (Path(version.path) / "sgt.npz").exists() and not (Path(version.path) / "train.extxyz").exists()
    assert "train_file" not in version.metadata and Path(version.metadata["all_file"]).is_file()
