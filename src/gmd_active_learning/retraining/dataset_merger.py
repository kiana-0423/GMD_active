"""Merge labeled samples into a new immutable dataset version (ledger DS-002).

The previous placeholder wrote a two-line text manifest containing the literal string
``existing_dataset`` (finding F-10). ``merge_datasets`` keeps its name and argument order but now
publishes through :class:`~gmd_active_learning.datasets.store.VersionedDatasetStore`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from gmd_active_learning.contracts import DatasetVersion
from gmd_active_learning.core.exceptions import DataContractError
from gmd_active_learning.datasets.config import DatasetsConfig
from gmd_active_learning.datasets.store import VersionedDatasetStore, collect_sample_files

LOGGER = logging.getLogger(__name__)


def merge_into_store(
    store: VersionedDatasetStore,
    labeled_data_dir: str | Path,
    *,
    parent: DatasetVersion | None = None,
) -> DatasetVersion:
    """Publish every canonical sample file under ``labeled_data_dir`` on top of ``parent``."""

    paths = collect_sample_files(labeled_data_dir)
    if not paths and parent is None:
        raise DataContractError(f"no labeled samples under {labeled_data_dir} and no parent dataset version")
    if not paths:
        return parent  # type: ignore[return-value]
    return store.publish(paths, parent=parent)


def merge_datasets(
    existing_dataset: str | None,
    labeled_data_dir: str | Path,
    output_dir: str | Path,
    config: DatasetsConfig | None = None,
) -> Path:
    """Publish samples from ``labeled_data_dir`` into the store at ``output_dir``.

    ``existing_dataset`` is the parent version ID in that store, or ``None`` for a first version;
    any other value raises (no placeholder parents). Returns the new version directory.
    """

    store = VersionedDatasetStore(output_dir, config)
    parent = None if existing_dataset in (None, "") else store.get(str(existing_dataset))
    version = merge_into_store(store, labeled_data_dir, parent=parent)
    LOGGER.info("published dataset version %s (%d structures)", version.version, version.n_structures)
    return Path(version.path)


__all__ = ["merge_datasets", "merge_into_store"]
