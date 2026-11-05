# Ported from PHAL (hotspot_al.utils.config) at 96263b6; MIT License, see NOTICE.
"""YAML loading and recursive merging (ledger UTIL-005).

Only the generic helpers are ported. PHAL's ``load_config`` (packaged default merged with an
override, legacy ``backend.md_engine`` normalisation, schema validation) is replaced by
:mod:`gmd_active_learning.config`, which owns the schema; combine the two with::

    raw = merge_yaml_files("configs/default.yaml", "my_run.yaml")
    config = build_config(raw, base_dir=Path("my_run.yaml").parent)

Merge semantics (as PHAL): mappings merge recursively, every other value (lists included)
replaces the base value, and the result shares no mutable state with either input.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from gmd_active_learning.core.exceptions import ConfigurationError
from gmd_active_learning.utils.logging import get_logger

LOGGER = get_logger(__name__)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML document; an empty document is ``{}``.

    Raises :class:`ConfigurationError` when the file cannot be read or parsed, or when its top
    level is not a mapping.
    """

    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigurationError(f"cannot read YAML file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"YAML file {path} must contain a mapping at the top level, got {type(data).__name__}")
    return data


def merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a deep copy of ``base``.

    Nested mappings merge key by key; lists and scalars from ``override`` replace the base value;
    ``None`` in ``override`` replaces as well (it does not delete). Inputs are never modified.
    """

    merged: dict[str, Any] = deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            merged[key] = merge_dicts(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def merge_yaml_files(*paths: str | Path) -> dict[str, Any]:
    """Load several YAML files and merge them left to right (later files override earlier ones)."""

    merged: dict[str, Any] = {}
    for path in paths:
        merged = merge_dicts(merged, load_yaml(path))
        LOGGER.debug("merged YAML layer %s", path)
    return merged


def dump_yaml(data: Mapping[str, Any], path: str | Path) -> Path:
    """Write ``data`` as block-style YAML (key order preserved) and return the path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(data), sort_keys=False, default_flow_style=False), encoding="utf-8")
    return path


__all__ = ["dump_yaml", "load_yaml", "merge_dicts", "merge_yaml_files"]
