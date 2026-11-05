"""Shared helpers for the configuration tests (frozen PHAL fixtures under tests/fixtures/phal_config)."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
PHAL_FIXTURES = REPO / "tests" / "fixtures" / "phal_config"
DEFAULT_CONFIG = REPO / "configs" / "default.yaml"
REF_ROOT = REPO / "ref" / "lammps_monitor"


@lru_cache(maxsize=None)
def _phal_default() -> dict[str, Any]:
    return yaml.safe_load((PHAL_FIXTURES / "default.yaml").read_text(encoding="utf-8"))


def phal_default() -> dict[str, Any]:
    return copy.deepcopy(_phal_default())


@lru_cache(maxsize=None)
def schema() -> dict[str, Any]:
    return json.loads((PHAL_FIXTURES / "schema_leaves.json").read_text(encoding="utf-8"))


def schema_leaves() -> list[str]:
    return [leaf["path"] for leaf in schema()["leaves"]]


def gmd_default() -> dict[str, Any]:
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def set_path(data: dict[str, Any], dotted: str, value: Any) -> dict[str, Any]:
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
    return data
