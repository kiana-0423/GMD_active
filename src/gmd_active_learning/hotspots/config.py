"""Configuration of hotspot clustering (YAML section ``hotspots``; defaults in configs/sections/hotspots.yaml).

There is deliberately no hotspot score threshold here: hotspot atoms are the monitor's
``hotspot_atom_ids`` (atom score >= ``monitor.stages.label_threshold``, argmax fallback on a
trigger). PHAL had a second, independent ``hotspot.threshold`` that could leave a triggered frame
with zero clusters (F-16).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gmd_active_learning.core.exceptions import ConfigurationError


class HotspotsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    merge_radius: float = Field(8.0, ge=0.0, description="Angstrom; single-linkage merge distance (PBC-aware)")


def load_hotspots_config(section: dict[str, Any] | HotspotsConfig | None) -> HotspotsConfig:
    if isinstance(section, HotspotsConfig):
        return section
    try:
        return HotspotsConfig.model_validate(section or {})
    except ValidationError as exc:
        raise ConfigurationError(f"invalid hotspots configuration: {exc}") from exc


__all__ = ["HotspotsConfig", "load_hotspots_config"]
