"""Central configuration: one YAML file, one strict pydantic model per section.

Replaces PHAL's ``config_schema.py`` / ``config.py`` (ledger CFG-001..003). Unknown sections and
unknown keys are errors. Section models live with the code they configure; this module only
maps section names to them. Relative paths in path-valued fields are resolved against the
directory of the config file, so a configuration is portable as a unit.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from gmd_active_learning.contracts import content_hash
from gmd_active_learning.core.exceptions import ConfigurationError

# section name -> "module:Model"
SECTIONS: dict[str, str] = {
    "workflow": "gmd_active_learning.workflow.controller:WorkflowConfig",
    "md": "gmd_active_learning.adapters.gmd.engine:GMDConfig",
    "mlip": "gmd_active_learning.adapters.sgt.backend:SGTConfig",
    "monitor": "gmd_active_learning.monitors.config:MonitorConfig",
    "events": "gmd_active_learning.events.config:EventsConfig",
    "hotspots": "gmd_active_learning.hotspots.config:HotspotsConfig",
    "extraction": "gmd_active_learning.extraction.config:ExtractionConfig",
    "capping": "gmd_active_learning.extraction.config:CappingConfig",
    "embedding": "gmd_active_learning.extraction.config:EmbeddingConfig",
    "datasets": "gmd_active_learning.datasets.config:DatasetsConfig",
    "backends": "gmd_active_learning.adapters.registry:BackendsConfig",
    "cp2k": "gmd_active_learning.adapters.dft.cp2k.config:CP2KConfig",
    "scheduler": "gmd_active_learning.adapters.schedulers.config:SchedulerConfig",
    "scf": "gmd_active_learning.adapters.dft.scf.config:ScfConfig",
}
REQUIRED_SECTIONS = ("workflow", "md", "mlip", "cp2k")
# Fields (by section) holding filesystem paths that are resolved relative to the config file.
PATH_FIELDS: dict[str, tuple[str, ...]] = {
    "md": ("executable", "initial_structure"),
    "mlip": ("source_path",),
}


def register_section(name: str, target: str) -> None:
    """Allow other packages (labeling, schedulers, SCF tooling) to add their sections."""

    SECTIONS[name] = target


def section_model(name: str) -> type[BaseModel]:
    try:
        module_name, attribute = SECTIONS[name].split(":")
    except KeyError as exc:
        raise ConfigurationError(f"unknown config section {name!r}; known: {sorted(SECTIONS)}") from exc
    return getattr(importlib.import_module(module_name), attribute)


@dataclass(frozen=True, slots=True)
class LoadedConfig:
    path: Path | None
    raw: dict[str, Any]
    sections: dict[str, BaseModel]
    config_hash: str

    def __getitem__(self, name: str) -> BaseModel:
        return self.sections[name]

    def get(self, name: str) -> BaseModel | None:
        return self.sections.get(name)


def _resolve_paths(raw: dict[str, Any], base: Path) -> dict[str, Any]:
    resolved = {name: dict(values) if isinstance(values, dict) else values for name, values in raw.items()}
    for section, fields in PATH_FIELDS.items():
        values = resolved.get(section)
        if not isinstance(values, dict):
            continue
        for field_name in fields:
            value = values.get(field_name)
            if isinstance(value, str) and value and not Path(value).expanduser().is_absolute():
                values[field_name] = str((base / value).resolve())
    return resolved


def build_config(raw: dict[str, Any], *, base_dir: Path | None = None, path: Path | None = None) -> LoadedConfig:
    if not isinstance(raw, dict):
        raise ConfigurationError("configuration must be a mapping of sections")
    unknown = sorted(set(raw) - set(SECTIONS))
    if unknown:
        raise ConfigurationError(f"unknown config section(s) {unknown}; known: {sorted(SECTIONS)}")
    missing = [name for name in REQUIRED_SECTIONS if name not in raw]
    if missing:
        raise ConfigurationError(f"missing required config section(s) {missing}")
    resolved = _resolve_paths(raw, base_dir or Path.cwd())
    sections: dict[str, BaseModel] = {}
    for name, values in resolved.items():
        model = section_model(name)
        try:
            sections[name] = model.model_validate(values or {})
        except ValidationError as exc:
            raise ConfigurationError(f"invalid config section {name!r}:\n{exc}") from exc
    digest = content_hash({name: section.model_dump(mode="json") for name, section in sorted(sections.items())})
    return LoadedConfig(path=path, raw=resolved, sections=sections, config_hash=digest)


def load_config(path: str | Path) -> LoadedConfig:
    path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read config {path}: {exc}") from exc
    return build_config(raw, base_dir=path.parent, path=path)
