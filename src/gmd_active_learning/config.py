"""Central configuration: one YAML file, one strict pydantic model per section.

Replaces PHAL's ``config_schema.py`` / ``config.py`` (ledger CFG-001..003). Unknown sections and
unknown keys are errors. Section models live with the code they configure; this module only
maps section names to them. Relative paths in path-valued fields are resolved against the
directory of the config file, so a configuration is portable as a unit; a bare executable name
(no path separator, e.g. ``gmd``) is resolved the same way only when that file exists next to the
config, and is otherwise left alone to be found on ``PATH``.

Validation is strict in PHAL's sense (CFG-001): values are checked as parsed YAML/JSON with
pydantic's strict mode, so ``true`` is never accepted for a number, ``"100"`` never for an integer
and ``1`` never for a boolean; integers are accepted for floats. Errors name the dotted path
(``cp2k.max_scf``). Unlike PHAL every key has a documented default, so an absent key is not an
error; absent *required* sections and fields are. ``configs/default.yaml`` is the complete
baseline (CFG-003), generated from the section models by ``default_config_dict``::

    python -m gmd_active_learning.config --write-default configs/default.yaml
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
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
# Placeholders for fields without a default, used only by ``default_config_dict``.
PLACEHOLDERS: dict[str, dict[str, Any]] = {
    "workflow": {"run_id": "default-run"},
    "md": {"executable": "gmd", "initial_structure": "initial_structure.xyz"},
}
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


def _has_separator(value: str) -> bool:
    return "/" in value or os.sep in value or (os.altsep is not None and os.altsep in value)


def _format_errors(section: str, exc: ValidationError) -> str:
    lines = []
    for error in exc.errors():
        dotted = ".".join([section, *(str(part) for part in error.get("loc", ()))])
        lines.append(f"  {dotted}: {error.get('msg')} (got {error.get('input')!r})")
    return "\n".join(lines)


def validate_section(name: str, values: Any) -> BaseModel:
    """Strictly validate one section (see module docstring); errors name the dotted path."""

    model = section_model(name)
    try:
        payload = json.dumps(values if values is not None else {})
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid config section {name!r}: values are not plain YAML data ({exc})") from exc
    try:
        return model.model_validate_json(payload, strict=True)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid config section {name!r}:\n{_format_errors(name, exc)}") from exc


def _resolve_paths(raw: dict[str, Any], base: Path) -> dict[str, Any]:
    resolved = {name: dict(values) if isinstance(values, dict) else values for name, values in raw.items()}
    for section, fields in PATH_FIELDS.items():
        values = resolved.get(section)
        if not isinstance(values, dict):
            continue
        for field_name in fields:
            value = values.get(field_name)
            if not isinstance(value, str) or not value or Path(value).expanduser().is_absolute():
                continue
            if field_name.endswith("executable") and not _has_separator(value) and not (base / value).is_file():
                continue  # bare command name with no such file next to the config: looked up on PATH
            values[field_name] = str((base / value).expanduser().resolve())
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
        if values is not None and not isinstance(values, dict):
            raise ConfigurationError(f"config section {name!r} must be a mapping, got {type(values).__name__}")
        sections[name] = validate_section(name, values or {})
    digest = content_hash({name: section.model_dump(mode="json") for name, section in sorted(sections.items())})
    return LoadedConfig(path=path, raw=resolved, sections=sections, config_hash=digest)


def load_config(path: str | Path) -> LoadedConfig:
    path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read config {path}: {exc}") from exc
    return build_config(raw, base_dir=path.parent, path=path)


def default_config_dict(sections: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Every section with all its defaults (JSON-compatible); required fields get ``PLACEHOLDERS``."""

    out: dict[str, Any] = {}
    for name in sections or tuple(SECTIONS):
        model = section_model(name)
        placeholders = PLACEHOLDERS.get(name, {})
        missing = [key for key, field in model.model_fields.items() if field.is_required() and key not in placeholders]
        if missing:
            raise ConfigurationError(f"section {name!r} has required fields {missing} without a placeholder")
        out[name] = model.model_validate(dict(placeholders)).model_dump(mode="json")
    return out


DEFAULT_CONFIG_HEADER = """\
# GMD_active packaged default configuration: every section with every key at its default.
# GENERATED from the section models; do not edit by hand. Regenerate with
#   python -m gmd_active_learning.config --write-default configs/default.yaml
# Copy it and change what you need; unknown keys and sections are errors. The placeholders
# workflow.run_id, md.executable (looked up on PATH) and md.initial_structure (relative to this
# file) must be set for a real run. Section documentation: configs/sections/*.yaml.
"""


def write_default_config(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(default_config_dict(), sort_keys=False, default_flow_style=False, width=100)
    target.write_text(DEFAULT_CONFIG_HEADER + body, encoding="utf-8")
    return target


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gmd_active_learning.config")
    parser.add_argument("--write-default", type=Path, required=True, metavar="PATH")
    args = parser.parse_args(argv)
    print(write_default_config(args.write_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
