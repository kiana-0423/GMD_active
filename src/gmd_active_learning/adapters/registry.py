# Ported from PHAL (hotspot_al.backends.registry, hotspot_al.backends.factory, hotspot_al.backends.base) at
# 96263b6; MIT License, see NOTICE.
"""Backend registry: configuration-driven construction by role, plugins, capability checks.

Roles are ``md``, ``mlip``, ``dft`` and ``scheduler``. A backend is any object exposing ``name``,
``role``, ``capabilities`` (a set of strings) and ``check_runtime() -> RuntimeStatus``; factories take
the complete application configuration mapping.

Sources of factories, in order of precedence:

1. programmatic :meth:`BackendRegistry.register` (``replace=True`` to override);
2. ``backends.<role>.factory: "package.module:callable"`` in the configuration;
3. entry points in the group ``gmd_active_learning.adapters`` named ``"<role>:<name>"``;
4. built-ins, imported lazily from ``module:attribute`` paths.

Fixes relative to PHAL (findings F-31): plugin import failures are keyed by the *normalized*
``role:name`` so they are reported with their import error instead of "unknown backend"; any
exception raised while importing a plugin is captured (not only ImportError & co.); and a plugin may
replace a built-in when ``backends.override_builtins`` lists it (otherwise the conflict is recorded and
reported, never silently dropped).
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from gmd_active_learning.core.exceptions import (
    AdapterError,
    CapabilityUnavailableError,
    ConfigurationError,
)

LOGGER = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "gmd_active_learning.adapters"


class BackendRole(str, Enum):
    """Roles that built-in or third-party backends can fill."""

    MD = "md"
    MLIP = "mlip"
    DFT = "dft"
    SCHEDULER = "scheduler"


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    """Result of a backend runtime health check."""

    backend: str
    role: BackendRole
    available: bool
    detail: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    """Structural contract every registered backend satisfies."""

    name: str
    role: BackendRole
    capabilities: frozenset[str]

    def check_runtime(self) -> RuntimeStatus: ...


BackendFactory = Callable[[Mapping[str, Any]], Any]


class BackendNotFoundError(ConfigurationError, KeyError):
    """No factory is registered for the requested ``role:name``."""

    def __str__(self) -> str:  # KeyError would quote the message
        return str(self.args[0]) if self.args else ""


class PluginLoadError(AdapterError, RuntimeError):
    """A plugin for the requested backend exists but could not be loaded."""


class BackendSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: str
    factory: str | None = None  # explicit "package.module:callable" plugin path
    required_capabilities: list[str] = Field(default_factory=list)


class BackendsConfig(BaseModel):
    """``backends`` section."""

    model_config = ConfigDict(extra="forbid")

    md: BackendSelection = Field(default_factory=lambda: BackendSelection(engine="gmd"))
    mlip: BackendSelection = Field(default_factory=lambda: BackendSelection(engine="gmd_sgt"))
    dft: BackendSelection = Field(default_factory=lambda: BackendSelection(engine="cp2k"))
    scheduler: BackendSelection = Field(default_factory=lambda: BackendSelection(engine="local"))
    discover_entry_points: bool = True
    entry_point_group: str = ENTRY_POINT_GROUP
    override_builtins: list[str] = Field(default_factory=list)
    check_runtime_on_create: bool = False

    def selection(self, role: BackendRole | str) -> BackendSelection:
        return getattr(self, BackendRole(role).value)


def normalize_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_")
    if not normalized:
        raise ConfigurationError("backend name must not be empty")
    return normalized


def _key(role: BackendRole | str, name: str) -> tuple[BackendRole, str]:
    try:
        resolved_role = BackendRole(str(role.value if isinstance(role, BackendRole) else role).strip().lower())
    except ValueError as exc:
        raise ConfigurationError(f"unknown backend role {role!r}; expected one of md, mlip, dft, scheduler") from exc
    return resolved_role, normalize_name(name)


def _label(key: tuple[BackendRole, str]) -> str:
    return f"{key[0].value}:{key[1]}"


def import_object(path: str) -> Any:
    """Resolve ``"package.module:attr.sub"``."""

    module_name, _, attribute = path.partition(":")
    if not module_name or not attribute:
        raise ConfigurationError(f"plugin path {path!r} must look like 'package.module:attribute'")
    target: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        target = getattr(target, part)
    return target


def _coerce_factory(value: Any) -> BackendFactory:
    from_config = getattr(value, "from_config", None)
    if isinstance(value, type) and callable(from_config):
        return from_config
    if callable(value):
        return value
    raise TypeError("a backend plugin must expose a class with from_config(config) or a factory callable")


# Built-ins are imported lazily so that, e.g., a broken optional dependency of one engine never
# prevents constructing another. Other areas register theirs with :func:`register_builtin`.
_BUILTIN_PATHS: dict[tuple[BackendRole, str], str] = {
    (BackendRole.DFT, "cp2k"): "gmd_active_learning.adapters.dft.cp2k.backend:CP2KBackend",
    (BackendRole.SCHEDULER, "local"): "gmd_active_learning.adapters.schedulers.local:LocalScheduler",
    (BackendRole.SCHEDULER, "slurm"): "gmd_active_learning.adapters.schedulers.slurm:SlurmScheduler",
    (BackendRole.SCHEDULER, "pbs"): "gmd_active_learning.adapters.schedulers.pbs:PBSScheduler",
}


def register_builtin(role: BackendRole | str, name: str, path: str) -> None:
    """Declare a built-in backend by lazy import path (used by other GMD_active areas)."""

    _BUILTIN_PATHS[_key(role, name)] = path


class BackendRegistry:
    """Factories keyed by (role, normalized name)."""

    def __init__(self, *, include_builtins: bool = True) -> None:
        self._factories: dict[tuple[BackendRole, str], BackendFactory] = {}
        self._sources: dict[tuple[BackendRole, str], str] = {}
        self._include_builtins = include_builtins
        self._entry_points_loaded: set[str] = set()
        self.failed_entry_points: dict[str, Exception] = {}

    # -- registration ------------------------------------------------------------------------------
    def register(
        self,
        role: BackendRole | str,
        name: str,
        factory: BackendFactory,
        *,
        replace: bool = False,
        source: str = "programmatic",
    ) -> None:
        key = _key(role, name)
        if key in self._factories and not replace:
            raise ValueError(f"backend already registered for {_label(key)} (from {self._sources[key]})")
        self._factories[key] = factory
        self._sources[key] = source
        self.failed_entry_points.pop(_label(key), None)

    def unregister(self, role: BackendRole | str, name: str) -> None:
        key = _key(role, name)
        self._factories.pop(key, None)
        self._sources.pop(key, None)

    def source(self, role: BackendRole | str, name: str) -> str | None:
        key = _key(role, name)
        if key in self._sources:
            return self._sources[key]
        if self._include_builtins and key in _BUILTIN_PATHS:
            return "builtin"
        return None

    def names(self, role: BackendRole | str | None = None) -> list[str]:
        selected = None if role is None else _key(role, "x")[0]
        keys = set(self._factories)
        if self._include_builtins:
            keys |= set(_BUILTIN_PATHS)
        return sorted(name for (item_role, name) in keys if selected is None or item_role == selected)

    def load_entry_points(
        self,
        points: Iterable[EntryPoint] | None = None,
        *,
        group: str = ENTRY_POINT_GROUP,
        override_builtins: Iterable[str] = (),
    ) -> None:
        """Register ``role:name`` factories from entry points; failures are recorded per backend."""

        if points is None:
            if group in self._entry_points_loaded:
                return
            points = entry_points().select(group=group)
            self._entry_points_loaded.add(group)
        overrides = {_label(_key(*item.split(":", 1))) for item in override_builtins if ":" in item}
        for point in points:
            label = point.name
            try:
                raw_role, raw_name = point.name.split(":", 1)
                key = _key(raw_role, raw_name)
                label = _label(key)
            except (ValueError, ConfigurationError) as exc:
                self.failed_entry_points[label] = ValueError(
                    f"entry point name {point.name!r} must be '<role>:<name>' with role md|mlip|dft|scheduler"
                )
                LOGGER.warning("skipping malformed backend entry point %r: %s", point.name, exc)
                continue
            if key in self._factories and self._sources.get(key, "").startswith("entry_point"):
                continue
            is_builtin = self._include_builtins and key in _BUILTIN_PATHS
            if (key in self._factories or is_builtin) and label not in overrides:
                conflict = ValueError(
                    f"plugin {point.value!r} would replace the existing backend {label}; "
                    f"list '{label}' in backends.override_builtins to allow it"
                )
                self.failed_entry_points[label + "#override"] = conflict
                LOGGER.warning("%s", conflict)
                continue
            try:
                factory = _coerce_factory(point.load())
            except Exception as exc:  # noqa: BLE001 - any plugin import failure must be contained
                self.failed_entry_points[label] = exc
                LOGGER.warning("backend plugin %s (%s) failed to load: %r", label, point.value, exc)
                continue
            self.register(key[0], key[1], factory, replace=True, source=f"entry_point:{point.value}")

    # -- construction ------------------------------------------------------------------------------
    def factory_for(self, role: BackendRole | str, name: str) -> BackendFactory:
        key = _key(role, name)
        factory = self._factories.get(key)
        if factory is not None:
            return factory
        failure = self.failed_entry_points.get(_label(key))
        if failure is not None:
            raise PluginLoadError(f"backend plugin {_label(key)} failed to load: {failure!r}") from failure
        if self._include_builtins and key in _BUILTIN_PATHS:
            path = _BUILTIN_PATHS[key]
            try:
                factory = _coerce_factory(import_object(path))
            except Exception as exc:  # noqa: BLE001
                raise PluginLoadError(f"built-in backend {_label(key)} ({path}) failed to import: {exc!r}") from exc
            self._factories[key] = factory
            self._sources[key] = "builtin"
            return factory
        available = ", ".join(self.names(key[0])) or "none"
        raise BackendNotFoundError(f"unknown {key[0].value} backend {key[1]!r}; registered engines: {available}")

    def create(
        self,
        role: BackendRole | str,
        name: str,
        config: Mapping[str, Any],
        *,
        required_capabilities: Iterable[str] = (),
        check_runtime: bool = False,
    ) -> Any:
        key = _key(role, name)
        backend = self.factory_for(*key)(config)
        declared_role = getattr(backend, "role", None)
        if declared_role is None or BackendRole(getattr(declared_role, "value", declared_role)) != key[0]:
            raise TypeError(f"backend {key[1]!r} declared role {declared_role!r}, expected {key[0].value!r}")
        check_capabilities(backend, required_capabilities)
        if check_runtime:
            status = backend.check_runtime()
            if not status.available:
                raise CapabilityUnavailableError(f"{_label(key)} runtime unavailable: {status.detail}")
        return backend


def check_capabilities(backend: Any, required: Iterable[str]) -> None:
    """Raise :class:`CapabilityUnavailableError` if ``backend`` lacks any required capability."""

    required_set = {str(item) for item in required}
    if not required_set:
        return
    declared = getattr(backend, "capabilities", frozenset())
    if callable(declared):
        declared = declared()
    missing = sorted(required_set - set(declared))
    if missing:
        name = getattr(backend, "name", type(backend).__name__)
        raise CapabilityUnavailableError(
            f"backend {name!r} does not provide required capabilities: {', '.join(missing)}"
        )


DEFAULT_REGISTRY = BackendRegistry()


def backends_config(config: Mapping[str, Any]) -> BackendsConfig:
    section = config.get("backends", {}) if isinstance(config, Mapping) else {}
    return BackendsConfig.model_validate(section or {})


def backend_engine(config: Mapping[str, Any], role: BackendRole | str) -> str:
    return normalize_name(backends_config(config).selection(role).engine)


def create_backend(
    config: Mapping[str, Any],
    role: BackendRole | str,
    *,
    registry: BackendRegistry | None = None,
    required_capabilities: Iterable[str] = (),
) -> Any:
    """Create the backend configured under ``backends.<role>``."""

    selected = registry or DEFAULT_REGISTRY
    settings = backends_config(config)
    choice = settings.selection(role)
    key = _key(role, choice.engine)
    if choice.factory:
        source = f"config:{choice.factory}"
        if selected.source(*key) != source:
            try:
                factory = _coerce_factory(import_object(choice.factory))
            except ConfigurationError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise PluginLoadError(f"configured factory {choice.factory!r} for {_label(key)} failed: {exc!r}") from exc
            selected.register(key[0], key[1], factory, replace=True, source=source)
    elif settings.discover_entry_points:
        selected.load_entry_points(group=settings.entry_point_group, override_builtins=settings.override_builtins)
    return selected.create(
        key[0],
        key[1],
        config,
        required_capabilities=[*choice.required_capabilities, *required_capabilities],
        check_runtime=settings.check_runtime_on_create,
    )


def create_typed_backend(
    config: Mapping[str, Any],
    role: BackendRole | str,
    backend_type: type[Any],
    *,
    registry: BackendRegistry | None = None,
) -> Any:
    backend = create_backend(config, role, registry=registry)
    if not isinstance(backend, backend_type):
        raise TypeError(f"configured {BackendRole(role).value} backend does not implement {backend_type.__name__}")
    return backend


def runtime_report(
    config: Mapping[str, Any],
    roles: Iterable[BackendRole | str] = (BackendRole.DFT, BackendRole.SCHEDULER),
    *,
    registry: BackendRegistry | None = None,
) -> list[RuntimeStatus]:
    """``check_runtime()`` of every configured backend; construction failures become unavailable rows."""

    report: list[RuntimeStatus] = []
    for role in roles:
        resolved = BackendRole(role)
        try:
            backend = create_backend(config, resolved, registry=registry)
            report.append(backend.check_runtime())
        except Exception as exc:  # noqa: BLE001 - a doctor report must list every failure
            engine = backends_config(config).selection(resolved).engine
            report.append(RuntimeStatus(engine, resolved, False, f"{type(exc).__name__}: {exc}"))
    return report


__all__ = [
    "DEFAULT_REGISTRY",
    "ENTRY_POINT_GROUP",
    "Backend",
    "BackendFactory",
    "BackendNotFoundError",
    "BackendRegistry",
    "BackendRole",
    "BackendSelection",
    "BackendsConfig",
    "PluginLoadError",
    "RuntimeStatus",
    "backend_engine",
    "backends_config",
    "check_capabilities",
    "create_backend",
    "create_typed_backend",
    "import_object",
    "normalize_name",
    "register_builtin",
    "runtime_report",
]
