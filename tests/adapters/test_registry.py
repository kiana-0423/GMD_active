"""Backend registry: config-driven construction, entry-point plugins, failures, overrides, capabilities."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import EntryPoint
from typing import Any

import pytest

from gmd_active_learning.adapters.dft.cp2k.backend import CP2KBackend
from gmd_active_learning.adapters.registry import (
    ENTRY_POINT_GROUP,
    BackendNotFoundError,
    BackendRegistry,
    BackendRole,
    PluginLoadError,
    RuntimeStatus,
    backend_engine,
    check_capabilities,
    create_backend,
    create_typed_backend,
    runtime_report,
)
from gmd_active_learning.adapters.schedulers.local import LocalScheduler
from gmd_active_learning.adapters.schedulers.pbs import PBSScheduler
from gmd_active_learning.adapters.schedulers.slurm import SlurmScheduler
from gmd_active_learning.core.exceptions import CapabilityUnavailableError, ConfigurationError


class ZeroMLIP:
    name = "zero"
    role = BackendRole.MLIP
    capabilities = frozenset({"forces", "committee"})

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ZeroMLIP:
        return cls(config)

    def check_runtime(self) -> RuntimeStatus:
        return RuntimeStatus(self.name, self.role, True, "test backend")


class ReplacementCP2K(ZeroMLIP):
    name = "cp2k"
    role = BackendRole.DFT


class BrokenRuntime(ZeroMLIP):
    def check_runtime(self) -> RuntimeStatus:
        return RuntimeStatus(self.name, self.role, False, "binary missing")


def _point(name: str, value: str) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=ENTRY_POINT_GROUP)


def _config(**backends: Any) -> dict[str, Any]:
    return {"backends": {"discover_entry_points": False, **backends}}


def test_third_party_backend_is_constructed_without_core_changes() -> None:
    registry = BackendRegistry()
    registry.register("mlip", "zero", ZeroMLIP.from_config)
    backend = create_backend(_config(mlip={"engine": "zero"}), "mlip", registry=registry)
    assert isinstance(backend, ZeroMLIP) and backend.config["backends"]["mlip"]["engine"] == "zero"
    with pytest.raises(ValueError, match="already registered"):
        registry.register("mlip", "ZERO", ZeroMLIP.from_config)


@pytest.mark.parametrize(
    "engine, expected", [("local", LocalScheduler), ("slurm", SlurmScheduler), ("pbs", PBSScheduler), ("PBS", PBSScheduler)]
)
def test_builtin_schedulers_by_config(engine: str, expected: type) -> None:
    backend = create_backend(_config(scheduler={"engine": engine}), "scheduler", registry=BackendRegistry())
    assert type(backend) is expected


def test_builtin_cp2k_by_config_reads_its_sections() -> None:
    config = {**_config(dft={"engine": "cp2k"}), "cp2k": {"executable": "my-cp2k"}, "embedding": {"enabled": False}}
    backend = create_typed_backend(config, "dft", CP2KBackend, registry=BackendRegistry())
    assert backend.executable == "my-cp2k"


def test_engine_names_are_normalized() -> None:
    assert backend_engine(_config(dft={"engine": " CP2K-ML "}), "dft") == "cp2k_ml"


def test_unknown_backend_lists_registered_engines() -> None:
    with pytest.raises(BackendNotFoundError, match="registered engines: local, pbs, slurm") as info:
        create_backend(_config(scheduler={"engine": "lsf"}), "scheduler", registry=BackendRegistry())
    assert isinstance(info.value, KeyError) and isinstance(info.value, ConfigurationError)
    with pytest.raises(ConfigurationError, match="unknown backend role"):
        BackendRegistry().names("qm")


def test_entry_point_plugin_is_discovered() -> None:
    registry = BackendRegistry()
    registry.load_entry_points([_point("mlip:zero", f"{__name__}:ZeroMLIP")])
    assert isinstance(registry.create("mlip", "zero", {}), ZeroMLIP)
    assert registry.source("mlip", "zero").startswith("entry_point")


def test_broken_plugin_is_reported_with_its_import_error_and_does_not_block_builtins() -> None:
    registry = BackendRegistry()
    registry.load_entry_points(
        [
            _point("mlip:My-Engine", "gmd_nonexistent_plugin_module:Factory"),  # PHAL keyed this by the raw name (F-31)
            _point("md:syntax", f"{__name__}:missing_attribute"),
        ]
    )
    with pytest.raises(PluginLoadError, match="gmd_nonexistent_plugin_module"):
        registry.create("mlip", "my_engine", {})
    with pytest.raises(PluginLoadError, match="missing_attribute"):
        registry.create("md", "syntax", {})
    assert isinstance(registry.create("scheduler", "local", {}), LocalScheduler)


def test_plugin_raising_any_exception_is_contained(monkeypatch: pytest.MonkeyPatch) -> None:
    point = _point("mlip:explodes", "x:y")
    monkeypatch.setattr(EntryPoint, "load", lambda self: (_ for _ in ()).throw(RuntimeError("boom at import")))
    registry = BackendRegistry()
    registry.load_entry_points([point])
    with pytest.raises(PluginLoadError, match="boom at import"):
        registry.create("mlip", "explodes", {})


def test_plugin_cannot_silently_replace_a_builtin() -> None:
    registry = BackendRegistry()
    registry.load_entry_points([_point("dft:cp2k", f"{__name__}:ReplacementCP2K")])
    assert "dft:cp2k#override" in registry.failed_entry_points
    assert isinstance(registry.create("dft", "cp2k", {}), CP2KBackend)


def test_plugin_replaces_a_builtin_when_explicitly_allowed() -> None:
    registry = BackendRegistry()
    registry.load_entry_points([_point("dft:cp2k", f"{__name__}:ReplacementCP2K")], override_builtins=["dft:cp2k"])
    assert isinstance(registry.create("dft", "cp2k", {}), ReplacementCP2K)


def test_config_factory_path_registers_a_plugin() -> None:
    config = _config(mlip={"engine": "zero", "factory": f"{__name__}:ZeroMLIP"})
    assert isinstance(create_backend(config, "mlip", registry=BackendRegistry()), ZeroMLIP)
    broken = _config(mlip={"engine": "zero", "factory": "gmd_nonexistent_plugin_module:Factory"})
    with pytest.raises(PluginLoadError):
        create_backend(broken, "mlip", registry=BackendRegistry())


def test_entry_point_discovery_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(BackendRegistry, "load_entry_points", lambda self, *a, **k: calls.append("loaded"))
    create_backend(_config(scheduler={"engine": "local"}), "scheduler", registry=BackendRegistry())
    assert calls == []
    create_backend({"backends": {"scheduler": {"engine": "local"}}}, "scheduler", registry=BackendRegistry())
    assert calls == ["loaded"]


def test_role_mismatch_raises_type_error() -> None:
    registry = BackendRegistry()
    registry.register("md", "zero", ZeroMLIP.from_config)
    with pytest.raises(TypeError, match="declared role"):
        registry.create("md", "zero", {})
    with pytest.raises(TypeError, match="does not implement"):
        create_typed_backend(_config(scheduler={"engine": "local"}), "scheduler", CP2KBackend, registry=BackendRegistry())


def test_capability_checks() -> None:
    registry = BackendRegistry()
    registry.register("mlip", "zero", ZeroMLIP.from_config)
    with pytest.raises(CapabilityUnavailableError, match="training"):
        create_backend(_config(mlip={"engine": "zero", "required_capabilities": ["forces", "training"]}), "mlip", registry=registry)
    check_capabilities(LocalScheduler(), ["restart_safe", "exit_code"])
    with pytest.raises(CapabilityUnavailableError):
        check_capabilities(SlurmScheduler(), ["restart_safe", "walltime"])


def test_runtime_check_on_create_and_doctor_report() -> None:
    registry = BackendRegistry()
    registry.register("mlip", "broken", BrokenRuntime.from_config)
    config = {"backends": {"discover_entry_points": False, "check_runtime_on_create": True, "mlip": {"engine": "broken"}}}
    with pytest.raises(CapabilityUnavailableError, match="binary missing"):
        create_backend(config, "mlip", registry=registry)
    report = runtime_report(
        {"backends": {"discover_entry_points": False, "scheduler": {"engine": "local"}, "dft": {"engine": "nope"}}},
        registry=BackendRegistry(),
    )
    by_role = {row.role: row for row in report}
    assert by_role[BackendRole.SCHEDULER].available is True
    assert by_role[BackendRole.DFT].available is False and "BackendNotFoundError" in by_role[BackendRole.DFT].detail


def test_backends_section_is_strict() -> None:
    with pytest.raises(Exception, match="extra"):
        create_backend({"backends": {"scheduler": {"engine": "local", "typo": 1}}}, "scheduler", registry=BackendRegistry())


def test_packaged_backends_section_equals_model_defaults() -> None:
    from pathlib import Path

    import yaml

    from gmd_active_learning.adapters.registry import BackendsConfig

    path = Path(__file__).resolve().parents[2] / "configs" / "sections" / "backends.yaml"
    data = yaml.safe_load(path.read_text())["backends"]
    assert BackendsConfig.model_validate(data) == BackendsConfig()
    assert data["dft"]["engine"] == "cp2k" and data["scheduler"]["engine"] == "local"
