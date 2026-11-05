# Ported from PHAL (hotspot_al.backends.cp2k:CP2KBackend) at 96263b6; MIT License, see NOTICE.
"""CP2K implementation of the DFT role: inputs, command, parsing, validation.

Extension points for the ML-SCF variant (``adapters/dft/scf``): override ``variant_id``,
``decorate_input`` / ``job_environment`` to add ML keywords and files, set ``ml_scf_enabled`` and
return the plain backend from ``native_backend()``. The labeling service then resubmits a failed
ML-SCF attempt natively (findings F-25).
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from gmd_active_learning.adapters.dft.cp2k.command import build_cp2k_command, merged_environment
from gmd_active_learning.adapters.dft.cp2k.config import CP2KConfig, RetryStep
from gmd_active_learning.adapters.dft.cp2k.input import (
    CP2KInput,
    build_cp2k_input,
    h_opt_fixed_atoms,
    load_template,
    needs_h_opt,
    settings_with_step,
    write_cp2k_inputs,
)
from gmd_active_learning.adapters.dft.cp2k.parse import CP2KOutput, parse_cp2k_forces, parse_cp2k_output_file
from gmd_active_learning.adapters.dft.cp2k.validation import output_is_complete, validation_reasons
from gmd_active_learning.adapters.registry import BackendRole, RuntimeStatus
from gmd_active_learning.adapters.schedulers.base import ExecutionRequest
from gmd_active_learning.adapters.schedulers.config import SchedulerResources
from gmd_active_learning.contracts.regions import ExtractedRegion
from gmd_active_learning.core.exceptions import ConfigurationError

STAGE_HOPT = "hopt"
STAGE_SINGLE_POINT = "sp"


class CP2KBackend:
    """Generate, run (via a scheduler) and parse CP2K labeling calculations."""

    name: ClassVar[str] = "cp2k"
    role: ClassVar[BackendRole] = BackendRole.DFT
    capabilities: ClassVar[frozenset[str]] = frozenset(
        {"energy", "forces", "stress", "h_only_opt", "non_periodic", "partially_periodic", "periodic", "charge", "spin"}
    )
    ml_scf_enabled: bool = False

    def __init__(
        self,
        config: CP2KConfig | None = None,
        *,
        embedding_enabled: bool = False,
        base_dir: str | Path | None = None,
    ) -> None:
        self.config = config or CP2KConfig()
        if not self.config.enabled:
            raise ConfigurationError("cp2k.enabled is false; the CP2K labeling backend is switched off")
        self.embedding_enabled = embedding_enabled
        self.base_dir = None if base_dir is None else Path(base_dir)
        self._template_text = load_template(self.config, base_dir=self.base_dir)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> CP2KBackend:
        section = config.get("cp2k", {}) if isinstance(config, Mapping) else {}
        embedding = config.get("embedding", {}) if isinstance(config, Mapping) else {}
        enabled = bool(embedding.get("enabled", False)) if isinstance(embedding, Mapping) else False
        return cls(CP2KConfig.model_validate(section or {}), embedding_enabled=enabled)

    # -- identity ----------------------------------------------------------------------------------
    @property
    def executable(self) -> str:
        return self.config.executable

    def variant_id(self) -> str:
        """Identifies the calculation flavour in the settings hash (ML-SCF variants override)."""

        return self.name

    def settings_hash(self) -> str:
        return self.config.settings_hash(template_text=self._template_text, extra=self.variant_id())

    def native_backend(self) -> CP2KBackend:
        return self

    # -- runtime -----------------------------------------------------------------------------------
    def check_runtime(self) -> RuntimeStatus:
        resolved = shutil.which(self.executable)
        if resolved is None:
            return RuntimeStatus(self.name, self.role, False, f"CP2K executable not found: {self.executable}")
        metadata: dict[str, Any] = {"executable": resolved}
        if self.config.data_dir and not Path(self.config.data_dir).is_dir():
            return RuntimeStatus(self.name, self.role, False, f"cp2k.data_dir does not exist: {self.config.data_dir}", metadata)
        try:
            probe = subprocess.run([resolved, "--version"], text=True, capture_output=True, timeout=60, check=False)
            first = next((line.strip() for line in probe.stdout.splitlines() if "version" in line.lower()), "")
            metadata["version"] = first or None
        except (OSError, subprocess.TimeoutExpired) as exc:
            return RuntimeStatus(self.name, self.role, False, f"{resolved} --version failed: {exc}", metadata)
        return RuntimeStatus(self.name, self.role, True, resolved, metadata)

    # -- inputs ------------------------------------------------------------------------------------
    def effective_config(self, step: RetryStep | None = None) -> CP2KConfig:
        return settings_with_step(self.config, step)

    def needs_h_opt(self, region: ExtractedRegion) -> bool:
        return needs_h_opt(region, self.config)

    def build_input(
        self,
        region: ExtractedRegion,
        *,
        stage: str,
        project_name: str,
        step: RetryStep | None = None,
    ) -> CP2KInput:
        config = self.effective_config(step)
        if stage == STAGE_HOPT:
            return build_cp2k_input(
                region,
                config,
                run_type="GEO_OPT",
                project_name=project_name,
                fixed_atom_indices=h_opt_fixed_atoms(region),
                embedding_enabled=self.embedding_enabled,
                template_text=self._template_text,
            )
        if stage == STAGE_SINGLE_POINT:
            return build_cp2k_input(
                region,
                config,
                run_type="ENERGY_FORCE",
                project_name=project_name,
                embedding_enabled=self.embedding_enabled,
                template_text=self._template_text,
            )
        raise ConfigurationError(f"unknown CP2K stage {stage!r}")

    def decorate_input(self, cp2k_input: CP2KInput, work_dir: Path, *, region: ExtractedRegion) -> str:
        """Hook for variants (ML-SCF) to add keywords/files; the plain backend returns the text."""

        return cp2k_input.text

    def job_environment(self, work_dir: Path) -> dict[str, str]:
        return merged_environment(self.config)

    def prepare_inputs(self, region: ExtractedRegion, output_dir: str | Path, *, task_id: str) -> dict[str, Path]:
        """PHAL-compatible: write ``<task_id>_hopt.inp`` (when capped) and ``<task_id>_sp.inp``."""

        return write_cp2k_inputs(
            region,
            output_dir,
            self.config,
            job_name=task_id,
            embedding_enabled=self.embedding_enabled,
            template_text=self._template_text,
        )

    # -- execution ---------------------------------------------------------------------------------
    def command(self, input_name: str, output_name: str, resources: SchedulerResources | None = None) -> list[str]:
        return build_cp2k_command(self.config, input_name, output_name, resources=resources or self.config.resources)

    def execution_request(
        self,
        input_file: str | Path,
        *,
        output_file: str | Path,
        resources: SchedulerResources | None = None,
    ) -> ExecutionRequest:
        input_file = Path(input_file)
        return ExecutionRequest.from_command(
            self.command(input_file.name, Path(output_file).name, resources),
            work_dir=input_file.parent,
            environment=self.job_environment(input_file.parent),
            metadata={"engine": self.name, "input_file": str(input_file)},
        )

    # -- results -----------------------------------------------------------------------------------
    def parse_output(self, output_path: str | Path) -> CP2KOutput:
        return parse_cp2k_output_file(output_path)

    def parse_forces(self, output_path: str | Path):  # noqa: ANN201 - numpy array
        return parse_cp2k_forces(output_path)

    def validate(self, output: CP2KOutput | None, *, n_atoms: int) -> list[str]:
        return validation_reasons(output, n_atoms=n_atoms, settings=self.config.validation)

    @staticmethod
    def output_is_complete(output_text: str) -> bool:
        return output_is_complete(output_text)


__all__ = ["CP2KBackend", "STAGE_HOPT", "STAGE_SINGLE_POINT"]
