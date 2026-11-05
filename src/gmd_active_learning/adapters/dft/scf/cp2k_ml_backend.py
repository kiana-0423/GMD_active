# Ported from PHAL (hotspot_al.backends.cp2k_ml_scf) at 96263b6; MIT License, see NOTICE.
"""Optional ML initial-guess wrapper around a CP2K labeling run (engine ``cp2k_ml_scf``).

**Status:** experimental, capability-gated and off by default
(``scf.ml_scf.enabled``). PHAL's recorded measurement of a trained density
guess is 63 SCF steps against 53 for CP2K's atomic guess (speedup 0.85); no
ML-SCF component has demonstrated an acceleration. The wrapper exists so the
experiment can be repeated against the cp2k-ml-scf build.

CP2K remains the backend: it builds the Kohn-Sham matrix, diagonalises,
computes energies/forces/stress and decides convergence. This wrapper only

1. predicts an initial density for a CP2K input that the labeling area has
   already written, validates it, and writes ``<input>.ml_density``
   (protocol ``phal-ml-scf-v2``, the only ML feature the cp2k-ml-scf
   reconstruction implements);
2. rewrites ``SCF_GUESS ATOMIC`` to ``SCF_GUESS ML`` only when a validated
   density exists;
3. builds the :class:`ExecutionRequest`, or a native one when the binary or
   the configuration cannot run ML-SCF;
4. reads CP2K's own ``ML-SCF|`` log lines to decide what actually ran.

PHAL subclassed its standard ``CP2KBackend``; GMD_active's CP2K labeler is
owned by the labeling area, so this is a composable wrapper that operates on
an input file instead. Deliberate deviations from PHAL (findings F-22, F-23):

* :meth:`degradation_reason` also degrades when ML mixing is requested but the
  binary does not advertise ``phal_ml_mixing`` (PHAL never checked);
* :meth:`retry_request` implements the documented "accelerated run fails ->
  retried once with plain CP2K";
* :meth:`parse_outcome` marks a run accelerated only when CP2K itself logged an
  accepted guess (PHAL marked every non-degraded run accelerated);
* ``scf_mode`` is always one of :data:`~.modes.SCF_MODES` (PHAL defaulted to "ml");
* the environment carries what the fork actually reads
  (``PHAL_ML_SCF_STRUCTURE_DIGEST``, ``_BASIS``, ``_GUESS_ALPHA``) instead of
  keys it ignores (``_MODEL``, ``_MANIFEST``, ``_MODEL_DIGEST``), and the
  configured environment (toolchain ``LD_LIBRARY_PATH``) reaches the capability
  probe and the native fallback too;
* predictors that accept an AO overlap get one (``density_gnn`` could never
  deliver a guess in PHAL because none was passed);
* an existing ``&MIXING`` section is replaced, not duplicated.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from gmd_active_learning.adapters.dft.scf.capabilities import CP2KCapabilities, detect_cp2k_capabilities
from gmd_active_learning.adapters.dft.scf.config import ScfConfig, labeling_cp2k_section, scf_config
from gmd_active_learning.adapters.dft.scf.execution import BackendRole, ExecutionRequest, RuntimeStatus
from gmd_active_learning.adapters.dft.scf.modes import (
    MODE_ML_FULL,
    MODE_ML_GUESS,
    MODE_NATIVE,
    MODE_NATIVE_FALLBACK,
)
from gmd_active_learning.contracts.ids import file_sha256
from gmd_active_learning.core.exceptions import CapabilityUnavailableError, ConfigurationError

LOGGER = logging.getLogger(__name__)

ENV_PREFIX = "PHAL_ML_SCF"
#: Environment variables the cp2k-ml-scf reconstruction reads (ml_scf_interface.F:44-50).
FORK_ENVIRONMENT = (
    "PHAL_ML_SCF_ENABLED",
    "PHAL_ML_SCF_DENSITY",
    "PHAL_ML_SCF_PROTOCOL",
    "PHAL_ML_SCF_FALLBACK",
    "PHAL_ML_SCF_GUESS_ALPHA",
    "PHAL_ML_SCF_STRUCTURE_DIGEST",
    "PHAL_ML_SCF_BASIS",
)

_ACCELERATOR_ENABLED = re.compile(r"ML[-_ ]SCF\|.*accelerator.*enabled", re.IGNORECASE)
_FALLBACK_USED = re.compile(r"ML[-_ ]SCF\|\s*fallback used:?\s*(.*)", re.IGNORECASE)
_GUESS_SOURCE = re.compile(r"ML[-_ ]SCF\|\s*ml_guess_source:\s*(\S+)", re.IGNORECASE)
_PREDICTION_SUCCESS = re.compile(r"ML[-_ ]SCF\|\s*prediction_success:\s*([TF])", re.IGNORECASE)
_ELECTRONS = re.compile(r"ML[-_ ]SCF\|\s*electrons:\s*(\S+)", re.IGNORECASE)
_SCF_SUCCESS_MARKERS = ("SCF run converged",)
_SCF_FAILURE_MARKERS = ("SCF run NOT converged", "ABORT")


@dataclass(frozen=True, slots=True)
class MLSCFOutcome:
    """What CP2K says happened to the ML guess, from its ``ML-SCF|`` lines."""

    ml_lines_present: bool
    accelerator_enabled: bool
    fallback_used: bool
    guess_source: str | None = None
    prediction_success: bool | None = None
    fallback_reason: str | None = None
    electrons: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ml_lines_present": self.ml_lines_present,
            "accelerator_enabled": self.accelerator_enabled,
            "fallback_used": self.fallback_used,
            "guess_source": self.guess_source,
            "prediction_success": self.prediction_success,
            "fallback_reason": self.fallback_reason,
            "electrons": self.electrons,
        }


def parse_ml_scf_log(output_text: str) -> MLSCFOutcome:
    """Parse both ``ML-SCF|`` dialects (legacy PHAL fork and the reconstruction)."""

    present = "ML-SCF|" in output_text or "ML_SCF|" in output_text
    fallback = _FALLBACK_USED.search(output_text)
    success = _PREDICTION_SUCCESS.search(output_text)
    source = _GUESS_SOURCE.search(output_text)
    electrons = _ELECTRONS.search(output_text)
    fallback_used = fallback is not None or (success is not None and success.group(1).upper() == "F")
    accepted = _ACCELERATOR_ENABLED.search(output_text) is not None and not fallback_used
    if success is not None:
        accepted = accepted and success.group(1).upper() == "T"
    try:
        electron_value = float(electrons.group(1)) if electrons else None
    except ValueError:
        electron_value = None
    return MLSCFOutcome(
        ml_lines_present=present,
        accelerator_enabled=accepted,
        fallback_used=fallback_used,
        guess_source=source.group(1) if source else None,
        prediction_success=None if success is None else success.group(1).upper() == "T",
        fallback_reason=(fallback.group(1).strip() or None) if fallback else None,
        electrons=electron_value,
    )


class CP2KMLSCFBackend:
    """Prepare, launch-describe and interpret one ML-SCF CP2K calculation."""

    backend_name = "cp2k_ml_scf"
    role = BackendRole.DFT

    def __init__(
        self,
        config: Any = None,
        *,
        capabilities: CP2KCapabilities | None = None,
        probe: Callable[..., CP2KCapabilities] | None = None,
    ) -> None:
        self.raw_config = config
        self.settings: ScfConfig = scf_config(config)
        self.ml = self.settings.ml_scf
        self.executable = str(self.ml.executable or self.settings.capabilities.executable or "cp2k.psmp")
        self.model_path = Path(self.ml.model_path) if self.ml.model_path else None
        self.model_digest = self.ml.model_digest
        self.fallback_to_standard_scf = self.ml.fallback_to_standard_scf
        self.require_final_scf_convergence = self.ml.require_final_scf_convergence
        self.controller_protocol = self.ml.protocol
        self._capabilities = capabilities
        self._probe = probe or detect_cp2k_capabilities

    # -- capabilities and provenance ------------------------------------

    def capabilities(self, *, refresh: bool = False) -> CP2KCapabilities:
        """Probe the configured CP2K build once (cached), with the configured environment."""

        if self._capabilities is None or refresh:
            self._capabilities = self._probe(
                executable=self.executable,
                config=self.settings,
                environment=dict(self.ml.environment),
            )
        return self._capabilities

    def ml_scf_provenance(self) -> dict[str, Any]:
        return self.capabilities().provenance(
            ml_model_digest=self._resolved_model_digest(),
            ml_scf_enabled=self.ml.enabled,
        )

    def check_runtime(self) -> RuntimeStatus:
        capabilities = self.capabilities()
        reason = self.degradation_reason()
        details = list(capabilities.details)
        if reason:
            details.append(reason)
        return RuntimeStatus(
            backend=self.backend_name,
            role=self.role,
            available=capabilities.available and reason is None,
            detail="; ".join(details) if details else str(capabilities.resolved_path),
            metadata={
                "model_path": None if self.model_path is None else str(self.model_path),
                "model_digest": self._resolved_model_digest(),
                **capabilities.to_dict(),
            },
        )

    # -- input preparation ------------------------------------------------

    def prepare_input(
        self,
        input_file: str | Path,
        structure: Any,
        *,
        task_id: str,
        electronic_state: Any = None,
        region_metadata: Mapping[str, Any] | None = None,
        overlap: np.ndarray | None = None,
    ) -> dict[str, Path]:
        """Add the ML guess to an existing CP2K input.

        ``structure`` is an :class:`~.schema.SCFStructure`, an ``ase.Atoms`` or
        a :class:`gmd_active_learning.contracts.ExtractedRegion`. ``overlap`` is
        the AO overlap matrix ``S`` for predictors that emit density matrices
        (``density_gnn``): CP2K checks ``Tr(P S)``, so a matrix without ``S`` is
        never written. Returns the written paths: ``input`` always,
        ``ml_density`` when a validated prediction was exported, and
        ``ml_scf_manifest``.
        """

        path = Path(input_file)
        if not path.is_file():
            raise ConfigurationError(f"CP2K input not found: {path}")
        written: dict[str, Path] = {"input": path}
        density = None
        if self.ml.enabled:
            density = self._write_density_guess(
                structure,
                path,
                task_id=task_id,
                electronic_state=electronic_state,
                region_metadata=region_metadata,
                overlap=overlap,
            )
        if density is not None:
            written["ml_density"] = density
        self._rewrite_ml_keywords(path, has_density=density is not None)
        # Written last: the manifest digests the input the keyword rewrite just changed.
        written["ml_scf_manifest"] = self._write_manifest(path, density)
        return written

    def _write_density_guess(
        self,
        structure: Any,
        input_path: Path,
        *,
        task_id: str,
        electronic_state: Any,
        region_metadata: Mapping[str, Any] | None,
        overlap: np.ndarray | None,
    ) -> Path | None:
        """Predict and export the initial density CP2K will read; never raises.

        Prediction happens before CP2K launches, so a model failure cannot
        interrupt a calculation: no file is written and CP2K runs ``ATOMIC``.
        """

        from gmd_active_learning.adapters.dft.scf.density.exchange import (
            exchange_path_for,
            write_density_exchange,
        )
        from gmd_active_learning.adapters.dft.scf.density.prediction import DENSITY_FORMAT_ATOMIC_POPULATION
        from gmd_active_learning.adapters.dft.scf.density.registry import create_predictor, record_prediction

        try:
            scf_structure = _as_scf_structure(structure)
            state = electronic_state or self._resolve_state(structure, region_metadata)
            predictor = create_predictor(self.settings)
            if getattr(predictor, "accepts_overlap", False):
                prediction = predictor.predict(scf_structure, state, overlap=overlap)
            else:
                prediction = predictor.predict(scf_structure, state)
            if self.ml.basis_set and not prediction.metadata.get("basis_set"):
                prediction = prediction.with_metadata(basis_set=self.ml.basis_set)
            validation = predictor.validate_prediction(
                prediction, state, structure=scf_structure, electron_tolerance=self.ml.electron_tolerance
            )
            if not validation.valid:
                self._log_prediction_skip(task_id, "; ".join(validation.reasons))
                return None
            if prediction.density_format == DENSITY_FORMAT_ATOMIC_POPULATION:
                LOGGER.warning(
                    "%s: writing an atomic-population guess; the cp2k-ml-scf reconstruction validates "
                    "DIM against n_ao and will fall back to SCF_GUESS ATOMIC for it",
                    task_id,
                )
            path = write_density_exchange(prediction, exchange_path_for(input_path))
        except Exception as exc:  # A prediction is optional; a label is not.
            self._log_prediction_skip(task_id, str(exc))
            return None

        if self.ml.prediction_dir:
            record_prediction(
                prediction,
                output_dir=self.ml.prediction_dir,
                task_id=task_id,
                extra={"density_file": str(path), "input_file": str(input_path)},
            )
        return path

    def _resolve_state(self, structure: Any, region_metadata: Mapping[str, Any] | None) -> Any:
        from gmd_active_learning.adapters.dft.scf.schema import SCFElectronicState
        from gmd_active_learning.utils.electronic_state import resolve_electronic_state

        metadata = dict(region_metadata or {})
        atoms = _as_atoms(structure)
        if hasattr(structure, "charge") and "charge" not in metadata:
            metadata["charge"] = int(structure.charge)
        if getattr(structure, "multiplicity", None) is not None and "multiplicity" not in metadata:
            metadata["multiplicity"] = int(structure.multiplicity)
        state = resolve_electronic_state(
            atoms, dict(labeling_cp2k_section(self.raw_config)), region_metadata=metadata
        )
        return SCFElectronicState.from_electronic_state(state)

    def _rewrite_ml_keywords(self, input_path: Path, *, has_density: bool) -> None:
        """``SCF_GUESS ML`` only with a validated density; ``&MIXING METHOD ML`` only when requested."""

        lines = input_path.read_text(encoding="utf-8").splitlines()
        mixing = self._ml_mixing_requested()
        updated: list[str] = []
        changed = False
        inside_mixing = False
        for line in lines:
            stripped = line.strip().upper()
            if mixing and stripped.startswith("&MIXING"):
                inside_mixing, changed = True, True  # replaced below, never duplicated
                continue
            if inside_mixing:
                if stripped.startswith("&END MIXING"):
                    inside_mixing = False
                continue
            if has_density and stripped == "SCF_GUESS ATOMIC":
                indent = line[: len(line) - len(line.lstrip())]
                updated.append(f"{indent}SCF_GUESS ML")
                changed = True
                continue
            updated.append(line)
            if mixing and stripped == "&SCF":
                indent = line[: len(line) - len(line.lstrip())]
                updated.extend(
                    [
                        f"{indent}  &MIXING T",
                        f"{indent}    METHOD ML",
                        f"{indent}    ALPHA {self.ml.mixing.alpha:.3f}",
                        f"{indent}    NBUFFER {self.ml.mixing.history_length}",
                        f"{indent}  &END MIXING",
                    ]
                )
                changed = True
        if changed:
            input_path.write_text("\n".join(updated) + "\n", encoding="utf-8")

    def _ml_mixing_requested(self) -> bool:
        return self.ml.enabled and self.ml.mixing.method == "ml"

    # -- execution ----------------------------------------------------------

    def degradation_reason(self) -> str | None:
        """Why ML-SCF cannot run as configured, or ``None`` when it can."""

        if not self.ml.enabled:
            return "scf.ml_scf.enabled is false"
        if self.model_path is not None and self.model_path.is_file():
            actual = _sha256(self.model_path)
            if self.model_digest is not None and self.model_digest != actual:
                return f"ML-SCF model digest mismatch: expected {self.model_digest}, got {actual}"
        elif self._ml_mixing_requested():
            return f"neural mixing needs a deployed model, which is missing: {self.model_path}"

        capabilities = self.capabilities()
        if capabilities.loader_failure:
            return f"CP2K at {capabilities.resolved_path} failed in the dynamic linker during the probe"
        if capabilities.available and not capabilities.ml_scf_supported:
            return (
                f"CP2K at {capabilities.resolved_path} does not support ML-SCF "
                f"(version {capabilities.version or 'unknown'})"
            )
        if self._ml_mixing_requested() and not capabilities.ml_mixing_supported:
            return (
                f"neural mixing requested but CP2K at {capabilities.resolved_path or self.executable} "
                "does not advertise phal_ml_mixing (no known build implements it)"
            )
        return None

    def _base_request(
        self, input_file: Path, *, output_file: Path, environment: Mapping[str, str], metadata: Mapping[str, Any]
    ) -> ExecutionRequest:
        return ExecutionRequest.from_command(
            [self.executable, "-i", input_file.name],
            work_dir=input_file.parent,
            stdout_path=Path(output_file).name,
            stderr_path=input_file.with_suffix(".err").name,
            environment=environment,
            metadata={"engine": self.backend_name, "input_file": str(input_file), **dict(metadata)},
        )

    def native_fallback_request(
        self, input_file: str | Path, *, output_file: str | Path, reason: str | None = None
    ) -> ExecutionRequest:
        """A plain CP2K request for ``input_file`` with every ML keyword and ML variable removed.

        The configured environment (e.g. the toolchain loader path) is kept:
        the same binary still has to start.
        """

        native_input = self._strip_ml_keywords(Path(input_file))
        environment = {
            key: value for key, value in self.ml.environment.items() if not key.startswith(ENV_PREFIX)
        }
        return self._base_request(
            native_input,
            output_file=Path(output_file),
            environment=environment,
            metadata={
                "scf_mode": MODE_NATIVE_FALLBACK,
                "fallback_used": True,
                "fallback_reason": reason,
                "input_file": str(native_input),
            },
        )

    def execution_request(self, input_file: str | Path, *, output_file: str | Path) -> ExecutionRequest:
        """The request to run ``input_file``; native when ML-SCF cannot run (or raise if configured)."""

        from gmd_active_learning.adapters.dft.scf.density.exchange import exchange_path_for, read_density_exchange

        path = Path(input_file)
        reason = self.degradation_reason()
        if reason is not None:
            if not self.fallback_to_standard_scf:
                raise CapabilityUnavailableError(reason)
            self._log_degradation(reason)
            return self.native_fallback_request(path, output_file=output_file, reason=reason)

        density_file = exchange_path_for(path)
        has_density = density_file.is_file()
        environment = dict(self.ml.environment)
        environment.update(
            {
                "PHAL_ML_SCF_ENABLED": "1",
                "PHAL_ML_SCF_PROTOCOL": self.controller_protocol,
                "PHAL_ML_SCF_FALLBACK": "1" if self.fallback_to_standard_scf else "0",
            }
        )
        structure_hash = None
        if has_density:
            environment["PHAL_ML_SCF_DENSITY"] = str(density_file)
            header = read_density_exchange(density_file)
            structure_hash = header["structure_hash"] or None
            if self.ml.export_structure_digest and structure_hash:
                environment["PHAL_ML_SCF_STRUCTURE_DIGEST"] = structure_hash
            if header["basis_set"] and header["basis_set"] != "unknown":
                environment["PHAL_ML_SCF_BASIS"] = header["basis_set"]
        if self.ml.guess_alpha is not None:
            environment["PHAL_ML_SCF_GUESS_ALPHA"] = repr(float(self.ml.guess_alpha))

        return self._base_request(
            path,
            output_file=Path(output_file),
            environment=environment,
            metadata={
                "scf_mode": self.scf_mode(has_density=has_density),
                "fallback_used": False,
                "ml_scf_manifest": str(path.with_suffix(".ml_scf.json")),
                "ml_density_file": str(density_file) if has_density else None,
                "structure_digest": structure_hash,
                "model_path": None if self.model_path is None else str(self.model_path),
                "model_digest": self._validated_model_digest(),
            },
        )

    def retry_request(
        self,
        request: ExecutionRequest,
        *,
        output_file: str | Path,
        failed_output: str | None = None,
    ) -> ExecutionRequest | None:
        """After a failed accelerated run: one native retry, or ``None``.

        Returns ``None`` when retrying is switched off
        (``scf.ml_scf.retry_native_on_failure``), when ``request`` was already
        native, or when CP2K's output shows the ML guess had already been
        rejected and the run *was* native (a native rerun would repeat it).
        """

        if not self.ml.retry_native_on_failure:
            return None
        if request.metadata.get("fallback_used") or request.metadata.get("scf_mode") in (
            MODE_NATIVE,
            MODE_NATIVE_FALLBACK,
        ):
            return None
        if failed_output is not None and parse_ml_scf_log(failed_output).fallback_used:
            return None
        return self.native_fallback_request(
            Path(str(request.metadata["input_file"])),
            output_file=output_file,
            reason="accelerated run failed; retried once with plain CP2K",
        )

    def scf_mode(self, *, has_density: bool) -> str:
        if self.degradation_reason() is not None:
            return MODE_NATIVE_FALLBACK
        if not has_density:
            return MODE_NATIVE
        return MODE_ML_FULL if self._ml_mixing_requested() else MODE_ML_GUESS

    # -- results ---------------------------------------------------------

    def parse_outcome(self, output_text: str) -> MLSCFOutcome:
        """What ran, from CP2K's own log; a degraded configuration is never accelerated."""

        outcome = parse_ml_scf_log(output_text)
        if self.degradation_reason() is not None:
            return MLSCFOutcome(
                ml_lines_present=outcome.ml_lines_present,
                accelerator_enabled=False,
                fallback_used=True,
                guess_source=outcome.guess_source,
                prediction_success=outcome.prediction_success,
                fallback_reason=outcome.fallback_reason or self.degradation_reason(),
                electrons=outcome.electrons,
            )
        return outcome

    def calculation_provenance(
        self,
        outcome: MLSCFOutcome | None = None,
        *,
        scf_mode: str | None = None,
        iterations: int | None = None,
        runtime_seconds: float | None = None,
    ) -> dict[str, Any]:
        """The record every ML-SCF calculation writes (build, model, mode, fallback)."""

        capabilities = self.capabilities()
        degraded = self.degradation_reason()
        if scf_mode is None:
            if degraded is not None or (outcome is not None and outcome.fallback_used):
                scf_mode = MODE_NATIVE_FALLBACK
            elif outcome is not None and outcome.accelerator_enabled:
                scf_mode = MODE_ML_FULL if self._ml_mixing_requested() else MODE_ML_GUESS
            else:
                scf_mode = MODE_NATIVE
        return {
            "capability_contract": capabilities.contract_version,
            "cp2k_version": capabilities.version,
            "cp2k_commit": capabilities.git_commit,
            "ml_model_digest": self._resolved_model_digest(),
            "scf_mode": scf_mode,
            "iterations": iterations,
            "runtime": runtime_seconds,
            "fallback_used": bool(degraded) or bool(outcome is not None and outcome.fallback_used),
            "ml_scf_enabled": degraded is None and (outcome is None or outcome.accelerator_enabled),
            "degradation_reason": degraded,
            "ml_guess_source": None if outcome is None else outcome.guess_source,
        }

    def output_is_complete(self, output_text: str) -> bool:
        if self.require_final_scf_convergence:
            if any(marker in output_text for marker in _SCF_FAILURE_MARKERS):
                return False
            return any(marker in output_text for marker in _SCF_SUCCESS_MARKERS)
        return "ABORT" not in output_text

    # -- helpers ------------------------------------------------------------

    def _strip_ml_keywords(self, input_file: Path) -> Path:
        """Write a sibling input with ``SCF_GUESS ML`` reverted and ML mixing removed."""

        if not input_file.is_file():
            return input_file
        target = input_file.with_name(f"{input_file.stem}_native{input_file.suffix}")
        cleaned: list[str] = []
        inside_ml_mixing = False
        block: list[str] = []
        for line in input_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip().upper()
            if stripped.startswith("&MIXING"):
                inside_ml_mixing, block = True, [line]
                continue
            if inside_ml_mixing:
                block.append(line)
                if stripped.startswith("&END MIXING"):
                    inside_ml_mixing = False
                    if not any(entry.strip().upper() in ("METHOD ML", "METHOD ML_PULAY") for entry in block):
                        cleaned.extend(block)  # a native &MIXING section is kept as written
                continue
            if stripped == "SCF_GUESS ML":
                indent = line[: len(line) - len(line.lstrip())]
                cleaned.append(f"{indent}SCF_GUESS ATOMIC")
                continue
            cleaned.append(line)
        target.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
        return target

    def _resolved_model_digest(self) -> str | None:
        if self.model_digest is not None:
            return self.model_digest
        return _sha256(self.model_path) if self.model_path is not None and self.model_path.is_file() else None

    def _validated_model_digest(self) -> str | None:
        if self.model_path is None or not self.model_path.is_file():
            return None
        actual = _sha256(self.model_path)
        if self.model_digest is not None and self.model_digest != actual:
            raise CapabilityUnavailableError(
                f"ML-SCF model digest mismatch: expected {self.model_digest}, got {actual}"
            )
        return actual

    def _write_manifest(self, input_path: Path, density: Path | None) -> Path:
        manifest = input_path.with_suffix(".ml_scf.json")
        payload = {
            "protocol": self.controller_protocol,
            "input_file": input_path.name,
            "input_digest": _sha256(input_path),
            "ml_density_file": None if density is None else density.name,
            "model_path": None if self.model_path is None else str(self.model_path),
            "model_digest": self._resolved_model_digest(),
            "fallback_to_standard_scf": self.fallback_to_standard_scf,
            "require_final_scf_convergence": self.require_final_scf_convergence,
            "provenance": self.ml_scf_provenance(),
        }
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest

    @staticmethod
    def _log_prediction_skip(task_id: str, reason: str) -> None:
        LOGGER.warning("ML-SCF initial guess not written for %s (%s); CP2K will use SCF_GUESS ATOMIC", task_id, reason)

    @staticmethod
    def _log_degradation(reason: str) -> None:
        LOGGER.warning("ML-SCF unavailable (%s); running standard CP2K instead", reason)


def _sha256(path: Path) -> str:
    return f"sha256:{file_sha256(path)}"


def _as_atoms(structure: Any) -> Any:
    from ase import Atoms

    if isinstance(structure, Atoms):
        return structure
    if hasattr(structure, "numbers") and hasattr(structure, "positions"):  # ExtractedRegion
        cell = structure.cell if getattr(structure, "cell", None) is not None else np.zeros((3, 3))
        return Atoms(numbers=structure.numbers, positions=structure.positions, cell=cell, pbc=structure.pbc)
    if hasattr(structure, "atomic_numbers"):  # SCFStructure
        return Atoms(
            numbers=list(structure.atomic_numbers),
            positions=np.asarray(structure.positions, dtype=float),
            cell=np.asarray(structure.cell, dtype=float),
            pbc=structure.pbc,
        )
    raise ConfigurationError(f"cannot interpret {type(structure).__name__} as a structure")


def _as_scf_structure(structure: Any) -> Any:
    from gmd_active_learning.adapters.dft.scf.schema import SCFStructure

    if isinstance(structure, SCFStructure):
        return structure
    return SCFStructure.from_atoms(_as_atoms(structure))


__all__ = ["CP2KMLSCFBackend", "FORK_ENVIRONMENT", "MLSCFOutcome", "parse_ml_scf_log"]
