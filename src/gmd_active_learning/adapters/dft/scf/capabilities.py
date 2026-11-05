# Ported from PHAL (hotspot_al.cp2k_ml_scf.cp2k_capabilities) at 96263b6; MIT License, see NOTICE.
"""Versioned capability contract for a CP2K binary.

What a CP2K build can do is *probed*, never assumed: assuming the modified
build is present is how a run silently degrades to a native SCF while still
recording itself as accelerated. The probe runs ``<cp2k> --version`` and
parses the banner. The result is a :class:`CP2KCapabilities` record carrying a
contract version (:data:`CAPABILITY_CONTRACT_VERSION`) and, per feature in
:data:`FEATURES`, whether the binary *advertises* it and on what evidence.

Three levels are kept apart and never conflated:

``advertised``
    the banner carries the exact build token for the feature;
``validated``
    a real calculation exercised the feature on this machine
    (:func:`validate_native_features` for the native features; nothing in this
    repository can validate the ML features without the modified binary);
``declared``
    what a build's own documentation claims (:data:`CP2K_ML_SCF_DECLARED`), recorded
    as "declared, not validated here".

Deliberate deviations from PHAL (findings F-22, SCF-032):

* The probe runs with ``os.environ`` *plus* ``scf.ml_scf.environment`` (PHAL
  ignored the configured environment, so a deployment that put the toolchain
  ``LD_LIBRARY_PATH`` only in config got an empty banner and silently degraded).
* A dynamic-linker failure (exit 127, "error while loading shared libraries")
  is reported as such, not as a missing executable.
* Build flags are matched as exact tokens; PHAL matched substrings, so any flag
  containing "torch" or "ml_scf" counted.
* ``parse_version_banner`` keeps PHAL's output keys for parity.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gmd_active_learning.core.exceptions import CapabilityUnavailableError

LOGGER = logging.getLogger(__name__)

#: Bumped whenever the meaning or the set of reported features changes.
CAPABILITY_CONTRACT_VERSION = "gmd-cp2k-capabilities-v1"

#: Searched in order when no executable is configured.
CP2K_CANDIDATES: tuple[str, ...] = ("cp2k.psmp", "cp2k.popt", "cp2k.ssmp", "cp2k.sopt", "cp2k")

VERSION_TIMEOUT_SECONDS = 30.0

_VERSION = re.compile(r"CP2K\s+version\s+(\S+)", re.IGNORECASE)
_VERSION_STRING = re.compile(r"CP2K\|\s*version string:\s*(.+)", re.IGNORECASE)
# v2024.1 printed "Source code revision git:b4a17a5"; the reconstructed v2026.2
# build prints "Source code revision 67b5da8". The prefix is optional.
_GIT_COMMIT = re.compile(
    r"(?:source code revision(?:\s+number)?|revision)\s*:?\s*(?:git:)?([0-9a-f]{6,40})\b",
    re.IGNORECASE,
)
_GIT_FALLBACK = re.compile(r"\bgit:([0-9a-f]{6,40})\b", re.IGNORECASE)
_FLAGS = re.compile(r"cp2kflags:\s*(.+)", re.IGNORECASE)
_LOADER_FAILURE = re.compile(r"error while loading shared libraries|Library not loaded|dyld", re.IGNORECASE)

#: Exact build tokens (``cp2kflags``) and what they prove.
TOKEN_ML_SCF = "phal_ml_scf"
TOKEN_ML_MIXING = "phal_ml_mixing"
TOKEN_LIBTORCH = "libtorch"

#: Feature name -> description. Every feature is reported in every probe.
FEATURES: Mapping[str, str] = {
    "native_scf": "CP2K Quickstep SCF (any working build)",
    "ao_matrix_printing": "&DFT &PRINT &AO_MATRICES DENSITY/OVERLAP (upstream CP2K; needed for density recording)",
    "ml_density_guess": "SCF_GUESS ML from a <input>.ml_density exchange file (mode A, protocol phal-ml-scf-v2)",
    "ml_inprocess_guess": "SCF_GUESS ML from an in-process TorchScript model (mode B, PHAL_ML_SCF_GUESS_MODEL)",
    "ml_mixing": "MIXING METHOD ML (neural mixing)",
    "scf_controller": "adaptive SCF controller (PHAL_ML_SCF_CONTROLLER)",
    "pulay_recorder": "Pulay matrix dumps (PHAL_ML_SCF_PULAY_RECORD_DIR)",
}

#: What the cp2k-ml-scf reconstruction's own documentation states (README.md,
#: RECONSTRUCTION.md, VERSION_LOCK.md). Declared, not validated here: its
#: binaries are Linux/aarch64 and cannot run on every development machine.
CP2K_ML_SCF_DECLARED: Mapping[str, Any] = {
    "source": "cp2k-ml-scf (clean-room reconstruction), VERSION_LOCK.md / RECONSTRUCTION.md",
    "upstream": {"version": "2026.2", "commit": "67b5da876dd6a76b8b021d5a04d1c81ba79a4c50"},
    "banner_flags": {
        "native": "omp libint fftw3 libxc elpa parallel scalapack mpi_f08",
        "ml": "omp libint fftw3 libxc elpa parallel scalapack mpi_f08 phal_ml_scf libtorch",
    },
    "protocols": {"density_exchange": "phal-ml-scf-v2", "deploy_manifest": "phal-scf-deploy-v1"},
    "environment_read": [
        "PHAL_ML_SCF_ENABLED",
        "PHAL_ML_SCF_DENSITY",
        "PHAL_ML_SCF_PROTOCOL",
        "PHAL_ML_SCF_FALLBACK",
        "PHAL_ML_SCF_GUESS_ALPHA",
        "PHAL_ML_SCF_STRUCTURE_DIGEST",
        "PHAL_ML_SCF_BASIS",
    ],
    "features": {
        "native_scf": "declared",
        "ao_matrix_printing": "declared (upstream CP2K)",
        "ml_density_guess": "declared (H2 replay: 1 step at -1.165933374533587 Ha, RECONSTRUCTION.md 4e)",
        "ml_inprocess_guess": "not implemented (hook 0009 is a placeholder; 'libtorch' in cp2kflags does not imply it)",
        "ml_mixing": "out of scope - not implemented, not advertised",
        "scf_controller": "not implemented",
        "pulay_recorder": "not implemented",
    },
    "runtime_requirement": (
        "the toolchain LD_LIBRARY_PATH must be in the environment of both the probe and the run "
        "(otherwise exit 127 from the dynamic linker)"
    ),
    "validated_here": False,
}

Runner = Callable[..., str]


@dataclass(frozen=True, slots=True)
class FeatureStatus:
    """One feature of one binary."""

    name: str
    advertised: bool
    evidence: str
    validated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"advertised": self.advertised, "validated": self.validated, "evidence": self.evidence}


@dataclass(frozen=True, slots=True)
class CP2KCapabilities:
    """What one CP2K installation supports, under :data:`CAPABILITY_CONTRACT_VERSION`."""

    executable: str | None
    resolved_path: str | None
    version: str | None
    git_commit: str | None
    ml_scf_supported: bool
    ml_mixing_supported: bool
    torch_supported: bool
    python_torch_available: bool
    flags: tuple[str, ...] = ()
    details: tuple[str, ...] = ()
    banner: str | None = field(default=None, repr=False)
    features: Mapping[str, FeatureStatus] = field(default_factory=dict)
    loader_failure: bool = False
    contract_version: str = CAPABILITY_CONTRACT_VERSION

    @property
    def available(self) -> bool:
        """Whether a usable CP2K executable was found and answered ``--version``."""

        return self.resolved_path is not None and not self.loader_failure

    def supports(self, feature: str) -> bool:
        """Whether ``feature`` is advertised by this binary (never inferred)."""

        if feature not in FEATURES:
            raise KeyError(f"unknown CP2K feature {feature!r}; known: {sorted(FEATURES)}")
        status = self.features.get(feature)
        return bool(status and status.advertised)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "executable": self.executable,
            "resolved_path": self.resolved_path,
            "available": self.available,
            "loader_failure": self.loader_failure,
            "version": self.version,
            "git_commit": self.git_commit,
            "ml_scf_supported": self.ml_scf_supported,
            "ml_mixing_supported": self.ml_mixing_supported,
            "torch_supported": self.torch_supported,
            "python_torch_available": self.python_torch_available,
            "flags": list(self.flags),
            "features": {name: status.to_dict() for name, status in sorted(self.features.items())},
            "details": list(self.details),
        }

    def provenance(self, *, ml_model_digest: str | None = None, ml_scf_enabled: bool = False) -> dict[str, Any]:
        """The provenance block recorded with every ML-SCF calculation."""

        return {
            "capability_contract": self.contract_version,
            "cp2k_version": self.version,
            "cp2k_commit": self.git_commit,
            "ml_model_digest": ml_model_digest,
            "ml_scf_enabled": bool(ml_scf_enabled and self.ml_scf_supported),
            "ml_scf_supported": self.ml_scf_supported,
            "ml_mixing_supported": self.ml_mixing_supported,
        }

    def with_validation(self, validated: Mapping[str, bool], evidence: Mapping[str, str]) -> "CP2KCapabilities":
        """A copy whose named features carry the outcome of a real run."""

        features = dict(self.features)
        for name, ok in validated.items():
            current = features.get(name)
            if current is None:
                raise KeyError(f"unknown CP2K feature {name!r}")
            features[name] = FeatureStatus(
                name=name,
                advertised=current.advertised,
                evidence=f"{current.evidence}; run: {evidence.get(name, '')}".strip("; "),
                validated=bool(ok),
            )
        return CP2KCapabilities(
            executable=self.executable,
            resolved_path=self.resolved_path,
            version=self.version,
            git_commit=self.git_commit,
            ml_scf_supported=self.ml_scf_supported,
            ml_mixing_supported=self.ml_mixing_supported,
            torch_supported=self.torch_supported,
            python_torch_available=self.python_torch_available,
            flags=self.flags,
            details=self.details,
            banner=self.banner,
            features=features,
            loader_failure=self.loader_failure,
            contract_version=self.contract_version,
        )


def detect_cp2k_capabilities(
    *,
    executable: str | Path | None = None,
    config: Any = None,
    runner: Runner | None = None,
    environment: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> CP2KCapabilities:
    """Probe a CP2K installation. Never raises.

    Executable resolution: the ``executable`` argument, then
    ``scf.ml_scf.executable``, then ``scf.capabilities.executable`` (PHAL:
    ``cp2k_ml_scf.executable`` then ``cp2k.executable``), then ``$CP2K_BIN``,
    then :data:`CP2K_CANDIDATES` on ``PATH``. The probe's environment is
    ``os.environ`` updated with ``scf.ml_scf.environment`` and then with
    ``environment``.
    """

    settings = _settings(config)
    configured = _configured_executable(executable, settings)
    probe_env = dict(settings.ml_scf.environment) if settings is not None else {}
    probe_env.update(dict(environment or {}))
    timeout_s = timeout if timeout is not None else (
        settings.capabilities.timeout_s if settings is not None else VERSION_TIMEOUT_SECONDS
    )
    details: list[str] = []

    search_path = probe_env.get("PATH") or os.environ.get("PATH")
    resolved: str | None = None
    name: str | None = configured
    if configured is not None:
        resolved = shutil.which(configured, path=search_path)
        if resolved is None and Path(configured).is_file() and os.access(configured, os.X_OK):
            resolved = str(Path(configured).resolve())
        if resolved is None:
            details.append(f"configured executable not found: {configured}")
    else:
        for candidate in CP2K_CANDIDATES:
            found = shutil.which(candidate, path=search_path)
            if found is not None:
                name, resolved = candidate, found
                break
        if resolved is None:
            details.append(f"no CP2K executable on PATH (tried: {', '.join(CP2K_CANDIDATES)})")

    python_torch = importlib.util.find_spec("torch") is not None
    if not python_torch:
        details.append("python torch is not importable; ML density models are unavailable")

    if resolved is None:
        return CP2KCapabilities(
            executable=name,
            resolved_path=None,
            version=None,
            git_commit=None,
            ml_scf_supported=False,
            ml_mixing_supported=False,
            torch_supported=False,
            python_torch_available=python_torch,
            details=tuple(details),
            features=_features(None, (), available=False),
        )

    banner, exit_code = _run_version(resolved, runner, probe_env, timeout_s)
    loader_failure = False
    if banner is None:
        details.append(f"could not run '{resolved} --version'; capabilities are unknown")
        banner = ""
    elif exit_code == 127 or (exit_code not in (0, None) and _LOADER_FAILURE.search(banner)):
        loader_failure = True
        details.append(
            f"'{resolved} --version' failed in the dynamic linker (exit {exit_code}): the binary is "
            "present but a shared library is missing from the probe environment; set the toolchain "
            "LD_LIBRARY_PATH in scf.ml_scf.environment"
        )

    parsed = parse_version_banner(banner)
    if parsed["version"] is None and banner and not loader_failure:
        details.append("CP2K banner did not contain a recognisable version string")
    if not parsed["ml_scf_supported"]:
        details.append(
            "this CP2K build does not advertise ML-SCF support (no 'phal_ml_scf' in cp2kflags); "
            "SCF_GUESS ML is not available"
        )

    return CP2KCapabilities(
        executable=name,
        resolved_path=resolved,
        version=parsed["version"],
        git_commit=parsed["git_commit"],
        ml_scf_supported=parsed["ml_scf_supported"],
        ml_mixing_supported=parsed["ml_mixing_supported"],
        torch_supported=parsed["torch_supported"],
        python_torch_available=python_torch,
        flags=tuple(parsed["flags"]),
        details=tuple(details),
        banner=banner or None,
        features=_features(parsed["version"], tuple(parsed["flags"]), available=not loader_failure),
        loader_failure=loader_failure,
    )


def capabilities_from_banner(
    banner: str,
    *,
    executable: str = "cp2k.psmp",
    resolved_path: str | None = "cp2k.psmp",
) -> CP2KCapabilities:
    """A capability record for a recorded ``--version`` banner (offline analysis and tests).

    Nothing is executed, so nothing is validated; the record says only what the
    banner advertises.
    """

    parsed = parse_version_banner(banner)
    return CP2KCapabilities(
        executable=executable,
        resolved_path=resolved_path,
        version=parsed["version"],
        git_commit=parsed["git_commit"],
        ml_scf_supported=parsed["ml_scf_supported"],
        ml_mixing_supported=parsed["ml_mixing_supported"],
        torch_supported=parsed["torch_supported"],
        python_torch_available=importlib.util.find_spec("torch") is not None,
        flags=tuple(parsed["flags"]),
        details=("from a recorded banner; nothing was executed",),
        banner=banner,
        features=_features(parsed["version"], tuple(parsed["flags"]), available=resolved_path is not None),
    )


def parse_version_banner(text: str) -> dict[str, Any]:
    """Parse a CP2K ``--version`` banner into capability fields (PHAL's keys)."""

    version_match = _VERSION_STRING.search(text) or _VERSION.search(text)
    version = version_match.group(1).strip() if version_match is not None else None
    if version is not None:
        inner = _VERSION.search(version)
        version = inner.group(1) if inner is not None else version

    commit_match = _GIT_COMMIT.search(text) or _GIT_FALLBACK.search(text)
    git_commit = commit_match.group(1).lower() if commit_match is not None else None

    flags_match = _FLAGS.search(text)
    flags = tuple(
        flag.strip().lower() for flag in re.split(r"[\s,]+", flags_match.group(1).strip()) if flag.strip()
    ) if flags_match is not None else ()

    return {
        "version": version,
        "git_commit": git_commit,
        "flags": flags,
        "ml_scf_supported": TOKEN_ML_SCF in flags,
        "ml_mixing_supported": TOKEN_ML_MIXING in flags,
        "torch_supported": TOKEN_LIBTORCH in flags,
    }


def require_ml_scf(capabilities: CP2KCapabilities) -> None:
    """Raise unless ``capabilities`` can run the ML density-guess path."""

    if not capabilities.available:
        raise CapabilityUnavailableError(
            "no working CP2K executable was found; the ML initial-guess path needs the external "
            "cp2k-ml-scf build (see docs/scf/architecture.md)"
            + (" -- the binary failed in the dynamic linker" if capabilities.loader_failure else "")
        )
    if not capabilities.ml_scf_supported:
        raise CapabilityUnavailableError(
            f"CP2K at {capabilities.resolved_path} does not support ML-SCF "
            f"(version {capabilities.version or 'unknown'}, commit {capabilities.git_commit or 'unknown'}). "
            "Build the external cp2k-ml-scf fork or disable scf.ml_scf."
        )


def require_feature(capabilities: CP2KCapabilities, feature: str) -> None:
    """Raise :class:`CapabilityUnavailableError` unless ``feature`` is advertised."""

    if not capabilities.supports(feature):
        status = capabilities.features.get(feature)
        raise CapabilityUnavailableError(
            f"CP2K at {capabilities.resolved_path or capabilities.executable} does not provide "
            f"{feature} ({FEATURES[feature]}): {status.evidence if status else 'not probed'}"
        )


def validate_native_features(
    capabilities: CP2KCapabilities,
    output_text: str,
) -> CP2KCapabilities:
    """Mark native features validated from the output of a real run of this binary.

    ``native_scf`` is validated when the output parses into a converged SCF
    history; ``ao_matrix_printing`` when it also contains a density *and* an
    overlap matrix whose ``Tr(P S)`` is finite. ML features are never touched:
    only a run of the modified binary could validate them.
    """

    from gmd_active_learning.adapters.dft.scf.ao_matrices import (
        electron_count,
        parse_density_matrix,
        parse_overlap_matrix,
    )
    from gmd_active_learning.adapters.dft.scf.scf_history import parse_scf_history

    history = parse_scf_history(output_text)
    scf_ok = bool(history.converged and history.iterations)
    density = parse_density_matrix(output_text)
    overlap = parse_overlap_matrix(output_text)
    ao_ok = density is not None and overlap is not None
    count = float(electron_count(density, overlap)) if ao_ok else None
    evidence = {
        "native_scf": f"converged={history.converged}, steps={len(history.iterations)}",
        "ao_matrix_printing": "no DENSITY/OVERLAP matrix printed" if not ao_ok else f"Tr(P S)={count:.9f}",
    }
    validated = {"native_scf": scf_ok}
    if ao_ok or "AO_MATRICES" in output_text or density is not None:
        validated["ao_matrix_printing"] = ao_ok
    return capabilities.with_validation(validated, evidence)


#: The H2 control case of cp2k-ml-scf (tests/inputs/h2.inp, reference: 10 steps,
#: -1.165933374533096 Ha) with AO-matrix printing added.
NATIVE_VALIDATION_INPUT = """&GLOBAL
  PROJECT h2_validation
  RUN_TYPE ENERGY
  PRINT_LEVEL MEDIUM
&END GLOBAL
&FORCE_EVAL
  METHOD QUICKSTEP
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    POTENTIAL_FILE_NAME POTENTIAL
    &SCF
      EPS_SCF 1.0E-6
      MAX_SCF 50
    &END SCF
    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
    &END XC
    &PRINT
      &AO_MATRICES
        DENSITY .TRUE.
        OVERLAP .TRUE.
        NDIGITS 12
      &END AO_MATRICES
    &END PRINT
  &END DFT
  &SUBSYS
    &CELL
      ABC 10.0 10.0 10.0
    &END CELL
    &COORD
      H 5.0 5.0 4.63
      H 5.0 5.0 5.37
    &END COORD
    &KIND H
      BASIS_SET DZVP-MOLOPT-GTH
      POTENTIAL GTH-PBE-q1
    &END KIND
  &END SUBSYS
&END FORCE_EVAL
"""


def run_native_validation(
    capabilities: CP2KCapabilities,
    work_dir: str | Path,
    *,
    config: Any = None,
    timeout: float = 600.0,
) -> CP2KCapabilities:
    """Run the H2 control case with this binary and validate the native features.

    Pinned to ``OMP_NUM_THREADS=1`` (SCF step counts are not thread-deterministic,
    cp2k-ml-scf RECONSTRUCTION.md 4g). Writes ``h2_validation.inp/.out`` into
    ``work_dir``. Raises :class:`CapabilityUnavailableError` when the binary is
    not available; a run that fails leaves the features unvalidated.
    """

    if not capabilities.available or capabilities.resolved_path is None:
        raise CapabilityUnavailableError("no working CP2K binary to validate")
    settings = _settings(config)
    directory = Path(work_dir)
    directory.mkdir(parents=True, exist_ok=True)
    input_path = directory / "h2_validation.inp"
    output_path = directory / "h2_validation.out"
    input_path.write_text(NATIVE_VALIDATION_INPUT, encoding="utf-8")
    environment = {**os.environ, **(dict(settings.ml_scf.environment) if settings else {}), "OMP_NUM_THREADS": "1"}
    try:
        completed = subprocess.run(
            [capabilities.resolved_path, "-i", input_path.name, "-o", output_path.name],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        LOGGER.warning("native validation run failed: %s", exc)
        return capabilities.with_validation(
            {"native_scf": False, "ao_matrix_printing": False}, {"native_scf": f"run failed: {exc}"}
        )
    text = output_path.read_text(encoding="utf-8", errors="ignore") if output_path.is_file() else ""
    if completed.returncode != 0:
        text = text + "\n" + (completed.stderr or "")
    return validate_native_features(capabilities, text)


def _features(version: str | None, flags: Sequence[str], *, available: bool) -> dict[str, FeatureStatus]:
    tokens = set(flags)
    native = available and version is not None
    ml = native and TOKEN_ML_SCF in tokens
    return {
        "native_scf": FeatureStatus(
            "native_scf", native, f"banner version {version}" if native else "no working CP2K banner"
        ),
        "ao_matrix_printing": FeatureStatus(
            "ao_matrix_printing",
            native,
            "upstream CP2K feature, present in every working build" if native else "no working CP2K banner",
        ),
        "ml_density_guess": FeatureStatus(
            "ml_density_guess",
            ml,
            f"cp2kflags contains '{TOKEN_ML_SCF}'" if ml else f"cp2kflags lacks '{TOKEN_ML_SCF}'",
        ),
        "ml_inprocess_guess": FeatureStatus(
            "ml_inprocess_guess",
            False,
            "no build token advertises mode B; 'libtorch' alone does not prove it "
            "(cp2k-ml-scf hook 0009 is a placeholder)",
        ),
        "ml_mixing": FeatureStatus(
            "ml_mixing",
            native and TOKEN_ML_MIXING in tokens,
            f"cp2kflags contains '{TOKEN_ML_MIXING}'"
            if TOKEN_ML_MIXING in tokens
            else f"cp2kflags lacks '{TOKEN_ML_MIXING}'",
        ),
        "scf_controller": FeatureStatus(
            "scf_controller", False, "no known build advertises it (existed only in the lost fork)"
        ),
        "pulay_recorder": FeatureStatus(
            "pulay_recorder", False, "no known build advertises it (existed only in the lost fork)"
        ),
    }


def _settings(config: Any) -> Any:
    if config is None:
        return None
    from gmd_active_learning.adapters.dft.scf.config import scf_config

    try:
        return scf_config(config)
    except Exception as exc:  # A probe must never take the caller down with it.
        LOGGER.warning("ignoring unreadable SCF configuration in capability probe: %s", exc)
        return None


def _configured_executable(executable: str | Path | None, settings: Any) -> str | None:
    if executable:
        return str(executable)
    if settings is not None:
        for value in (settings.ml_scf.executable, settings.capabilities.executable):
            if value:
                return str(value)
    return os.environ.get("CP2K_BIN") or None


def _run_version(
    executable: str,
    runner: Runner | None,
    environment: Mapping[str, str],
    timeout: float,
) -> tuple[str | None, int | None]:
    command = [executable, "--version"]
    full_env = {**os.environ, **environment}
    if runner is not None:
        try:
            accepts_env = len(inspect.signature(runner).parameters) >= 2
        except (TypeError, ValueError):
            accepts_env = False
        try:
            text = runner(command, full_env) if accepts_env else runner(command)
        except Exception:  # A probe must never take the caller down with it.
            return None, None
        return text, 0
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False, env=full_env
        )
    except (OSError, subprocess.SubprocessError) as exc:
        LOGGER.warning("CP2K probe failed: %s", exc)
        return None, None
    return (completed.stdout or "") + (completed.stderr or ""), completed.returncode


__all__ = [
    "CAPABILITY_CONTRACT_VERSION",
    "CP2K_CANDIDATES",
    "CP2K_ML_SCF_DECLARED",
    "FEATURES",
    "CP2KCapabilities",
    "FeatureStatus",
    "capabilities_from_banner",
    "detect_cp2k_capabilities",
    "parse_version_banner",
    "require_feature",
    "require_ml_scf",
    "run_native_validation",
    "validate_native_features",
]
