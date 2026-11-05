# Ported from PHAL (hotspot_al.cp2k.scf_history) at 96263b6; MIT License, see NOTICE.
"""Parse and persist conventional CP2K SCF convergence trajectories."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from gmd_active_learning.utils.atomic_io import write_json_atomic

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
_EXPLICIT_ITERATION = re.compile(
    rf"SCF\s+ITERATION\s+(?P<step>\d+)\s+RESIDUAL\s+(?P<residual>{_NUMBER})\s+ENERGY\s+(?P<energy>{_NUMBER})",
    re.IGNORECASE,
)
_SCF_ITERATIONS = re.compile(r"SCF run converged in\s+(\d+)\s+steps?", re.IGNORECASE)
_RESIDUAL = re.compile(rf"(?:SCF|OT).*?residual.*?({_NUMBER})", re.IGNORECASE)
# v2026.2 prints "ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]   -17.22...";
# earlier releases printed "... energy (a.u.):   -17.22...". Requiring the colon
# found no energy at all on v2026.2 output.
_FINAL_ENERGY = re.compile(rf"ENERGY\|.*?energy.*?(?::|\])\s*({_NUMBER})", re.IGNORECASE)
_VERSION_STRING = re.compile(r"CP2K\|\s*version string:\s*(.+)", re.IGNORECASE)
_VERSION_FALLBACK = re.compile(r"\bCP2K\s+version\s+([^\s]+)", re.IGNORECASE)
_SCF_SUCCESS = re.compile(r"SCF (?:run )?converged", re.IGNORECASE)
_SCF_FAILURE = re.compile(r"SCF (?:run )?(?:NOT converged|not converged)|\bABORT\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SCFIteration:
    """One iteration in CP2K native units (energy in Hartree)."""

    step: int
    residual: float | None
    energy: float | None


@dataclass(frozen=True, slots=True)
class SCFHistory:
    """SCF trajectory and the settings required to interpret it."""

    cp2k_version: str | None
    iterations: tuple[SCFIteration, ...]
    eps_scf: float | None
    max_scf: int | None
    converged: bool
    initial_guess: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cp2k_version": self.cp2k_version,
            "iterations": [asdict(iteration) for iteration in self.iterations],
            "eps_scf": self.eps_scf,
            "max_scf": self.max_scf,
            "converged": self.converged,
            "initial_guess": self.initial_guess,
            "energy_unit": "hartree",
        }


def parse_scf_history(output_text: str, *, input_text: str = "") -> SCFHistory:
    """Parse SCF iterations from CP2K text without requiring an executable."""

    iterations = _parse_explicit_iterations(output_text)
    if not iterations:
        iterations = _parse_cp2k_iteration_tables(output_text)
    if not iterations:
        iterations = _parse_final_iteration(output_text)

    version_match = _VERSION_STRING.search(output_text)
    if version_match is not None:
        version = version_match.group(1).strip()
    else:
        fallback = _VERSION_FALLBACK.search(output_text)
        version = fallback.group(1).strip() if fallback is not None else None

    failed = _SCF_FAILURE.search(output_text) is not None
    return SCFHistory(
        cp2k_version=version,
        iterations=tuple(iterations),
        eps_scf=_input_float(input_text, "EPS_SCF"),
        max_scf=_input_int(input_text, "MAX_SCF"),
        converged=not failed and _SCF_SUCCESS.search(output_text) is not None,
        initial_guess=_input_word(input_text, "SCF_GUESS"),
    )


def record_scf_history(
    output_path: str | Path,
    input_path: str | Path,
    *,
    config: Mapping[str, Any],
    task_id: str | None = None,
    electronic_state: Mapping[str, Any] | None = None,
) -> Path | None:
    """Write an enabled task's SCF history and return its path."""

    output = Path(output_path)
    input_file = Path(input_path)
    return record_scf_history_text(
        output.read_text(encoding="utf-8", errors="ignore"),
        input_path=input_file,
        config=config,
        task_id=task_id,
        electronic_state=electronic_state,
    )


def record_scf_history_text(
    output_text: str,
    *,
    input_path: str | Path,
    config: Mapping[str, Any],
    task_id: str | None = None,
    electronic_state: Mapping[str, Any] | None = None,
) -> Path | None:
    """Parse in-memory CP2K stdout and atomically persist it when enabled."""

    recording = _recording_config(config)
    if not bool(recording.get("enabled", False)):
        return None
    input_file = Path(input_path)
    input_text = input_file.read_text(encoding="utf-8", errors="ignore") if input_file.is_file() else ""
    history = parse_scf_history(output_text, input_text=input_text)
    output_root = Path(str(recording["output_dir"]))
    calculation_id = task_id or input_file.stem
    target = output_root / calculation_id / "scf_history.json"
    written = write_json_atomic(target, history.to_dict())

    # The density is an optional companion to the trajectory. A failure to
    # record it must not discard the trajectory that was already parsed, so it
    # is attempted after the JSON is on disk.
    record_scf_density(
        output_text,
        config=config,
        output_root=output_root,
        task_id=calculation_id,
        history=history,
        electronic_state=electronic_state,
    )
    return written


def record_scf_density(
    output_text: str,
    *,
    config: Mapping[str, Any],
    output_root: str | Path,
    task_id: str,
    history: SCFHistory | None = None,
    electronic_state: Mapping[str, Any] | None = None,
) -> Path | None:
    """Persist the AO density CP2K printed, when density recording is on."""

    from gmd_active_learning.adapters.dft.scf.recording.density import density_recording_enabled, record_density

    if not density_recording_enabled(config):
        return None

    recording = _recording_config(config)
    artifacts = record_density(
        output_text,
        output_root=output_root,
        task_id=task_id,
        config=config,
        electronic_state=electronic_state,
        cp2k_version=None if history is None else history.cp2k_version,
        scf_converged=bool(history.converged) if history is not None else False,
        strict=bool(recording.get("density_strict", False)),
    )
    # Deliberate extension over PHAL (ledger SCF-001): with record_trajectory on,
    # the labeling path now also keeps every printed iteration, honouring
    # trajectory_strict (declared but never read in PHAL, finding F-23).
    from gmd_active_learning.adapters.dft.scf.recording.trajectory import (
        record_scf_trajectory,
        trajectory_recording_enabled,
    )

    if trajectory_recording_enabled(config):
        record_scf_trajectory(
            output_text,
            output_root=output_root,
            task_id=task_id,
            config=config,
            electronic_state=electronic_state,
            cp2k_version=None if history is None else history.cp2k_version,
            scf_converged=bool(history.converged) if history is not None else False,
            strict=bool(recording.get("trajectory_strict", False)),
        )
    return None if artifacts is None else artifacts.directory


def _recording_config(config: Any) -> Mapping[str, Any]:
    from gmd_active_learning.adapters.dft.scf.config import scf_config

    return scf_config(config).recording.model_dump()


def _parse_explicit_iterations(text: str) -> list[SCFIteration]:
    return [
        SCFIteration(
            step=int(match.group("step")),
            residual=_float(match.group("residual")),
            energy=_float(match.group("energy")),
        )
        for match in _EXPLICIT_ITERATION.finditer(text)
    ]


def _parse_cp2k_iteration_tables(text: str) -> list[SCFIteration]:
    """Parse CP2K's aligned ``Step / Convergence / Total energy`` tables."""

    lines = text.splitlines()
    iterations: list[SCFIteration] = []
    for header_index, header in enumerate(lines):
        lowered = header.lower()
        if "step" not in lowered or "convergence" not in lowered or "total energy" not in lowered:
            continue
        convergence_column = lowered.index("convergence")
        energy_column = lowered.index("total energy")
        if energy_column <= convergence_column:
            continue
        for line in lines[header_index + 1 :]:
            stripped = line.strip()
            if not stripped or set(stripped) <= {"-", " "}:
                continue
            step_match = re.match(r"^\s*(\d+)\s+", line)
            if step_match is None:
                if iterations and ("SCF run" in line or "ENERGY|" in line):
                    break
                continue
            # An AO matrix row also starts with an integer:
            #
            #     Step  Update method  ...          <- header
            #        1  P_Mix/Diag. 0.40E+00 ...    <- an SCF iteration
            #        1     1  H  1s   0.123456      <- a DENSITY MATRIX row
            #
            # Without this check every printed matrix row is counted as an SCF
            # iteration, which is only visible once AO_MATRICES printing is
            # switched on -- i.e. exactly when trajectories are being recorded.
            # A real row names its update method, so the second field carries
            # letters ("P_Mix/Diag.", "DIIS/Diag.", "OT/DIIS"); a matrix row's
            # second field is another index.
            fields = line.split()
            if len(fields) < 2 or not any(character.isalpha() for character in fields[1]):
                continue
            residual, energy = _row_tail_values(line)
            if residual is None or energy is None:
                residual = _first_number(line[convergence_column:energy_column])
                energy = _first_number(line[energy_column:])
            iterations.append(SCFIteration(step=int(step_match.group(1)), residual=residual, energy=energy))
    return iterations


def _parse_final_iteration(text: str) -> list[SCFIteration]:
    steps = _SCF_ITERATIONS.findall(text)
    energies = _FINAL_ENERGY.findall(text)
    residuals = _RESIDUAL.findall(text)
    if not steps and not energies and not residuals:
        return []
    return [
        SCFIteration(
            step=int(steps[-1]) if steps else 0,
            residual=_float(residuals[-1]) if residuals else None,
            energy=_float(energies[-1]) if energies else None,
        )
    ]


def _row_tail_values(line: str) -> tuple[float | None, float | None]:
    values = [_float(value) for value in re.findall(_NUMBER, line)]
    # Common CP2K rows end with convergence, total energy, and energy change.
    return (values[-3], values[-2]) if len(values) >= 4 else (None, None)


def _first_number(value: str) -> float | None:
    match = re.search(_NUMBER, value)
    return _float(match.group(0)) if match is not None else None


def _input_float(text: str, keyword: str) -> float | None:
    match = re.search(rf"^\s*{re.escape(keyword)}\s+({_NUMBER})", text, re.IGNORECASE | re.MULTILINE)
    return _float(match.group(1)) if match is not None else None


def _input_int(text: str, keyword: str) -> int | None:
    value = _input_float(text, keyword)
    return int(value) if value is not None else None


def _input_word(text: str, keyword: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(keyword)}\s+(\S+)", text, re.IGNORECASE | re.MULTILINE)
    return match.group(1).upper() if match is not None else None


def _float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


__all__ = [
    "SCFHistory",
    "SCFIteration",
    "parse_scf_history",
    "record_scf_density",
    "record_scf_history",
    "record_scf_history_text",
]
