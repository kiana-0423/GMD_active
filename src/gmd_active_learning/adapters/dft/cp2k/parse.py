# Ported from PHAL (hotspot_al.cp2k.cp2k_force_parser, hotspot_al.cp2k.cp2k_result) at 96263b6;
# MIT License, see NOTICE.
"""Parse CP2K output (validated on CP2K 2026.2) into SI-free GMD units (eV, eV/A, eV/A^3).

Fixes relative to PHAL, reproduced on real CP2K 2026.2 output (``tests/fixtures/cp2k/``):

- F-12: CP2K 2026.2 prints ``ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]   -1.14`` without a
  colon, so PHAL's ``ENERGY\\|.*?energy.*?:`` never matched and every label lost its energy. The
  energy regex now accepts the bracketed/parenthesised unit with or without a colon and converts by
  the printed unit.
- The stress tensor is printed as ``STRESS| Analytical stress tensor [bar]`` with lower-case
  ``x/y/z`` rows; PHAL only recognised ``[GPa]``. Both are read, converted to eV/A^3 and flipped to
  the ASE sign convention (CP2K reports the negative, as in ``ase.calculators.cp2k``).
- ``FORCES| Sum`` is indented by one space in real output; the block end no longer depends on column 0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ase import units

from gmd_active_learning.core.exceptions import LabelingError

HARTREE_PER_BOHR_TO_EV_PER_A = units.Hartree / units.Bohr

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
# Old style (ATOMIC FORCES in [a.u.]) and FORCE_EVAL print (FORCES| ...) blocks, as in PHAL.
_FORCE_HEADER_OLD = re.compile(r"ATOMIC FORCES in \[a\.u\.\]", re.IGNORECASE)
_FORCE_LINE_OLD = re.compile(rf"^\s*\d+\s+\d+\s+[A-Za-z]{{1,3}}\s+({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\s*$")
_FORCE_HEADER = re.compile(r"FORCES\|\s+Atomic forces \[hartree/bohr\]", re.IGNORECASE)
_FORCE_LINE = re.compile(rf"^\s*FORCES\|\s+\d+\s+({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\s+{_NUMBER}\s*$")
_ENERGY = re.compile(
    rf"^\s*ENERGY\|\s*Total FORCE_EVAL\s*\(\s*\w+\s*\)\s*energy\s*"
    rf"(?:\[(?P<unit>[^\]]+)\]|\((?P<unit2>[^)]+)\))?\s*:?\s*(?P<value>{_NUMBER})\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# PHAL's pattern, kept only to document F-12 in tests.
LEGACY_PHAL_ENERGY = re.compile(rf"ENERGY\|.*?energy.*?:\s*({_NUMBER})", re.IGNORECASE)
_SCF_STEPS = re.compile(r"SCF run converged in\s+(\d+)\s+steps?", re.IGNORECASE)
_SCF_SUCCESS = re.compile(r"SCF (?:run )?converged", re.IGNORECASE)
_SCF_NOT_CONVERGED = re.compile(r"SCF (?:run )?NOT converged", re.IGNORECASE)
_ABORT = re.compile(r"\bABORT\b", re.IGNORECASE)
_VERSION_STRING = re.compile(r"CP2K\|\s*version string:\s*(.+)", re.IGNORECASE)
_VERSION_FALLBACK = re.compile(r"\bCP2K\s+version\s+([^\s,]+)", re.IGNORECASE)
_VERSION_NUMBER = re.compile(r"(\d{4}\.\d+(?:\.\d+)?)")
_GIT_COMMIT = re.compile(r"\bgit:([0-9a-f]{6,40})\b", re.IGNORECASE)
_STRESS_HEADER = re.compile(r"STRESS\|?\s*(?:Analytical|Numerical)\s+stress tensor\s*\[(?P<unit>GPa|bar)\]", re.IGNORECASE)
_STRESS_ROW = re.compile(rf"^\s*(?:STRESS\|)?\s*[XYZxyz]\s+({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\s*$")
_ML_ACCELERATOR = re.compile(r"ML[-_ ]SCF\|.*accelerator.*enabled", re.IGNORECASE)
_ML_FALLBACK = re.compile(r"ML[-_ ]SCF\|.*fallback.*(?:used|true|enabled)", re.IGNORECASE)
_NORMAL_END = re.compile(r"PROGRAM ENDED AT")
_GEO_OPT_DONE = re.compile(r"GEOMETRY OPTIMIZATION COMPLETED", re.IGNORECASE)
_GEO_OPT_MAX_ITER = re.compile(r"MAXIMUM NUMBER OF OPTIMIZATION STEPS REACHED", re.IGNORECASE)
_SCF_TABLE_HEADER = re.compile(r"^\s*Step\s+Update method\s+Time\s+Convergence")
_ART = re.compile(r"^[\s*\\/|_O]+")


@dataclass(frozen=True, slots=True)
class CP2KOutput:
    """Values and markers from one CP2K output, in eV, eV/A and eV/A^3 (ASE sign convention)."""

    energy: float | None
    forces: np.ndarray | None
    stress: np.ndarray | None
    scf_converged: bool
    scf_success_marker: bool
    scf_not_converged_marker: bool
    aborted: bool
    abort_message: str | None
    scf_steps: int | None
    scf_residual: float | None
    version: str | None
    version_string: str | None
    git_revision: str | None
    normal_termination: bool
    geo_opt_completed: bool
    geo_opt_max_iter_reached: bool
    ml_scf_accelerator: bool
    ml_scf_fallback: bool
    stress_raw: tuple[tuple[float, ...], ...] | None = None
    stress_unit: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_force_rows(self) -> int:
        return 0 if self.forces is None else int(self.forces.shape[0])

    def provenance(self) -> dict[str, Any]:
        """JSON-safe summary without the force array."""

        return {
            "energy_ev": self.energy,
            "stress_ev_per_a3": None if self.stress is None else self.stress.tolist(),
            "stress_raw": None if self.stress_raw is None else [list(row) for row in self.stress_raw],
            "stress_unit": self.stress_unit,
            "scf_converged": self.scf_converged,
            "scf_steps": self.scf_steps,
            "scf_residual": self.scf_residual,
            "aborted": self.aborted,
            "abort_message": self.abort_message,
            "cp2k_version": self.version,
            "cp2k_version_string": self.version_string,
            "cp2k_git_revision": self.git_revision,
            "normal_termination": self.normal_termination,
            "geo_opt_completed": self.geo_opt_completed,
            "geo_opt_max_iter_reached": self.geo_opt_max_iter_reached,
            "ml_scf_accelerator": self.ml_scf_accelerator,
            "ml_scf_fallback": self.ml_scf_fallback,
        }


def _force_eval_blocks(lines: list[str]) -> list[list[list[float]]]:
    blocks: list[list[list[float]]] = []
    current: list[list[float]] = []
    capture = False
    for line in lines:
        if _FORCE_HEADER.search(line):
            if current:
                blocks.append(current)
            current, capture = [], True
            continue
        if not capture:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("FORCES| Sum"):
            if current:
                blocks.append(current)
                current = []
            capture = False
            continue
        match = _FORCE_LINE.match(line)
        if match:
            current.append([float(match.group(i)) for i in range(1, 4)])
    if current:
        blocks.append(current)
    return blocks


def _atomic_force_blocks(lines: list[str]) -> list[list[list[float]]]:
    blocks: list[list[list[float]]] = []
    current: list[list[float]] = []
    seen_header = capture = False
    for line in lines:
        if _FORCE_HEADER_OLD.search(line):
            if current:
                blocks.append(current)
            current, seen_header, capture = [], True, False
            continue
        if seen_header and "Atom   Kind   Element" in line:
            capture = True
            continue
        if not capture:
            continue
        if not line.strip() or line.lstrip().startswith("SUM"):
            if current:
                blocks.append(current)
                current = []
            capture = seen_header = False
            continue
        match = _FORCE_LINE_OLD.match(line)
        if match:
            current.append([float(match.group(i)) for i in range(1, 4)])
    if current:
        blocks.append(current)
    return blocks


def parse_forces_text(text: str) -> np.ndarray | None:
    """Last force block (eV/A), preferring the ``FORCES|`` print; None if there is none."""

    lines = text.splitlines()
    blocks = _force_eval_blocks(lines) or _atomic_force_blocks(lines)
    if not blocks:
        return None
    return np.asarray(blocks[-1], dtype=float) * HARTREE_PER_BOHR_TO_EV_PER_A


def parse_cp2k_forces(path: str | Path) -> np.ndarray:
    """PHAL-compatible entry point: forces in eV/A or :class:`LabelingError` if no block exists."""

    forces = parse_forces_text(Path(path).read_text(encoding="utf-8", errors="ignore"))
    if forces is None:
        raise LabelingError(f"No CP2K force block found in {path}")
    return forces


def _energy_unit_factor(unit: str | None) -> float:
    normalized = (unit or "a.u.").strip().lower()
    if normalized in {"hartree", "a.u.", "au", "a.u", "ha"}:
        return float(units.Hartree)
    if normalized == "ev":
        return 1.0
    raise LabelingError(f"unrecognised CP2K energy unit {unit!r}")


def parse_energy_text(text: str) -> float | None:
    """Last ``ENERGY| Total FORCE_EVAL`` value in eV."""

    matches = list(_ENERGY.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    return float(last.group("value")) * _energy_unit_factor(last.group("unit") or last.group("unit2"))


def parse_stress_text(text: str) -> tuple[np.ndarray | None, tuple[tuple[float, ...], ...] | None, str | None]:
    """Last analytic/numeric stress tensor: (eV/A^3 in ASE convention, raw rows, raw unit)."""

    lines = text.splitlines()
    result: tuple[np.ndarray | None, tuple[tuple[float, ...], ...] | None, str | None] = (None, None, None)
    for index, line in enumerate(lines):
        header = _STRESS_HEADER.search(line)
        if header is None or "eigen" in line.lower():
            continue
        rows: list[list[float]] = []
        for candidate in lines[index + 1 : index + 12]:
            match = _STRESS_ROW.match(candidate)
            if match is not None:
                rows.append([float(match.group(i)) for i in range(1, 4)])
                if len(rows) == 3:
                    break
        if len(rows) == 3:
            unit = header.group("unit")
            factor = units.GPa if unit.lower() == "gpa" else units.bar
            raw = np.asarray(rows, dtype=float)
            result = (-raw * factor, tuple(tuple(row) for row in rows), unit)
    return result


def _scf_residual(lines: list[str]) -> float | None:
    residual: float | None = None
    in_table = False
    for line in lines:
        if _SCF_TABLE_HEADER.match(line):
            in_table = True
            continue
        if not in_table:
            continue
        stripped = line.strip()
        if "SCF run" in line or stripped.startswith("***"):
            in_table = False
            continue
        tokens = stripped.split()
        if tokens and tokens[0].isdigit() and len(tokens) >= 5:
            try:
                residual = float(tokens[-3])
            except ValueError:
                continue
    return residual


def _abort_message(lines: list[str]) -> str | None:
    for index, line in enumerate(lines):
        if "[ABORT]" not in line:
            continue
        parts: list[str] = []
        for following in lines[index + 1 : index + 6]:
            stripped = following.strip()
            if not stripped or set(stripped) <= {"*"}:
                break
            cleaned = _ART.sub("", stripped).rstrip("*").strip()
            if cleaned:
                parts.append(cleaned)
        return " ".join(parts) or "CP2K aborted"
    return None


def parse_cp2k_output(text: str) -> CP2KOutput:
    lines = text.splitlines()
    success = _SCF_SUCCESS.search(text) is not None
    not_converged = _SCF_NOT_CONVERGED.search(text) is not None
    aborted = _ABORT.search(text) is not None
    steps = _SCF_STEPS.findall(text)
    version_match = _VERSION_STRING.search(text)
    version_string = version_match.group(1).strip() if version_match else None
    if version_string is None:
        fallback = _VERSION_FALLBACK.search(text)
        version_string = f"CP2K version {fallback.group(1)}" if fallback else None
    number = _VERSION_NUMBER.search(version_string) if version_string else None
    git = _GIT_COMMIT.search(text)
    stress, stress_raw, stress_unit = parse_stress_text(text)
    return CP2KOutput(
        energy=parse_energy_text(text),
        forces=parse_forces_text(text),
        stress=stress,
        # F-27 (PHAL rule, kept): a success marker and neither a non-convergence nor an ABORT marker.
        scf_converged=success and not not_converged and not aborted,
        scf_success_marker=success,
        scf_not_converged_marker=not_converged,
        aborted=aborted,
        abort_message=_abort_message(lines) if aborted else None,
        scf_steps=int(steps[-1]) if steps else None,
        scf_residual=_scf_residual(lines),
        version=number.group(1) if number else None,
        version_string=version_string,
        git_revision=git.group(1).lower() if git else None,
        normal_termination=_NORMAL_END.search(text) is not None,
        geo_opt_completed=_GEO_OPT_DONE.search(text) is not None,
        geo_opt_max_iter_reached=_GEO_OPT_MAX_ITER.search(text) is not None,
        ml_scf_accelerator=_ML_ACCELERATOR.search(text) is not None,
        ml_scf_fallback=_ML_FALLBACK.search(text) is not None,
        stress_raw=stress_raw,
        stress_unit=stress_unit,
    )


def parse_cp2k_output_file(path: str | Path) -> CP2KOutput:
    return parse_cp2k_output(Path(path).read_text(encoding="utf-8", errors="ignore"))


def read_last_xyz_frame(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Symbols and positions (A) of the last frame of a CP2K ``*-pos-1.xyz`` trajectory."""

    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    frames: list[tuple[list[str], np.ndarray]] = []
    index = 0
    while index < len(lines):
        header = lines[index].strip()
        if not header:
            index += 1
            continue
        try:
            count = int(header)
        except ValueError as exc:
            raise LabelingError(f"malformed XYZ trajectory {path} at line {index + 1}") from exc
        block = lines[index + 2 : index + 2 + count]
        if len(block) < count:
            break  # truncated trailing frame (job killed mid-write): use the previous complete frame
        symbols = [row.split()[0] for row in block]
        positions = np.asarray([[float(value) for value in row.split()[1:4]] for row in block], dtype=float)
        frames.append((symbols, positions))
        index += 2 + count
    if not frames:
        raise LabelingError(f"no complete frame in XYZ trajectory {path}")
    return frames[-1]


__all__ = [
    "CP2KOutput",
    "HARTREE_PER_BOHR_TO_EV_PER_A",
    "LEGACY_PHAL_ENERGY",
    "parse_cp2k_forces",
    "parse_cp2k_output",
    "parse_cp2k_output_file",
    "parse_energy_text",
    "parse_forces_text",
    "parse_stress_text",
    "read_last_xyz_frame",
]
