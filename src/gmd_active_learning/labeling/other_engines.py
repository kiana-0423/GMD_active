# Ported from PHAL (hotspot_al.labeling.gaussian_input, hotspot_al.labeling.orca_input) at 96263b6;
# MIT License, see NOTICE.
"""UNVALIDATED Gaussian / ORCA input writers (optional tooling; CP2K is the labeling engine).

Input generation only: there is no backend, runner or parser for these engines, and nothing here has
been checked against a real Gaussian or ORCA run. Fixes relative to PHAL: functional and basis are
explicit, validated settings (PHAL read a ``dft`` section the strict schema rejected, so the defaults
always applied, and the Gaussian default ``DZVP`` is not a Gaussian keyword); the Gaussian H-only
optimisation now freezes every non-cap atom with Gaussian's freeze code (PHAL's ``opt=modredundant``
had no frozen-atom section, so it optimised everything).
"""

from __future__ import annotations

from pathlib import Path

from ase.data import chemical_symbols
from pydantic import BaseModel, ConfigDict

from gmd_active_learning.contracts.regions import ExtractedRegion, RegionRole
from gmd_active_learning.utils.electronic_state import resolve_electronic_state

VALIDATED = False


class OtherEngineSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    functional: str = "PBE"
    basis: str
    multiplicity: int | None = None
    spin_policy: str = "auto"


def _state(region: ExtractedRegion, settings: OtherEngineSettings):  # noqa: ANN202
    metadata = {"charge": region.charge, **({"multiplicity": region.multiplicity} if region.multiplicity else {})}
    return resolve_electronic_state(
        region.to_atoms(),
        {"multiplicity": settings.multiplicity, "spin_policy": settings.spin_policy},
        region_metadata=metadata,
    )


def build_gaussian_input(region: ExtractedRegion, settings: OtherEngineSettings, *, optimize_h_only: bool = False) -> str:
    state = _state(region, settings)
    method = f"U{settings.functional}" if state.is_open_shell else settings.functional
    caps = {index for index, role in enumerate(region.roles) if role is RegionRole.H_CAP}
    optimize = optimize_h_only and bool(caps)
    route = f"#p {method}/{settings.basis} {'opt' if optimize else 'force'}"
    lines = [route, "", "Hotspot region", "", f"{state.charge} {state.multiplicity}"]
    for index, (number, position) in enumerate(zip(region.numbers, region.positions, strict=True)):
        symbol = chemical_symbols[int(number)]
        freeze = "" if not optimize else (" 0" if index in caps else " -1")
        lines.append(f"{symbol:2s}{freeze} {position[0]: .8f} {position[1]: .8f} {position[2]: .8f}")
    lines.append("")
    return "\n".join(lines)


def build_orca_input(region: ExtractedRegion, settings: OtherEngineSettings) -> str:
    state = _state(region, settings)
    method = "UKS" if state.is_open_shell else "RKS"
    lines = [f"! {method} {settings.functional} {settings.basis} EnGrad", "", f"* xyz {state.charge} {state.multiplicity}"]
    for number, position in zip(region.numbers, region.positions, strict=True):
        lines.append(f"  {chemical_symbols[int(number)]:2s} {position[0]: .8f} {position[1]: .8f} {position[2]: .8f}")
    lines.append("*")
    return "\n".join(lines)


def write_gaussian_inputs(region: ExtractedRegion, output_dir: str | Path, settings: OtherEngineSettings, *, job_name: str = "hotspot_region") -> dict[str, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{job_name}.gjf"
    path.write_text(build_gaussian_input(region, settings), encoding="utf-8")
    return {"single_point_input": path}


def write_orca_inputs(region: ExtractedRegion, output_dir: str | Path, settings: OtherEngineSettings, *, job_name: str = "hotspot_region") -> dict[str, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{job_name}.inp"
    path.write_text(build_orca_input(region, settings), encoding="utf-8")
    return {"single_point_input": path}


__all__ = [
    "OtherEngineSettings",
    "VALIDATED",
    "build_gaussian_input",
    "build_orca_input",
    "write_gaussian_inputs",
    "write_orca_inputs",
]
