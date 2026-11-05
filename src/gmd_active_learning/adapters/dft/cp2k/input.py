# Ported from PHAL (hotspot_al.cp2k.cp2k_input, hotspot_al.cp2k.cp2k_templates) at 96263b6; MIT License, see NOTICE.
"""CP2K input generation for an :class:`~gmd_active_learning.contracts.ExtractedRegion`.

The text layout follows PHAL byte for byte for equivalent settings (parity test in
``tests/adapters/dft/test_cp2k_input.py``). Deliberate deviations:

- Boundary conditions come from ``region.pbc`` exactly: ``PERIODIC NONE`` for clusters, ``XY`` etc.
  for slabs, ``XYZ`` for blocks, in both ``&CELL`` and ``&POISSON``; the Poisson solver is chosen per
  periodicity class and an unsupported solver/periodicity pair raises instead of silently changing
  the boundary conditions (PHAL used one global ``cp2k.periodic`` for every region).
- An isolated region whose cell is missing, skewed, short of ``min_vacuum`` or (for the WAVELET
  solver) not cubic gets a padded box (cubic for WAVELET: CP2K 2026.2 aborts with "Poisson solver for
  non cubic cells not yet implemented", which made PHAL's padded non-cubic cluster cells unlabelable);
  the coordinate shift is recorded so H-cap relaxations are mapped back exactly. Periodic regions keep
  their cell; non-periodic directions of a slab must keep ``min_vacuum`` to the faces.
- Charge and multiplicity come from the region (``ExtractedRegion.charge``/``multiplicity``), resolved
  through :func:`gmd_active_learning.utils.electronic_state.resolve_electronic_state`.
- Point-charge embedding is refused with :class:`UnsupportedCombinationError` (known issue: the
  generator emits a pure Quickstep input and cannot consume point charges).
- Basis/potential file names, per-element kinds and extra sections are configurable; the analytic
  stress tensor is requested for fully periodic regions (``stress: auto``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any

import numpy as np
from ase.data import chemical_symbols

from gmd_active_learning.adapters.dft.cp2k.config import CP2KConfig, RetryStep
from gmd_active_learning.contracts.regions import ExtractedRegion, RegionRole
from gmd_active_learning.core.exceptions import (
    ConfigurationError,
    LabelingError,
    UnsupportedCombinationError,
)
from gmd_active_learning.utils.electronic_state import ElectronicState, resolve_electronic_state

LOGGER = logging.getLogger(__name__)

_AXES = "XYZ"
SOLVER_SUPPORT: dict[str, frozenset[str]] = {
    "PERIODIC": frozenset({"XYZ"}),
    "WAVELET": frozenset({"NONE", "XZ", "XYZ"}),
    "MT": frozenset({"NONE", "X", "Y", "Z", "XY", "XZ", "YZ"}),
    "ANALYTIC": frozenset({"NONE", "X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"}),
    "MULTIPOLE": frozenset({"NONE"}),
}


def periodicity(pbc: tuple[bool, bool, bool] | Any) -> str:
    """``(True, True, False)`` -> ``"XY"``; all False -> ``"NONE"``."""

    label = "".join(axis for axis, flag in zip(_AXES, pbc) if bool(flag))
    return label or "NONE"


def poisson_solver(periodic: str, config: CP2KConfig) -> str:
    if periodic == "NONE":
        solver = config.poisson.non_periodic_solver
    elif periodic == "XYZ":
        solver = config.poisson.periodic_solver
    else:
        solver = config.poisson.partially_periodic_solver
    if periodic not in SOLVER_SUPPORT[solver]:
        raise UnsupportedCombinationError(
            f"CP2K Poisson solver {solver} does not support PERIODIC {periodic}; choose another solver in "
            "cp2k.poisson (the region's boundary conditions are never changed to fit the solver)"
        )
    return solver


def wants_stress(periodic: str, config: CP2KConfig) -> bool:
    if config.stress == "never":
        return False
    if config.stress == "always":
        if periodic == "NONE":
            raise UnsupportedCombinationError("cp2k.stress=always is meaningless for a non-periodic region")
        return True
    return periodic == "XYZ"


@dataclass(frozen=True, slots=True)
class PreparedGeometry:
    """Coordinates and cell in the CP2K frame: ``cp2k_positions = region_positions + offset``."""

    positions: np.ndarray
    cell: np.ndarray
    offset: np.ndarray
    periodic: str
    rebuilt_cell: bool

    def to_region_frame(self, positions: np.ndarray) -> np.ndarray:
        return np.asarray(positions, dtype=float) - self.offset


def _face_distances(positions: np.ndarray, cell: np.ndarray, axis: int) -> tuple[float, float]:
    fractional = np.linalg.solve(cell.T, positions.T).T[:, axis]
    others = [cell[(axis + 1) % 3], cell[(axis + 2) % 3]]
    spacing = abs(float(np.linalg.det(cell))) / float(np.linalg.norm(np.cross(*others)))
    return float(fractional.min() * spacing), float((1.0 - fractional.max()) * spacing)


def prepare_geometry(region: ExtractedRegion, config: CP2KConfig) -> PreparedGeometry:
    periodic = periodicity(region.pbc)
    positions = np.asarray(region.positions, dtype=float)
    cell = None if region.cell is None else np.asarray(region.cell, dtype=float)
    usable = cell is not None and abs(float(np.linalg.det(cell))) > 1.0e-8
    if periodic == "NONE":
        # CP2K's WAVELET solver aborts for non-cubic cells ("Poisson solver for non cubic cells not yet
        # implemented", CP2K 2026.2), so isolated regions get a cubic box when that solver is used.
        cubic = config.poisson.non_periodic_solver == "WAVELET"
        acceptable = usable and np.allclose(cell - np.diag(np.diag(cell)), 0.0, atol=1.0e-10)  # type: ignore[operator]
        if acceptable and cubic:
            acceptable = bool(np.allclose(np.diag(cell), cell[0, 0], atol=1.0e-8))  # type: ignore[index]
        if acceptable and all(min(_face_distances(positions, cell, axis)) >= config.min_vacuum for axis in range(3)):  # type: ignore[arg-type]
            return PreparedGeometry(positions, cell, np.zeros(3), periodic, rebuilt_cell=False)  # type: ignore[arg-type]
        low, high = positions.min(axis=0), positions.max(axis=0)
        lengths = (high - low) + 2.0 * config.cluster_padding
        if cubic:
            lengths = np.full(3, float(lengths.max()))
        offset = 0.5 * lengths - 0.5 * (low + high)
        if usable:
            LOGGER.info(
                "region %s: isolated region's cell is skewed, non-cubic for WAVELET or lacks vacuum; using a padded box",
                region.region_id,
            )
        return PreparedGeometry(positions + offset, np.diag(lengths), offset, periodic, rebuilt_cell=True)
    if not usable:
        raise LabelingError(f"region {region.region_id}: periodic ({periodic}) region needs a non-degenerate cell")
    for axis, name in enumerate(_AXES):
        if name in periodic:
            continue
        low, high = _face_distances(positions, cell, axis)  # type: ignore[arg-type]
        if min(low, high) < config.min_vacuum:
            raise LabelingError(
                f"region {region.region_id}: only {min(low, high):.2f} A between atoms and the cell face along the "
                f"non-periodic {name} direction (cp2k.min_vacuum = {config.min_vacuum} A); extend the vacuum in "
                "extraction instead of letting CP2K wrap atoms"
            )
    return PreparedGeometry(positions, cell, np.zeros(3), periodic, rebuilt_cell=False)  # type: ignore[arg-type]


def region_electronic_state(region: ExtractedRegion, config: CP2KConfig) -> ElectronicState:
    metadata: dict[str, Any] = {"charge": int(region.charge)}
    if region.multiplicity is not None:
        metadata["multiplicity"] = int(region.multiplicity)
    settings = {"charge": int(region.charge), "multiplicity": config.multiplicity, "spin_policy": config.spin_policy}
    return resolve_electronic_state(region.to_atoms(), settings, region_metadata=metadata, logger=LOGGER)


def check_uncapped_isolated_region(region: ExtractedRegion, config: CP2KConfig, periodic: str) -> None:
    """Fail closed on an isolated fragment whose broken bonds were neither capped nor embedded."""

    if not bool(region.metadata.get("h_capping_skipped")) or periodic != "NONE":
        return
    if config.uncapped_fragment_policy == "allow":
        return
    message = (
        f"region {region.region_id} has uncapped broken bonds "
        f"({region.metadata.get('h_capping_skipped_reason', 'unknown')}) and would be computed as a non-periodic "
        "isolated cluster; reference forces near the boundary would include artificial surface states. Use a "
        "chemically valid boundary treatment or a periodic region; cp2k.uncapped_fragment_policy=warn/allow is "
        "an explicit expert override only."
    )
    if config.uncapped_fragment_policy == "error":
        raise UnsupportedCombinationError(message)
    LOGGER.warning("%s", message)


def check_embedding(region: ExtractedRegion, *, embedding_enabled: bool) -> None:
    embedding = region.metadata.get("embedding")
    region_embedded = isinstance(embedding, dict) and (
        embedding.get("status") == "ok" or bool(embedding.get("enabled")) or bool(embedding.get("point_charges"))
    )
    if embedding_enabled or region_embedded:
        raise UnsupportedCombinationError(
            "point-charge embedding is enabled, but the CP2K generator emits a pure Quickstep input and cannot "
            "consume point charges (known issue). Disable embedding for a pure-QM label, or provide a separately "
            "validated QM/MM backend."
        )


def settings_with_step(config: CP2KConfig, step: RetryStep | None, *, max_scf_cap: int | None = None) -> CP2KConfig:
    """Apply one retry step to the *base* settings; multipliers never compound across retries."""

    if step is None:
        return config
    cap = max_scf_cap if max_scf_cap is not None else config.retry.max_scf_cap
    data = config.model_dump()
    if step.max_scf is not None:
        data["max_scf"] = min(step.max_scf, cap)
    if step.max_scf_multiplier is not None:
        data["max_scf"] = min(int(math.ceil(config.max_scf * step.max_scf_multiplier)), cap)
    if step.use_ot is not None:
        data["use_ot"] = step.use_ot
    if step.scf_guess is not None:
        data["scf_guess"] = step.scf_guess
    for key, value in (("method", step.mixing_method), ("alpha", step.mixing_alpha), ("nbuffer", step.mixing_nbuffer)):
        if value is not None:
            data["mixing"][key] = value
    if step.h_opt_max_iter_multiplier is not None:
        data["h_only_opt"]["max_iter"] = int(math.ceil(config.h_only_opt.max_iter * step.h_opt_max_iter_multiplier))
    return CP2KConfig.model_validate(data)


def global_block(run_type: str, project_name: str = "hotspot_region", *, print_level: str = "MEDIUM", extra: list[str] | None = None) -> str:
    return "\n".join(
        [
            "&GLOBAL",
            f"  PROJECT {project_name}",
            f"  RUN_TYPE {run_type}",
            f"  PRINT_LEVEL {print_level}",
            *[f"  {line}" for line in (extra or [])],
            "&END GLOBAL",
        ]
    )


def constraint_block(fixed_atom_indices: list[int] | None) -> str:
    if not fixed_atom_indices:
        return ""
    indices = " ".join(str(index + 1) for index in sorted(fixed_atom_indices))
    return "\n".join(
        [
            "  &CONSTRAINT",
            "    &FIXED_ATOMS",
            "      COMPONENTS_TO_FIX XYZ",
            f"      LIST {indices}",
            "    &END FIXED_ATOMS",
            "  &END CONSTRAINT",
        ]
    )


@dataclass(frozen=True, slots=True)
class CP2KInput:
    text: str
    project: str
    run_type: str
    geometry: PreparedGeometry
    electronic_state: ElectronicState
    poisson_solver: str
    stress: bool
    fixed_atoms: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def _symbols(region: ExtractedRegion) -> list[str]:
    return [chemical_symbols[int(number)] for number in region.numbers]


def build_cp2k_input(
    region: ExtractedRegion,
    config: CP2KConfig,
    *,
    run_type: str = "ENERGY_FORCE",
    project_name: str = "hotspot_region",
    fixed_atom_indices: list[int] | None = None,
    embedding_enabled: bool = False,
    template_text: str | None = None,
) -> CP2KInput:
    """Render one CP2K input (``ENERGY_FORCE`` single point or cap-only ``GEO_OPT``)."""

    run_type = run_type.upper()
    if run_type not in {"ENERGY_FORCE", "GEO_OPT"}:
        raise ConfigurationError(f"unsupported CP2K run type {run_type!r}")
    check_embedding(region, embedding_enabled=embedding_enabled)
    geometry = prepare_geometry(region, config)
    periodic = geometry.periodic
    solver = poisson_solver(periodic, config)
    stress = wants_stress(periodic, config)
    if stress and solver == "WAVELET" and periodic != "XYZ":
        raise UnsupportedCombinationError("CP2K has no stress tensor for the 2D WAVELET solver; use MT or ANALYTIC")
    state = region_electronic_state(region, config)
    check_uncapped_isolated_region(region, config, periodic)

    symbols = _symbols(region)
    spin_lines = ["    UKS"] if state.is_open_shell else []
    kind_blocks = [
        "\n".join(
            [
                f"    &KIND {symbol}",
                f"      BASIS_SET {config.basis_for(symbol)}",
                f"      POTENTIAL {config.potential_for(symbol)}",
                "    &END KIND",
            ]
        )
        for symbol in sorted(set(symbols))
    ]
    motion_block = ""
    if run_type == "GEO_OPT":
        motion_block = "\n".join(
            [
                "&MOTION",
                "  &GEO_OPT",
                f"    OPTIMIZER {config.h_only_opt.optimizer}",
                f"    MAX_ITER {int(config.h_only_opt.max_iter)}",
                "  &END GEO_OPT",
                constraint_block(fixed_atom_indices),
                "&END MOTION",
            ]
        )
    coord_lines = [
        f"    {symbol:<2} {position[0]:>16.8f} {position[1]:>16.8f} {position[2]:>16.8f}"
        for symbol, position in zip(symbols, geometry.positions, strict=True)
    ]
    if config.use_ot:
        scf_method_block = "\n".join(
            [
                "      &OT",
                f"        MINIMIZER {config.ot_minimizer}",
                f"        PRECONDITIONER {config.ot_preconditioner}",
                "      &END OT",
            ]
        )
    else:
        mixing = config.mixing
        scf_method_block = "\n".join(
            [
                "      &DIAGONALIZATION",
                "        ALGORITHM STANDARD",
                "      &END DIAGONALIZATION",
                "      &MIXING",
                f"        METHOD {mixing.method}",
                f"        ALPHA {mixing.alpha}",
                *([f"        NBUFFER {mixing.nbuffer}"] if mixing.nbuffer is not None else []),
                "      &END MIXING",
            ]
        )
    dispersion_block = ""
    if config.dispersion:
        dispersion_block = "\n".join(
            [
                "      &VDW_POTENTIAL",
                "        POTENTIAL_TYPE PAIR_POTENTIAL",
                "        &PAIR_POTENTIAL",
                "          TYPE DFTD3(BJ)",
                f"          PARAMETER_FILE_NAME {config.dispersion_parameter_file}",
                f"          REFERENCE_FUNCTIONAL {config.functional}",
                "        &END PAIR_POTENTIAL",
                "      &END VDW_POTENTIAL",
            ]
        )
    cell = geometry.cell
    global_text = global_block(run_type, project_name, print_level=config.print_level, extra=config.extra_sections.global_lines)
    force_eval_text = "\n".join(
        [
            "&FORCE_EVAL",
            "  METHOD Quickstep",
            *(["  STRESS_TENSOR ANALYTICAL"] if stress else []),
            *[f"  {line}" for line in config.extra_sections.force_eval_lines],
            "  &DFT",
            f"    CHARGE {state.charge}",
            f"    MULTIPLICITY {state.multiplicity}",
            *spin_lines,
            f"    BASIS_SET_FILE_NAME {config.basis_set_file}",
            f"    POTENTIAL_FILE_NAME {config.potential_file}",
            "    &MGRID",
            f"      CUTOFF {float(config.cutoff)}",
            f"      REL_CUTOFF {float(config.rel_cutoff)}",
            "    &END MGRID",
            "    &QS",
            f"      EPS_DEFAULT {_cp2k_float(config.eps_default)}",
            "    &END QS",
            "    &POISSON",
            f"      PERIODIC {periodic}",
            f"      POISSON_SOLVER {solver}",
            "    &END POISSON",
            "    &SCF",
            f"      MAX_SCF {int(config.max_scf)}",
            f"      EPS_SCF {config.scf_eps}",
            f"      SCF_GUESS {config.scf_guess}",
            scf_method_block,
            "    &END SCF",
            "    &XC",
            f"      &XC_FUNCTIONAL {config.functional}",
            "      &END XC_FUNCTIONAL",
            dispersion_block,
            "    &END XC",
            *[f"    {line}" for line in config.extra_sections.dft_lines],
            "  &END DFT",
            "  &SUBSYS",
            "    &CELL",
            f"      A {cell[0, 0]:.8f} {cell[0, 1]:.8f} {cell[0, 2]:.8f}",
            f"      B {cell[1, 0]:.8f} {cell[1, 1]:.8f} {cell[1, 2]:.8f}",
            f"      C {cell[2, 0]:.8f} {cell[2, 1]:.8f} {cell[2, 2]:.8f}",
            f"      PERIODIC {periodic}",
            "    &END CELL",
            "    &COORD",
            *coord_lines,
            "    &END COORD",
            *kind_blocks,
            "  &END SUBSYS",
            "  &PRINT",
            "    &FORCES MEDIUM",
            "    &END FORCES",
            *(["    &STRESS_TENSOR MEDIUM", "    &END STRESS_TENSOR"] if stress else []),
            "  &END PRINT",
            "&END FORCE_EVAL",
        ]
    )
    if template_text is None:
        text = "\n".join([global_text, force_eval_text, motion_block, ""])
    else:
        try:
            text = Template(template_text).substitute(
                global_section=global_text,
                force_eval_section=force_eval_text,
                motion_section=motion_block,
                project=project_name,
                run_type=run_type,
            )
        except (KeyError, ValueError) as exc:
            raise ConfigurationError(f"cp2k.template_path: invalid placeholder {exc}") from exc
    return CP2KInput(
        text=text,
        project=project_name,
        run_type=run_type,
        geometry=geometry,
        electronic_state=state,
        poisson_solver=solver,
        stress=stress,
        fixed_atoms=tuple(sorted(fixed_atom_indices or ())),
        metadata={"periodic": periodic, "rebuilt_cell": geometry.rebuilt_cell, **state.as_metadata()},
    )


def _cp2k_float(value: float) -> str:
    return "1.0E-12" if value == 1.0e-12 else f"{value:.6E}"


def cap_indices(region: ExtractedRegion) -> list[int]:
    return [index for index, role in enumerate(region.roles) if role is RegionRole.H_CAP]


def h_opt_fixed_atoms(region: ExtractedRegion) -> list[int]:
    """Every atom except the H caps (PHAL also fixed boundary atoms, which are never caps)."""

    caps = set(cap_indices(region))
    return [index for index in range(region.n_atoms) if index not in caps]


def needs_h_opt(region: ExtractedRegion, config: CP2KConfig) -> bool:
    return bool(config.h_only_opt.enabled) and bool(cap_indices(region))


def load_template(config: CP2KConfig, *, base_dir: str | Path | None = None) -> str | None:
    if not config.template_path:
        return None
    path = Path(config.template_path)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"cannot read cp2k.template_path {path}: {exc}") from exc


def write_cp2k_inputs(
    region: ExtractedRegion,
    output_dir: str | Path,
    config: CP2KConfig,
    *,
    job_name: str = "hotspot_region",
    embedding_enabled: bool = False,
    template_text: str | None = None,
) -> dict[str, Path]:
    """Write ``<job>_hopt.inp`` (when caps exist and h_only_opt is enabled) and ``<job>_sp.inp``.

    The H-only input is written from the unrelaxed geometry; the labeling service submits it and
    regenerates the single point from the relaxed caps (F-17).
    """

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    if needs_h_opt(region, config):
        hopt = build_cp2k_input(
            region,
            config,
            run_type="GEO_OPT",
            project_name=f"{job_name}_hopt",
            fixed_atom_indices=h_opt_fixed_atoms(region),
            embedding_enabled=embedding_enabled,
            template_text=template_text,
        )
        path = target / f"{job_name}_hopt.inp"
        path.write_text(hopt.text, encoding="utf-8")
        written["hopt_input"] = path
    single_point = build_cp2k_input(
        region,
        config,
        run_type="ENERGY_FORCE",
        project_name=f"{job_name}_sp",
        embedding_enabled=embedding_enabled,
        template_text=template_text,
    )
    path = target / f"{job_name}_sp.inp"
    path.write_text(single_point.text, encoding="utf-8")
    written["single_point_input"] = path
    return written


__all__ = [
    "CP2KInput",
    "PreparedGeometry",
    "SOLVER_SUPPORT",
    "build_cp2k_input",
    "cap_indices",
    "check_embedding",
    "check_uncapped_isolated_region",
    "constraint_block",
    "global_block",
    "h_opt_fixed_atoms",
    "load_template",
    "needs_h_opt",
    "periodicity",
    "poisson_solver",
    "prepare_geometry",
    "region_electronic_state",
    "settings_with_step",
    "wants_stress",
    "write_cp2k_inputs",
]
