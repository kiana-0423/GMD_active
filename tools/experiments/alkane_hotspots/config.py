"""Typed configuration of the alkane hotspot demonstration (YAML, ``extra="forbid"`` everywhere).

Every grid, threshold, size and budget used by the demo lives here; code never hard-codes them
(AGENTS.md rule 6). Part beta of the demo (receptive field, ground truth, acquisition arms, figures)
keeps its own sections under ``model_side``, validated by its own modules.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gmd_active_learning.contracts.ids import content_hash

LOGGER = logging.getLogger(__name__)

SPLITS: tuple[str, ...] = ("initial_train", "calibration", "validation", "final_eval")
FAMILIES: tuple[str, ...] = ("normal", "torsion", "bend", "contact")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SeedsConfig(_Strict):
    generation: int = 20261107
    split: int = 7


class SystemsConfig(_Strict):
    """Chain lengths (number of carbons). Chains longer than ``long_chain_threshold`` need
    ``allow_long_chains: true`` (campaign only: C64/C128 parents are very expensive in CP2K)."""

    chain_lengths: list[int] = Field(default_factory=lambda: [16, 32])
    allow_long_chains: bool = False
    long_chain_threshold: int = Field(32, ge=2)

    @model_validator(mode="after")
    def _long_chains_need_flag(self) -> SystemsConfig:
        if not self.chain_lengths:
            raise ValueError("systems.chain_lengths must not be empty")
        if any(n < 4 for n in self.chain_lengths):
            raise ValueError("systems.chain_lengths: every chain needs at least 4 carbons")
        if len(set(self.chain_lengths)) != len(self.chain_lengths):
            raise ValueError("systems.chain_lengths must be unique")
        long = [n for n in self.chain_lengths if n > self.long_chain_threshold]
        if long and not self.allow_long_chains:
            raise ValueError(f"chain lengths {long} exceed {self.long_chain_threshold}; set systems.allow_long_chains: true explicitly")
        return self


class GeometryConfig(_Strict):
    cc_bond: float = Field(1.53, gt=0.0)
    ch_bond: float = Field(1.09, gt=0.0)
    angle_deg: float = Field(109.47122063449069, gt=90.0, lt=180.0)
    covalent_scale: float = Field(1.2, gt=0.0)  # bond inference (utils.bonds.infer_bonds)


class ThermalConfig(_Strict):
    amplitude: float = Field(0.03, ge=0.0)  # A, standard deviation per Cartesian component


class TorsionFamily(_Strict):
    enabled: bool = True
    deltas_deg: list[float] = Field(default_factory=lambda: [120.0, -120.0, 60.0, -60.0])
    locations: list[float] = Field(default_factory=lambda: [0.35, 0.5, 0.65])  # fraction along the chain


class BendFamily(_Strict):
    enabled: bool = True
    deltas_deg: list[float] = Field(default_factory=lambda: [-12.0, 12.0, 20.0])
    locations: list[float] = Field(default_factory=lambda: [0.35, 0.5, 0.65])


class ContactFamily(_Strict):
    """Fold the chain with ``fold_bonds`` consecutive backbone torsions starting at ``location`` so that
    atoms of two segments at least ``min_segment_separation`` carbons apart reach ``target`` (min
    interatomic distance between the two segments), within ``target_tolerance`` before thermal noise."""

    enabled: bool = True
    target_distances: list[float] = Field(default_factory=lambda: [2.0, 2.3, 2.6])
    target_tolerance: float = Field(0.02, gt=0.0)
    locations: list[float] = Field(default_factory=lambda: [0.4, 0.5])
    fold_bonds: int = Field(5, ge=2, le=8)
    min_segment_separation: int = Field(6, ge=3)
    n_starts: int = Field(4, ge=1)
    clash_margin: float = Field(0.05, ge=0.0)  # A added to the nonbonded minima inside the fold
    max_evaluations: int = Field(4000, ge=10)


class FamiliesConfig(_Strict):
    thermal: ThermalConfig = Field(default_factory=ThermalConfig)
    torsion: TorsionFamily = Field(default_factory=TorsionFamily)
    bend: BendFamily = Field(default_factory=BendFamily)
    contact: ContactFamily = Field(default_factory=ContactFamily)


class GenerationConfig(_Strict):
    """``n_parents`` = len(chain_lengths) x bases per length x sum(per_base) (must divide exactly)."""

    n_parents: int = Field(24, ge=1)
    per_base: dict[str, int] = Field(default_factory=lambda: {"normal": 1, "torsion": 1, "bend": 1, "contact": 1})
    max_candidates_per_slot: int = Field(50, ge=1)

    @field_validator("per_base")
    @classmethod
    def _families(cls, value: dict[str, int]) -> dict[str, int]:
        unknown = set(value) - set(FAMILIES)
        if unknown:
            raise ValueError(f"generation.per_base: unknown families {sorted(unknown)}")
        if any(int(count) < 0 for count in value.values()) or sum(value.values()) == 0:
            raise ValueError("generation.per_base counts must be >= 0 with a positive sum")
        if value.get("normal", 0) > 1:
            raise ValueError("generation.per_base.normal must be 0 or 1 (a base conformation has one unperturbed parent)")
        return {key: int(value.get(key, 0)) for key in FAMILIES}


class ChecksConfig(_Strict):
    """Geometry acceptance thresholds. Distances in A, angles in degrees."""

    bond_tolerance: float = Field(0.25, gt=0.0)  # |d - (r_cov,i + r_cov,j)|
    exact_tolerance: float = Field(1.0e-8, gt=0.0)  # rigid operations: bond-length change before noise
    exact_angle_tolerance_deg: float = Field(1.0e-6, gt=0.0)  # untouched angles before noise
    max_angle_deviation_deg: float = Field(15.0, gt=0.0)  # untouched angles vs ideal after noise
    nonbonded_min_bond_separation: int = Field(3, ge=2)
    min_nonbonded_hh: float = Field(1.8, gt=0.0)
    min_nonbonded_ch: float = Field(2.2, gt=0.0)
    min_nonbonded_cc: float = Field(2.5, gt=0.0)
    contact_window: tuple[float, float] = (1.8, 2.9)

    @model_validator(mode="after")
    def _window(self) -> ChecksConfig:
        low, high = self.contact_window
        if not 0.0 < low < high:
            raise ValueError("checks.contact_window must be (low, high) with 0 < low < high")
        return self


class SplitsConfig(_Strict):
    fractions: dict[str, float] = Field(
        default_factory=lambda: {"initial_train": 0.34, "calibration": 0.33, "validation": 0.0, "final_eval": 0.33}
    )
    group_key: Literal["base_conformation"] = "base_conformation"
    stratify_by: Literal["chain_length", "none"] = "chain_length"
    assignment_order: list[str] = Field(default_factory=lambda: ["calibration", "final_eval", "initial_train", "validation"])

    @model_validator(mode="after")
    def _fractions(self) -> SplitsConfig:
        if set(self.fractions) != set(SPLITS):
            raise ValueError(f"splits.fractions needs exactly {list(SPLITS)}")
        if any(value < 0 for value in self.fractions.values()) or abs(sum(self.fractions.values()) - 1.0) > 1.0e-6:
            raise ValueError("splits.fractions must be >= 0 and sum to 1")
        if sorted(self.assignment_order) != sorted(SPLITS):
            raise ValueError(f"splits.assignment_order must be a permutation of {list(SPLITS)}")
        return self


class HOptConfig(_Strict):
    enabled: bool = False
    max_iter: int = Field(50, ge=1)


class CP2KSettingsConfig(_Strict):
    """Electronic-structure settings, identical for full parents and fragments (one settings hash).

    ``smoke_only`` marks settings that are not production quality (e.g. a low cutoff); every label
    record and manifest carries the flag.
    """

    executable: str = "cp2k.psmp"
    data_dir: str | None = None  # null: $CP2K_DATA_DIR, else <prefix>/share/cp2k/data next to the executable
    launcher: list[str] = Field(default_factory=list)  # e.g. [mpirun, -np, "{ntasks}"]
    omp_threads: int = Field(5, ge=1)
    mpi_ranks: int = Field(1, ge=1)
    functional: Literal["PBE"] = "PBE"
    basis: str = "DZVP-MOLOPT-SR-GTH"
    potential: str = "GTH-PBE"
    dispersion: bool = True  # DFT-D3(BJ)
    charge: Literal[0] = 0
    multiplicity: Literal[1] = 1
    cutoff: float = Field(400.0, gt=0.0)
    rel_cutoff: float = Field(60.0, gt=0.0)
    eps_scf: float = Field(1.0e-6, gt=0.0)
    max_scf: int = Field(100, ge=1)
    poisson_solver: Literal["WAVELET"] = "WAVELET"
    box_padding: float = Field(6.0, gt=0.0)  # A; isolated cubic box = extent + 2 * padding
    h_only_opt: HOptConfig = Field(default_factory=HOptConfig)
    smoke_only: bool = False

    @model_validator(mode="after")
    def _launcher(self) -> CP2KSettingsConfig:
        if self.mpi_ranks > 1 and not self.launcher:
            raise ValueError("cp2k.mpi_ranks > 1 needs cp2k.launcher (e.g. [mpirun, -np, '{ntasks}'])")
        return self

    @property
    def cores(self) -> int:
        return int(self.omp_threads * self.mpi_ranks)

    def resolved_data_dir(self) -> str | None:
        if self.data_dir:
            return self.data_dir
        env = os.environ.get("CP2K_DATA_DIR")
        if env:
            return env
        found = shutil.which(self.executable)
        if found:
            candidate = Path(found).resolve().parent.parent / "share" / "cp2k" / "data"
            if candidate.is_dir():
                return str(candidate)
        return None


class LabelsConfig(_Strict):
    full_splits: list[str] = Field(default_factory=lambda: list(SPLITS))
    poll_interval_s: float = Field(5.0, gt=0.0)
    max_concurrent_jobs: int = Field(1, ge=1)

    @field_validator("full_splits")
    @classmethod
    def _splits(cls, value: list[str]) -> list[str]:
        unknown = set(value) - set(SPLITS)
        if unknown:
            raise ValueError(f"labels.full_splits: unknown splits {sorted(unknown)}")
        return value


class ClusterSweep(_Strict):
    core_radii: list[float] = Field(default_factory=lambda: [1.5])
    buffer_radii: list[float] = Field(default_factory=lambda: [3.0, 4.0, 5.0])

    @model_validator(mode="after")
    def _radii(self) -> ClusterSweep:
        if any(r < 0 for r in self.core_radii) or any(r <= 1.2 for r in self.buffer_radii):
            raise ValueError("fragments.cluster: core radii >= 0 and buffer radii > 1.2 A (one C-H bond) are required")
        return self


class FragmentsConfig(_Strict):
    """Fragment sweep. ``graph_hops``: carbons within that many C-C bonds of the core carbons plus all
    their hydrogens. Cluster: core = atoms within ``core_radius`` of the site atoms, fragment = atoms
    within ``core_radius + buffer_radius`` (whole CH_n units). Every cut C-C bond is H-capped."""

    sites: list[Literal["perturbation_site", "normal_control"]] = Field(default_factory=lambda: ["perturbation_site", "normal_control"])
    normal_location: float = Field(0.5, gt=0.0, lt=1.0)
    graph_hops: list[int] = Field(default_factory=lambda: [2, 3, 4])
    cluster: ClusterSweep = Field(default_factory=ClusterSweep)
    vacuum_padding: float = Field(4.0, ge=0.0)  # extraction box only; CP2K rebuilds a cubic WAVELET box
    cap_clash_distance: float = Field(1.5, gt=0.0)  # A; a cap closer than this to a nonbonded atom fails

    @field_validator("graph_hops")
    @classmethod
    def _hops(cls, value: list[int]) -> list[int]:
        if any(h < 1 for h in value):
            raise ValueError("fragments.graph_hops must be >= 1")
        return value


class FidelityConfig(_Strict):
    """Pre-registered acceptance targets (eV/A). They are frozen into the preregistration manifest at
    the first ``fidelity-prepare``; a later change is refused, never applied retroactively."""

    core_force_rmse: float = Field(0.03, gt=0.0)
    max_core_force_error: float = Field(0.10, gt=0.0)
    max_failures: int = Field(0, ge=0)
    selection_objective: Literal["min_measured_cost"] = "min_measured_cost"
    distance_bins: list[float] = Field(default_factory=lambda: [0, 1, 2, 3, 4, 6, 100])  # bonds to the nearest cut
    environment_radius: float = Field(4.0, gt=0.0)  # A; parent atoms this close to a core atom but outside the fragment = missing environment


class BudgetsConfig(_Strict):
    max_full_labels: int = Field(24, ge=0)
    max_fragment_labels: int = Field(120, ge=0)
    max_total_cpu_core_hours: float = Field(80.0, ge=0.0)
    per_job_walltime_s: int = Field(7200, ge=1)
    max_atoms_per_fragment: int = Field(80, ge=1)
    max_atoms_full: int = Field(200, ge=1)


class AlkaneConfig(_Strict):
    name: str
    description: str = ""
    artifacts_root: str | None = None  # default ../alkane_artifacts/<name> relative to the GMD_active root
    seeds: SeedsConfig = Field(default_factory=SeedsConfig)
    systems: SystemsConfig = Field(default_factory=SystemsConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    families: FamiliesConfig = Field(default_factory=FamiliesConfig)
    checks: ChecksConfig = Field(default_factory=ChecksConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    cp2k: CP2KSettingsConfig = Field(default_factory=CP2KSettingsConfig)
    labels: LabelsConfig = Field(default_factory=LabelsConfig)
    fragments: FragmentsConfig = Field(default_factory=FragmentsConfig)
    fidelity: FidelityConfig = Field(default_factory=FidelityConfig)
    budgets: BudgetsConfig = Field(default_factory=BudgetsConfig)
    model_side: dict[str, Any] = Field(default_factory=dict)  # part beta's sections; validated by its modules

    @model_validator(mode="after")
    def _consistency(self) -> AlkaneConfig:
        per_base = sum(self.generation.per_base.values())
        n_lengths = len(self.systems.chain_lengths)
        if self.generation.n_parents % (n_lengths * per_base):
            raise ValueError(
                f"generation.n_parents={self.generation.n_parents} is not a multiple of "
                f"len(chain_lengths) x sum(per_base) = {n_lengths} x {per_base}"
            )
        for family in ("torsion", "bend", "contact"):
            if self.generation.per_base.get(family, 0) and not getattr(self.families, family).enabled:
                raise ValueError(f"generation.per_base.{family} > 0 but families.{family}.enabled is false")
        return self

    @property
    def bases_per_length(self) -> int:
        return self.generation.n_parents // (len(self.systems.chain_lengths) * sum(self.generation.per_base.values()))

    def config_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"), length=16)

    def generation_hash(self) -> str:
        """Hash of everything that determines the parent structures and splits."""

        keys = ("seeds", "systems", "geometry", "generation", "families", "checks", "splits")
        return content_hash({key: self.model_dump(mode="json")[key] for key in keys}, length=16)


def find_repo_root(start: Path) -> Path:
    """Nearest ancestor holding ``pyproject.toml`` (the GMD_active root); falls back to the CWD."""

    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "gmd_active_learning").is_dir():
            return candidate
    return Path.cwd()


class LoadedConfig:
    """A validated config plus where it came from and where its artifacts go."""

    def __init__(self, config: AlkaneConfig, path: Path | None, *, artifacts_override: str | Path | None = None) -> None:
        self.config = config
        self.path = path
        self.repo_root = find_repo_root(path.resolve().parent if path else Path.cwd())
        root = artifacts_override or config.artifacts_root or f"../alkane_artifacts/{config.name}"
        root_path = Path(root)
        self.artifacts_root = (root_path if root_path.is_absolute() else self.repo_root / root_path).resolve()

    @property
    def name(self) -> str:
        return self.config.name


def load_config(path: str | Path, *, artifacts_root: str | Path | None = None) -> LoadedConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping")
    return LoadedConfig(AlkaneConfig.model_validate(data), path, artifacts_override=artifacts_root)


__all__ = ["FAMILIES", "SPLITS", "AlkaneConfig", "LoadedConfig", "find_repo_root", "load_config"]
