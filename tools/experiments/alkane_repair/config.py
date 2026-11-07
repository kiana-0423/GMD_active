"""Typed configuration of the alkane repair benchmark (YAML, ``extra="forbid"`` everywhere).

The benchmark asks whether PHAL repairs failures of a machine-learned force field on alkane
conformations withheld from the initial training coverage. Every partition size, coverage window,
acceptance threshold, budget and endpoint criterion lives in the benchmark YAML (AGENTS.md rule 6);
the code never hard-codes them.

The YAML references two existing configuration files of the alkane demonstration by path:

* ``alpha_config``: an ``AlkaneConfig`` (CP2K settings, geometry, checks, fragment and fidelity
  settings, label budgets). Its ``artifacts_root`` is the benchmark's artifacts root. Its
  ``generation``/``families``/``splits`` sections are not used: structures come from :mod:`generate`.
* ``model_config``: a part-beta study file (``monitor`` + ``training``); the monitor section is the
  pre-registered staged online monitor of arm D.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Partitions, the closed-loop split each maps to, and what the split enforces. Validation/final_eval
#: parents can never be an MD start or enter training through the closed loop; final_test parents and
#: labels live in a separate (sealed) artifacts root as well.
PARTITIONS: tuple[str, ...] = ("id_train", "id_val", "calibration", "development", "repair", "repair_val", "final_test")
PARTITION_SPLIT: dict[str, str] = {
    "id_train": "initial_train",
    "id_val": "validation",
    "repair_val": "validation",
    "calibration": "calibration",
    "development": "calibration",
    "repair": "calibration",
    "final_test": "final_eval",
}
CHALLENGE_FAMILIES: tuple[str, ...] = ("torsion", "bend", "contact")
ARM_KINDS: tuple[str, ...] = ("frozen", "continued_training", "closed_loop")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------------------------- structures


class CoverageConfig(_Strict):
    """Operational definition of the initial training coverage (the "in-domain" region).

    A structure is in-domain when every descriptor lies inside its window. These windows define the
    withheld coverage; they are not a proof that the model extrapolates outside them.
    """

    max_torsion_deviation_deg: float = Field(..., gt=0.0, lt=180.0, description="every backbone C-C-C-C dihedral within 180 +- this")
    max_angle_deviation_deg: float = Field(..., gt=0.0, description="every C-C-C angle within the ideal angle +- this")
    min_segment_contact: float = Field(..., gt=0.0, description="A; min distance between atoms of chain segments >= min_segment_separation carbons apart")
    min_segment_separation: int = Field(6, ge=3)


class OrdinarySamplerConfig(_Strict):
    """Ordinary configurations: a base backbone with random dihedral/angle deviations, then per-sample jitter."""

    base_torsion_sigma_deg: float = Field(..., ge=0.0, description="per backbone dihedral, Gaussian, of a base conformation")
    base_angle_sigma_deg: float = Field(..., ge=0.0, description="per C-C-C angle, Gaussian, of a base conformation")
    sample_torsion_sigma_deg: float = Field(..., ge=0.0, description="extra per-sample dihedral jitter")
    thermal_amplitude: float = Field(..., ge=0.0, description="A, Gaussian per Cartesian component (every structure)")
    max_attempts: int = Field(200, ge=1)


class TorsionChallenge(_Strict):
    enabled: bool = True
    deltas_deg: list[float] = Field(..., min_length=1, description="change of one backbone dihedral (120 = gauche)")
    locations: list[float] = Field(..., min_length=1, description="fraction along the chain")
    min_excess_deg: float = Field(..., ge=0.0, description="challenged dihedral deviation must exceed the coverage window by this")
    max_deviation_deg: float = Field(180.0, gt=0.0, le=180.0, description="upper bound of the challenged deviation (e.g. stay in the gauche well)")


class BendChallenge(_Strict):
    enabled: bool = True
    deltas_deg: list[float] = Field(..., min_length=1)
    locations: list[float] = Field(..., min_length=1)
    min_excess_deg: float = Field(..., ge=0.0)


class ContactChallenge(_Strict):
    enabled: bool = True
    target_distances: list[float] = Field(..., min_length=1, description="A, segment contact distance of the fold (before noise)")
    locations: list[float] = Field(..., min_length=1)
    fold_bonds: int = Field(5, ge=2, le=8)
    target_tolerance: float = Field(0.02, gt=0.0)
    n_starts: int = Field(4, ge=1)
    clash_margin: float = Field(0.05, ge=0.0)
    max_evaluations: int = Field(4000, ge=10)
    min_excess: float = Field(..., ge=0.0, description="A; segment contact must lie below the coverage minimum by this")


class ChallengesConfig(_Strict):
    torsion: TorsionChallenge
    bend: BendChallenge
    contact: ContactChallenge

    def enabled(self) -> list[str]:
        return [family for family in CHALLENGE_FAMILIES if getattr(self, family).enabled]


class PartitionSpec(_Strict):
    bases: int = Field(..., ge=0, description="independent base conformations (the experimental unit)")
    ordinary_per_base: int = Field(0, ge=0)
    challenges_per_family: int = Field(0, ge=0, description="challenge structures per enabled family and base")


class PartitionsConfig(_Strict):
    id_train: PartitionSpec
    id_val: PartitionSpec
    calibration: PartitionSpec
    development: PartitionSpec
    repair: PartitionSpec
    repair_val: PartitionSpec
    final_test: PartitionSpec

    @model_validator(mode="after")
    def _roles(self) -> PartitionsConfig:
        for name in ("id_train", "id_val", "calibration"):
            if getattr(self, name).challenges_per_family:
                raise ValueError(f"partitions.{name} is in-domain only (challenges_per_family must be 0)")
        if self.repair.ordinary_per_base:
            raise ValueError("partitions.repair holds challenge MD starts only")
        if self.id_train.bases < 1 or self.id_train.ordinary_per_base < 1:
            raise ValueError("partitions.id_train needs at least one base with one ordinary structure")
        return self

    def total_bases(self) -> int:
        return sum(getattr(self, name).bases for name in PARTITIONS)


# ---------------------------------------------------------------------------------------------- initial model


class MDCheckConfig(_Strict):
    """Short GMD runs of a candidate initial model; the parent bond topology must survive."""

    start_partitions: list[Literal["calibration", "id_val"]] = Field(default_factory=lambda: ["calibration", "id_val"])
    seeds: list[int] = Field(..., min_length=1)
    steps: int = Field(..., ge=1)
    frame_interval: int = Field(10, ge=1)
    min_intact_fraction: float = Field(1.0, gt=0.0, le=1.0, description="per trajectory; every trajectory counts")
    covalent_scale: float = Field(1.3, gt=0.0)


class AcceptanceConfig(_Strict):
    id_val_force_rmse_max: float = Field(..., gt=0.0, description="eV/A, per Cartesian component, all atoms of the id_val structures")
    id_val_atom_error_p95_max: float = Field(..., gt=0.0, description="eV/A, 95th percentile of |F_model - F_DFT| per atom")
    md: MDCheckConfig
    forbidden_directive_keys: list[str] = Field(default_factory=lambda: ["force_cap", "max_force", "force_clip", "fix", "constraint",
                                                                         "constraints", "shake", "rattle", "restraint", "wall"],
                                                description="GMD directive keys that would hide instability (clipping, constraints)")
    required_force_field: Literal["ml"] = "ml"


class QualificationStage(_Strict):
    id_train_bases: int = Field(..., ge=1, description="the first N id_train bases (all their ordinary structures)")


class InitialModelConfig(_Strict):
    seed: int = 11
    train_overrides: dict[str, Any] = Field(default_factory=dict, description="overrides of model_config training.train")
    stages: list[QualificationStage] = Field(..., min_length=1, description="data growth; a later stage is labeled only if the earlier failed")
    acceptance: AcceptanceConfig


# ---------------------------------------------------------------------------------------------- failure and repair


class FailureConfig(_Strict):
    """A challenge exposes a reference-confirmed failure when the initial model's force error in the
    challenged region exceeds both bounds (DFT force magnitude is reported alongside, so a legitimately
    large force that the model reproduces is not counted as a failure)."""

    region_bond_hops: int = Field(1, ge=0, description="challenged region: perturbation-site atoms + carbons within this many bonds (+H)")
    abs_threshold: float = Field(..., gt=0.0, description="eV/A, region force RMSE")
    ratio_to_id_val: float = Field(..., gt=0.0, description="region force RMSE / qualified id_val force RMSE")


class MonitorCalibrationConfig(_Strict):
    """In-domain trajectories of the frozen initial model that calibrate the monitor's running statistics
    before any repair run (the challenge trajectory itself is never used for warm-up)."""

    partitions: list[Literal["calibration"]] = Field(default_factory=lambda: ["calibration"])
    seeds: list[int] = Field(..., min_length=1)
    steps: int = Field(..., ge=1)
    frame_interval: int = Field(10, ge=1)
    max_alarm_fraction: float = Field(..., ge=0.0, le=1.0, description="refuse an arm whose calibrated monitor triggers on more in-domain frames")


class ArmConfig(_Strict):
    name: str = Field(..., pattern=r"^[A-Za-z0-9_]+$")
    kind: Literal["frozen", "continued_training", "closed_loop"]
    enabled: bool = True
    description: str = ""
    monitor_overrides: dict[str, Any] = Field(default_factory=dict, description="closed_loop only; merged into model_config monitor")
    events: dict[str, Any] | None = Field(None, description="closed_loop only; null = repair.events")
    required_warm_metrics: list[str] = Field(default_factory=list, description="closed_loop only: running statistics that must be past warm-up")

    @model_validator(mode="after")
    def _kind(self) -> ArmConfig:
        if self.kind != "closed_loop" and (self.monitor_overrides or self.events is not None or self.required_warm_metrics):
            raise ValueError(f"arm {self.name}: monitor/events settings apply to closed_loop arms only")
        return self


class PromotionConfig(_Strict):
    """Repair promotion, on repair-validation challenge structures of the run's family and on id_val."""

    max_force_rmse: float = Field(..., gt=0.0, description="eV/A, all validation structures (sanity bound only)")
    overall_max_relative_regression: float = Field(..., ge=0.0, description="all validation structures vs the active model")
    family_min_relative_improvement: float = Field(..., ge=0.0, lt=1.0)
    id_val_max_relative_regression: float = Field(..., ge=0.0)
    require_export_validation: bool = True
    export_max_force_deviation: float = Field(1.0e-4, gt=0.0)
    require_engine_smoke_test: bool = True


class RepairLoopConfig(_Strict):
    seeds: list[int] = Field(..., min_length=1, description="paired across arms: MD velocity seed and fine-tune seed offset")
    cycles: int = Field(..., ge=1)
    segment_steps: int = Field(..., ge=1)
    labels_per_cycle: int = Field(..., ge=1)
    max_per_event: int = Field(1, ge=1)
    diversity_threshold: float = Field(0.0, ge=0.0)
    min_new_samples: int = Field(1, ge=1)
    budget_core_h_per_run: float = Field(..., gt=0.0)
    md_directives: dict[str, Any] = Field(..., description="GMD directives (velocity_seed is set per seed)")
    frame_interval: int = Field(10, ge=1)
    md_box: dict[str, Any] = Field(default_factory=dict)
    label_geometry: dict[str, Any] = Field(default_factory=dict)
    events: dict[str, Any] = Field(default_factory=lambda: {"pre_trigger_frames": 1, "post_trigger_frames": 1, "cooldown_steps": 200})
    hotspots: dict[str, Any] = Field(default_factory=lambda: {"merge_radius": 3.0})
    finetune_train: dict[str, Any] = Field(default_factory=dict, description="overrides of training.train for fine-tuning")
    finetune_seed_offset: int = Field(1000, ge=0, description="fine-tune seed = offset + repetition seed (paired across arms)")
    promotion: PromotionConfig
    poll_interval_s: float = Field(10.0, ge=0.0)
    max_poll_wait_s: float = Field(900.0, ge=0.0)
    executable: str = "../GMD/build-torch/gmd"
    env: dict[str, str] = Field(default_factory=lambda: {"OMP_NUM_THREADS": "1"})


class SuccessConfig(_Strict):
    challenge_region_rmse_max: float = Field(..., gt=0.0, description="eV/A on sealed challenge structures of the run's family")
    min_relative_improvement: float = Field(..., ge=0.0, lt=1.0, description="vs the frozen initial model on the same structures")
    id_retention_max_relative_regression: float = Field(..., ge=0.0, description="sealed in-domain test force RMSE vs the initial model")
    require_stability: bool = True


class StabilityConfig(_Strict):
    seeds: list[int] = Field(..., min_length=1)
    steps: int = Field(..., ge=1)
    frame_interval: int = Field(10, ge=1)
    covalent_scale: float = Field(1.3, gt=0.0)
    min_intact_fraction: float = Field(1.0, gt=0.0, le=1.0)


class EvaluationConfig(_Strict):
    success: SuccessConfig
    stability: StabilityConfig
    bootstrap: dict[str, Any] = Field(default_factory=lambda: {"n": 2000, "seed": 20261108, "confidence": 0.95})
    consistency_check: bool = Field(True, description="trained-model core forces: capped fragment vs full parent input")


# ---------------------------------------------------------------------------------------------- budgets and plan


class StageBudgets(_Strict):
    """Measured DFT CPU core-hours per stage (failed attempts and retries count)."""

    initial_model: float = Field(..., ge=0.0)
    development: float = Field(..., ge=0.0)
    fidelity_audit: float = Field(..., ge=0.0)
    repair_val: float = Field(..., ge=0.0)
    final_test: float = Field(..., ge=0.0)
    repair_runs: float = Field(..., ge=0.0)

    def total(self) -> float:
        return float(sum(self.model_dump().values()))


class PlanConfig(_Strict):
    timing_sources: list[str] = Field(default_factory=list, description="artifacts roots with measured labels/trainings/MD (relative to GMD_active)")
    fidelity_audit_fragments_per_challenge: int = Field(2, ge=0)
    fidelity_audit_event_full_labels: int = Field(0, ge=0, description="full-parent labels of repair event frames (fragment check on MD geometry)")
    expected_labels_per_run_fraction: float = Field(1.0, gt=0.0, le=1.0, description="planning: fraction of the per-run label cap used")


class FidelitySource(_Strict):
    selection_dir: str = Field(..., description="frozen fragment setting (fidelity/selection.json), relative to GMD_active")
    require_audit: bool = True


class SealingConfig(_Strict):
    sealed_subdir: str = "sealed"


class RepairBenchmarkConfig(_Strict):
    name: str = Field(..., pattern=r"^[A-Za-z0-9_\-]+$")
    protocol_version: str
    protocol_document: str = Field(..., description="the frozen English protocol (relative to this file)")
    description: str = ""
    smoke_only: bool = False
    alpha_config: str
    model_config_file: str = Field(..., alias="model_config")
    seeds: dict[str, int] = Field(default_factory=lambda: {"generation": 20261108, "assignment": 11})
    chain_length: int = Field(16, ge=4)
    coverage: CoverageConfig
    ordinary: OrdinarySamplerConfig
    challenges: ChallengesConfig
    partitions: PartitionsConfig
    initial_model: InitialModelConfig
    failure: FailureConfig
    monitor_calibration: MonitorCalibrationConfig
    arms: list[ArmConfig] = Field(..., min_length=1)
    repair: RepairLoopConfig
    evaluation: EvaluationConfig
    budgets: StageBudgets
    plan: PlanConfig = Field(default_factory=PlanConfig)
    fidelity: FidelitySource
    sealing: SealingConfig = Field(default_factory=SealingConfig)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _arms(self) -> RepairBenchmarkConfig:
        names = [arm.name for arm in self.arms]
        if len(set(names)) != len(names):
            raise ValueError("arm names must be unique")
        kinds = {arm.kind for arm in self.arms if arm.enabled}
        if "frozen" not in kinds:
            raise ValueError("an enabled frozen arm (A) is required: every outcome is measured against the frozen initial model")
        if not self.challenges.enabled():
            raise ValueError("at least one challenge family must be enabled")
        for stage in self.initial_model.stages:
            if stage.id_train_bases > self.partitions.id_train.bases:
                raise ValueError(f"qualification stage uses {stage.id_train_bases} id_train bases; partitions.id_train.bases = {self.partitions.id_train.bases}")
        bases = [s.id_train_bases for s in self.initial_model.stages]
        if bases != sorted(set(bases)):
            raise ValueError("initial_model.stages must grow strictly")
        if self.partitions.repair.bases and not self.partitions.repair.challenges_per_family:
            raise ValueError("partitions.repair needs challenges_per_family >= 1")
        if not self.fidelity.require_audit and not self.smoke_only:
            raise ValueError("fidelity.require_audit: false is allowed for smoke-only profiles only")
        return self

    def arm(self, name: str) -> ArmConfig:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise KeyError(name)


# ---------------------------------------------------------------------------------------------- loading


def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class LoadedBenchmark:
    config: RepairBenchmarkConfig
    path: Path
    raw: dict[str, Any]
    alpha: Any  # alkane_hotspots.config.LoadedConfig (main artifacts root)
    model: Any  # alkane_hotspots.arms.LoadedConfig (monitor + training)

    @property
    def root(self) -> Path:
        return Path(self.alpha.artifacts_root)

    @property
    def sealed_root(self) -> Path:
        return self.root / self.config.sealing.sealed_subdir

    def sealed_alpha(self) -> Any:
        """alpha's configuration with the sealed root as artifacts root (final-test parents and labels)."""

        from tools.experiments.alkane_hotspots.config import LoadedConfig

        return LoadedConfig(self.alpha.config, self.alpha.path, artifacts_override=self.sealed_root)

    def resolve(self, relative: str) -> Path:
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else (self.path.parent / candidate).resolve()

    def repo_path(self, relative: str) -> Path:
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()

    @property
    def protocol_path(self) -> Path:
        return self.resolve(self.config.protocol_document)

    def protocol_files(self) -> dict[str, Path]:
        """Every file that defines the protocol (hashed together at freeze time)."""

        return {"benchmark": self.path, "alpha_config": self.resolve(self.config.alpha_config),
                "model_config": self.resolve(self.config.model_config_file), "protocol": self.protocol_path}

    def protocol_hash(self) -> dict[str, Any]:
        files = {name: {"path": str(path), "sha256": file_digest(path)} for name, path in self.protocol_files().items()}
        combined = hashlib.sha256(json.dumps({k: v["sha256"] for k, v in files.items()}, sort_keys=True).encode()).hexdigest()
        return {"protocol_version": self.config.protocol_version, "files": files, "sha256": combined}


def load_benchmark(path: str | Path) -> LoadedBenchmark:
    from tools.experiments.alkane_hotspots.arms import load_beta_config
    from tools.experiments.alkane_hotspots.config import load_config

    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = RepairBenchmarkConfig.model_validate(raw)
    alpha = load_config(path.parent / config.alpha_config)
    if alpha.config.cp2k.smoke_only and not config.smoke_only:
        raise ValueError(f"{config.alpha_config} has smoke-only CP2K settings; set smoke_only: true (no scientific claim)")
    if config.chain_length not in alpha.config.systems.chain_lengths:
        raise ValueError(f"chain_length {config.chain_length} is not in {config.alpha_config} systems.chain_lengths")
    model = load_beta_config(path.parent / config.model_config_file, alpha=alpha)
    if model.config.training.model.get("long_range_type", "none") != "none":
        raise ValueError("the repair benchmark needs a local model (training.model.long_range_type: none)")
    if Path(model.artifacts_root) != Path(alpha.artifacts_root):
        raise ValueError("model_config and alpha_config must share one artifacts root")
    return LoadedBenchmark(config, path, raw, alpha, model)


__all__ = [
    "ARM_KINDS",
    "CHALLENGE_FAMILIES",
    "PARTITIONS",
    "PARTITION_SPLIT",
    "LoadedBenchmark",
    "RepairBenchmarkConfig",
    "load_benchmark",
]
