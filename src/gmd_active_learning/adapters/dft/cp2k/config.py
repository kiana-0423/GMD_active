"""Typed ``cp2k`` configuration section (input settings, execution, validation, retry ladder).

Defaults equal PHAL's ``default.yaml`` values unless noted. Deliberate changes:

- ``periodic`` is gone: the Poisson/cell periodicity is taken from the region's PBC (a cluster is
  never computed periodic and a slab never isolated). ``poisson`` selects the solver per class.
- ``charge`` is gone: the region's ``charge`` (set by extraction) is used.
- ``executable`` defaults to ``cp2k.psmp`` (the validated build), ``data_dir``/``environment`` and the
  basis/potential file names are configurable instead of hard-coded.
- ``single_point.enabled`` is gone: a labeling task without a single point produces no label.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gmd_active_learning.adapters.schedulers.config import SchedulerResources
from gmd_active_learning.contracts.ids import content_hash


class KindSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    basis: str | None = None
    potential: str | None = None


class PoissonSettings(BaseModel):
    """Solver per periodicity class; the periodicity itself always comes from the region."""

    model_config = ConfigDict(extra="forbid")

    non_periodic_solver: Literal["WAVELET", "MT", "ANALYTIC", "MULTIPOLE"] = "WAVELET"
    partially_periodic_solver: Literal["MT", "ANALYTIC", "WAVELET"] = "MT"
    periodic_solver: Literal["PERIODIC"] = "PERIODIC"


class MixingSettings(BaseModel):
    """Density mixing; CP2K ignores it under OT, so it is emitted only for diagonalization."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["BROYDEN_MIXING", "PULAY_MIXING", "DIRECT_P_MIXING", "KERKER_MIXING"] = "BROYDEN_MIXING"
    alpha: float = Field(default=0.4, gt=0.0, le=1.0)
    nbuffer: int | None = Field(default=None, ge=1)


class HOnlyOptSettings(BaseModel):
    """GEO_OPT of the H caps (every other atom fixed) before the single point (F-17)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_iter: int = Field(default=50, ge=1)
    optimizer: Literal["BFGS", "LBFGS", "CG"] = "BFGS"
    require_converged: bool = True
    fixed_atom_tolerance: float = Field(default=1.0e-4, gt=0.0)  # Angstrom


class ValidationSettings(BaseModel):
    """Scientific validity rule (F-27) plus the GMD additions (energy present, atom count, termination)."""

    model_config = ConfigDict(extra="forbid")

    require_scf_convergence: bool = True
    require_energy: bool = True
    require_normal_termination: bool = True


class FailureClass(str, Enum):
    SCF_NOT_CONVERGED = "scf_not_converged"
    GEO_OPT_NOT_CONVERGED = "geo_opt_not_converged"
    WALLTIME = "walltime"
    EXECUTION_FAILED = "execution_failed"
    INVALID_RESULT = "invalid_result"
    ML_SCF_FAILED = "ml_scf_failed"
    CANCELLED = "cancelled"


class RetryStep(BaseModel):
    """Overrides applied to the *base* settings for one retry (never to a previous retry input)."""

    model_config = ConfigDict(extra="forbid")

    max_scf: int | None = Field(default=None, ge=1)
    max_scf_multiplier: float | None = Field(default=None, gt=0.0)
    use_ot: bool | None = None
    mixing_method: Literal["BROYDEN_MIXING", "PULAY_MIXING", "DIRECT_P_MIXING", "KERKER_MIXING"] | None = None
    mixing_alpha: float | None = Field(default=None, gt=0.0, le=1.0)
    mixing_nbuffer: int | None = Field(default=None, ge=1)
    scf_guess: str | None = None
    h_opt_max_iter_multiplier: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _exclusive_max_scf(self) -> RetryStep:
        if self.max_scf is not None and self.max_scf_multiplier is not None:
            raise ValueError("set either max_scf or max_scf_multiplier in one retry step, not both")
        return self


def _default_ladders() -> dict[FailureClass, list[RetryStep]]:
    return {
        # NBUFFER 4 needs diagonalization (CP2K ignores &MIXING under OT), then more SCF steps on OT.
        FailureClass.SCF_NOT_CONVERGED: [
            RetryStep(use_ot=False, mixing_method="BROYDEN_MIXING", mixing_nbuffer=4),
            RetryStep(max_scf_multiplier=3.0),
        ],
        FailureClass.GEO_OPT_NOT_CONVERGED: [RetryStep(h_opt_max_iter_multiplier=2.0)],
        FailureClass.WALLTIME: [RetryStep()],
        FailureClass.EXECUTION_FAILED: [RetryStep()],
        FailureClass.INVALID_RESULT: [],
        FailureClass.CANCELLED: [],
    }


class RetrySettings(BaseModel):
    """Bounded retry ladder per failure class (fixes F-25)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_retries: int = Field(default=3, ge=0)
    max_scf_cap: int = Field(default=2000, ge=1)
    ml_scf_native_fallback: bool = True
    ladders: dict[FailureClass, list[RetryStep]] = Field(default_factory=_default_ladders)

    @field_validator("ladders", mode="before")
    @classmethod
    def _fill_defaults(cls, value: Any) -> Any:
        if value is None:
            return _default_ladders()
        merged: dict[Any, Any] = {key.value: steps for key, steps in _default_ladders().items()}
        merged.update({str(getattr(key, "value", key)): steps for key, steps in dict(value).items()})
        return merged


class ExtraSections(BaseModel):
    """Raw CP2K lines appended inside GLOBAL, FORCE_EVAL or FORCE_EVAL/DFT (hashed into settings)."""

    model_config = ConfigDict(extra="forbid")

    global_lines: list[str] = Field(default_factory=list)
    force_eval_lines: list[str] = Field(default_factory=list)
    dft_lines: list[str] = Field(default_factory=list)


class CP2KConfig(BaseModel):
    """The ``cp2k`` section."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # execution
    executable: str = "cp2k.psmp"
    launcher: list[str] = Field(default_factory=list)  # e.g. ["mpiexec", "-n", "{ntasks}"]
    data_dir: str | None = None  # exported as CP2K_DATA_DIR when set
    environment: dict[str, str] = Field(default_factory=dict)
    task_dir: str = "cp2k_tasks"
    resources: SchedulerResources = Field(default_factory=SchedulerResources)
    max_unknown_polls_warning: int = Field(default=5, ge=1)
    # electronic structure
    basis_set_file: str = "BASIS_MOLOPT"
    potential_file: str = "GTH_POTENTIALS"
    basis: str = "DZVP-MOLOPT-SR-GTH"
    potential: str = "GTH-PBE"
    kinds: dict[str, KindSettings] = Field(default_factory=dict)
    functional: str = "PBE"
    cutoff: float = Field(default=400.0, gt=0.0)
    rel_cutoff: float = Field(default=60.0, gt=0.0)
    eps_default: float = Field(default=1.0e-12, gt=0.0)
    max_scf: int = Field(default=100, ge=1)
    scf_eps: float = Field(default=1.0e-6, gt=0.0)
    scf_guess: str = "ATOMIC"
    use_ot: bool = True
    ot_minimizer: str = "DIIS"
    ot_preconditioner: str = "FULL_SINGLE_INVERSE"
    mixing: MixingSettings = Field(default_factory=MixingSettings)
    dispersion: bool = False
    dispersion_parameter_file: str = "dftd3.dat"
    print_level: Literal["SILENT", "LOW", "MEDIUM", "HIGH", "DEBUG"] = "MEDIUM"
    stress: Literal["auto", "always", "never"] = "auto"
    poisson: PoissonSettings = Field(default_factory=PoissonSettings)
    cluster_padding: float = Field(default=6.0, gt=0.0)  # Angstrom of vacuum when a region has no usable cell
    min_vacuum: float = Field(default=2.0, ge=0.0)  # Angstrom to the box faces in non-periodic directions
    multiplicity: int | None = Field(default=None, ge=1)
    spin_policy: Literal["auto", "strict"] = "auto"
    uncapped_fragment_policy: Literal["error", "warn", "allow"] = "error"
    h_only_opt: HOnlyOptSettings = Field(default_factory=HOnlyOptSettings)
    template_path: str | None = None
    extra_sections: ExtraSections = Field(default_factory=ExtraSections)
    # labeling
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)

    @field_validator("kinds")
    @classmethod
    def _element_keys(cls, value: dict[str, KindSettings]) -> dict[str, KindSettings]:
        from ase.data import atomic_numbers

        for symbol in value:
            if symbol not in atomic_numbers:
                raise ValueError(f"cp2k.kinds key {symbol!r} is not an element symbol")
        return value

    def basis_for(self, symbol: str) -> str:
        kind = self.kinds.get(symbol)
        return kind.basis if kind is not None and kind.basis else self.basis

    def potential_for(self, symbol: str) -> str:
        kind = self.kinds.get(symbol)
        return kind.potential if kind is not None and kind.potential else self.potential

    # Fields that change the calculation (and therefore the label). Execution, retry and storage
    # settings are excluded so that moving a campaign to another machine keeps its task IDs.
    PHYSICS_FIELDS: ClassVar[tuple[str, ...]] = (
        "basis_set_file",
        "potential_file",
        "basis",
        "potential",
        "kinds",
        "functional",
        "cutoff",
        "rel_cutoff",
        "eps_default",
        "max_scf",
        "scf_eps",
        "scf_guess",
        "use_ot",
        "ot_minimizer",
        "ot_preconditioner",
        "mixing",
        "dispersion",
        "dispersion_parameter_file",
        "stress",
        "poisson",
        "cluster_padding",
        "multiplicity",
        "spin_policy",
        "h_only_opt",
        "extra_sections",
    )

    def settings_hash(self, *, template_text: str | None = None, extra: Any = None) -> str:
        """Deterministic hash of everything that changes the CP2K calculation."""

        payload = {name: self.model_dump(mode="json")[name] for name in self.PHYSICS_FIELDS}
        payload["h_only_opt"] = {
            key: value for key, value in payload["h_only_opt"].items() if key in {"enabled", "max_iter", "optimizer"}
        }
        payload["template"] = template_text
        payload["extra"] = extra
        return content_hash(payload, length=16)


__all__ = [
    "CP2KConfig",
    "ExtraSections",
    "FailureClass",
    "HOnlyOptSettings",
    "KindSettings",
    "MixingSettings",
    "PoissonSettings",
    "RetrySettings",
    "RetryStep",
    "ValidationSettings",
]
