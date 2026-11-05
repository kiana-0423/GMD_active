"""Configuration for the SCF subsystem (section ``scf`` in ``configs/sections/scf.yaml``).

Every switch defaults to *off* except plain parsing: nothing in the default
labeling path depends on this package beyond SCF-history parsing/recording,
and recording itself is opt-in (``scf.recording.enabled``).

PHAL spread these settings over ``cp2k.scf_recording``, ``cp2k.ml_scf``,
``cp2k_ml_scf`` and ``backend.dft.engine``. :meth:`ScfConfig.from_phal`
translates such a mapping, so PHAL configurations (and PHAL's tests) can be
replayed. Keys PHAL declared but never read (``cp2k_ml_scf.runtime_dir``,
``cp2k.ml_scf.mixing.fallback_method``) are accepted by the translator and
dropped with a warning; every key that exists in this model is read somewhere
(finding F-23).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gmd_active_learning.core.exceptions import ConfigurationError

LOGGER = logging.getLogger(__name__)

#: Density exchange / controller protocol spoken by the cp2k-ml-scf fork (VERSION_LOCK.md).
EXCHANGE_PROTOCOL = "phal-ml-scf-v2"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecordingConfig(_Strict):
    """``scf.recording``: what to keep from every CP2K labeling run."""

    #: Master switch; when false nothing is written.
    enabled: bool = False
    #: Root for ``<output_dir>/<task_id>/scf_history.json`` (relative to the working directory).
    output_dir: str = "runtime/cp2k/scf"
    #: Keep the converged AO density and overlap (needs &AO_MATRICES printing in the input).
    record_density: bool = False
    #: Keep every SCF iteration's density (needs record_density; prints &EACH QS_SCF 1).
    record_trajectory: bool = False
    #: Raise instead of skipping when the density cannot be recorded.
    density_strict: bool = False
    #: Raise instead of skipping when the trajectory cannot be recorded (PHAL declared, never read).
    trajectory_strict: bool = False
    #: Digits printed per AO matrix element.
    ndigits: int = Field(default=12, ge=6, le=16)
    #: Relative tolerance on Tr(P S) = N_e for recorded frames.
    electron_tolerance: float = Field(default=1.0e-3, gt=0.0)
    #: Hermiticity tolerance for recorded frames.
    hermitian_tolerance: float = Field(default=1.0e-8, gt=0.0)


class StrategyBase(_Strict):
    """The baseline ``&SCF`` settings a retry ladder starts from.

    PHAL's baseline used ``NBUFFER 8``; CP2K v2026.2's own default is 4
    (``qs_density_mixing_types.F``). The ``nbuffer4`` rung therefore returns to
    upstream's default.
    """

    mixing_method: Literal["PULAY_MIXING", "BROYDEN_MIXING", "DIRECT_P_MIXING"] = "PULAY_MIXING"
    alpha: float = Field(default=0.4, gt=0.0, le=1.0)
    nbuffer: int = Field(default=8, ge=1)
    smearing: bool = False
    added_mos: int = Field(default=0, ge=0)
    electronic_temperature: float = Field(default=500.0, gt=0.0)
    max_scf: int = Field(default=120, ge=1)
    eps_scf: float = Field(default=1.0e-6, gt=0.0)


class RetryConfig(_Strict):
    """``scf.retry``: the deterministic, model-free retry ladder (PHAL's one positive SCF result).

    Measured by PHAL on 12 solids: ``default -> nbuffer4`` converges 12/12 against
    9/12 for the default alone, for 4.7% more SCF iterations. Off by default so
    the labeling path's behaviour does not change unless asked.
    """

    enabled: bool = False
    #: Strategy names from :func:`~gmd_active_learning.adapters.dft.scf.strategy.strategy_grid`.
    order: list[str] = Field(default_factory=lambda: ["default", "nbuffer4"])
    #: SCF iterations charged for a failed attempt when *evaluating* a ladder.
    failure_cost: int = Field(default=120, ge=0)
    base: StrategyBase = Field(default_factory=StrategyBase)

    @field_validator("order")
    @classmethod
    def _non_empty_unique(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("retry order must name at least one strategy")
        if len(set(value)) != len(value):
            raise ValueError(f"retry order repeats a strategy: {value}")
        return value


class DatasetConfig(_Strict):
    """``scf.dataset``: the SCF sample store (PHAL's ``cp2k.ml_scf.dataset_dir``, previously unread)."""

    root: str = "runtime/cp2k/scf_dataset"
    split_ratios: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.8, "validation": 0.1, "test": 0.1}
    )

    @field_validator("split_ratios")
    @classmethod
    def _ratios(cls, value: dict[str, float]) -> dict[str, float]:
        if set(value) != {"train", "validation", "test"}:
            raise ValueError("split_ratios needs exactly train, validation and test")
        if any(ratio < 0 for ratio in value.values()) or abs(sum(value.values()) - 1.0) > 1e-9:
            raise ValueError(f"split_ratios must be non-negative and sum to 1, got {value}")
        return value


class PredictorConfig(_Strict):
    """``scf.ml_scf.predictor``: which density predictor produces the exchange file."""

    type: str = "dummy"
    model_path: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class MixingConfig(_Strict):
    """``scf.ml_scf.mixing``. ``ml`` is a documented negative result and no CP2K build implements it."""

    method: Literal["pulay", "ml"] = "pulay"
    history_length: int = Field(default=8, ge=1)
    alpha: float = Field(default=0.4, gt=0.0, le=1.0)


class MLSCFConfig(_Strict):
    """``scf.ml_scf``: the optional ML initial-guess wrapper around a modified CP2K.

    Off by default. No recorded ML-SCF component has shown an acceleration
    (PHAL measured the ML density guess at 63 SCF steps against 53 for the
    atomic guess, speedup 0.85); this exists so the experiment can be repeated.
    """

    enabled: bool = False
    #: Required for the torch MLP predictor and neural mixing (documented negative results).
    experimental: bool = False
    #: The modified CP2K executable; ``None`` falls back to ``scf.capabilities.executable``.
    executable: str | None = None
    #: Extra environment for both the capability probe and the CP2K run
    #: (e.g. the toolchain LD_LIBRARY_PATH the cp2k-ml-scf build needs at run time).
    environment: dict[str, str] = Field(default_factory=dict)
    #: A deployed in-process model (mode B). The cp2k-ml-scf reconstruction does not implement mode B.
    model_path: str | None = None
    model_digest: str | None = None
    #: Degrade to native CP2K before the run when ML-SCF cannot run (otherwise raise).
    fallback_to_standard_scf: bool = True
    #: Retry once with plain CP2K when an accelerated run fails (PHAL documented, never implemented).
    retry_native_on_failure: bool = True
    require_final_scf_convergence: bool = True
    protocol: str = EXCHANGE_PROTOCOL
    #: Directory for prediction provenance (``predictions.jsonl``); ``None`` disables logging.
    prediction_dir: str | None = None
    #: Relative tolerance for declared vs expected electron counts of a prediction.
    electron_tolerance: float = Field(default=1.0e-3, gt=0.0)
    #: Exported as PHAL_ML_SCF_GUESS_ALPHA; ``None`` leaves CP2K's default (1.0).
    guess_alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Export PHAL_ML_SCF_STRUCTURE_DIGEST so the fork can verify the exchange file's identity.
    export_structure_digest: bool = True
    #: Basis set name written into the exchange file and exported as PHAL_ML_SCF_BASIS.
    basis_set: str | None = None
    predictor: PredictorConfig = Field(default_factory=PredictorConfig)
    mixing: MixingConfig = Field(default_factory=MixingConfig)

    @model_validator(mode="after")
    def _experimental_gates(self) -> "MLSCFConfig":
        if self.mixing.method == "ml" and not self.experimental:
            raise ValueError(
                "scf.ml_scf.mixing.method=ml is a documented negative result with no CP2K "
                "consumer; set scf.ml_scf.experimental=true to request it anyway"
            )
        if self.predictor.type == "torch" and not self.experimental:
            raise ValueError(
                "predictor type 'torch' (atomic-population MLP) is experimental; set "
                "scf.ml_scf.experimental=true to use it"
            )
        return self


class CapabilityConfig(_Strict):
    """``scf.capabilities``: how CP2K builds are probed."""

    #: Executable to probe when nothing more specific is configured.
    executable: str | None = None
    timeout_s: float = Field(default=30.0, gt=0.0)


class ScfConfig(_Strict):
    """The ``scf`` configuration section."""

    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    ml_scf: MLSCFConfig = Field(default_factory=MLSCFConfig)
    capabilities: CapabilityConfig = Field(default_factory=CapabilityConfig)

    @classmethod
    def from_phal(cls, config: Mapping[str, Any]) -> "ScfConfig":
        """Translate a PHAL-shaped configuration (``cp2k.scf_recording`` ...)."""

        cp2k = _mapping(config.get("cp2k"))
        ml_block = _mapping(config.get("cp2k_ml_scf"))
        recording = dict(_mapping(cp2k.get("scf_recording")))
        ml = dict(_mapping(cp2k.get("ml_scf")))
        engine = _mapping(_mapping(config.get("backend")).get("dft")).get("engine")
        engine_selected = isinstance(engine, str) and engine.strip().lower().replace("-", "_") == "cp2k_ml_scf"

        mixing = dict(_mapping(ml.get("mixing")))
        if "fallback_method" in mixing:
            LOGGER.warning("PHAL key cp2k.ml_scf.mixing.fallback_method was never read; dropped")
            mixing.pop("fallback_method")
        if "runtime_dir" in ml_block:
            LOGGER.warning("PHAL key cp2k_ml_scf.runtime_dir was never read; dropped")

        ml_section: dict[str, Any] = {
            "enabled": bool(engine_selected and ml.get("enabled", True)),
            "experimental": bool(ml.get("experimental", False) or mixing.get("method") == "ml"
                                 or _mapping(ml.get("predictor")).get("type") == "torch"),
            "predictor": dict(_mapping(ml.get("predictor"))) or {},
            "mixing": mixing,
        }
        for key in ("prediction_dir", "electron_tolerance"):
            if ml.get(key) is not None:
                ml_section[key] = ml[key]
        for key in ("executable", "model_path", "model_digest", "fallback_to_standard_scf",
                    "require_final_scf_convergence", "environment"):
            if key in ml_block and ml_block[key] is not None:
                ml_section[key] = ml_block[key]
        if "controller_protocol" in ml_block:
            ml_section["protocol"] = ml_block["controller_protocol"]
        if "basis" in cp2k and cp2k["basis"]:
            ml_section["basis_set"] = str(cp2k["basis"])

        section: dict[str, Any] = {
            "recording": {k: v for k, v in recording.items() if k in RecordingConfig.model_fields},
            "ml_scf": ml_section,
        }
        if ml.get("dataset_dir"):
            section["dataset"] = {"root": str(ml["dataset_dir"])}
        if cp2k.get("executable"):
            section["capabilities"] = {"executable": str(cp2k["executable"])}
        return cls.model_validate(section)


def scf_config(config: "ScfConfig | Mapping[str, Any] | None") -> ScfConfig:
    """Coerce the accepted configuration shapes into :class:`ScfConfig`.

    Accepted: an :class:`ScfConfig`; ``None`` (defaults); a full configuration
    mapping with an ``scf`` section; a PHAL-shaped mapping (``cp2k`` /
    ``cp2k_ml_scf`` / ``backend`` keys); or the bare section mapping.
    """

    if isinstance(config, ScfConfig):
        return config
    if config is None:
        return ScfConfig()
    if not isinstance(config, Mapping):
        raise ConfigurationError(f"SCF configuration must be a mapping, got {type(config).__name__}")
    try:
        if "scf" in config:
            return ScfConfig.model_validate(config.get("scf") or {})
        if any(key in config for key in ("cp2k", "cp2k_ml_scf", "backend")):
            return ScfConfig.from_phal(config)
        return ScfConfig.model_validate(config)
    except ValueError as exc:  # pydantic.ValidationError is a ValueError
        raise ConfigurationError(f"invalid SCF configuration: {exc}") from exc


def labeling_cp2k_section(config: Any) -> Mapping[str, Any]:
    """The CP2K *labeling* settings (basis, potential, charge, ...) if the mapping carries them.

    Those belong to the CP2K labeling area, not to this section; the SCF code
    only reads them for provenance.
    """

    if isinstance(config, Mapping):
        return _mapping(config.get("cp2k"))
    return {}


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a mapping (PHAL's ``utils.config.load_config`` for the CLI)."""

    import yaml

    source = Path(path)
    if not source.is_file():
        raise ConfigurationError(f"configuration file not found: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ConfigurationError(f"{source}: configuration must be a mapping")
    return dict(payload)


def default_section_path() -> Path:
    """``configs/sections/scf.yaml`` in a source checkout."""

    return Path(__file__).resolve().parents[5] / "configs" / "sections" / "scf.yaml"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "EXCHANGE_PROTOCOL",
    "CapabilityConfig",
    "DatasetConfig",
    "MLSCFConfig",
    "MixingConfig",
    "PredictorConfig",
    "RecordingConfig",
    "RetryConfig",
    "ScfConfig",
    "StrategyBase",
    "default_section_path",
    "labeling_cp2k_section",
    "load_config",
    "scf_config",
]
