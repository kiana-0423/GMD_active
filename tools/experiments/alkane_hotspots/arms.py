"""Acquisition arms on a fixed, recorded candidate pool (A random local, B committee local, C PHAL local,
D PHAL acquisition with full-system labels) plus configurable ablations of C.

What is identical across arms (enforced by sharing code and inputs): model architecture and training
budget (``training`` section), initial data (full-system labels of ``training.initial_splits``),
validation data (``training.validation_splits``), reference settings (the label records), the
evaluation protocol (``evaluation`` on final_eval full-DFT configurations), the per-round acquisition
budget and the per-repetition seed. Arms differ only in how candidates are ranked and what is labeled.

Candidate pool: for every configuration of ``pool.splits`` (never final_eval), one candidate region per
carbon (``pool.stride``) cut with GMD_active's extraction (``pool.extraction`` / ``pool.capping``). The
pool is written once to ``pool.json`` and re-used; a fragment label belongs to a candidate when its
region (``region_path``) has the same parent, the same core parent atoms and the same non-cap parent
atoms. Labels come from a :class:`LabelProvider` (recorded: ``labels.jsonl``; on demand: a callable that
runs alpha's label runner). A label acquired by several arms is the same physical calculation and is
charged to each of them at its measured ``cost.cpu_core_s``; within one arm it is charged once.

Arms:

* ``random_local``: uniformly random order of unlabeled candidates (seeded per repetition and round).
* ``committee_local``: candidates ranked by the max committee force deviation over their core atoms
  (``committee_force_deviation``); no physical signals, no events, no diversity filter.
* ``phal_local``: each split_group's pool configurations are streamed through GMD_active's
  ``OnlineMonitorDriver`` (staged monitor with physical signals + committee, event buffering, hotspot
  clustering); every event's trigger-frame hotspot groups map to the best-scoring candidate whose core
  holds a hotspot atom; ``CandidateSelector`` applies the diversity filter (also against labeled
  fingerprints) and the per-event budget; only labels from a fidelity-gate-approved fragment setting
  with status ``valid`` enter training (validated local labeling).
* ``phal_full_system``: the same acquisition as ``phal_local``; each selected candidate's parent is
  labeled as a full system (one charge per parent).

Ablations (variants of ``phal_local``): ``physical_signals: false`` (committee only, every frame
escalated), ``lj: false``, ``event_buffering: false`` (every triggered frame is its own event),
``force_weights`` overrides (core/buffer/boundary weighting) and ``extraction`` overrides (a separate
pool, e.g. another buffer size). Combinations that do not apply raise ``ConfigurationError``.

The per-round budget is ``budget.cpu_core_s`` (measured DFT CPU core-seconds) and/or ``budget.labels``;
acquisition stops when either is reached. With ``overshoot: allow`` the calculation that crosses the
budget is kept (the overshoot is recorded); ``forbid`` skips a recorded label whose measured cost would
exceed the budget (only possible for recorded labels).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gmd_active_learning.active_learning.candidates import CandidateConfig, CandidateSelector, fingerprint_region
from gmd_active_learning.contracts import ExtractedRegion, Frame, RegionRole
from gmd_active_learning.core.exceptions import ConfigurationError, DataContractError
from gmd_active_learning.datasets.config import RoleWeights
from gmd_active_learning.events.config import load_events_config
from gmd_active_learning.extraction.config import CappingConfig, ExtractionConfig
from gmd_active_learning.extraction.dispatch import extract_and_prepare
from gmd_active_learning.hotspots.config import load_hotspots_config
from gmd_active_learning.hotspots.detector import cluster_atom_ids
from gmd_active_learning.monitors.config import MonitorConfig, load_monitor_config
from gmd_active_learning.monitors.ensemble_deviation import committee_force_deviation
from gmd_active_learning.monitors.online import OnlineMonitorDriver
from gmd_active_learning.monitors.staged_monitor import StagedOODMonitor

from .evaluation import EvaluationConfig, evaluate_frames
from .ground_truth import GroundTruthConfig, GTFrame, deep_merge, parent_frame, read_jsonl, resolve_artifact_path
from .receptive_field import ReceptiveFieldConfig
from .training import TrainingConfig, sample_for_label, write_dataset

LOGGER = logging.getLogger(__name__)

ARM_KINDS = ("random_local", "committee_local", "phal_local", "phal_full_system")
LOCAL_KINDS = ("random_local", "committee_local", "phal_local")
TEMPORAL_METRICS = ("delta_force", "displacement", "delta_q")
REPO_ROOT = Path(__file__).resolve().parents[3]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------------------------- config


class BudgetConfig(_Strict):
    cpu_core_s: float | None = Field(None, gt=0.0, description="measured DFT CPU core-seconds per round")
    labels: int | None = Field(None, ge=1, description="label count per round")
    overshoot: Literal["allow", "forbid"] = "allow"

    @model_validator(mode="after")
    def _one(self) -> BudgetConfig:
        if self.cpu_core_s is None and self.labels is None:
            raise ValueError("arms.budget needs cpu_core_s and/or labels")
        return self


class PoolConfig(_Strict):
    source: Literal["gate_selection", "explicit"] = Field(
        "gate_selection", description="gate_selection: alpha's build_fragment with the setting frozen by the fidelity gate "
                                      "(fidelity/selection.json), so pool fragments are exactly alpha's label requests; explicit: "
                                      "GMD_active extraction with `extraction` / `capping`")
    splits: list[str] = Field(default_factory=lambda: ["validation"])
    stride: int = Field(1, ge=1, description="one candidate per stride-th carbon")
    extraction: dict[str, Any] = Field(
        default_factory=lambda: {"mode": "cluster", "terminal_atoms": "follow_partner", "core_radius": 1.2, "buffer_radius": 2.0,
                                 "boundary_thickness": 1.0, "extract_radius": 3.2},
        description="must equal the fragment setting approved by alpha's fidelity gate (labels are matched by atom sets)")
    capping: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> PoolConfig:
        if "final_eval" in self.splits:
            raise ValueError("arms.pool.splits must not contain final_eval")
        if self.source == "explicit":
            ExtractionConfig.from_mapping(self.extraction)
            CappingConfig.from_mapping(self.capping)
        return self


class FidelityGate(_Strict):
    required: bool = True
    allow_unvalidated_overrides: bool = Field(False, description="permit ablations whose fragment setting did not pass the gate "
                                                                   "(recorded in the progress file; their labels are not validated)")
    directory: str = Field("fidelity", description="alpha's gate directory: selection.json = passed, failure.json = failed")


class ArmVariant(_Strict):
    name: str
    kind: Literal["random_local", "committee_local", "phal_local", "phal_full_system"]
    physical_signals: bool | None = None
    lj: bool | None = None
    event_buffering: bool | None = None
    force_weights: dict[str, float] | None = None
    extraction: dict[str, Any] | None = Field(None, description="pool override: an ExtractionConfig mapping (pool.source explicit) "
                                                                "or an alpha fragment setting (gate_selection), e.g. another buffer size")

    @model_validator(mode="after")
    def _supported(self) -> ArmVariant:
        phal_only = {"physical_signals": self.physical_signals, "lj": self.lj, "event_buffering": self.event_buffering}
        if self.kind in ("random_local", "committee_local"):
            used = [key for key, value in phal_only.items() if value is not None]
            if used:
                raise ValueError(f"variant {self.name}: {used} apply to PHAL arms only, not {self.kind}")
        if self.kind == "phal_full_system" and (self.force_weights is not None or self.extraction is not None):
            raise ValueError(f"variant {self.name}: force_weights/extraction overrides apply to local labels, not full-system labeling")
        if self.physical_signals is False and self.lj is False:
            raise ValueError(f"variant {self.name}: physical_signals=false already disables LJ; lj=false is redundant")
        if self.force_weights is not None:
            RoleWeights.model_validate(self.force_weights)
        return self


class ArmsConfig(_Strict):
    arms: list[Literal["random_local", "committee_local", "phal_local", "phal_full_system"]] = Field(default_factory=lambda: list(ARM_KINDS))
    ablations: list[ArmVariant] = Field(default_factory=list)
    repetitions: int = Field(1, ge=1)
    seeds: list[int] | None = None
    base_seed: int = 1000
    n_rounds: int = Field(1, ge=1)
    budget: BudgetConfig
    pool: PoolConfig = Field(default_factory=PoolConfig)
    selection: dict[str, Any] = Field(default_factory=lambda: {"max_per_cycle": 1000000, "max_per_event": 4, "min_score": None})
    monitor_overrides: dict[str, Any] = Field(default_factory=dict)
    events: dict[str, Any] = Field(default_factory=lambda: {"pre_trigger_frames": 0, "post_trigger_frames": 2, "cooldown_steps": 0})
    hotspots: dict[str, Any] = Field(default_factory=lambda: {"merge_radius": 3.0})
    calibration_splits: list[str] = Field(default_factory=lambda: ["calibration"])
    calibration_families: list[str] = Field(default_factory=lambda: ["normal"])
    label_mode: Literal["recorded", "on_demand"] = "recorded"
    missing_label_policy: Literal["error", "skip"] = "error"
    fidelity_gate: FidelityGate = Field(default_factory=FidelityGate)

    @model_validator(mode="after")
    def _check(self) -> ArmsConfig:
        if self.seeds is not None:
            if len(self.seeds) != self.repetitions:
                raise ValueError(f"arms.seeds has {len(self.seeds)} entries for {self.repetitions} repetitions")
            if len(set(self.seeds)) != len(self.seeds):
                raise ValueError("arms.seeds must be distinct (independent repetitions)")
        if "final_eval" in self.calibration_splits:
            raise ValueError("arms.calibration_splits must not contain final_eval")
        CandidateConfig.model_validate(self.selection)
        load_events_config(self.events)
        load_hotspots_config(self.hotspots)
        names = [variant.name for variant in self.variants()]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate arm/ablation names: {names}")
        return self

    def seed_for(self, repetition: int) -> int:
        return int(self.seeds[repetition]) if self.seeds is not None else self.base_seed + 1009 * repetition

    def variants(self) -> list[ArmVariant]:
        return [ArmVariant(name=kind, kind=kind) for kind in self.arms] + list(self.ablations)


class ReportConfig(_Strict):
    results_dir: str = Field("docs/demos/alkane_hotspots/results", description="committed summaries (relative to GMD_active)")


class FiguresConfig(_Strict):
    formats: list[Literal["pdf", "svg", "png"]] = Field(default_factory=lambda: ["pdf", "svg", "png"])
    dpi: int = Field(200, ge=50)
    fidelity_results: str = Field("fidelity/report_calibration.json", description="alpha's fidelity report (relative to the artifacts root)")
    timeline_report: str | None = Field(None, description="closed-loop controller report JSON (relative to the artifacts root)")


class InputsConfig(_Strict):
    parents: str = "parents.jsonl"
    labels: str = "labels.jsonl"


class BetaConfig(_Strict):
    """Top-level YAML of the hotspot-model study (configs/demos/alkane_hotspots/arms_*.yaml)."""

    name: str
    base_config: str | None = Field(None, description="alpha's demo config (relative to this file); supplies artifacts_root")
    artifacts_root: str | None = Field(None, description="overrides the base config; relative to GMD_active")
    inputs: InputsConfig = Field(default_factory=InputsConfig)
    output_subdir: str = "model_study"
    monitor: dict[str, Any] = Field(default_factory=dict, description="GMD_active monitor section (defaults from configs/sections)")
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    receptive_field: ReceptiveFieldConfig = Field(default_factory=ReceptiveFieldConfig)
    ground_truth: GroundTruthConfig = Field(default_factory=GroundTruthConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    arms: ArmsConfig | None = None
    figures: FiguresConfig = Field(default_factory=FiguresConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)


@dataclass(slots=True)
class LoadedConfig:
    config: BetaConfig
    path: Path
    artifacts_root: Path
    config_hash: str
    raw: dict[str, Any]
    base_raw: dict[str, Any] | None
    alpha: Any = None  # alpha's LoadedConfig (fragment builder, label runner) when available

    @property
    def output_dir(self) -> Path:
        return self.artifacts_root / self.config.output_subdir

    def input_path(self, relative: str) -> Path:
        return resolve_artifact_path(relative, self.artifacts_root)

    def bases(self) -> tuple[Path, ...]:
        return (self.artifacts_root, self.artifacts_root / Path(self.config.inputs.parents).parent)


def canonical_hash(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def load_beta_config(path: str | Path, *, alpha: Any = None) -> LoadedConfig:
    """Validate the study YAML. Artifacts root: the study's ``artifacts_root`` > alpha's config (``alpha`` or
    ``base_config``) > ``../alkane_artifacts/<name>`` (relative to GMD_active)."""

    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    return beta_from_mapping(raw, path=path, alpha=alpha)


def beta_from_mapping(raw: Mapping[str, Any], *, path: Path, alpha: Any = None) -> LoadedConfig:
    raw = dict(raw)
    config = BetaConfig.model_validate(raw)
    if alpha is None and config.base_config is not None:
        from .config import load_config  # alpha's loader resolves the artifacts root of the demo

        alpha = load_config((path.parent / config.base_config).resolve())
    if config.artifacts_root is not None:
        root = Path(config.artifacts_root)
        artifacts_root = root if root.is_absolute() else (REPO_ROOT / root).resolve()
    elif alpha is not None:
        artifacts_root = Path(alpha.artifacts_root)
    else:
        artifacts_root = (REPO_ROOT / f"../alkane_artifacts/{config.name}").resolve()
    base_hash = None if alpha is None else alpha.config.config_hash()
    digest = canonical_hash({"beta": raw, "alpha_config_hash": base_hash})
    return LoadedConfig(config, path, artifacts_root, digest, raw, None if alpha is None else alpha.config.model_dump(mode="json"), alpha)


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, target)
    return target


def json_safe(value: Any) -> Any:
    """NaN/inf -> None, numpy scalars/arrays -> Python, recursively."""

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# ---------------------------------------------------------------------------------------------- pool


@dataclass(slots=True)
class PoolCandidate:
    candidate_id: str
    parent_id: str
    split_group: str
    center: int
    region: ExtractedRegion
    core_ids: frozenset[int]
    atom_ids: frozenset[int]

    @property
    def key(self) -> tuple[str, frozenset[int], frozenset[int]]:
        return (self.parent_id, self.core_ids, self.atom_ids)


def region_key(parent_id: str, region: ExtractedRegion) -> tuple[str, frozenset[int], frozenset[int]]:
    roles = region.roles
    core = frozenset(int(i) for i, role in zip(region.parent_atom_ids, roles) if role is RegionRole.CORE)
    atoms = frozenset(int(i) for i, role in zip(region.parent_atom_ids, roles) if role is not RegionRole.H_CAP)
    return (str(parent_id), core, atoms)


#: ``builder(parent_record, frame, center_atom_id, override) -> prepared region`` (override = variant.extraction)
FragmentBuilder = Callable[[Mapping[str, Any], Frame, int, "Mapping[str, Any] | None"], ExtractedRegion]


def explicit_builder(pool: PoolConfig) -> FragmentBuilder:
    """GMD_active extraction + capping with ``pool.extraction`` (or the variant's override)."""

    capping_cfg = CappingConfig.from_mapping(dict(pool.capping))

    def build(record: Mapping[str, Any], frame: Frame, center: int, override: Mapping[str, Any] | None) -> ExtractedRegion:
        extraction_cfg = ExtractionConfig.from_mapping(dict(override if override is not None else pool.extraction))
        (region,) = extract_and_prepare(frame, [[center]], extraction=extraction_cfg, capping=capping_cfg, event_id=None)
        return region

    return build


def gate_builder(alpha_loaded: Any, selected_setting: Mapping[str, Any]) -> FragmentBuilder:
    """alpha's own fragment construction with the gate-selected setting (identical systems -> shared label cache)."""

    from .fidelity import build_fragment

    def build(record: Mapping[str, Any], frame: Frame, center: int, override: Mapping[str, Any] | None) -> ExtractedRegion:
        setting = dict(override) if override is not None else dict(selected_setting)
        region, _core = build_fragment(frame, [int(center)], setting, alpha_loaded.config)
        return region

    return build


def build_pool(parents: Sequence[Mapping[str, Any]], pool: PoolConfig, bases: Sequence[Path], *,
               extraction: Mapping[str, Any] | None = None, builder: FragmentBuilder | None = None) -> list[PoolCandidate]:
    """One candidate per (pool configuration, stride-th carbon); deterministic order."""

    if builder is None:
        if pool.source != "explicit":
            raise ConfigurationError("arms.pool.source=gate_selection needs alpha's fragment builder and a passed fidelity gate")
        builder = explicit_builder(pool)
    candidates: list[PoolCandidate] = []
    for record in sorted((p for p in parents if p.get("split") in pool.splits), key=lambda p: str(p["parent_id"])):
        frame = parent_frame(record, *bases)
        carbons = [int(i) for i in np.flatnonzero(frame.numbers == 6)][:: pool.stride]
        for center in carbons:
            atom_id = int(frame.atom_ids[center])
            region = builder(record, frame, atom_id, extraction)
            key = region_key(str(record["parent_id"]), region)
            candidates.append(PoolCandidate(candidate_id=region.region_id, parent_id=str(record["parent_id"]),
                                            split_group=str(record.get("split_group", record["parent_id"])), center=atom_id,
                                            region=region, core_ids=key[1], atom_ids=key[2]))
    ids = [c.candidate_id for c in candidates]
    if len(set(ids)) != len(ids):
        raise DataContractError("candidate pool has duplicate region ids")
    return candidates


def save_pool(candidates: Sequence[PoolCandidate], path: Path, *, config_hash: str) -> None:
    payload = {"config_hash": config_hash, "candidates": [
        {"candidate_id": c.candidate_id, "parent_id": c.parent_id, "split_group": c.split_group, "center": c.center,
         "core_ids": sorted(c.core_ids), "atom_ids": sorted(c.atom_ids), "region": c.region.to_dict()} for c in candidates]}
    write_json_atomic(path, payload)


def load_pool(path: Path, *, config_hash: str) -> list[PoolCandidate] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if payload.get("config_hash") != config_hash:
        raise ConfigurationError(f"{path} was built with another pool configuration; delete it or use a new output directory")
    return [PoolCandidate(item["candidate_id"], item["parent_id"], item["split_group"], int(item["center"]),
                          ExtractedRegion.from_dict(item["region"]), frozenset(item["core_ids"]), frozenset(item["atom_ids"]))
            for item in payload["candidates"]]


# ---------------------------------------------------------------------------------------------- labels


class LabelProvider(Protocol):
    def fragment(self, candidate: PoolCandidate) -> dict[str, Any] | None: ...

    def full(self, parent_id: str) -> dict[str, Any] | None: ...


class RecordedLabelProvider:
    """Label records from ``labels.jsonl``; fragments matched by region key, full systems by parent."""

    def __init__(self, labels: Sequence[Mapping[str, Any]], bases: Sequence[Path]):
        self.by_key: dict[Any, dict[str, Any]] = {}
        self.full_by_parent: dict[str, dict[str, Any]] = {}
        for record in labels:
            if record.get("kind") == "full":
                self._put(self.full_by_parent, str(record["parent_id"]), record)
            elif record.get("kind") == "fragment" and record.get("region_path"):
                region = ExtractedRegion.from_dict(json.loads(resolve_artifact_path(record["region_path"], *bases).read_text()))
                self._put(self.by_key, region_key(str(record["parent_id"]), region), record)

    @staticmethod
    def _put(table: dict[Any, dict[str, Any]], key: Any, record: Mapping[str, Any]) -> None:
        current = table.get(key)
        # prefer a valid record; among equals keep the first (file order)
        if current is None or (current.get("status") != "valid" and record.get("status") == "valid"):
            table[key] = dict(record)

    def fragment(self, candidate: PoolCandidate) -> dict[str, Any] | None:
        return self.by_key.get(candidate.key)

    def full(self, parent_id: str) -> dict[str, Any] | None:
        return self.full_by_parent.get(str(parent_id))


class CallableLabelProvider:
    """On-demand labels through callables (alpha's label runner); results are records of the shared schema."""

    def __init__(self, fragment_fn: Callable[[PoolCandidate], dict[str, Any] | None], full_fn: Callable[[str], dict[str, Any] | None]):
        self.fragment_fn, self.full_fn = fragment_fn, full_fn

    def fragment(self, candidate: PoolCandidate) -> dict[str, Any] | None:
        return self.fragment_fn(candidate)

    def full(self, parent_id: str) -> dict[str, Any] | None:
        return self.full_fn(parent_id)


def measured_cost(record: Mapping[str, Any]) -> float:
    cost = (record.get("cost") or {}).get("cpu_core_s")
    if cost is None or not np.isfinite(float(cost)) or float(cost) < 0:
        raise DataContractError(f"label {record.get('label_id')} (status {record.get('status')}) has no measured cpu_core_s")
    return float(cost)


# ---------------------------------------------------------------------------------------------- ranking


@dataclass(slots=True)
class PoolFrames:
    """Model predictions on the pool and calibration configurations for one round."""

    items: dict[str, GTFrame]  # parent_id -> frame (dft_forces may be None)
    calibration: list[GTFrame]


def arm_monitor_config(base: Mapping[str, Any], overrides: Mapping[str, Any], variant: ArmVariant) -> MonitorConfig:
    data = deep_merge(load_monitor_config(dict(base)).model_dump(), dict(overrides))
    enabled = [name for name in TEMPORAL_METRICS if data["metrics"][name]["enabled"]]
    if enabled:
        raise ConfigurationError(f"temporal metrics {enabled} need an MD trajectory; on static pool configurations they "
                                 "would leak the perturbation site. Disable them in monitor/arms.monitor_overrides.")
    if variant.physical_signals is False:
        for name in ("force", "rmin", "lj_residual", "mlip_force_deviation"):
            data["metrics"][name]["enabled"] = False
        data["actions"]["stop_min_distance"] = None
        data["actions"]["stop_max_force"] = None
        data["stages"].update(physics_interval="always", full_interval="always")
    if variant.lj is False:
        data["metrics"]["lj_residual"]["enabled"] = False
    return load_monitor_config(data)


def _frame_for(item: GTFrame, *, run_id: str, step: int) -> Frame:
    return Frame(run_id=run_id, step=step, atom_ids=np.arange(len(item.numbers)), numbers=item.numbers, positions=item.positions,
                 cell=None, pbc=(False, False, False), forces=item.model_forces, committee_forces=item.committee_forces)


def calibrated_state(config: MonitorConfig, frames: Sequence[GTFrame]) -> dict[str, Any]:
    monitor = StagedOODMonitor(config)
    for step, item in enumerate(frames):
        monitor.reset_temporal_state()
        monitor.evaluate_detailed(_frame_for(item, run_id="calibration", step=step))
    state = monitor.to_state()
    state["previous"] = None
    return state


def rank_random(candidates: Sequence[PoolCandidate], seed: int) -> tuple[list[PoolCandidate], dict[str, Any]]:
    order = np.random.default_rng(seed).permutation(len(candidates))
    return [candidates[i] for i in order], {"seed": seed}


def rank_committee(candidates: Sequence[PoolCandidate], frames: PoolFrames) -> tuple[list[PoolCandidate], dict[str, Any]]:
    sigma: dict[str, np.ndarray] = {}
    scored = []
    for candidate in candidates:
        item = frames.items[candidate.parent_id]
        if item.committee_forces is None:
            raise ConfigurationError("committee_local needs committee forces (training.sgt.committee_size >= 1)")
        if candidate.parent_id not in sigma:
            sigma[candidate.parent_id] = committee_force_deviation(item.committee_forces)
        score = float(np.max(sigma[candidate.parent_id][sorted(candidate.core_ids)]))
        scored.append((score, candidate))
    scored.sort(key=lambda pair: (-pair[0], pair[1].candidate_id))
    return [c for _, c in scored], {"scores": {c.candidate_id: s for s, c in scored}}


def rank_phal(candidates: Sequence[PoolCandidate], frames: PoolFrames, *, monitor_cfg: MonitorConfig, events: Mapping[str, Any],
              hotspots: Mapping[str, Any], selection: CandidateConfig, labeled: Sequence[PoolCandidate],
              event_buffering: bool) -> tuple[list[PoolCandidate], dict[str, Any]]:
    """Staged monitor + events + hotspot grouping + CandidateSelector over the fixed pool."""

    events_cfg = dict(events)
    if not event_buffering:
        events_cfg.update(pre_trigger_frames=0, post_trigger_frames=0, cooldown_steps=0)
    hotspots_cfg = load_hotspots_config(dict(hotspots))
    state = calibrated_state(monitor_cfg, frames.calibration)
    by_parent: dict[str, list[PoolCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_parent[candidate.parent_id].append(candidate)
    groups: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        if candidate.parent_id not in groups[candidate.split_group]:
            groups[candidate.split_group].append(candidate.parent_id)
    selector = CandidateSelector(selection)
    pool_candidates = []
    diagnostics: dict[str, Any] = {"events": [], "n_triggered_frames": 0, "n_frames": 0}
    for group in sorted(groups):
        parents = sorted(groups[group])
        monitor = StagedOODMonitor(monitor_cfg)
        monitor.load_state(state)
        found: list[Any] = []
        results: dict[int, Any] = {}
        driver = OnlineMonitorDriver(monitor_cfg, events_cfg, hotspots_cfg, monitor=monitor, on_event=found.append)
        for step, parent_id in enumerate(parents):
            result, _ = driver.process(_frame_for(frames.items[parent_id], run_id=group, step=step))
            results[step] = result
            diagnostics["n_frames"] += 1
            diagnostics["n_triggered_frames"] += int(result.triggered)
        driver.finish("end_of_stream")
        for event in found:
            parent_id = parents[event.trigger_frame.step]
            result = results[event.trigger_frame.step]
            ids = (event.metadata or {}).get("trigger_hotspot_atom_ids") or event.hotspot_atom_ids
            scores = np.asarray(result.atom_scores, dtype=float)
            clusters = cluster_atom_ids(event.trigger_frame, list(ids), scores, merge_radius=hotspots_cfg.merge_radius)
            chosen = []
            for hotspot in clusters:
                members = set(hotspot.atom_ids)
                options = [c for c in by_parent[parent_id] if c.core_ids & members]
                if not options:
                    continue
                best = max(options, key=lambda c: (float(np.max(scores[sorted(c.core_ids)])), -abs(c.center - min(members)), c.candidate_id))
                if best.candidate_id in {c.candidate_id for c, _ in chosen}:
                    continue
                chosen.append((best, float(np.max(scores[sorted(best.core_ids)]))))
            for candidate, score in chosen:
                region = candidate.region
                region_for_selection = ExtractedRegion.from_dict({**region.to_dict(), "event_id": event.event_id})
                pool_candidates.append(selector.make_candidate(region_for_selection, score=score,
                                                               metadata={"candidate_id": candidate.candidate_id}))
            diagnostics["events"].append({"event_id": event.event_id, "group": group, "parent_id": parent_id,
                                          "trigger_steps": list(event.trigger_steps), "n_hotspots": len(clusters),
                                          "candidates": [c.candidate_id for c, _ in chosen], "max_score": event.max_score})
    labeled_fps = [fingerprint_region(c.region, selection) for c in labeled]
    decision = selector.select(pool_candidates, labeled_fingerprints=labeled_fps)
    by_id = {c.candidate_id: c for c in candidates}
    ordered = [by_id[item.metadata["candidate_id"]] for item in decision.selected]
    diagnostics["rejected"] = dict(sorted(decision.rejected.items()))
    diagnostics["n_event_candidates"] = len(pool_candidates)
    return ordered, diagnostics


# ---------------------------------------------------------------------------------------------- acquisition


@dataclass(slots=True)
class Acquisition:
    label_id: str
    kind: str
    status: str
    cpu_core_s: float
    candidate_id: str | None
    parent_id: str
    round: int

    def to_dict(self) -> dict[str, Any]:
        return {"label_id": self.label_id, "kind": self.kind, "status": self.status, "cpu_core_s": self.cpu_core_s,
                "candidate_id": self.candidate_id, "parent_id": self.parent_id, "round": self.round}


def acquire(ordered: Sequence[PoolCandidate], provider: LabelProvider, budget: BudgetConfig, *, kind: str, round_index: int,
            charged_labels: set[str], labeled_parents: set[str], label_mode: str, missing_policy: str,
            record_sink: dict[str, dict[str, Any]] | None = None) -> tuple[list[Acquisition], dict[str, Any]]:
    """Walk the ranking and acquire labels until the per-round budget is reached."""

    acquisitions: list[Acquisition] = []
    skipped: dict[str, str] = {}
    spent, count = 0.0, 0
    for candidate in ordered:
        if budget.labels is not None and count >= budget.labels:
            break
        if budget.cpu_core_s is not None and spent >= budget.cpu_core_s:
            break
        if kind == "phal_full_system":
            if candidate.parent_id in labeled_parents:
                skipped[candidate.candidate_id] = "parent already labeled"
                continue
            record = provider.full(candidate.parent_id)
        else:
            record = provider.fragment(candidate)
        if record is None:
            if missing_policy == "error":
                raise DataContractError(f"no label record for candidate {candidate.candidate_id} ({kind}); "
                                        "run the label runner for the pool or set arms.missing_label_policy: skip")
            skipped[candidate.candidate_id] = "no label record"
            continue
        if record.get("status") == "budget_skipped":
            skipped[candidate.candidate_id] = "label budget_skipped (not computed)"
            continue
        label_id = str(record["label_id"])
        if label_id in charged_labels:
            skipped[candidate.candidate_id] = "label already acquired by this arm"
            continue
        if record.get("status") == "failed" and (record.get("cost") or {}).get("cpu_core_s") is None:
            skipped[candidate.candidate_id] = "label failed before any measured CPU time (not charged)"
            continue
        cost = measured_cost(record)
        if budget.overshoot == "forbid" and budget.cpu_core_s is not None and spent + cost > budget.cpu_core_s:
            if label_mode != "recorded":
                raise ConfigurationError("overshoot=forbid needs recorded labels (the cost of an on-demand label is known only afterwards)")
            skipped[candidate.candidate_id] = "measured cost would exceed the round budget"
            continue
        acquisitions.append(Acquisition(label_id, "full" if kind == "phal_full_system" else "fragment", str(record.get("status")),
                                        cost, candidate.candidate_id, candidate.parent_id, round_index))
        charged_labels.add(label_id)
        if record_sink is not None:
            record_sink[label_id] = dict(record)
        if kind == "phal_full_system":
            labeled_parents.add(candidate.parent_id)
        spent += cost
        count += 1
    overshoot = None if budget.cpu_core_s is None else max(0.0, spent - budget.cpu_core_s)
    return acquisitions, {"spent_cpu_core_s": spent, "n_charged": count, "overshoot_cpu_core_s": overshoot, "skipped": skipped}


# ---------------------------------------------------------------------------------------------- runner


@dataclass
class ArmContext:
    """Everything the runner needs besides the configuration; injectable for tests."""

    parents: list[dict[str, Any]]
    labels: list[dict[str, Any]]
    bases: tuple[Path, ...]
    provider: LabelProvider
    trainer: Callable[[Any, Path, int], Any]  # (dataset, work_dir, seed) -> model with .checkpoints
    predictor_factory: Callable[[Any], Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray | None]]]
    out_dir: Path
    config_hash: str
    gate_passed: bool | None = None
    fragment_builder: FragmentBuilder | None = None
    gate_setting: dict[str, Any] | None = None
    gate_passing_settings: frozenset[str] = frozenset()
    label_lookup: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.label_lookup = {str(r["label_id"]): dict(r) for r in self.labels}


def _predict_frames(parents: Sequence[Mapping[str, Any]], splits: Sequence[str], predictor: Any, bases: Sequence[Path],
                    labels_full: Mapping[str, Mapping[str, Any]] | None = None, families: Sequence[str] | None = None) -> list[GTFrame]:
    from .ground_truth import label_forces

    frames = []
    for record in sorted((p for p in parents if p.get("split") in splits), key=lambda p: str(p["parent_id"])):
        if families is not None and record.get("family") not in families:
            continue
        frame = parent_frame(record, *bases)
        forces, committee = predictor(frame.numbers, frame.positions)
        dft = None
        if labels_full is not None:
            label = labels_full.get(str(record["parent_id"]))
            if label is None:
                continue
            dft = label_forces(label, *bases)
        frames.append(GTFrame(parent_id=str(record["parent_id"]), family=str(record.get("family")), split=str(record.get("split")),
                              split_group=str(record.get("split_group", record["parent_id"])), numbers=frame.numbers, positions=frame.positions,
                              model_forces=forces, dft_forces=dft, committee_forces=committee,
                              perturbation_atoms=[int(a) for a in (record.get("perturbation") or {}).get("atoms") or []]))
    return frames


def _dataset_key(sample_ids: Sequence[str], seed: int, training_hash: str) -> str:
    return canonical_hash({"samples": sorted(sample_ids), "seed": seed, "training": training_hash})[:20]


class ArmsRunner:
    """Resumable execution of every (variant, repetition): rounds of rank -> acquire -> train -> evaluate."""

    def __init__(self, beta: BetaConfig, context: ArmContext) -> None:
        if beta.arms is None:
            raise ConfigurationError("the configuration has no arms section")
        self.beta = beta
        self.cfg = beta.arms
        self.ctx = context
        self.parents_by_id = {str(p["parent_id"]): p for p in context.parents}
        self.full_labels = {}
        for record in context.labels:
            if record.get("kind") == "full" and record.get("status") == "valid":
                self.full_labels[str(record["parent_id"])] = record
        self.training_hash = canonical_hash(beta.training.model_dump())
        self._pools: dict[str, list[PoolCandidate]] = {}
        self.evaluated: dict[str, dict[str, Any]] = {}

    # -- inputs shared by every arm ----------------------------------------------------------------
    def pool(self, variant: ArmVariant) -> list[PoolCandidate]:
        key = canonical_hash({"pool": self.cfg.pool.model_dump(), "override": variant.extraction, "gate_setting": self.ctx.gate_setting})
        if key not in self._pools:
            path = self.ctx.out_dir / f"pool_{key[:12]}.json"
            pool = load_pool(path, config_hash=key)
            if pool is None:
                pool = build_pool(self.ctx.parents, self.cfg.pool, self.ctx.bases, extraction=variant.extraction,
                                  builder=self.ctx.fragment_builder)
                save_pool(pool, path, config_hash=key)
            self._pools[key] = pool
        return self._pools[key]

    def _split_samples(self, splits: Sequence[str]) -> list[Any]:
        samples = []
        for parent_id, record in sorted(self.full_labels.items()):
            parent = self.parents_by_id.get(parent_id)
            if parent is not None and parent.get("split") in splits:
                samples.append(sample_for_label(record, self.parents_by_id, self.beta.training, *self.ctx.bases))
        return samples

    def _train(self, samples: Sequence[Any], val: Sequence[Any], seed: int) -> tuple[Any, str]:
        key = _dataset_key([s.sample_id for s in samples] + [f"val:{s.sample_id}" for s in val], seed, self.training_hash)
        work = self.ctx.out_dir / "models" / key
        dataset = write_dataset(list(samples), list(val), work / "dataset", version=key)
        return self.ctx.trainer(dataset, work / "train", seed), key

    def _evaluate(self, model: Any, model_key: str) -> dict[str, Any]:
        if model_key not in self.evaluated:
            predictor = self.ctx.predictor_factory(model)
            frames = _predict_frames(self.ctx.parents, self.beta.evaluation.splits, predictor, self.ctx.bases, self.full_labels)
            self.evaluated[model_key] = evaluate_frames(frames, self.beta.evaluation)
        return self.evaluated[model_key]

    def _check_gate(self, variant: ArmVariant) -> None:
        if variant.kind == "phal_full_system" or not self.cfg.fidelity_gate.required:
            return
        if variant.extraction is not None and self.cfg.pool.source == "gate_selection":
            setting_id = variant.extraction.get("setting_id")
            if setting_id not in self.ctx.gate_passing_settings and not self.cfg.fidelity_gate.allow_unvalidated_overrides:
                raise ConfigurationError(f"ablation {variant.name}: fragment setting {setting_id!r} did not pass the fidelity gate; "
                                         "set arms.fidelity_gate.allow_unvalidated_overrides: true to run it as an explicit, recorded "
                                         "unvalidated ablation")
        if self.ctx.gate_passed is not True:
            raise ConfigurationError(f"arm {variant.name} labels fragments, but the fragment-fidelity gate has "
                                     f"{'not been run' if self.ctx.gate_passed is None else 'failed'}; local labels are not validated "
                                     "(set arms.fidelity_gate.required: false only for a recorded negative-control run)")

    def _check_monitor_calibration(self, variant: ArmVariant) -> None:
        """A PHAL arm whose staged monitor cannot leave warm-up never triggers: refuse it instead of recording 0 labels."""

        if variant.kind not in ("phal_local", "phal_full_system"):
            return
        warmup = arm_monitor_config(self.beta.monitor, self.cfg.monitor_overrides, variant).running_stats.warmup_frames
        n = sum(1 for p in self.ctx.parents
                if p.get("split") in self.cfg.calibration_splits and p.get("family") in self.cfg.calibration_families)
        if n < warmup:
            raise ConfigurationError(
                f"arm {variant.name}: the monitor is calibrated on {n} configuration(s) (arms.calibration_splits="
                f"{self.cfg.calibration_splits}, families={self.cfg.calibration_families}) but monitor.running_stats.warmup_frames="
                f"{warmup}; its z-scored terms would stay suppressed and the arm could never trigger. Provide at least {warmup} "
                "calibration configurations (more base conformations, or further non-final_eval splits)")

    # -- one (variant, repetition) ----------------------------------------------------------------
    def run_one(self, variant: ArmVariant, repetition: int) -> dict[str, Any]:
        self._check_gate(variant)
        self._check_monitor_calibration(variant)
        seed = self.cfg.seed_for(repetition)
        run_dir = self.ctx.out_dir / "arms" / variant.name / f"rep{repetition}"
        progress_path = run_dir / "progress.json"
        variant_hash = canonical_hash({"variant": variant.model_dump(), "arms": self.cfg.model_dump(), "training": self.training_hash,
                                       "evaluation": self.beta.evaluation.model_dump(), "monitor": self.beta.monitor, "seed": seed})
        progress: dict[str, Any] = {"variant": variant.model_dump(), "repetition": repetition, "seed": seed, "config_hash": variant_hash,
                                    "rounds": [], "status": "running"}
        if progress_path.is_file():
            stored = json.loads(progress_path.read_text())
            if stored.get("config_hash") != variant_hash:
                raise ConfigurationError(f"{progress_path} belongs to another configuration; use a new output directory")
            progress = stored
            if progress.get("status") == "complete":
                return progress
        training = self.beta.training
        if variant.force_weights is not None:
            training = training.model_copy(update={"force_weights": RoleWeights.model_validate(variant.force_weights)})
        initial = self._split_samples(training.initial_splits)
        val = self._split_samples(training.validation_splits)
        if not initial:
            raise DataContractError(f"no valid full-system labels for the initial splits {training.initial_splits}")
        pool = self.pool(variant)
        by_id = {c.candidate_id: c for c in pool}
        acquired = [Acquisition(**item) for r in progress["rounds"] for item in r["acquisitions"]]
        charged = {a.label_id for a in acquired}
        labeled_candidates = {a.candidate_id for a in acquired if a.candidate_id}
        labeled_parents = {a.parent_id for a in acquired if a.kind == "full"}
        if not progress["rounds"]:
            model, key = self._train(initial, val, seed)
            evaluation = self._evaluate(model, key)
            progress["rounds"].append(self._round_record(0, [], {}, model, key, evaluation, acquired))
            write_json_atomic(progress_path, progress)
        for round_index in range(len(progress["rounds"]), self.cfg.n_rounds + 1):
            previous = progress["rounds"][-1]
            model = _ModelRef(previous["model"]["checkpoints"])
            predictor = self.ctx.predictor_factory(model)
            available = [c for c in pool if c.candidate_id not in labeled_candidates
                         and not (variant.kind == "phal_full_system" and c.parent_id in labeled_parents)]
            ordered, diagnostics = self._rank(variant, available, [by_id[c] for c in labeled_candidates if c in by_id],
                                              predictor, seed=seed * 1000 + round_index)
            new, accounting = acquire(ordered, self.ctx.provider, self.cfg.budget, kind=variant.kind, round_index=round_index,
                                      charged_labels=charged, labeled_parents=labeled_parents, label_mode=self.cfg.label_mode,
                                      missing_policy=self.cfg.missing_label_policy, record_sink=self.ctx.label_lookup)
            acquired.extend(new)
            labeled_candidates |= {a.candidate_id for a in new if a.candidate_id}
            samples = list(initial) + self._acquired_samples(acquired, training)
            model, key = self._train(samples, val, seed)
            evaluation = self._evaluate(model, key)
            progress["rounds"].append(self._round_record(round_index, new, {**diagnostics, "accounting": accounting}, model, key, evaluation, acquired))
            write_json_atomic(progress_path, progress)
        progress["status"] = "complete"
        write_json_atomic(progress_path, progress)
        return progress

    def _acquired_samples(self, acquired: Sequence[Acquisition], training: TrainingConfig) -> list[Any]:
        samples = []
        for item in acquired:
            if item.status != "valid":
                continue  # charged (the calculation ran) but never trained on
            record = self.ctx.label_lookup.get(item.label_id)
            if record is None:
                raise DataContractError(f"acquired label {item.label_id} is missing from the label records")
            samples.append(sample_for_label(record, self.parents_by_id, training, *self.ctx.bases))
        return samples

    def _rank(self, variant: ArmVariant, available: Sequence[PoolCandidate], labeled: Sequence[PoolCandidate], predictor: Any,
              *, seed: int) -> tuple[list[PoolCandidate], dict[str, Any]]:
        if variant.kind == "random_local":
            return rank_random(available, seed)
        parents_needed = sorted({c.parent_id for c in available})
        items = {f.parent_id: f for f in _predict_frames([self.parents_by_id[p] for p in parents_needed], self.cfg.pool.splits,
                                                          predictor, self.ctx.bases)}
        calibration = _predict_frames(self.ctx.parents, self.cfg.calibration_splits, predictor, self.ctx.bases,
                                      families=self.cfg.calibration_families)
        frames = PoolFrames(items, calibration)
        if variant.kind == "committee_local":
            return rank_committee(available, frames)
        monitor_cfg = arm_monitor_config(self.beta.monitor, self.cfg.monitor_overrides, variant)
        selection = CandidateConfig.model_validate(self.cfg.selection)
        return rank_phal(available, frames, monitor_cfg=monitor_cfg, events=self.cfg.events, hotspots=self.cfg.hotspots,
                         selection=selection, labeled=labeled, event_buffering=variant.event_buffering is not False)

    @staticmethod
    def _round_record(index: int, new: Sequence[Acquisition], diagnostics: Mapping[str, Any], model: Any, key: str,
                      evaluation: Mapping[str, Any], acquired: Sequence[Acquisition]) -> dict[str, Any]:
        overall = evaluation.get("overall", {}) if evaluation.get("status") == "executed" else {}
        return {
            "round": index,
            "acquisitions": [a.to_dict() for a in new],
            "n_labels": len(acquired),
            "n_valid_labels": sum(1 for a in acquired if a.status == "valid"),
            "cumulative_cpu_core_s": float(sum(a.cpu_core_s for a in acquired)),
            "model": {"checkpoints": list(model.checkpoints), "key": key},
            "evaluation": dict(evaluation),
            "force_rmse": overall.get("force_rmse"),
            "atom_error_p99": overall.get("atom_error_p99"),
            "core_rmse": (evaluation.get("core") or {}).get("force_rmse"),
            "diagnostics": dict(diagnostics),
        }

    def run(self, *, only: Sequence[str] | None = None, repetitions: Sequence[int] | None = None) -> dict[str, Any]:
        results: dict[str, Any] = {}
        selected = [v for v in self.cfg.variants() if not only or v.name in only]
        for variant in selected:  # pre-flight: refuse before any training
            self._check_gate(variant)
            self._check_monitor_calibration(variant)
        for variant in selected:
            reps = range(self.cfg.repetitions) if repetitions is None else repetitions
            results[variant.name] = [self.run_one(variant, rep) for rep in reps]
        return results


@dataclass(slots=True)
class _ModelRef:
    checkpoints: list[str]

    @property
    def primary(self) -> str:
        return self.checkpoints[0]


def learning_curves(results: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[list[dict[str, Any]]]]:
    """``{variant: [[point per round] per repetition]}`` from progress records."""

    return {name: [[{k: r.get(k) for k in ("round", "n_labels", "n_valid_labels", "cumulative_cpu_core_s", "force_rmse",
                                             "atom_error_p99", "core_rmse")} for r in rep["rounds"]] for rep in reps]
            for name, reps in results.items()}


def collect_progress(out_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """All progress records found under ``<out_dir>/arms`` (complete or not)."""

    results: dict[str, list[dict[str, Any]]] = {}
    root = out_dir / "arms"
    if not root.is_dir():
        return results
    for variant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        reps = []
        for rep_dir in sorted(variant_dir.glob("rep*")):
            path = rep_dir / "progress.json"
            if path.is_file():
                reps.append({**json.loads(path.read_text()), "path": str(path)})
        if reps:
            results[variant_dir.name] = reps
    return results


def read_gate(loaded: LoadedConfig) -> tuple[bool | None, dict[str, Any] | None]:
    """alpha's fragment-fidelity gate: (True, selected setting) | (False, None) | (None, None) when not decided."""

    cfg = loaded.config.arms.fidelity_gate if loaded.config.arms is not None else FidelityGate()
    directory = loaded.artifacts_root / cfg.directory
    selection, failure = directory / "selection.json", directory / "failure.json"
    if selection.is_file():
        data = json.loads(selection.read_text())
        if data.get("status") in ("selected", "frozen") and data.get("setting"):
            return True, dict(data["setting"])
    if failure.is_file():
        return False, None
    return None, None


def gate_passing_settings(loaded: LoadedConfig) -> frozenset[str]:
    cfg = loaded.config.arms.fidelity_gate if loaded.config.arms is not None else FidelityGate()
    path = loaded.artifacts_root / cfg.directory / "report_calibration.json"
    if not path.is_file():
        return frozenset()
    return frozenset(str(s["setting_id"]) for s in json.loads(path.read_text()).get("settings", []) if s.get("passes"))


def alpha_label_provider(loaded: LoadedConfig, bases: Sequence[Path]) -> CallableLabelProvider:
    """On-demand labels through alpha's budgeted, cache-reusing LabelRunner (CP2K)."""

    if loaded.alpha is None:
        raise ConfigurationError("arms.label_mode=on_demand needs alpha's configuration (run through alpha's CLI or set base_config)")
    from .labels import LabelRequest, LabelRunner, LabelStore, build_service, cp2k_provenance, label_id_for, parent_region
    from .families import load_positions

    alpha = loaded.alpha
    service = build_service(alpha)
    runner = LabelRunner(service, LabelStore(alpha.artifacts_root), alpha, cp2k_provenance(service, smoke_only=alpha.config.cp2k.smoke_only))
    parents = {str(p["parent_id"]): p for p in read_jsonl(loaded.input_path(loaded.config.inputs.parents))}

    def run(request: Any) -> dict[str, Any] | None:
        runner.run([request])
        return LabelStore(alpha.artifacts_root).get(label_id_for(request.region, service.settings_hash))

    def fragment(candidate: PoolCandidate) -> dict[str, Any] | None:
        parent = parents[candidate.parent_id]
        setting = (candidate.region.metadata or {}).get("fragment_setting") or {}
        info = {"policy": setting.get("policy", candidate.region.mode), "setting_id": setting.get("setting_id"), "site_kind": "pool",
                "buffer": setting.get("buffer"), "core_parent_ids": sorted(candidate.core_ids), "n_atoms": candidate.region.n_atoms,
                "n_core": len(candidate.core_ids), "n_caps": int(sum(1 for r in candidate.region.roles if r is RegionRole.H_CAP))}
        return run(LabelRequest("fragment", candidate.parent_id, int(parent["chain_length"]), candidate.region, info,
                                [f"pool:{candidate.candidate_id}"]))

    def full(parent_id: str) -> dict[str, Any] | None:
        parent = parents[parent_id]
        return run(LabelRequest("full", parent_id, int(parent["chain_length"]), parent_region(parent, load_positions(alpha.artifacts_root, parent))))

    return CallableLabelProvider(fragment, full)


def default_context(loaded: LoadedConfig) -> ArmContext:
    """Context backed by the real artifacts and GMD_SGT (CLI path)."""

    from .training import make_backend, predictor_for, train_model

    beta = loaded.config
    parents = read_jsonl(loaded.input_path(beta.inputs.parents))
    labels_path = loaded.artifacts_root / beta.inputs.labels
    labels = read_jsonl(labels_path) if labels_path.is_file() else []
    bases = loaded.bases()
    gate_passed, setting = read_gate(loaded)
    builder = None
    if beta.arms is not None and beta.arms.pool.source == "gate_selection" and setting is not None:
        if loaded.alpha is None:
            raise ConfigurationError("arms.pool.source=gate_selection needs alpha's configuration")
        builder = gate_builder(loaded.alpha, setting)
    if beta.arms is not None and beta.arms.label_mode == "on_demand":
        provider: LabelProvider = alpha_label_provider(loaded, bases)
    else:
        provider = RecordedLabelProvider(labels, bases)
    backend = make_backend(beta.training, root=REPO_ROOT)
    return ArmContext(parents=parents, labels=labels, bases=bases, provider=provider,
                      trainer=lambda dataset, work, seed: train_model(backend, dataset, work, seed=seed, config=beta.training),
                      predictor_factory=lambda model: predictor_for(backend, model), out_dir=loaded.output_dir,
                      config_hash=loaded.config_hash, gate_passed=gate_passed, fragment_builder=builder, gate_setting=setting,
                      gate_passing_settings=gate_passing_settings(loaded))


__all__ = [
    "ARM_KINDS",
    "ArmContext",
    "ArmVariant",
    "ArmsConfig",
    "ArmsRunner",
    "BetaConfig",
    "BudgetConfig",
    "CallableLabelProvider",
    "PoolCandidate",
    "RecordedLabelProvider",
    "acquire",
    "build_pool",
    "collect_progress",
    "default_context",
    "learning_curves",
    "load_beta_config",
    "rank_committee",
    "rank_phal",
    "rank_random",
    "write_json_atomic",
]
