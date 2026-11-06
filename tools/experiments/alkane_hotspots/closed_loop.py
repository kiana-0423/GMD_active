"""The real closed loop of the alkane demonstration on GMD_active's persistent controller.

    GMD sampling of a generated C16 parent in a vacuum-padded periodic box (``md_box``)
    -> online monitoring (part beta's monitor configuration, committee forces from the active model)
    -> events -> extraction (whole molecule, or a gate-validated capped fragment) -> selection
    -> CP2K (isolated, identical settings to part alpha's labels) -> validation -> dataset publication
    -> GMD_SGT fine-tuning -> export validation -> GMD smoke test -> gated promotion
    -> GMD checkpoint restart with the new model.

Nothing of the loop is re-implemented here. ``workflow.assembly.build_controller`` builds the
controller with GMD_active's GMD adapter, GMD_SGT adapter, ``LabelingService`` (CP2K + local
scheduler), ``VersionedDatasetStore``, ``ModelRegistry``, promotion gates and rollback. This module
supplies the alkane-specific parts by composition, each implementing a ``workflow.protocols``
interface around the existing component:

* :class:`AlkaneRegionExtractor` (``RegionExtractor``): the event's trigger frame is unwrapped along the
  bonds into an isolated molecule; ``full`` mode labels the whole chain, ``local`` mode cuts alpha's
  fragment (``fidelity.build_fragment``, existing extraction + H capping) with the setting frozen by the
  fragment-fidelity gate around each hotspot group. Reference calculations stay isolated.
* :class:`BudgetedLabeler` (``Labeler``): refuses a new CP2K job beyond the YAML CPU-hour budget or atom
  limits (measured cost of finished closed-loop tasks + the measured mean of alpha's labels as estimate).
* :class:`HeldOutValidationStore` (``DatasetStore``): the training side is the cumulative versioned store;
  the validation side is always the fixed ``validation`` split (full DFT labels), so early stopping and
  the promotion gates see the validation split only. final_eval is never read.
* :class:`RecordingTrainer` (``Trainer``): delegates to ``GMDSGTBackend`` and records the evaluation,
  export check and engine smoke results so every gate decision can be re-derived in the report.
* ``md_box.ImageGuard`` (``FrameEnricher``): checks every monitored frame for periodic-image interaction.

Labeling mode (``labeling.mode``): ``local`` needs a passed fidelity gate (``fidelity/selection.json``);
it is refused otherwise. ``auto`` uses local labels only with a passed gate and falls back to full-system
labels when the gate failed (``fidelity/failure.json``) or is undecided. Unvalidated local labels are
never produced.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gmd_active_learning.contracts import (
    DatasetVersion,
    EventRecord,
    ExtractedRegion,
    Frame,
    LabelResult,
    LabelTask,
    ModelArtifact,
    RegionRole,
    content_hash,
    stable_id,
)
from gmd_active_learning.core.exceptions import ConfigurationError, DataContractError, LabelingError, WorkflowError
from gmd_active_learning.utils.atomic_io import write_json_atomic

from .config import SPLITS, LoadedConfig, load_config
from .geometry import AlkaneTopology

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_KINDS = ("main", "recovery")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ============================================================================================ config


class ParentSelector(_Strict):
    parent_id: str | None = Field(None, description="explicit parent; otherwise the index-th parent of split/family (file order)")
    split: str = "initial_train"
    family: str = "normal"
    index: int = Field(0, ge=0)


class MDSection(_Strict):
    executable: str = Field("../GMD/build-torch/gmd", description="GMD Torch build (relative to the GMD_active root)")
    parent: ParentSelector = Field(default_factory=ParentSelector)
    allowed_splits: list[str] = Field(default_factory=lambda: ["initial_train"],
                                      description="splits the MD parent may come from (validation/final_eval are always refused)")
    directives: dict[str, Any] = Field(default_factory=lambda: {"time_step": 0.5, "velocity": 300.0, "velocity_init": "random",
                                                                "velocity_seed": 7, "thermostat": "nose_hoover", "thermostat_tau": 100.0,
                                                                "remove_com_velocity": "true"})
    frame_interval: int = Field(10, ge=1)
    control_poll_interval: int = Field(1, ge=1)
    segment_timeout_s: float = Field(3600.0, gt=0.0)
    env: dict[str, str] = Field(default_factory=lambda: {"OMP_NUM_THREADS": "1"})
    smoke_steps: int = Field(5, ge=1)

    @field_validator("allowed_splits")
    @classmethod
    def _splits(cls, value: list[str]) -> list[str]:
        forbidden = sorted(set(value) & {"validation", "final_eval"})
        if forbidden:
            raise ValueError(f"md.allowed_splits must not contain {forbidden}: sampled configurations enter training")
        unknown = sorted(set(value) - set(SPLITS))
        if unknown:
            raise ValueError(f"md.allowed_splits: unknown splits {unknown}")
        return value


class LoopSection(_Strict):
    run_id: str = "alkane-closed-loop"
    cycles: int = Field(2, ge=1)
    segment_steps: int = Field(2000, ge=1)
    segments_per_cycle: int = Field(1, ge=1)
    labels_per_cycle: int = Field(2, ge=1)
    max_per_event: int = Field(1, ge=1)
    diversity_threshold: float = Field(0.0, ge=0.0)
    min_score: float | None = None
    min_new_samples: int = Field(1, ge=1)
    max_pending_tasks: int = Field(8, ge=1)
    poll_interval_s: float = Field(10.0, ge=0.0)
    max_poll_wait_s: float = Field(600.0, ge=0.0, description="the controller returns WAITING after this; closed-loop-run resumes it")


class BudgetSection(_Strict):
    max_cpu_core_hours: float = Field(..., gt=0.0, description="measured CPU core-hours of closed-loop CP2K labels (DFT only)")
    max_atoms_full: int | None = Field(None, ge=1, description="null = alpha budgets.max_atoms_full")
    max_atoms_fragment: int | None = Field(None, ge=1, description="null = alpha budgets.max_atoms_per_fragment")


class LabelingSection(_Strict):
    mode: Literal["auto", "full", "local"] = "auto"
    fidelity_dir: str = "fidelity"


class InitialModelSection(_Strict):
    train_splits: list[str] = Field(default_factory=lambda: ["initial_train"])
    validation_splits: list[str] = Field(default_factory=lambda: ["validation"])
    train_parents: list[str] | None = Field(None, description="smoke only: explicit parents instead of whole splits")
    validation_parents: list[str] | None = Field(None, description="smoke only: explicit parents instead of whole splits")
    seed: int = 11
    train_overrides: dict[str, Any] = Field(default_factory=dict, description="overrides of the study's training.train for the initial "
                                            "model only (empty in the pilot: the study's training budget)")

    @model_validator(mode="after")
    def _no_final_eval(self) -> InitialModelSection:
        if "final_eval" in self.train_splits or "final_eval" in self.validation_splits:
            raise ValueError("final_eval is never used for training or model selection")
        return self


class LabelGeometrySection(_Strict):
    """Physical sanity of an event geometry before any CP2K job (a collapsed MD frame is not labeled)."""

    require_parent_topology: bool = Field(True, description="the bond graph (utils.bonds.infer_bonds) must equal the alkane's")
    covalent_scale: float = Field(1.2, gt=0.0)
    min_distance: float = Field(0.7, gt=0.0, description="A; any interatomic distance below rejects the event")


class FinetuneSection(_Strict):
    train: dict[str, Any] = Field(default_factory=dict, description="overrides of the study's training.train for fine-tuning")
    seed: int = 101


class FrozenEvalSection(_Strict):
    seeds: list[int] = Field(default_factory=lambda: [1, 2])
    steps: int = Field(2000, ge=1)
    frame_interval: int = Field(10, ge=1)
    discard_steps: int = Field(200, ge=0, description="equilibration frames excluded from the observables")
    dihedral_bins: int = Field(36, ge=4)
    gauche_window_deg: float = Field(120.0, gt=0.0, lt=180.0, description="|phi| below this counts as gauche (trans = 180)")


class RecoverySection(_Strict):
    run_id: str = "alkane-recovery"
    cycles: int = Field(1, ge=1)
    labels_per_cycle: int = Field(1, ge=1)
    promotion: dict[str, Any] | None = Field(None, description="gates of the recovery run (null = the loop's promotion gates)")
    strict_promotion: dict[str, Any] = Field(..., description="gate config audited on the recovery run's promoted model (real metrics)")
    kill_after: Literal["submitted"] = "submitted"
    kill_timeout_s: float = Field(3600.0, gt=0.0, description="give up waiting for a submission after this")
    restart_steps: int = Field(200, ge=1)
    cancellation_check: bool = True
    cancellation_wait_s: float = Field(120.0, gt=0.0)


class ClosedLoopConfig(_Strict):
    name: str
    description: str = ""
    base_config: str = Field(..., description="alpha's demo configuration (relative to this file)")
    study: str = Field(..., description="part beta's study configuration (relative to this file): monitor + training")
    output_subdir: str = "closed_loop"
    exclusive_cp2k: bool = Field(True, description="refuse to start CP2K work while a CP2K process outside this run is running")
    md: MDSection = Field(default_factory=MDSection)
    md_box: dict[str, Any] = Field(default_factory=dict)
    loop: LoopSection = Field(default_factory=LoopSection)
    budget: BudgetSection
    labeling: LabelingSection = Field(default_factory=LabelingSection)
    label_geometry: LabelGeometrySection = Field(default_factory=LabelGeometrySection)
    initial_model: InitialModelSection = Field(default_factory=InitialModelSection)
    finetune: FinetuneSection = Field(default_factory=FinetuneSection)
    promotion: dict[str, Any] = Field(default_factory=dict, description="workflow.promotion gates (validation split)")
    monitor_overrides: dict[str, Any] = Field(default_factory=dict)
    events: dict[str, Any] = Field(default_factory=lambda: {"pre_trigger_frames": 1, "post_trigger_frames": 1, "cooldown_steps": 200})
    hotspots: dict[str, Any] | None = Field(None, description="null = the study's arms.hotspots")
    frozen_eval: FrozenEvalSection = Field(default_factory=FrozenEvalSection)
    recovery: RecoverySection | None = None


@dataclass
class LoadedClosedLoop:
    config: ClosedLoopConfig
    path: Path
    alpha: LoadedConfig
    beta: Any  # arms.LoadedConfig
    raw: dict[str, Any]

    @property
    def out_dir(self) -> Path:
        return self.alpha.artifacts_root / self.config.output_subdir / self.config.name

    @property
    def config_hash(self) -> str:
        return content_hash({"closed_loop": self.raw, "alpha": self.alpha.config.config_hash(), "beta": self.beta.config_hash}, length=16)

    def run_dir(self, kind: str) -> Path:
        if kind not in RUN_KINDS:
            raise ConfigurationError(f"unknown run kind {kind!r}")
        return self.out_dir / ("run" if kind == "main" else "recovery/run")


def load_closed_loop(path: str | Path, *, alpha: LoadedConfig | None = None) -> LoadedClosedLoop:
    """Validate the closed-loop YAML; ``alpha`` (from ``--config``) must be the YAML's ``base_config``."""

    from .arms import load_beta_config

    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = ClosedLoopConfig.model_validate(raw)
    base = load_config(path.parent / config.base_config)
    if alpha is None:
        alpha = base
    elif alpha.config.config_hash() != base.config.config_hash():  # an --artifacts-root override is allowed
        raise ConfigurationError(f"--config {alpha.path} is not the closed-loop base_config {config.base_config} (different configuration hash)")
    beta = load_beta_config(path.parent / config.study, alpha=alpha)
    if beta.config.training.model.get("long_range_type", "none") != "none":
        raise ConfigurationError("the closed loop needs a local model (training.model.long_range_type: none)")
    return LoadedClosedLoop(config, path, alpha, beta, raw)


# ============================================================================================ labeling mode


@dataclass(frozen=True)
class LabelingDecision:
    mode: Literal["full", "local"]
    requested: str
    gate_status: Literal["passed", "failed", "undecided"]
    reason: str
    setting: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "requested": self.requested, "gate_status": self.gate_status, "reason": self.reason,
                "setting": self.setting}


def decide_labeling_mode(fidelity_dir: str | Path, requested: str) -> LabelingDecision:
    """Local fragment labels only with a passed fidelity gate; otherwise full-system labels or a refusal."""

    directory = Path(fidelity_dir)
    selection_path, failure_path = directory / "selection.json", directory / "failure.json"
    setting = None
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("status") in ("selected", "frozen") and selection.get("setting"):
            setting = dict(selection["setting"])
    if setting is not None and failure_path.is_file():
        raise ConfigurationError(f"{directory}: both selection.json and failure.json exist; the fidelity gate state is inconsistent")
    status: Literal["passed", "failed", "undecided"] = "passed" if setting is not None else ("failed" if failure_path.is_file() else "undecided")
    if requested == "full":
        return LabelingDecision("full", requested, status, "full-system labels requested", None)
    if requested == "local":
        if status != "passed":
            raise ConfigurationError(
                f"labeling.mode=local refused: the fragment-fidelity gate is {status} ({directory}); local labels are used only "
                "with a frozen, passed setting (fidelity/selection.json)")
        return LabelingDecision("local", requested, status, f"fidelity gate passed; frozen setting {setting['setting_id']}", setting)
    if requested != "auto":
        raise ConfigurationError(f"unknown labeling.mode {requested!r}")
    if status == "passed":
        return LabelingDecision("local", requested, status, f"fidelity gate passed; frozen setting {setting['setting_id']}", setting)  # type: ignore[index]
    reason = ("fidelity gate failed (failure.json): local labels are not validated" if status == "failed"
              else "fidelity gate undecided (no selection.json): local labels are not validated")
    return LabelingDecision("full", requested, status, reason + "; full-system labels are used", None)


# ============================================================================================ region extraction


def sorted_frame_positions(frame: Frame, topology: AlkaneTopology) -> np.ndarray:
    """Positions in parent atom order (atom_id = parent index); species are checked."""

    order = np.argsort(np.asarray(frame.atom_ids))
    ids = np.asarray(frame.atom_ids)[order]
    if not np.array_equal(ids, np.arange(topology.n_atoms)):
        raise DataContractError(f"frame {frame.step}: atom ids are not 0..{topology.n_atoms - 1}")
    numbers = np.asarray(frame.numbers)[order]
    if not np.array_equal(numbers, topology.numbers):
        raise DataContractError(f"frame {frame.step}: species do not match the C{topology.n_carbons} topology")
    return np.asarray(frame.positions, dtype=float)[order]


def isolated_frame(frame: Frame, topology: AlkaneTopology) -> Frame:
    """The MD frame's molecule made whole along its bonds and centred, without cell (isolated)."""

    from .md_box import orthorhombic_lengths, unwrap_molecule

    positions = sorted_frame_positions(frame, topology)
    whole = unwrap_molecule(positions, orthorhombic_lengths(frame.cell), topology.bonds)
    whole = whole - whole.mean(axis=0)
    return Frame(run_id=frame.run_id, step=frame.step, atom_ids=np.arange(topology.n_atoms), numbers=topology.numbers, positions=whole,
                 cell=None, pbc=(False, False, False), model_version=frame.model_version)


def hotspot_site(topology: AlkaneTopology, group: Sequence[int], scores: np.ndarray | None) -> tuple[int, list[int]]:
    """The fidelity gate's site shape around a hotspot group: carbons k-1..k+2 (+H), k = the group's top carbon."""

    from .families import torsion_site

    group = [int(atom) for atom in group]
    if scores is not None:
        top = max(group, key=lambda atom: (float(scores[atom]), -atom))
    else:
        top = group[0]
    carbon = top if top < topology.n_carbons else topology.carbon_of[top]
    k = min(max(carbon, 1), topology.n_carbons - 3)
    return k, torsion_site(topology, k)


class AlkaneRegionExtractor:
    """``RegionExtractor``: event -> isolated whole-molecule region (full) or gate-validated fragments (local)."""

    def __init__(self, *, decision: LabelingDecision, alpha_config: Any, topology: AlkaneTopology, force_weights: Any,
                 full_force_weight: float, merge_radius: float, parent_id: str, geometry: LabelGeometrySection | None = None,
                 rejected_path: str | Path | None = None) -> None:
        from gmd_active_learning.workflow.assembly import hotspot_grouper

        if decision.mode == "local" and decision.setting is None:
            raise ConfigurationError("local extraction needs the frozen fidelity setting")
        self.decision = decision
        self.alpha_config = alpha_config
        self.topology = topology
        self.force_weights = force_weights
        self.full_force_weight = float(full_force_weight)
        self.grouper = hotspot_grouper(merge_radius)
        self.parent_id = parent_id
        self.geometry = geometry or LabelGeometrySection()
        self.rejected_path = None if rejected_path is None else Path(rejected_path)

    def check_geometry(self, frame: Frame) -> list[str]:
        """Reasons why the isolated event geometry must not be labeled (empty = fine)."""

        from gmd_active_learning.utils.bonds import infer_bonds

        reasons = []
        positions = np.asarray(frame.positions, dtype=float)
        distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
        np.fill_diagonal(distances, np.inf)
        if float(distances.min()) < self.geometry.min_distance:
            reasons.append(f"min interatomic distance {float(distances.min()):.3f} A < label_geometry.min_distance {self.geometry.min_distance} A")
        if self.geometry.require_parent_topology:
            bonds = sorted(tuple(sorted(pair)) for pair in infer_bonds(positions, frame.numbers, scale=self.geometry.covalent_scale))
            if bonds != list(self.topology.bonds):
                missing = sorted(set(self.topology.bonds) - set(bonds))
                extra = sorted(set(bonds) - set(self.topology.bonds))
                reasons.append(f"bond topology changed (missing {missing[:5]}, extra {extra[:5]})")
        return reasons

    def extract(self, event: EventRecord) -> list[tuple[ExtractedRegion, float]]:
        from gmd_active_learning.extraction.dispatch import EventRegionExtractor

        trigger = event.trigger_frame
        frame = isolated_frame(trigger, self.topology)
        reasons = self.check_geometry(frame)
        if reasons:
            if self.rejected_path is not None:
                self.rejected_path.parent.mkdir(parents=True, exist_ok=True)
                with self.rejected_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"event_id": event.event_id, "trigger_step": int(trigger.step), "reasons": reasons}) + "\n")
            raise DataContractError(f"event {event.event_id} (step {trigger.step}) is not labeled: {'; '.join(reasons)}")
        score = float(event.max_score)
        model_version = trigger.model_version or event.model_version
        provenance = {"md_parent_id": self.parent_id, "event_id": event.event_id, "trigger_step": int(trigger.step),
                      "labeling_mode": self.decision.mode}
        if self.decision.mode == "full":
            n = self.topology.n_atoms
            region = ExtractedRegion(
                region_id=stable_id("alkane-cl-full", event.event_id, int(trigger.step), content_hash(np.round(frame.positions, 8))),
                mode="full_system", run_id=trigger.run_id, parent_step=int(trigger.step), event_id=event.event_id,
                numbers=self.topology.numbers, positions=frame.positions, cell=None, pbc=(False, False, False),
                parent_atom_ids=np.arange(n), image_shifts=np.zeros((n, 3), dtype=np.int64), roles=[RegionRole.CORE] * n,
                force_weights=np.full(n, self.full_force_weight), frozen=np.zeros(n, dtype=bool),
                hotspot_atom_ids=[int(a) for a in event.hotspot_atom_ids], charge=0, multiplicity=1, model_version=model_version,
                metadata={**provenance, "kind": "full"})
            return [(region, score)]
        from gmd_active_learning.datasets.masks import reweight

        from .fidelity import build_fragment

        ids, scores = EventRegionExtractor.trigger_hotspots(event)
        groups = self.grouper(trigger, ids, scores)
        index_scores = None
        if scores is not None:
            index_scores = np.zeros(self.topology.n_atoms)
            index_scores[np.asarray(trigger.atom_ids, dtype=int)] = scores
        out: list[tuple[ExtractedRegion, float]] = []
        seen: set[int] = set()
        for group in groups:
            k, site = hotspot_site(self.topology, group, index_scores)
            if k in seen:
                continue
            seen.add(k)
            region, core = build_fragment(frame, site, self.decision.setting, self.alpha_config)  # type: ignore[arg-type]
            region = reweight(region, self.force_weights)
            data = region.to_dict()
            data.update(region_id=stable_id("alkane-cl-frag", event.event_id, int(trigger.step), self.decision.setting["setting_id"], k),  # type: ignore[index]
                        run_id=trigger.run_id, parent_step=int(trigger.step), event_id=event.event_id, model_version=model_version,
                        hotspot_atom_ids=[int(a) for a in group])
            data["metadata"] = {**data["metadata"], **provenance, "kind": "fragment", "site_carbon": k, "core_parent_ids": core}
            out.append((ExtractedRegion.from_dict(data), score))
        return out


# ============================================================================================ budgeted labeling


class BudgetExceededError(LabelingError):
    """A new CP2K job would exceed the closed loop's CPU-hour budget or atom limits."""


class BudgetedLabeler:
    """``Labeler`` around ``ServiceLabeler``: checks the measured CPU budget before each new CP2K job."""

    def __init__(self, inner: Any, *, max_core_s: float, cores: int, max_atoms: Mapping[str, int],
                 estimate: Callable[[str], tuple[float, str]], ledger_path: str | Path) -> None:
        self.inner = inner
        self.service = inner.service
        self.max_core_s = float(max_core_s)
        self.cores = int(cores)
        self.max_atoms = dict(max_atoms)
        self.estimate = estimate
        self.ledger_path = Path(ledger_path)

    @staticmethod
    def kind_of(region: ExtractedRegion) -> str:
        return "full" if region.mode == "full_system" else "fragment"

    def committed(self) -> dict[str, Any]:
        """Measured core-seconds of finished closed-loop tasks plus estimates of running ones."""

        from gmd_active_learning.labeling.task_store import load_task

        from .cost import measured_cost

        spent, inflight, tasks = 0.0, 0.0, []
        root = Path(self.service.task_root)
        for task_file in sorted(root.glob("*/task.json")) if root.is_dir() else []:
            task = load_task(task_file.parent)
            attempts = task.metadata.get("attempts") or []
            if not attempts:
                continue
            region = ExtractedRegion.from_dict(json.loads((task_file.parent / "region.json").read_text()))
            kind = self.kind_of(region)
            if task.metadata.get("stage") == "done":
                cost = measured_cost(task.work_dir, attempts, self.cores)
                spent += cost["cpu_core_s"]
                tasks.append({"task_id": task.task_id, "kind": kind, "state": "done", "cpu_core_s": cost["cpu_core_s"]})
            else:
                value, source = self.estimate(kind)
                inflight += value
                tasks.append({"task_id": task.task_id, "kind": kind, "state": "running", "estimate_core_s": value, "source": source})
        return {"spent_core_s": spent, "inflight_core_s": inflight, "tasks": tasks}

    def check(self, region: ExtractedRegion) -> dict[str, Any]:
        kind = self.kind_of(region)
        limit = self.max_atoms[kind]
        if region.n_atoms > limit:
            raise BudgetExceededError(f"{kind} region of {region.n_atoms} atoms exceeds the closed-loop limit {limit}")
        state = self.committed()
        estimate, source = self.estimate(kind)
        projected = state["spent_core_s"] + state["inflight_core_s"] + estimate
        decision = {"kind": kind, "n_atoms": region.n_atoms, "estimate_core_s": estimate, "estimate_source": source,
                    "spent_core_s": state["spent_core_s"], "inflight_core_s": state["inflight_core_s"], "projected_core_s": projected,
                    "limit_core_s": self.max_core_s, "allowed": projected <= self.max_core_s}
        ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.is_file() else []
        ledger.append(decision)
        write_json_atomic(self.ledger_path, ledger)
        if not decision["allowed"]:
            raise BudgetExceededError(f"projected {projected / 3600:.3f} core-h > budget.max_cpu_core_hours={self.max_core_s / 3600:.3f} "
                                      f"(spent {state['spent_core_s'] / 3600:.3f}, in flight {state['inflight_core_s'] / 3600:.3f}, "
                                      f"this job {estimate / 3600:.3f} [{source}])")
        return decision

    # -- Labeler protocol -------------------------------------------------------------------------------
    def prepare(self, region: ExtractedRegion) -> LabelTask:
        return self.inner.prepare(region)

    def find_submission(self, task: LabelTask) -> LabelTask | None:
        return self.inner.find_submission(task)

    def submit(self, task: LabelTask) -> LabelTask:
        region = ExtractedRegion.from_dict(json.loads((Path(task.work_dir) / "region.json").read_text()))
        self.check(region)
        return self.inner.submit(task)

    def poll(self, task: LabelTask) -> LabelTask:
        return self.inner.poll(task)

    def reconcile(self, task: LabelTask) -> LabelTask:
        return self.inner.reconcile(task)

    def collect(self, task: LabelTask) -> LabelResult:
        return self.inner.collect(task)

    def retry(self, task: LabelTask, result: LabelResult | None) -> LabelTask | None:
        return self.inner.retry(task, result)

    def cancel(self, task: LabelTask) -> LabelTask:
        return self.inner.cancel(task)


# ============================================================================================ datasets


class HeldOutValidationStore:
    """``DatasetStore``: cumulative training versions + one fixed, verified validation version.

    The wrapped store publishes with its split disabled (every sample is training data). Each returned
    ``DatasetVersion`` points ``val_file`` at the fixed validation version (the ``validation`` split's
    full DFT labels), whose checksum is re-verified every time, so early stopping and the promotion
    gates are always evaluated on the validation split.
    """

    def __init__(self, inner: Any, validation_store: Any, validation: DatasetVersion) -> None:
        if inner.config.split.enabled:
            raise ConfigurationError("the training store must publish with datasets.split.enabled: false (validation is held out)")
        self.inner = inner
        self.validation_store = validation_store
        self.validation = validation
        self.config = inner.config

    def _patch(self, version: DatasetVersion) -> DatasetVersion:
        self.validation_store.verify(self.validation)
        overlap = set(version.sample_ids) & set(self.validation.sample_ids)
        if overlap:
            raise DataContractError(f"dataset {version.version} contains validation samples {sorted(overlap)[:3]}")
        return replace(version, n_train=version.n_structures, n_val=self.validation.n_structures, metadata={
            **version.metadata, "val_file": self.validation.metadata["all_file"], "validation_version": self.validation.version,
            "validation_sha256": self.validation.sha256, "validation_policy": "held-out validation split (fixed, verified)"})

    def add_sample(self, region: ExtractedRegion, label: LabelResult, *, provenance: dict[str, Any]) -> tuple[str, Path]:
        return self.inner.add_sample(region, label, provenance=provenance)

    def publish(self, sample_paths: list[Path], *, parent: DatasetVersion | None, split_groups: dict[str, str]) -> DatasetVersion:
        return self._patch(self.inner.publish(sample_paths, parent=parent, split_groups=split_groups))

    def publish_samples(self, samples: list[Any], *, parent: DatasetVersion | None = None) -> DatasetVersion:
        return self._patch(self.inner.publish_samples(samples, parent=parent))

    def get(self, version: str) -> DatasetVersion:
        return self._patch(self.inner.get(version))

    def read_samples(self, version: str | DatasetVersion) -> list[Any]:
        return self.inner.read_samples(version if isinstance(version, str) else version.version)


# ============================================================================================ trainer recording


class RecordingTrainer:
    """``Trainer`` around ``GMDSGTBackend`` that appends every evaluation/export check/smoke test to a JSONL history."""

    def __init__(self, inner: Any, history_path: str | Path) -> None:
        self.inner = inner
        self.history_path = Path(history_path)

    def _record(self, kind: str, **payload: Any) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": kind, "at": time.time(), **payload}, default=str) + "\n")

    def train(self, dataset: DatasetVersion, *, parent: ModelArtifact | None, work_dir: Path) -> Any:
        started = time.monotonic()
        output = self.inner.train(dataset, parent=parent, work_dir=work_dir)
        self._record("train", dataset=dataset.version, parent=None if parent is None else parent.model_id, work_dir=str(work_dir),
                     checkpoint=str(output.checkpoint), mode=output.mode, wall_s=time.monotonic() - started,
                     best_val=output.summary.get("best_val"))
        return output

    def evaluate(self, checkpoint: Path, dataset: DatasetVersion) -> dict[str, float]:
        metrics = self.inner.evaluate(checkpoint, dataset)
        self._record("evaluate", checkpoint=str(checkpoint), dataset=dataset.version, validation=dataset.metadata.get("validation_version"),
                     metrics=metrics)
        return metrics

    def export(self, checkpoint: Path, work_dir: Path) -> Path:
        return self.inner.export(checkpoint, work_dir)

    def validate_export(self, checkpoint: Path, export: Path, dataset: DatasetVersion) -> dict[str, Any]:
        result = self.inner.validate_export(checkpoint, export, dataset)
        self._record("validate_export", checkpoint=str(checkpoint), export=str(export), result=result)
        return result

    def engine_smoke_test(self, export: Path, work_dir: Path) -> dict[str, Any]:
        result = self.inner.engine_smoke_test(export, work_dir)
        self._record("engine_smoke", export=str(export), result=result)
        return result

    def history(self) -> list[dict[str, Any]]:
        if not self.history_path.is_file():
            return []
        return [json.loads(line) for line in self.history_path.read_text().splitlines() if line.strip()]


# ============================================================================================ CP2K exclusivity


def foreign_cp2k_processes(own_roots: Sequence[Path], executable: str = "cp2k") -> list[dict[str, Any]]:
    """Running CP2K processes whose working directory is not below one of ``own_roots``."""

    listing = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=True).stdout
    roots = [str(Path(root).resolve()) for root in own_roots]
    foreign = []
    for line in listing.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[1].split():
            continue
        if executable not in Path(parts[1].split()[0]).name:
            continue
        pid = int(parts[0])
        cwd = None
        lsof = subprocess.run(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"], capture_output=True, text=True)
        for entry in lsof.stdout.splitlines():
            if entry.startswith("n"):
                cwd = entry[1:]
        if cwd is not None and any(cwd == root or cwd.startswith(root + os.sep) for root in roots):
            continue
        foreign.append({"pid": pid, "command": parts[1][:200], "cwd": cwd})
    return foreign


def require_exclusive_cp2k(cl: LoadedClosedLoop, own_roots: Sequence[Path]) -> None:
    if not cl.config.exclusive_cp2k:
        return
    foreign = foreign_cp2k_processes(own_roots)
    if foreign:
        raise WorkflowError(f"exclusive_cp2k: {len(foreign)} other CP2K process(es) are running ({foreign}); one CP2K job at a time. "
                            "Re-run when they have finished.")


# ============================================================================================ assembly


@dataclass
class Assembled:
    controller: Any
    decision: LabelingDecision
    parent: dict[str, Any]
    topology: AlkaneTopology
    box: Any
    receptive_field: Any
    guard: Any
    trainer: RecordingTrainer
    labeler: BudgetedLabeler
    validation: DatasetVersion
    work_dir: Path
    run_id: str
    gmd_config: Any
    service_settings_hash: str


def select_md_parent(cl: LoadedClosedLoop, parents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    md = cl.config.md
    if md.parent.parent_id is not None:
        matches = [p for p in parents if p["parent_id"] == md.parent.parent_id]
    else:
        matches = [p for p in parents if p["split"] == md.parent.split and p["family"] == md.parent.family]
        matches = matches[md.parent.index : md.parent.index + 1]
    if not matches:
        raise ConfigurationError(f"md.parent {md.parent.model_dump()} matches no parent in parents.jsonl")
    parent = dict(matches[0])
    if parent["split"] not in md.allowed_splits:
        raise ConfigurationError(f"md parent {parent['parent_id']} is in split {parent['split']!r}, not in md.allowed_splits {md.allowed_splits}")
    return parent


def _parents_for(cl: LoadedClosedLoop, parents: Sequence[Mapping[str, Any]], splits: Sequence[str], explicit: Sequence[str] | None) -> list[dict[str, Any]]:
    if explicit is not None:
        if not cl.alpha.config.cp2k.smoke_only:
            raise ConfigurationError("initial_model.train_parents/validation_parents are allowed for smoke-only configurations only")
        chosen = [dict(p) for p in parents if p["parent_id"] in set(explicit)]
        missing = sorted(set(explicit) - {p["parent_id"] for p in chosen})
        if missing:
            raise ConfigurationError(f"initial_model parents {missing} are not in parents.jsonl")
        return chosen
    return [dict(p) for p in parents if p["split"] in splits]


class MissingInputsError(WorkflowError):
    """Required full DFT labels do not exist yet."""


def initial_samples(cl: LoadedClosedLoop, settings_hash: str) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """beta's full-system samples for the initial-model train and validation parents (valid labels only)."""

    from .families import load_parents
    from .labels import LabelStore
    from .training import full_sample

    root = cl.alpha.artifacts_root
    parents = load_parents(root)
    init = cl.config.initial_model
    train_parents = _parents_for(cl, parents, init.train_splits, init.train_parents)
    val_parents = _parents_for(cl, parents, init.validation_splits, init.validation_parents)
    if any(p["split"] == "final_eval" for p in train_parents + val_parents):
        raise ConfigurationError("final_eval parents are never used")
    overlap = {p["split_group"] for p in train_parents} & {p["split_group"] for p in val_parents}
    if overlap and not cl.alpha.config.cp2k.smoke_only:
        raise ConfigurationError(f"train and validation parents share split groups {sorted(overlap)}")
    labels = [r for r in LabelStore(root).all() if r.get("kind") == "full" and r.get("status") == "valid"]
    by_parent: dict[str, dict[str, Any]] = {}
    for record in labels:
        if (record.get("cp2k") or {}).get("settings_hash") != settings_hash:
            continue
        by_parent.setdefault(str(record["parent_id"]), record)
    missing = [p["parent_id"] for p in train_parents + val_parents if p["parent_id"] not in by_parent]
    if missing or not train_parents or not val_parents:
        raise MissingInputsError(
            f"initial model needs valid full-system labels (settings {settings_hash}) for {len(train_parents)} train and "
            f"{len(val_parents)} validation parents; missing: {missing}. Produce them first, e.g. `python -m tools.experiments.alkane_hotspots "
            f"label-full --config {cl.alpha.path} --splits {' '.join(init.train_splits + init.validation_splits)}`")
    training = cl.beta.config.training
    train = [full_sample(by_parent[p["parent_id"]], p, training, root) for p in train_parents]
    val = [full_sample(by_parent[p["parent_id"]], p, training, root) for p in val_parents]
    record = {"train": [{"parent_id": p["parent_id"], "label_id": by_parent[p["parent_id"]]["label_id"], "split": p["split"],
                         "split_group": p["split_group"], "sample_id": s.sample_id} for p, s in zip(train_parents, train)],
              "validation": [{"parent_id": p["parent_id"], "label_id": by_parent[p["parent_id"]]["label_id"], "split": p["split"],
                              "split_group": p["split_group"], "sample_id": s.sample_id} for p, s in zip(val_parents, val)]}
    return train, val, record


def _resolve(path: str) -> str:
    candidate = Path(path).expanduser()
    return str(candidate if candidate.is_absolute() else (REPO_ROOT / candidate).resolve())


def gmd_active_raw(cl: LoadedClosedLoop, *, kind: str, initial_structure: Path, task_dir: Path) -> dict[str, Any]:
    """The GMD_active configuration mapping for ``build_config`` / ``build_controller``."""

    from gmd_active_learning.monitors.config import load_monitor_config

    from .ground_truth import deep_merge
    from .labels import cp2k_raw_config

    config = cl.config
    beta = cl.beta.config
    loop = config.loop
    promotion = dict(config.promotion)
    run_id, cycles, labels_per_cycle = loop.run_id, loop.cycles, loop.labels_per_cycle
    if kind == "recovery":
        if config.recovery is None:
            raise ConfigurationError("the closed-loop YAML has no recovery section")
        run_id, cycles, labels_per_cycle = config.recovery.run_id, config.recovery.cycles, config.recovery.labels_per_cycle
        if config.recovery.promotion is not None:
            promotion = dict(config.recovery.promotion)
    monitor = load_monitor_config(deep_merge(beta.monitor, config.monitor_overrides)).model_dump(mode="json")
    hotspots = config.hotspots if config.hotspots is not None else (dict(beta.arms.hotspots) if beta.arms is not None else {})
    train_section = beta.training.sgt_train_section()
    train_section["train"] = {**train_section["train"], **config.finetune.train}
    cp2k = dict(cp2k_raw_config(cl.alpha)["cp2k"])
    cp2k["task_dir"] = str(task_dir)
    source = beta.training.sgt.source_path
    return {
        "workflow": {
            "run_id": run_id, "max_cycles": cycles, "segment_steps": loop.segment_steps, "segments_per_cycle": loop.segments_per_cycle,
            "max_pending_tasks": loop.max_pending_tasks, "poll_interval_s": loop.poll_interval_s, "max_poll_wait_s": loop.max_poll_wait_s,
            "save_event_frames": True,
            "candidates": {"max_per_cycle": labels_per_cycle, "max_per_event": loop.max_per_event,
                           "diversity_threshold": loop.diversity_threshold, "min_score": loop.min_score},
            "retraining": {"min_new_samples": loop.min_new_samples, "train_on_first_samples": True, "wait_for_pending_labels": True},
            "promotion": promotion,
        },
        "md": {"executable": _resolve(config.md.executable), "initial_structure": str(initial_structure),
               "directives": dict(config.md.directives), "frame_interval": config.md.frame_interval,
               "control_poll_interval": config.md.control_poll_interval, "segment_timeout_s": config.md.segment_timeout_s,
               "env": dict(config.md.env), "smoke_steps": config.md.smoke_steps},
        "mlip": {"source_path": None if source is None else _resolve(source), "device": beta.training.sgt.device, "train": train_section,
                 "finetune_from_active": True, "seed": config.finetune.seed, "committee_size": beta.training.sgt.committee_size},
        "monitor": monitor,
        "events": dict(config.events),
        "hotspots": dict(hotspots),
        "datasets": {"split": {"enabled": False},
                     "energy": {"nonperiodic_weight": float(beta.training.energy.full_system_weight), "periodic_weight": 0.0,
                                "allow_capped": False}},
        "backends": {"dft": {"engine": "cp2k"}, "scheduler": {"engine": "local"}, "discover_entry_points": False},
        "scheduler": {},
        "cp2k": cp2k,
    }


def _estimator(cl: LoadedClosedLoop, chain_length: int) -> Callable[[str], tuple[float, str]]:
    from .cost import BudgetLedger
    from .labels import LabelStore

    alpha = cl.alpha

    def estimate(kind: str) -> tuple[float, str]:
        ledger = BudgetLedger(LabelStore(alpha.artifacts_root).all(), alpha.config.budgets, alpha.config.cp2k.cores)
        return ledger.estimate(kind, chain_length)

    return estimate


def assemble(cl: LoadedClosedLoop, kind: str = "main") -> Assembled:
    """Build the controller (``build_controller``) and attach the alkane components; idempotent."""

    from gmd_active_learning.config import build_config
    from gmd_active_learning.datasets.config import DatasetsConfig
    from gmd_active_learning.datasets.store import VersionedDatasetStore
    from gmd_active_learning.workflow.assembly import build_controller

    from .families import load_parents, load_positions
    from .labels import build_service
    from .md_box import ImageGuard, MDBoxConfig, build_box, force_receptive_field, write_gmd_xyz

    work = cl.run_dir(kind)
    work.mkdir(parents=True, exist_ok=True)
    alpha_root = cl.alpha.artifacts_root
    decision = decide_labeling_mode(alpha_root / cl.config.labeling.fidelity_dir, cl.config.labeling.mode)
    parents = load_parents(alpha_root)
    parent = select_md_parent(cl, parents)
    topology = AlkaneTopology(int(parent["chain_length"]))
    box_config = MDBoxConfig.model_validate(cl.config.md_box)
    receptive = force_receptive_field(cl.beta.config.training.model, override=box_config.receptive_field_override)
    box = build_box(load_positions(alpha_root, parent), receptive, box_config)
    xyz = write_gmd_xyz(cl.out_dir / "md_box" / f"{parent['parent_id']}.xyz", topology.numbers, box.positions, box.lengths)
    write_json_atomic(cl.out_dir / "md_box" / "box.json", {"parent_id": parent["parent_id"], "xyz": str(xyz), **box.to_dict()})

    raw = gmd_active_raw(cl, kind=kind, initial_structure=xyz, task_dir=work / "cp2k_tasks")
    gmd_config = build_config(raw, base_dir=work, path=None)
    controller = build_controller(gmd_config, work_dir=work)
    controller.config_hash = content_hash({"gmd_active": gmd_config.config_hash, "closed_loop": cl.config_hash, "kind": kind,
                                           "labeling": decision.to_dict()}, length=16)
    alpha_hash = build_service(cl.alpha).settings_hash
    service = controller.labeler.service
    if service.settings_hash != alpha_hash:
        raise ConfigurationError(f"closed-loop CP2K settings hash {service.settings_hash} differs from alpha's {alpha_hash}")

    beta_training = cl.beta.config.training
    controller.extractor = AlkaneRegionExtractor(
        decision=decision, alpha_config=cl.alpha.config, topology=topology, force_weights=beta_training.force_weights,
        full_force_weight=beta_training.full_system_force_weight, merge_radius=float(raw["hotspots"].get("merge_radius", 8.0)),
        parent_id=parent["parent_id"], geometry=cl.config.label_geometry, rejected_path=work / "rejected_events.jsonl")
    budgets = cl.alpha.config.budgets
    labeler = BudgetedLabeler(
        controller.labeler, max_core_s=cl.config.budget.max_cpu_core_hours * 3600.0, cores=cl.alpha.config.cp2k.cores,
        max_atoms={"full": cl.config.budget.max_atoms_full or budgets.max_atoms_full,
                   "fragment": cl.config.budget.max_atoms_fragment or budgets.max_atoms_per_fragment},
        estimate=_estimator(cl, topology.n_carbons), ledger_path=work / "budget_ledger.json")
    controller.labeler = labeler
    guard = ImageGuard(controller.enricher, bonds=topology.bonds, receptive_field=receptive, margin=box_config.margin,
                       stats_path=work / "image_check.json", enabled=box_config.frame_check)
    controller.enricher = guard
    trainer = RecordingTrainer(controller.trainer, work / "trainer_history.jsonl")
    controller.trainer = trainer

    # Validation split: one fixed, immutable version; training store publishes without a split.
    train_samples, val_samples, _record = initial_samples(cl, alpha_hash)
    validation_store = VersionedDatasetStore(work / "validation_store", DatasetsConfig.model_validate({"split": {"enabled": False}}))
    validation = validation_store.publish_samples(val_samples)
    controller.datasets = HeldOutValidationStore(controller.datasets, validation_store, validation)
    return Assembled(controller, decision, parent, topology, box, receptive, guard, trainer, labeler, validation, work,
                     controller.run_id, gmd_config, alpha_hash)


# ============================================================================================ initial model


def ensure_initial_model(cl: LoadedClosedLoop, asm: Assembled) -> ModelArtifact:
    """Train (once, shared by every run of this config) and register the initial model; initialise the run."""

    from gmd_active_learning.retraining.registry import PromotionGateConfig, evaluate_gates

    from .training import make_backend, train_model

    controller = asm.controller
    run = controller.store.ensure_run(asm.run_id, config_hash=controller.config_hash, initial_model_id=None, stage="sampling")
    if run.active_model_id is not None:
        return controller.registry.get(run.active_model_id)
    train_samples, val_samples, record = initial_samples(cl, asm.service_settings_hash)
    datasets: HeldOutValidationStore = controller.datasets
    for sample in train_samples:
        datasets.inner.write_sample(sample)
    dataset = datasets.publish_samples(train_samples, parent=None)
    training = cl.beta.config.training
    if cl.config.initial_model.train_overrides:
        training = training.model_copy(update={"train": {**training.train, **cl.config.initial_model.train_overrides}})
    training_hash = content_hash({"training": training.model_dump(mode="json"), "seed": cl.config.initial_model.seed}, length=16)
    shared = cl.out_dir / "initial_model"
    manifest_path = shared / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["dataset_version"], manifest["validation_version"], manifest.get("training_hash")) != (
                dataset.version, asm.validation.version, training_hash):
            raise WorkflowError(f"{shared} holds an initial model trained on other data or settings ({manifest['dataset_version']}, "
                                f"training {manifest.get('training_hash')}); use a new output directory")
    backend = make_backend(training, root=REPO_ROOT)
    started = time.monotonic()
    trained = train_model(backend, dataset, shared / "train", seed=cl.config.initial_model.seed)
    train_wall = time.monotonic() - started
    trainer = controller.trainer
    if manifest_path.is_file() and Path(json.loads(manifest_path.read_text())["export"]).is_file():
        # Reuse the recorded export and checks: a TorchScript re-export is not byte-identical, and the
        # initial model must be the same artifact (same model id) in every run of this configuration.
        manifest = json.loads(manifest_path.read_text())
        export, export_check, smoke, metrics = Path(manifest["export"]), manifest["export_check"], manifest["engine_smoke"], manifest["metrics_validation"]
    else:
        export = trainer.export(Path(trained.primary), shared / "export")
        export_check = trainer.validate_export(Path(trained.primary), export, dataset)
        smoke = trainer.engine_smoke_test(export, shared / "engine_smoke")
        metrics = trainer.evaluate(Path(trained.primary), dataset)
        gate_config = PromotionGateConfig.model_validate(controller.config.promotion.model_dump())
        gates = evaluate_gates(gate_config, metrics=metrics, parent_metrics=None, export_check=export_check, engine_smoke=smoke)
        deviation = export_check.get("max_force_deviation")
        if deviation is None or float(deviation) > gate_config.export_max_force_deviation:
            raise WorkflowError(f"initial model export does not reproduce the checkpoint: {export_check}")
        if not smoke.get("passed"):
            raise WorkflowError(f"initial model failed the GMD engine smoke test: {smoke}")
        write_json_atomic(manifest_path, {"dataset_version": dataset.version, "validation_version": asm.validation.version,
                                          "training_hash": training_hash, "samples": record, "checkpoints": trained.checkpoints,
                                          "export": str(export), "train_wall_s": train_wall, "best_val": trained.summary.get("best_val"),
                                          "metrics_validation": metrics, "export_check": export_check, "engine_smoke": smoke,
                                          "gates_recorded_not_enforced": gates.to_dict(), "seed": cl.config.initial_model.seed,
                                          "training_section": training.model_dump(mode="json")})
    artifact = controller.registry.register(
        checkpoint=trained.primary, export=export, dataset_version=dataset.version, parent_model_id=None, species=trained.species,
        cutoff=trained.cutoff, metrics=metrics, train_mode=str(trained.summary.get("mode", "fresh")), backend="gmd_sgt",
        metadata={"initial_model": True, "initial_samples": record, "export_check": export_check, "engine_smoke": smoke},
        extra_files={f"committee_{k}": path for k, path in enumerate(trained.checkpoints[1:], start=1)})
    controller.initialise(artifact)
    controller.store.update_run(asm.run_id, active_dataset_version=dataset.version)
    return artifact


# ============================================================================================ bookkeeping


def read_segment_frames(segment: Mapping[str, Any]) -> list[Frame]:
    from gmd_active_learning.adapters.gmd.engine import parse_frames

    path = Path(segment["work_dir"]) / "frames.extxyz"
    if not path.is_file():
        return []
    return parse_frames(path.read_text(), run_id=str(segment.get("run_id", "segment")))


def model_change_jumps(segments: Sequence[Mapping[str, Any]], frames_of: Callable[[Mapping[str, Any]], list[Frame]],
                       reevaluate: Callable[[str, Frame], float] | None = None, *, position_tolerance: float = 1.0e-5) -> list[dict[str, Any]]:
    """Potential-energy jump at every model change between consecutive segments.

    The last frame of the old segment and the restarted segment's first frame (the same MD step, the
    same coordinates, recomputed with the new model) give ``E_new - E_old`` at fixed coordinates; this
    is a model change, not integration drift. ``reevaluate(model_id, frame)`` optionally recomputes both
    energies on the boundary configuration independently of GMD.
    """

    records = []
    finished = [s for s in segments if s.get("status") in ("completed", "stopped")]
    for previous, current in zip(finished, finished[1:]):
        if previous.get("model_id") == current.get("model_id"):
            continue
        old_frames, new_frames = frames_of(previous), frames_of(current)
        record: dict[str, Any] = {
            "kind": "model_change",
            "note": "potential-energy jump from replacing the model at fixed coordinates (restart boundary); not integration drift",
            "old_segment": previous.get("segment_index"), "new_segment": current.get("segment_index"),
            "old_model_id": previous.get("model_id"), "new_model_id": current.get("model_id"),
            "boundary_step": previous.get("end_step"), "energy_old_model_ev": None, "energy_new_model_ev": None, "jump_ev": None,
            "same_coordinates": None, "max_position_difference": None, "reevaluated": None,
        }
        if old_frames and new_frames:
            last_old = max(old_frames, key=lambda frame: frame.step)
            first_new = min(new_frames, key=lambda frame: frame.step)
            difference = float(np.max(np.abs(np.asarray(last_old.positions) - np.asarray(first_new.positions))))
            same = last_old.step == first_new.step and difference <= position_tolerance
            record.update(boundary_step=int(last_old.step), new_first_step=int(first_new.step), same_coordinates=bool(same),
                          max_position_difference=difference, energy_old_model_ev=last_old.energy, energy_new_model_ev=first_new.energy)
            if same and last_old.energy is not None and first_new.energy is not None:
                record["jump_ev"] = float(first_new.energy) - float(last_old.energy)
                record["jump_ev_per_atom"] = record["jump_ev"] / len(last_old.numbers)
            if reevaluate is not None:
                e_old = reevaluate(str(previous["model_id"]), last_old)
                e_new = reevaluate(str(current["model_id"]), last_old)
                record["reevaluated"] = {"configuration_step": int(last_old.step), "energy_old_model_ev": e_old, "energy_new_model_ev": e_new,
                                         "jump_ev": e_new - e_old, "source": "both exports re-evaluated on the old segment's last frame"}
        records.append(record)
    return records


def provenance_chain(store: Any, run_id: str, model_id: str, *, dataset_reader: Callable[[str], Sequence[str]],
                     initial_samples_record: Mapping[str, Any] | None, md_parent: Mapping[str, Any] | None) -> dict[str, Any]:
    """model -> dataset version -> samples -> tasks -> regions -> events -> parent frames -> parent structure."""

    model = store.model(model_id)
    if model is None:
        raise WorkflowError(f"unknown model {model_id}")
    version = model["dataset_version"]
    sample_ids = list(dataset_reader(version)) if version is not None else []
    rows = {row["sample_id"]: row for row in store.samples(run_id)}
    events = {row["event_id"]: row for row in store.events(run_id)}
    initial = {item["sample_id"]: item for item in (initial_samples_record or {}).get("train", [])}
    chain = []
    for sample_id in sample_ids:
        if sample_id in rows:
            row = rows[sample_id]
            task = store.task(row["task_id"]) or {}
            region_file = Path(task.get("work_dir", "")) / "region.json" if task else None
            region = json.loads(region_file.read_text()) if region_file is not None and region_file.is_file() else {}
            event_id = region.get("event_id")
            event = events.get(event_id) or {}
            summary = event.get("summary") or {}
            frame_steps = [*summary.get("pre_frame_steps", []), summary.get("trigger_step"), *summary.get("post_frame_steps", [])]
            chain.append({
                "sample_id": sample_id, "origin": "closed_loop", "cycle": row["cycle"], "task_id": row["task_id"],
                "job_id": ((task.get("job") or {}).get("job_id")), "task_work_dir": task.get("work_dir"), "task_state": task.get("state"),
                "region_id": region.get("region_id"), "region_mode": region.get("mode"), "region_n_atoms": len(region.get("numbers", [])),
                "region_kind": (region.get("metadata") or {}).get("kind"), "event_id": event_id, "event_trigger_step": event.get("trigger_step"),
                "event_frames_path": event.get("path"), "parent_frame_steps": [s for s in frame_steps if s is not None],
                "parent_step": region.get("parent_step"), "model_version_at_event": region.get("model_version"),
                "parent_structure": None if md_parent is None else {"parent_id": md_parent.get("parent_id"), "structure_path": md_parent.get("structure_path"),
                                                                     "split": md_parent.get("split")},
            })
        elif sample_id in initial:
            item = initial[sample_id]
            chain.append({"sample_id": sample_id, "origin": "initial_train_full_dft", "label_id": item["label_id"],
                          "parent_id": item["parent_id"], "split": item["split"], "split_group": item["split_group"]})
        else:
            chain.append({"sample_id": sample_id, "origin": "unknown"})
    unknown = [item["sample_id"] for item in chain if item["origin"] == "unknown"]
    return {"model_id": model_id, "state": model["state"], "parent_model_id": model.get("parent_model_id"), "dataset_version": version, "n_samples": len(sample_ids), "unresolved_samples": unknown, "samples": chain}


def gate_reports(store: Any, history: Sequence[Mapping[str, Any]], gate_config: Any) -> list[dict[str, Any]]:
    """Re-derive each registered (non-initial) model's gate report from the recorded inputs."""

    from gmd_active_learning.retraining.registry import evaluate_gates

    exports = {item["export"]: item["result"] for item in history if item["kind"] == "validate_export"}
    smokes = {item["export"]: item["result"] for item in history if item["kind"] == "engine_smoke"}
    out = []
    for row in store.models():
        if (row.get("metadata") or {}).get("initial_model"):
            continue
        source_export = (row.get("metadata") or {}).get("source_export")
        parent_metrics = (row.get("metadata") or {}).get("parent_metrics_on_this_dataset")
        report = evaluate_gates(gate_config, metrics=row["metrics"], parent_metrics=parent_metrics,
                                export_check=exports.get(source_export), engine_smoke=smokes.get(source_export))
        out.append({"model_id": row["model_id"], "state": row["state"], "parent_model_id": row.get("parent_model_id"),
                    "dataset_version": row.get("dataset_version"), "metrics": row["metrics"], "parent_metrics": parent_metrics,
                    "gate_report": report.to_dict()})
    return out


def task_costs(store: Any, run_id: str, cores: int) -> list[dict[str, Any]]:
    from gmd_active_learning.labeling.task_store import load_task

    from .cost import measured_cost

    out = []
    for row in store.tasks(run_id):
        work = Path(row["work_dir"])
        if not (work / "task.json").is_file():
            continue
        task = load_task(work)
        attempts = task.metadata.get("attempts") or []
        cost = measured_cost(task.work_dir, attempts, cores)
        out.append({"task_id": row["task_id"], "state": row["state"], "cycle": row["cycle"], "n_atoms": task.metadata.get("n_atoms"),
                    "attempts": len(attempts), "job_ids": [a.get("job_id") for a in attempts], "execution": task.execution.value,
                    "validation": task.validation.value, "wall_s": cost["wall_s"], "cpu_core_s": cost["cpu_core_s"], "complete": cost["complete"]})
    return out


def build_report(cl: LoadedClosedLoop, asm: Assembled) -> dict[str, Any]:
    from gmd_active_learning.workflow.controller import summarize

    from .manifest import utc_now

    controller = asm.controller
    store = controller.store
    summary = summarize(store, asm.run_id)
    segments = [dict(row, run_id=asm.run_id) for row in store.segments(asm.run_id)]
    backend = controller.trainer.inner

    def reevaluate(model_id: str, frame: Frame) -> float:
        model = controller.registry.get(model_id)
        return backend.predict(model.export_path, frame.positions, frame.numbers, frame.cell, frame.pbc, species=model.species).energy

    jumps = model_change_jumps(segments, read_segment_frames, reevaluate)
    init_manifest = json.loads((cl.out_dir / "initial_model" / "manifest.json").read_text())
    promoted = [p["model_id"] for p in store.promotions(asm.run_id) if p["action"] in ("promote", "rollback")]
    chains = {mid: provenance_chain(store, asm.run_id, mid, dataset_reader=lambda v: controller.datasets.get(v).sample_ids,
                                    initial_samples_record=init_manifest["samples"], md_parent=asm.parent) for mid in dict.fromkeys(promoted)}
    events = [{"event_id": row["event_id"], "cycle": row["cycle"], "trigger_step": row["trigger_step"], "state": row["state"],
               "max_score": row["summary"].get("max_score"), "n_hotspot_atoms": len(row["summary"].get("hotspot_atom_ids", [])),
               "trigger_reasons": row["summary"].get("trigger_reasons")} for row in store.events(asm.run_id)]
    costs = task_costs(store, asm.run_id, cl.alpha.config.cp2k.cores)
    gates = gate_reports(store, controller.trainer.history(), controller.config.promotion)
    trainings = [item for item in controller.trainer.history() if item["kind"] == "train"]
    return {
        "created_at": utc_now(), "config": str(cl.path), "config_hash": cl.config_hash, "run_kind": "main" if asm.work_dir == cl.run_dir("main") else "recovery",
        "smoke_only": cl.alpha.config.cp2k.smoke_only, "summary": summary, "labeling": asm.decision.to_dict(),
        "md_parent": {k: asm.parent[k] for k in ("parent_id", "split", "family", "chain_length", "structure_path")},
        "md_box": asm.box.to_dict(), "image_check": asm.guard.flush(), "cp2k_settings_hash": asm.service_settings_hash,
        "events": events, "event_steps": [e["trigger_step"] for e in events], "label_tasks": costs,
        "events_rejected_before_labeling": [json.loads(line) for line in (asm.work_dir / "rejected_events.jsonl").read_text().splitlines()]
        if (asm.work_dir / "rejected_events.jsonl").is_file() else [],
        "measured_dft_cpu_core_hours": sum(c["cpu_core_s"] for c in costs) / 3600.0,
        "budget": {"max_cpu_core_hours": cl.config.budget.max_cpu_core_hours, "ledger": json.loads((asm.work_dir / "budget_ledger.json").read_text())
                   if (asm.work_dir / "budget_ledger.json").is_file() else []},
        "trainings": trainings, "gate_reports": gates, "model_changes": jumps, "provenance": chains,
        "initial_model": {k: init_manifest[k] for k in ("dataset_version", "validation_version", "metrics_validation", "export_check",
                                                         "engine_smoke", "gates_recorded_not_enforced", "best_val")},
        "final_model_id": summary["active_model_id"],
        "initial_model_id": next((p["model_id"] for p in store.promotions(asm.run_id) if p["previous_model_id"] is None), None),
        "transitions": [{k: t[k] for k in ("stage_from", "stage_to", "cycle", "at")} for t in store.transitions(asm.run_id)],
    }


# ============================================================================================ run


def run_closed_loop(cl: LoadedClosedLoop, *, kind: str = "main", max_wall_s: float | None = None,
                    sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Run (or resume) the closed loop until DONE/STOPPED, or until ``max_wall_s``; writes the report."""

    from .manifest import write_manifest

    require_exclusive_cp2k(cl, [cl.out_dir])
    asm = assemble(cl, kind)
    write_manifest(cl.alpha, f"closed-loop-{cl.config.name}-{kind}", {
        "closed_loop_config": cl.raw, "closed_loop_config_hash": cl.config_hash, "labeling": asm.decision.to_dict(),
        "md_box": asm.box.to_dict(), "cp2k_settings_hash": asm.service_settings_hash, "work_dir": str(asm.work_dir),
        "gmd_active_config_hash": asm.gmd_config.config_hash})
    ensure_initial_model(cl, asm)
    started = time.monotonic()
    while True:
        stage = asm.controller.run()
        LOGGER.info("controller returned %s", stage)
        if stage != "waiting":
            break
        if max_wall_s is not None and time.monotonic() - started > max_wall_s:
            LOGGER.warning("stopping after %.0f s with labels outstanding; re-run to resume", max_wall_s)
            break
        sleep(cl.config.loop.poll_interval_s)
    report = build_report(cl, asm)
    report["final_stage"] = stage
    write_json_atomic(asm.work_dir.parent / ("closed_loop_report.json" if kind == "main" else "recovery_run_report.json"), report)
    if kind == "main":
        timeline = {**report["summary"], "event_steps": report["event_steps"]}
        write_json_atomic(cl.out_dir / "timeline.json", timeline)
    return report


# ============================================================================================ frozen-model evaluation


def chain_observables(frame: Frame, topology: AlkaneTopology, *, covalent_scale: float | None = None) -> dict[str, Any]:
    """End-to-end distance, radius of gyration, C-C-C-C dihedrals and (with ``covalent_scale``) topology integrity."""

    from gmd_active_learning.utils.bonds import infer_bonds

    from .geometry import dihedral_deg
    from .md_box import orthorhombic_lengths, unwrap_molecule

    positions = sorted_frame_positions(frame, topology)
    if frame.cell is not None and any(frame.pbc):
        positions = unwrap_molecule(positions, orthorhombic_lengths(frame.cell), topology.bonds)
    carbons = positions[: topology.n_carbons]
    centre = carbons.mean(axis=0)
    dihedrals = [dihedral_deg(positions, i, i + 1, i + 2, i + 3) for i in range(topology.n_carbons - 3)]
    intact = None
    if covalent_scale is not None:
        bonds = sorted(tuple(sorted(pair)) for pair in infer_bonds(positions, topology.numbers, scale=covalent_scale))
        intact = bonds == list(topology.bonds)
    return {"step": int(frame.step), "end_to_end": float(np.linalg.norm(carbons[-1] - carbons[0])),
            "radius_of_gyration": float(np.sqrt(np.mean(np.sum((carbons - centre) ** 2, axis=1)))),
            "dihedrals_deg": [float(d) for d in dihedrals], "energy": frame.energy, "topology_intact": intact}


def summarize_observables(rows: Sequence[Mapping[str, Any]], config: FrozenEvalSection) -> dict[str, Any]:
    """Chain statistics over frames whose bond topology is still the parent's (a dissociated chain is not an alkane)."""

    if not rows:
        return {"n_frames": 0}
    checked = [r for r in rows if r.get("topology_intact") is not None]
    integrity: dict[str, Any] = {"checked": bool(checked)}
    if checked:
        broken = [r["step"] for r in checked if not r["topology_intact"]]
        integrity.update(n_intact=len(checked) - len(broken), intact_fraction=(len(checked) - len(broken)) / len(checked),
                         first_broken_step=min(broken) if broken else None)
        rows = [r for r in rows if r.get("topology_intact") is not False]
        if not rows:
            return {"n_frames": len(checked), "topology": integrity,
                    "reason": "no frame keeps the parent bond topology: chain observables are undefined"}
    n_total = len(checked) if checked else len(rows)
    e2e = np.asarray([r["end_to_end"] for r in rows])
    rg = np.asarray([r["radius_of_gyration"] for r in rows])
    dihedrals = np.abs(np.concatenate([np.asarray(r["dihedrals_deg"]) for r in rows]))
    counts, edges = np.histogram(np.concatenate([np.asarray(r["dihedrals_deg"]) for r in rows]), bins=config.dihedral_bins, range=(-180.0, 180.0))
    return {"n_frames": n_total, "n_frames_used": len(rows), "topology": integrity,
            "observables_over": "frames with the parent bond topology" if checked else "all frames (topology not checked)",
            "end_to_end": {"mean": float(e2e.mean()), "std": float(e2e.std()), "min": float(e2e.min()), "max": float(e2e.max())},
            "radius_of_gyration": {"mean": float(rg.mean()), "std": float(rg.std())},
            "dihedral_gauche_fraction": float(np.mean(dihedrals < config.gauche_window_deg)),
            "dihedral_definition": f"C-C-C-C, gauche = |phi| < {config.gauche_window_deg} deg, trans otherwise",
            "dihedral_histogram": {"edges_deg": edges.tolist(), "counts": counts.tolist()}}


def frozen_eval(cl: LoadedClosedLoop, *, steps: int | None = None) -> dict[str, Any]:
    """Fresh GMD trajectories with the frozen initial and final models (same seeds, same start structure)."""

    from gmd_active_learning.adapters.gmd.engine import GMDConfig, GMDEngine
    from gmd_active_learning.contracts import FrameAction
    from gmd_active_learning.workflow.protocols import SegmentRequest

    from .manifest import utc_now

    report_path = cl.out_dir / "closed_loop_report.json"
    if not report_path.is_file():
        raise WorkflowError(f"{report_path} does not exist: run closed-loop-run first")
    loop_report = json.loads(report_path.read_text())
    if loop_report.get("final_stage") != "done":
        raise WorkflowError(f"the closed loop is not finished (stage {loop_report.get('final_stage')}); frozen-eval needs the final model")
    asm = assemble(cl, "main")
    registry = asm.controller.registry
    models = {"initial": registry.get(loop_report["initial_model_id"]), "final": registry.get(loop_report["final_model_id"])}
    section = cl.config.frozen_eval
    n_steps = int(steps or section.steps)
    out = cl.out_dir / "frozen_eval"
    results: dict[str, Any] = {"created_at": utc_now(), "steps": n_steps, "seeds": section.seeds, "frame_interval": section.frame_interval,
                               "discard_steps": section.discard_steps, "md_parent": asm.parent["parent_id"],
                               "models": {k: m.model_id for k, m in models.items()}, "same_model": models["initial"].model_id == models["final"].model_id,
                               "scope": "pilot-scale comparison of two frozen models on identical seeds and start structure; no claim beyond these trajectories",
                               "per_model": {}}
    base = asm.gmd_config.raw["md"]
    from .md_box import ImageGuard

    guard = ImageGuard(None, bonds=asm.topology.bonds, receptive_field=asm.receptive_field, margin=asm.guard.margin,
                       stats_path=out / "image_check.json")
    for label, model in models.items():
        rows_all = []
        per_seed = {}
        for seed in section.seeds:
            directives = {**base["directives"], "velocity_seed": int(seed), "velocity_init": "random"}
            engine = GMDEngine(GMDConfig.model_validate({**base, "directives": directives, "frame_interval": section.frame_interval}))
            request = SegmentRequest(f"frozen-{label}-s{seed}", 0, out / label / f"seed_{seed}", model, n_steps, None, None)
            rows: list[dict[str, Any]] = []

            def on_frame(frame: Frame, rows: list[dict[str, Any]] = rows) -> FrameAction:
                guard.check(frame)
                rows.append(chain_observables(frame, asm.topology, covalent_scale=cl.config.label_geometry.covalent_scale))
                return FrameAction.CONTINUE

            result = engine.completed_segment(request)
            if result is not None and result.status == "completed":
                for frame in engine.read_frames(result, request):
                    on_frame(frame)
            else:
                result = engine.run_segment(request, on_frame)
            if result.status != "completed":
                raise WorkflowError(f"frozen-eval GMD run {label} seed {seed} ended {result.status}: {result.detail}")
            kept = [r for r in rows if r["step"] >= section.discard_steps]
            per_seed[str(seed)] = summarize_observables(kept, section)
            rows_all.extend(kept)
        results["per_model"][label] = {"model_id": model.model_id, "pooled": summarize_observables(rows_all, section), "per_seed": per_seed}
    results["image_check"] = guard.flush()
    write_json_atomic(out / "frozen_eval.json", results)
    return results


# ============================================================================================ CLI


def _closed_loop_argument(parser: Any) -> None:
    parser.add_argument("--closed-loop", required=True, help="configs/demos/alkane_hotspots/closed_loop_{smoke,pilot}.yaml")


def _run_arguments(parser: Any) -> None:
    _closed_loop_argument(parser)
    parser.add_argument("--run-kind", choices=list(RUN_KINDS), default="main", help="main loop or the recovery demo's run")
    parser.add_argument("--max-wall-s", type=float, default=None, help="stop waiting for labels after this (resumable)")


def _frozen_arguments(parser: Any) -> None:
    _closed_loop_argument(parser)
    parser.add_argument("--steps", type=int, default=None, help="override frozen_eval.steps")


def _short(report: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("final_stage", "labeling", "measured_dft_cpu_core_hours", "final_model_id", "initial_model_id", "event_steps")
    out = {k: report.get(k) for k in keys}
    out["summary"] = {k: report["summary"][k] for k in ("stage", "cycle", "events", "samples", "tasks", "promotions")}
    out["model_changes"] = [{k: j.get(k) for k in ("boundary_step", "jump_ev", "same_coordinates")} for j in report.get("model_changes", [])]
    return out


def cmd_closed_loop_run(args: Any, loaded: LoadedConfig) -> int:
    cl = load_closed_loop(args.closed_loop, alpha=loaded)
    report = run_closed_loop(cl, kind=args.run_kind, max_wall_s=args.max_wall_s)
    print(json.dumps(_short(report), indent=1, default=str))
    return 0 if report["final_stage"] in ("done", "waiting") else 1


def cmd_frozen_eval(args: Any, loaded: LoadedConfig) -> int:
    cl = load_closed_loop(args.closed_loop, alpha=loaded)
    result = frozen_eval(cl, steps=args.steps)
    print(json.dumps({k: {"model_id": v["model_id"], **{key: v["pooled"].get(key) for key in ("n_frames", "topology", "end_to_end", "radius_of_gyration",
                                                                                              "dihedral_gauche_fraction", "reason")}}
                      for k, v in result["per_model"].items()}, indent=1))
    return 0


def register_cli(register: Callable[..., None]) -> None:
    register("closed-loop-run", "Real closed loop on GMD_active's controller (resumable)", _run_arguments, cmd_closed_loop_run)
    register("frozen-eval", "Fresh GMD runs with the frozen initial and final models; chain observables", _frozen_arguments, cmd_frozen_eval)


__all__ = [
    "AlkaneRegionExtractor",
    "BudgetExceededError",
    "BudgetedLabeler",
    "ClosedLoopConfig",
    "HeldOutValidationStore",
    "LabelingDecision",
    "MissingInputsError",
    "RecordingTrainer",
    "assemble",
    "chain_observables",
    "decide_labeling_mode",
    "ensure_initial_model",
    "frozen_eval",
    "gate_reports",
    "hotspot_site",
    "isolated_frame",
    "load_closed_loop",
    "model_change_jumps",
    "provenance_chain",
    "run_closed_loop",
    "summarize_observables",
]
