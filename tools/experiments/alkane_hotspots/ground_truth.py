"""Hotspot ground truth from reference (DFT) force errors, and how well each monitor signal finds it.

Definitions (also in docs/demos/alkane_hotspots.md, section "Hotspot ground truth"):

* Per-atom reference error: ``e_i = |F_model_i - F_DFT_i|`` (eV/A, Euclidean norm of the force-vector
  difference) on configurations that carry a valid **full-system** DFT label. ``F_model`` is the
  primary model's force, i.e. the force that would drive the MD.
* Regions: one region per heavy atom (``centers: heavy_atoms``) or per atom (``all_atoms``). A
  region is the set of heavy atoms within ``radius_bonds`` bond hops of its centre in the covalent
  bond graph (:func:`gmd_active_learning.utils.bonds.infer_bonds`, ``bond_scale``), plus the hydrogens
  bonded to them (``include_hydrogens``). Regions overlap; they are correlated samples.
* Region reference error ``E_R`` = ``aggregate`` (max | rms | mean) of ``e_i`` over the region.
* Reference high-error region at threshold ``tau`` (YAML ``reference_thresholds``): ``E_R >= tau``.
* Region signal score ``S_R`` = ``score_aggregate`` (max | mean) of the per-atom signal over the region.
* A signal *flags* a region when ``S_R >= signal threshold`` (YAML ``signals.<name>.threshold``; signals
  without a threshold are evaluated threshold-free only).
* precision = TP / (TP + FP), recall = TP / (TP + FN), FPR = FP / (FP + TN) over regions.
* Average precision (AP, the PR-curve area) = sum_k (R_k - R_{k-1}) P_k over the ranking by
  descending ``S_R`` (ties form one step); ROC-AUC is the Mann-Whitney probability with ties counted
  1/2. Undefined values (no positives / no negatives) are ``null`` with a reason, never 0.
* ``fpr_normal``: region-level FPR on configurations of family ``normal`` (reference labels).
  ``alarm_rate_normal``: fraction of ``normal`` configurations in which at least one region is flagged
  (independent of the reference error).
* Uncertainty: percentile bootstrap over ``split_group`` (perturbations of one base conformation are
  resampled together), ``bootstrap.n`` resamples with ``bootstrap.seed``.

Signals are computed with GMD_active's real monitor code on the same frames:

* raw per-atom metric values from :class:`~gmd_active_learning.monitors.StagedOODMonitor` run with every
  stage forced and lazy LJ off (``committee``: sqrt(mean_m |F^m - mean F|^2); ``lj_residual``: local LJ
  projection residual of the model forces; ``force``: |F_i|; ``rmin``: score = -r_min (closer = more
  suspicious); ``delta_force``: |F_i(t) - F_i(t_prev)|, only with a temporal reference, see below);
* ``coordination``: |q_i - mean q_Z| where q is GMD_active's smooth coordination number and the mean
  over element Z comes from the calibration ``normal`` frames (a static analogue of ``delta_q``);
* monitor variants (``monitor_variants``: deep-merged overrides of the ``monitor`` section), scored as
  the variant's final-stage per-atom ``atom_scores``, e.g. the full PHAL staged score, the same with LJ
  disabled, committee-only and committee+LJ. Running statistics are calibrated on the calibration
  frames before any evaluated frame (each evaluated frame starts from the same calibrated state, so
  frames do not contaminate each other).

Temporal metrics (delta_force, displacement, delta_q) need a previous MD frame. The configurations
here are static; using the unperturbed base conformation as "previous frame" would hand the monitor
the perturbation site (displacement = the applied perturbation), so ``temporal_reference: none`` is
the default and those signals are reported as unavailable.
"""

from __future__ import annotations

import copy
import json
import logging
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from gmd_active_learning.contracts import Frame
from gmd_active_learning.core.exceptions import ConfigurationError, DataContractError
from gmd_active_learning.monitors.config import MonitorConfig, load_monitor_config
from gmd_active_learning.monitors.coordination import smooth_coordination_numbers
from gmd_active_learning.monitors.staged_monitor import StagedOODMonitor
from gmd_active_learning.utils.bonds import infer_bonds
from gmd_active_learning.utils.neighbors import MonitorNeighbors

LOGGER = logging.getLogger(__name__)

RAW_SIGNALS = ("committee", "lj_residual", "force", "delta_force", "rmin", "coordination")
FAMILY_NORMAL = "normal"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------------------------- records


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """All JSON objects of a JSONL file (blank lines skipped); a malformed line raises."""

    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DataContractError(f"{path}:{number}: invalid JSON: {exc}") from exc
    return rows


def resolve_artifact_path(path: str | Path, *bases: str | Path | None) -> Path:
    """An absolute path as is; a relative one against the first base under which it exists."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    tried = []
    for base in bases:
        if base is None:
            continue
        resolved = Path(base) / candidate
        tried.append(str(resolved))
        if resolved.exists():
            return resolved
    raise DataContractError(f"artifact {path} not found (tried {tried})")


def parent_frame(record: Mapping[str, Any], *bases: str | Path | None) -> Frame:
    """The parent structure of a ``parents.jsonl`` record as a Frame (atom_id array = index)."""

    from ase.io import read

    path = resolve_artifact_path(record["structure_path"], *bases)
    atoms = read(str(path), index=0)
    ids = atoms.arrays.get("atom_id")
    atom_ids = np.arange(len(atoms)) if ids is None else np.asarray(ids, dtype=np.int64)
    cell = atoms.cell.array if any(atoms.pbc) else None
    return Frame(run_id=str(record["parent_id"]), step=0, atom_ids=atom_ids, numbers=atoms.get_atomic_numbers(),
                 positions=atoms.get_positions(), cell=cell, pbc=tuple(bool(x) for x in atoms.pbc))


def label_forces(record: Mapping[str, Any], *bases: str | Path | None) -> np.ndarray:
    if record.get("status") != "valid":
        raise DataContractError(f"label {record.get('label_id')} has status {record.get('status')!r}; only valid labels carry forces")
    forces = np.load(resolve_artifact_path(record["forces_path"], *bases))
    forces = np.asarray(forces, dtype=float)
    if forces.ndim != 2 or forces.shape[1] != 3 or not np.all(np.isfinite(forces)):
        raise DataContractError(f"label {record.get('label_id')}: forces have shape {forces.shape} or are non-finite")
    return forces


def full_labels_by_parent(labels: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The valid full-system label of each parent (several valid ones for one parent raise)."""

    out: dict[str, dict[str, Any]] = {}
    for record in labels:
        if record.get("kind") != "full" or record.get("status") != "valid":
            continue
        parent = str(record["parent_id"])
        if parent in out and out[parent]["label_id"] != record["label_id"]:
            raise DataContractError(f"parent {parent} has two different valid full labels: {out[parent]['label_id']}, {record['label_id']}")
        out[parent] = dict(record)
    return out


# ---------------------------------------------------------------------------------------------- graphs


def bond_adjacency(numbers: np.ndarray, positions: np.ndarray, *, scale: float) -> list[list[int]]:
    """Covalent bond graph (non-periodic) as adjacency lists."""

    adjacency: list[list[int]] = [[] for _ in range(len(numbers))]
    for i, j in infer_bonds(positions, numbers, scale=scale):
        adjacency[i].append(j)
        adjacency[j].append(i)
    return adjacency


def graph_hops(adjacency: Sequence[Sequence[int]], sources: Iterable[int]) -> np.ndarray:
    """Breadth-first hop count from the nearest source (-1 = unreachable)."""

    hops = np.full(len(adjacency), -1, dtype=np.int64)
    queue: deque[int] = deque()
    for source in sources:
        if hops[int(source)] != 0:
            hops[int(source)] = 0
            queue.append(int(source))
    while queue:
        node = queue.popleft()
        for other in adjacency[node]:
            if hops[other] < 0:
                hops[other] = hops[node] + 1
                queue.append(other)
    return hops


# ---------------------------------------------------------------------------------------------- config


class RegionDefinition(_Strict):
    centers: Literal["heavy_atoms", "all_atoms"] = "heavy_atoms"
    radius_bonds: int = Field(1, ge=0, description="bond hops between heavy atoms around each centre")
    include_hydrogens: bool = True
    bond_scale: float = Field(1.2, gt=0.0)
    aggregate: Literal["max", "rms", "mean"] = "max"
    score_aggregate: Literal["max", "mean"] = "max"


class SignalSpec(_Strict):
    enabled: bool = True
    threshold: float | None = Field(None, description="operating threshold on the region score; null = threshold-free only")


class BootstrapConfig(_Strict):
    n: int = Field(1000, ge=0, description="0 disables the bootstrap")
    seed: int = 20261107
    confidence: float = Field(0.95, gt=0.0, lt=1.0)


class CalibrationConfig(_Strict):
    splits: list[str] = Field(default_factory=lambda: ["calibration"])
    families: list[str] = Field(default_factory=lambda: [FAMILY_NORMAL])
    stages_always: bool = Field(True, description="evaluate every metric on every calibration frame so all running statistics fill")


_ONLY_PHYSICS_OFF = {name: {"enabled": False} for name in ("force", "delta_force", "displacement", "rmin", "delta_q", "mlip_force_deviation")}
_ALWAYS = {"stages": {"physics_interval": "always", "full_interval": "always"}, "actions": {"stop_min_distance": None, "stop_max_force": None}}
#: monitor variants scored by default: the configured staged monitor, the same without LJ, the committee
#: term alone and committee + LJ (both evaluated on every atom of every frame).
DEFAULT_VARIANTS: dict[str, dict[str, Any]] = {
    "phal_staged": {},
    "phal_staged_no_lj": {"metrics": {"lj_residual": {"enabled": False}}},
    "committee_monitor": {**_ALWAYS, "metrics": {**_ONLY_PHYSICS_OFF, "lj_residual": {"enabled": False}}},
    "committee_plus_lj": {**_ALWAYS, "metrics": {**_ONLY_PHYSICS_OFF, "lj_residual": {"lazy": False}}},
}


class GroundTruthConfig(_Strict):
    enabled: bool = True
    splits: list[str] = Field(default_factory=lambda: ["validation"], description="configurations evaluated")
    region: RegionDefinition = Field(default_factory=RegionDefinition)
    reference_thresholds: list[float] = Field(default_factory=lambda: [0.1, 0.2, 0.5], description="eV/A on E_R")
    signals: dict[str, SignalSpec] = Field(default_factory=dict)
    monitor_variants: dict[str, dict[str, Any]] = Field(default_factory=lambda: copy.deepcopy(DEFAULT_VARIANTS))
    comparisons: list[tuple[str, str]] = Field(
        default_factory=lambda: [("committee_plus_lj", "committee_monitor"), ("phal_staged", "phal_staged_no_lj")],
        description="(a, b) pairs: AP(a) - AP(b) with bootstrap interval",
    )
    temporal_reference: Literal["none", "group_normal"] = "none"
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    sweep_points: int = Field(25, ge=2)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)

    @model_validator(mode="after")
    def _check(self) -> GroundTruthConfig:
        if "final_eval" in self.calibration.splits:
            raise ValueError("ground_truth.calibration.splits must not contain final_eval (never used for any selection)")
        if not self.reference_thresholds:
            raise ValueError("ground_truth.reference_thresholds must not be empty")
        known = set(RAW_SIGNALS) | set(self.monitor_variants)
        unknown = sorted(set(self.signals) - known)
        if unknown:
            raise ValueError(f"ground_truth.signals has unknown names {unknown}; known: {sorted(known)}")
        for left, right in self.comparisons:
            for name in (left, right):
                if name not in known:
                    raise ValueError(f"comparison names unknown signal {name!r}")
        if self.temporal_reference != "none":
            LOGGER.warning("temporal_reference=%s hands the perturbation site to temporal metrics (leakage)", self.temporal_reference)
        return self

    def signal_names(self) -> list[str]:
        names = [*RAW_SIGNALS, *self.monitor_variants]
        return [name for name in names if self.signals.get(name, SignalSpec()).enabled]

    def threshold(self, name: str) -> float | None:
        return self.signals.get(name, SignalSpec()).threshold


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# ---------------------------------------------------------------------------------------------- frames


@dataclass(slots=True)
class GTFrame:
    """One configuration with model predictions and its full-system DFT reference."""

    parent_id: str
    family: str
    split: str
    split_group: str
    numbers: np.ndarray
    positions: np.ndarray
    model_forces: np.ndarray
    dft_forces: np.ndarray | None
    committee_forces: np.ndarray | None = None
    perturbation_atoms: list[int] = field(default_factory=list)
    label_id: str | None = None

    def __post_init__(self) -> None:
        n = len(self.numbers)
        self.numbers = np.asarray(self.numbers, dtype=np.int64)
        self.positions = np.asarray(self.positions, dtype=float).reshape(n, 3)
        self.model_forces = np.asarray(self.model_forces, dtype=float).reshape(n, 3)
        if self.dft_forces is not None:
            self.dft_forces = np.asarray(self.dft_forces, dtype=float)
            if self.dft_forces.shape != (n, 3):
                raise DataContractError(f"{self.parent_id}: DFT forces {self.dft_forces.shape} != ({n}, 3)")
        if self.committee_forces is not None:
            self.committee_forces = np.asarray(self.committee_forces, dtype=float)
            if self.committee_forces.ndim != 3 or self.committee_forces.shape[1:] != (n, 3):
                raise DataContractError(f"{self.parent_id}: committee forces {self.committee_forces.shape}")

    def frame(self, *, step: int = 0) -> Frame:
        return Frame(run_id=self.parent_id, step=step, atom_ids=np.arange(len(self.numbers)), numbers=self.numbers,
                     positions=self.positions, cell=None, pbc=(False, False, False), forces=self.model_forces,
                     committee_forces=self.committee_forces)

    def atom_errors(self) -> np.ndarray:
        if self.dft_forces is None:
            raise DataContractError(f"{self.parent_id} has no DFT reference")
        return np.linalg.norm(self.model_forces - self.dft_forces, axis=1)


#: ``predictor(numbers, positions) -> (primary forces (N, 3), committee forces (M, N, 3) or None)``
Predictor = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, "np.ndarray | None"]]


def build_gt_frames(
    parents: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    predictor: Predictor,
    *,
    splits: Sequence[str],
    bases: Sequence[str | Path | None],
    require_dft: bool = True,
) -> tuple[list[GTFrame], dict[str, str]]:
    """Frames of the requested splits; parents without a valid full label are skipped with a reason."""

    full = full_labels_by_parent(labels)
    frames: list[GTFrame] = []
    skipped: dict[str, str] = {}
    for record in parents:
        if record.get("split") not in splits:
            continue
        parent_id = str(record["parent_id"])
        label = full.get(parent_id)
        if label is None and require_dft:
            skipped[parent_id] = "no valid full-system DFT label"
            continue
        frame = parent_frame(record, *bases)
        forces, committee = predictor(frame.numbers, frame.positions)
        dft = None if label is None else label_forces(label, *bases)
        frames.append(GTFrame(parent_id=parent_id, family=str(record.get("family")), split=str(record.get("split")),
                              split_group=str(record.get("split_group", parent_id)), numbers=frame.numbers,
                              positions=frame.positions, model_forces=forces, dft_forces=dft, committee_forces=committee,
                              perturbation_atoms=[int(a) for a in (record.get("perturbation") or {}).get("atoms") or []],
                              label_id=None if label is None else str(label["label_id"])))
    return frames, skipped


# ---------------------------------------------------------------------------------------------- regions


def define_regions(numbers: np.ndarray, positions: np.ndarray, config: RegionDefinition) -> list[np.ndarray]:
    """Atom index sets of the regions of one configuration (see module docstring)."""

    numbers = np.asarray(numbers)
    adjacency = bond_adjacency(numbers, positions, scale=config.bond_scale)
    heavy = numbers > 1
    heavy_adjacency = [[j for j in adjacency[i] if heavy[j]] if heavy[i] else [] for i in range(len(numbers))]
    centers = np.flatnonzero(heavy) if config.centers == "heavy_atoms" else np.arange(len(numbers))
    regions = []
    for center in centers:
        if heavy[center]:
            hops = graph_hops(heavy_adjacency, [int(center)])
            members = set(np.flatnonzero((hops >= 0) & (hops <= config.radius_bonds)).tolist())
        else:  # hydrogen centre (all_atoms): its heavy neighbour's neighbourhood, plus itself
            members = {int(center)}
            for other in adjacency[center]:
                if heavy[other]:
                    hops = graph_hops(heavy_adjacency, [other])
                    members |= set(np.flatnonzero((hops >= 0) & (hops <= config.radius_bonds)).tolist())
        if config.include_hydrogens:
            members |= {j for i in list(members) for j in adjacency[i] if not heavy[j]}
        regions.append(np.asarray(sorted(members), dtype=np.int64))
    return regions


def aggregate(values: np.ndarray, how: str) -> float:
    if len(values) == 0:
        return float("nan")
    if how == "max":
        return float(np.max(values))
    if how == "mean":
        return float(np.mean(values))
    if how == "rms":
        return float(np.sqrt(np.mean(np.square(values))))
    raise ConfigurationError(f"unknown aggregate {how!r}")


# ---------------------------------------------------------------------------------------------- signals


def _raw_monitor_config(monitor: MonitorConfig) -> MonitorConfig:
    data = monitor.model_dump()
    data["enabled"] = True
    data["stages"].update(light_interval=1, physics_interval="always", full_interval="always", scheduled_full_interval=1)
    data["metrics"]["lj_residual"]["lazy"] = False
    for name in ("force", "rmin", "lj_residual", "committee"):
        data["metrics"][name]["enabled"] = True
    return load_monitor_config(data)


def _calibrate(monitor: StagedOODMonitor, frames: Sequence[Frame]) -> dict[str, Any]:
    for frame in frames:
        monitor.reset_temporal_state()
        monitor.evaluate_detailed(frame)
    state = monitor.to_state()
    state["previous"] = None
    return state


def _with_previous(state: dict[str, Any], monitor_cls_cfg: MonitorConfig, previous: Frame | None) -> dict[str, Any]:
    """``state`` with ``previous`` set to the given frame (stats untouched)."""

    if previous is None:
        return state
    helper = StagedOODMonitor(monitor_cls_cfg)
    helper.load_state(state)
    helper.evaluate_detailed(previous)
    out = helper.to_state()
    out["stats"], out["frame_index"] = state["stats"], state["frame_index"]
    return out


@dataclass(slots=True)
class SignalTable:
    """Per-atom signals of one configuration (None = unavailable, with ``reasons``)."""

    values: dict[str, np.ndarray | None]
    reasons: dict[str, str]
    monitor_results: dict[str, dict[str, Any]]


class SignalComputer:
    """Computes every configured signal on GTFrames with the real monitor code."""

    def __init__(self, monitor_section: Mapping[str, Any] | None, config: GroundTruthConfig, calibration_frames: Sequence[GTFrame]):
        self.config = config
        self.base = load_monitor_config(dict(monitor_section or {}))
        self.raw_cfg = _raw_monitor_config(self.base)
        self.variant_cfgs = {name: load_monitor_config(deep_merge(self.base.model_dump(), override))
                             for name, override in config.monitor_variants.items()}
        frames = [item.frame(step=k) for k, item in enumerate(calibration_frames)]
        self.n_calibration_frames = len(frames)
        self.states: dict[str, dict[str, Any]] = {}
        for name, cfg in [("__raw__", self.raw_cfg), *self.variant_cfgs.items()]:
            calib_cfg = cfg
            if config.calibration.stages_always:
                calib_cfg = _raw_monitor_config(cfg)
                data = calib_cfg.model_dump()
                for metric, settings in cfg.model_dump()["metrics"].items():  # keep the variant's enabled flags
                    data["metrics"][metric]["enabled"] = settings["enabled"]
                calib_cfg = load_monitor_config(data)
            self.states[name] = _calibrate(StagedOODMonitor(calib_cfg), frames)
        self.q_reference = self._coordination_reference(frames)
        warmup = self.base.running_stats.warmup_frames
        self.warmup_satisfied = self.n_calibration_frames >= warmup
        if not self.warmup_satisfied:
            LOGGER.warning("only %d calibration frames (< warmup_frames=%d): z-based score parts are suppressed or raw",
                           self.n_calibration_frames, warmup)

    def _neighbors(self) -> MonitorNeighbors:
        return MonitorNeighbors(lj_cutoff=self.base.neighbors.lj_cutoff, coordination_cutoff=self.base.neighbors.coordination_cutoff)

    def _q(self, frame: Frame) -> np.ndarray:
        pairs = self._neighbors().rebuild(frame.positions, frame.cell, frame.pbc)
        cfg = self.base.metrics.delta_q
        return smooth_coordination_numbers(pairs, frame.numbers, cutoff=self.base.neighbors.coordination_cutoff, scale=cfg.scale, power=cfg.power)

    def _coordination_reference(self, frames: Sequence[Frame]) -> dict[int, float]:
        sums: dict[int, list[float]] = defaultdict(list)
        for frame in frames:
            for z, q in zip(frame.numbers.tolist(), self._q(frame).tolist()):
                sums[int(z)].append(float(q))
        return {z: float(np.mean(values)) for z, values in sums.items()}

    def _evaluate(self, name: str, cfg: MonitorConfig, item: GTFrame, previous: Frame | None) -> Any:
        monitor = StagedOODMonitor(cfg)
        monitor.load_state(_with_previous(self.states[name], cfg, previous))
        return monitor.evaluate_detailed(item.frame(step=1))

    def compute(self, item: GTFrame, previous: GTFrame | None = None) -> SignalTable:
        values: dict[str, np.ndarray | None] = {}
        reasons: dict[str, str] = {}
        previous_frame = None if previous is None else previous.frame(step=0)
        raw = self._evaluate("__raw__", self.raw_cfg, item, previous_frame).result
        for name in ("committee", "lj_residual", "force", "delta_force", "rmin"):
            metric = raw.metrics[name]
            if metric.status.value != "measured":
                values[name], reasons[name] = None, f"{metric.status.value}: {metric.reason}"
                continue
            data = np.where(metric.evaluated, metric.values, np.nan)
            values[name] = -data if name == "rmin" else data
        if self.q_reference:
            q = self._q(item.frame())
            ref = np.asarray([self.q_reference.get(int(z), np.nan) for z in item.numbers])
            values["coordination"] = np.abs(q - ref)
            if np.any(~np.isfinite(ref)):
                reasons["coordination"] = "elements without calibration reference are NaN"
        else:
            values["coordination"], reasons["coordination"] = None, "no calibration frames for the coordination reference"
        results: dict[str, dict[str, Any]] = {}
        for name, cfg in self.variant_cfgs.items():
            evaluation = self._evaluate(name, cfg, item, previous_frame)
            values[name] = np.asarray(evaluation.result.atom_scores, dtype=float)
            results[name] = {"stage": evaluation.result.stage.value, "triggered": evaluation.result.triggered,
                             "max_score": evaluation.result.max_score, "hotspot_atom_ids": evaluation.result.hotspot_atom_ids,
                             "trigger_reasons": evaluation.result.trigger_reasons}
        return SignalTable(values, reasons, results)


# ---------------------------------------------------------------------------------------------- metrics


def _ranking_steps(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative TP/FP at each distinct score (descending): thresholds, tp, fp."""

    order = np.argsort(-scores, kind="mergesort")
    scores, labels = scores[order], labels[order]
    distinct = np.flatnonzero(np.diff(scores)) if len(scores) > 1 else np.zeros(0, dtype=int)
    ends = np.r_[distinct, len(scores) - 1]
    tp = np.cumsum(labels)[ends]
    fp = np.cumsum(~labels)[ends]
    return scores[ends], tp.astype(float), fp.astype(float)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum())
    if n_pos == 0 or len(scores) == 0:
        return None
    _, tp, fp = _ranking_steps(np.asarray(scores, dtype=float), labels)
    precision = tp / (tp + fp)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    from scipy.stats import rankdata

    ranks = rankdata(np.asarray(scores, dtype=float))
    return float((np.sum(ranks[labels]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def confusion_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=bool)
    flagged = np.asarray(scores, dtype=float) >= threshold
    tp = int(np.sum(flagged & labels))
    fp = int(np.sum(flagged & ~labels))
    fn = int(np.sum(~flagged & labels))
    tn = int(np.sum(~flagged & ~labels))
    def ratio(a: int, b: int) -> float | None:
        return None if b == 0 else a / b
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall,
            "fpr": ratio(fp, fp + tn), "f1": f1}


def threshold_sweep(scores: np.ndarray, labels: np.ndarray, n_points: int) -> list[dict[str, Any]]:
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return []
    thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_points)))
    return [confusion_at(scores, labels, float(t)) for t in thresholds]


def _bootstrap(groups: np.ndarray, statistic: Callable[[np.ndarray], float | None], config: BootstrapConfig) -> dict[str, Any] | None:
    if config.n == 0:
        return None
    unique = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(config.seed)
    values = []
    for _ in range(config.n):
        pick = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in pick])
        value = statistic(idx)
        if value is not None:
            values.append(value)
    if not values:
        return {"n_defined": 0, "low": None, "high": None}
    alpha = (1.0 - config.confidence) / 2.0
    return {"n_defined": len(values), "n_resamples": config.n, "n_groups": int(len(unique)),
            "low": float(np.quantile(values, alpha)), "high": float(np.quantile(values, 1.0 - alpha)), "confidence": config.confidence}


# ---------------------------------------------------------------------------------------------- analysis


@dataclass(slots=True)
class RegionTable:
    """Flat table: one row per (configuration, region)."""

    parent_id: list[str] = field(default_factory=list)
    family: list[str] = field(default_factory=list)
    split_group: list[str] = field(default_factory=list)
    center: list[int] = field(default_factory=list)
    atoms: list[list[int]] = field(default_factory=list)
    contains_site: list[bool] = field(default_factory=list)
    error: list[float] = field(default_factory=list)
    scores: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def arrays(self) -> dict[str, np.ndarray]:
        return {"error": np.asarray(self.error, dtype=float), "family": np.asarray(self.family), "group": np.asarray(self.split_group),
                "parent": np.asarray(self.parent_id), "site": np.asarray(self.contains_site, dtype=bool)}


def region_table(frames: Sequence[GTFrame], tables: Sequence[SignalTable], config: GroundTruthConfig, signal_names: Sequence[str]) -> RegionTable:
    out = RegionTable()
    for item, table in zip(frames, tables):
        errors = item.atom_errors()
        site = set(item.perturbation_atoms)
        for members in define_regions(item.numbers, item.positions, config.region):
            out.parent_id.append(item.parent_id)
            out.family.append(item.family)
            out.split_group.append(item.split_group)
            out.center.append(int(members[0]))
            out.atoms.append(members.tolist())
            out.contains_site.append(bool(site & set(members.tolist())))
            out.error.append(aggregate(errors[members], config.region.aggregate))
            for name in signal_names:
                values = table.values.get(name)
                if values is None:
                    out.scores[name].append(float("nan"))
                else:
                    sub = values[members]
                    sub = sub[np.isfinite(sub)]
                    out.scores[name].append(aggregate(sub, config.region.score_aggregate) if len(sub) else float("nan"))
    return out


def evaluate_signal(scores: np.ndarray, labels: np.ndarray, family: np.ndarray, parent: np.ndarray, groups: np.ndarray,
                    threshold: float | None, config: GroundTruthConfig) -> dict[str, Any]:
    """Threshold-free and operating-point metrics of one signal against one reference labelling."""

    valid = np.isfinite(scores)
    result: dict[str, Any] = {"n_regions": int(len(scores)), "n_scored": int(valid.sum()), "n_positive": int(labels[valid].sum())}
    if not valid.any():
        result["reason"] = "signal unavailable on every region"
        return result
    s, y, fam, par, grp = scores[valid], labels[valid], family[valid], parent[valid], groups[valid]
    ap = average_precision(s, y)
    result["average_precision"] = ap
    result["prevalence"] = float(y.mean())
    result["roc_auc"] = roc_auc(s, y)
    if ap is None:
        result["average_precision_reason"] = "no positive regions"
    if result["roc_auc"] is None:
        result["roc_auc_reason"] = "needs positive and negative regions"
    result["average_precision_ci"] = _bootstrap(grp, lambda idx: average_precision(s[idx], y[idx]), config.bootstrap)
    result["sweep"] = threshold_sweep(s, y, config.sweep_points)
    if threshold is not None:
        result["operating_point"] = confusion_at(s, y, threshold)
        normal = fam == FAMILY_NORMAL
        if normal.any():
            point = confusion_at(s[normal], y[normal], threshold)
            flagged_parents = {p for p, flag in zip(par[normal], s[normal] >= threshold) if flag}
            n_normal = len(set(par[normal]))
            result["normal"] = {"fpr_normal": point["fpr"], "n_regions": int(normal.sum()), "n_configurations": n_normal,
                                "alarm_rate_normal": len(flagged_parents) / n_normal, "tp": point["tp"], "fp": point["fp"]}
        else:
            result["normal"] = {"reason": "no normal-family configurations among the evaluated splits"}
    else:
        result["operating_point"] = {"reason": "no operating threshold configured (threshold-free metrics only)"}
    return result


def localisation(frames: Sequence[GTFrame], table: RegionTable, signal_names: Sequence[str]) -> dict[str, Any]:
    """Per perturbed configuration: is the max-error region at the site; does each signal's top region hit the max-error region."""

    arrays = table.arrays()
    out: dict[str, Any] = {"n_perturbed": 0, "max_error_region_contains_site": 0, "top1_hit": {name: 0 for name in signal_names}}
    for item in frames:
        if not item.perturbation_atoms:
            continue
        rows = np.flatnonzero(arrays["parent"] == item.parent_id)
        if len(rows) == 0:
            continue
        out["n_perturbed"] += 1
        worst = rows[int(np.argmax(arrays["error"][rows]))]
        out["max_error_region_contains_site"] += int(arrays["site"][worst])
        worst_atoms = set(table.atoms[worst])
        for name in signal_names:
            scores = np.asarray(table.scores[name], dtype=float)[rows]
            if not np.isfinite(scores).any():
                continue
            top = rows[int(np.nanargmax(scores))]
            out["top1_hit"][name] += int(bool(worst_atoms & set(table.atoms[top])))
    n = out["n_perturbed"]
    out["fraction_site_is_max_error"] = None if n == 0 else out["max_error_region_contains_site"] / n
    out["top1_hit_rate"] = {name: (None if n == 0 else hits / n) for name, hits in out["top1_hit"].items()}
    return out


def analyse(frames: Sequence[GTFrame], tables: Sequence[SignalTable], config: GroundTruthConfig) -> dict[str, Any]:
    """The full ground-truth report (JSON-serialisable)."""

    if not frames:
        return {"status": "no_data", "reason": "no configuration with model predictions and a full-system DFT label"}
    names = config.signal_names()
    table = region_table(frames, tables, config, names)
    arrays = table.arrays()
    per_threshold: dict[str, Any] = {}
    for tau in config.reference_thresholds:
        labels = arrays["error"] >= tau
        signals = {}
        for name in names:
            scores = np.asarray(table.scores[name], dtype=float)
            signals[name] = evaluate_signal(scores, labels, arrays["family"], arrays["parent"], arrays["group"], config.threshold(name), config)
        comparisons = []
        for left, right in config.comparisons:
            if left not in names or right not in names:
                comparisons.append({"a": left, "b": right, "reason": "signal disabled"})
                continue
            a, b = np.asarray(table.scores[left], dtype=float), np.asarray(table.scores[right], dtype=float)
            both = np.isfinite(a) & np.isfinite(b)
            ap_a, ap_b = average_precision(a[both], labels[both]), average_precision(b[both], labels[both])
            delta = None if ap_a is None or ap_b is None else ap_a - ap_b
            def stat(idx: np.ndarray, a=a[both], b=b[both], y=labels[both]) -> float | None:
                x, z = average_precision(a[idx], y[idx]), average_precision(b[idx], y[idx])
                return None if x is None or z is None else x - z
            comparisons.append({"a": left, "b": right, "average_precision_a": ap_a, "average_precision_b": ap_b, "delta": delta,
                                "delta_ci": _bootstrap(arrays["group"][both], stat, config.bootstrap), "n_regions": int(both.sum())})
        per_threshold[f"{tau:g}"] = {"tau": tau, "n_positive": int(labels.sum()), "signals": signals, "comparisons": comparisons}
    families = sorted(set(arrays["family"].tolist()))
    return {
        "status": "executed",
        "definitions": {
            "atom_error": "|F_model - F_DFT| (eV/A)",
            "region": config.region.model_dump(),
            "positive": "region error >= tau",
            "fpr_normal": "FP/(FP+TN) over regions of normal-family configurations",
            "alarm_rate_normal": "fraction of normal configurations with >= 1 flagged region",
        },
        "n_configurations": len(frames),
        "n_regions": len(table.error),
        "configurations_per_family": {f: int(len(set(arrays["parent"][arrays["family"] == f]))) for f in families},
        "error_quantiles": {q: float(np.quantile(arrays["error"], float(q))) for q in ("0.5", "0.9", "0.99")},
        "signals_unavailable": sorted({f"{name}: {reason}" for t in tables for name, reason in t.reasons.items()}),
        "per_threshold": per_threshold,
        "localisation": localisation(frames, table, names),
    }


def run_ground_truth(
    frames: Sequence[GTFrame],
    calibration_frames: Sequence[GTFrame],
    monitor_section: Mapping[str, Any] | None,
    config: GroundTruthConfig,
    *,
    references: Mapping[str, GTFrame] | None = None,
) -> tuple[dict[str, Any], list[SignalTable]]:
    """Compute the signals on ``frames`` and analyse them; ``references`` maps parent_id -> temporal reference."""

    if config.temporal_reference != "none" and references is None:
        raise ConfigurationError(f"temporal_reference={config.temporal_reference} needs reference frames")
    computer = SignalComputer(monitor_section, config, calibration_frames)
    tables = [computer.compute(item, None if references is None else references.get(item.parent_id)) for item in frames]
    report = analyse(frames, tables, config)
    report["calibration"] = {"n_frames": computer.n_calibration_frames, "warmup_satisfied": computer.warmup_satisfied,
                             "splits": config.calibration.splits, "families": config.calibration.families,
                             "coordination_reference": {str(k): v for k, v in computer.q_reference.items()}}
    return report, tables


def per_atom_dump(frames: Sequence[GTFrame], tables: Sequence[SignalTable]) -> list[dict[str, Any]]:
    """Per-atom errors and signals (for the hotspot figure)."""

    rows = []
    for item, table in zip(frames, tables):
        rows.append({"parent_id": item.parent_id, "family": item.family, "positions": item.positions.tolist(),
                     "numbers": item.numbers.tolist(), "perturbation_atoms": item.perturbation_atoms,
                     "atom_error": item.atom_errors().tolist(),
                     "signals": {k: (None if v is None else np.where(np.isfinite(v), v, np.nan).tolist()) for k, v in table.values.items()}})
    return rows


__all__ = [
    "GTFrame",
    "GroundTruthConfig",
    "RegionDefinition",
    "SignalComputer",
    "analyse",
    "average_precision",
    "bond_adjacency",
    "build_gt_frames",
    "confusion_at",
    "define_regions",
    "graph_hops",
    "read_jsonl",
    "roc_auc",
    "run_ground_truth",
]
