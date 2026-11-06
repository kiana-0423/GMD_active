"""Held-out evaluation on final_eval full-DFT configurations and budget/accuracy analysis across seeds.

Definitions (docs/demos/alkane_hotspots.md, "Acquisition arms" / "Cost accounting"):

* ``force_rmse`` = sqrt(mean over all atoms and Cartesian components of (F_model - F_DFT)^2), eV/A,
  pooled over every evaluated configuration.
* atom error e_i = |F_model_i - F_DFT_i|; tail statistics are the 95th / 99th percentile and the
  maximum of e_i over all evaluated atoms.
* ``core_rmse``: the same RMSE restricted to the perturbation-site region of each perturbed
  configuration (heavy atoms within ``core_radius_bonds`` bond hops of the perturbed heavy atoms, plus
  their hydrogens); configurations without a perturbation (family ``normal``) do not enter it.
* per-family breakdown: the above per ``family``.
* accuracy at a fixed DFT budget B: linear interpolation of the metric between the two measured
  learning-curve points that bracket B on the cumulative-cost axis. B outside the measured range gives
  ``null`` with the reason (no extrapolation).
* cost to reach a target metric A: the first measured point with metric <= A; if it is not the first
  point, linear interpolation between it and the previous point; never reached gives ``null``.
* variation across independent seeds (repetitions with distinct seeds): mean, sample std (n >= 2),
  min, max, n -- never across atoms or frames.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .ground_truth import GTFrame, bond_adjacency, graph_hops

LOGGER = logging.getLogger(__name__)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvaluationConfig(_Strict):
    splits: list[str] = Field(default_factory=lambda: ["final_eval"])
    core_radius_bonds: int = Field(1, ge=0)
    bond_scale: float = Field(1.2, gt=0.0)
    budgets_cpu_core_s: list[float] = Field(default_factory=list, description="fixed DFT budgets for accuracy-at-budget")
    budgets_labels: list[int] = Field(default_factory=list)
    accuracy_targets: list[float] = Field(default_factory=list, description="force RMSE targets (eV/A) for cost-to-reach")
    metric: str = Field("force_rmse", description="learning-curve metric (key of evaluate_frames' overall block)")


def site_region(numbers: np.ndarray, positions: np.ndarray, site_atoms: Sequence[int], *, radius: int, bond_scale: float) -> np.ndarray:
    numbers = np.asarray(numbers)
    adjacency = bond_adjacency(numbers, positions, scale=bond_scale)
    heavy = numbers > 1
    seeds = set()
    for atom in site_atoms:
        if heavy[atom]:
            seeds.add(int(atom))
        else:
            seeds |= {j for j in adjacency[atom] if heavy[j]}
    if not seeds:
        return np.zeros(0, dtype=np.int64)
    heavy_adjacency = [[j for j in adjacency[i] if heavy[j]] if heavy[i] else [] for i in range(len(numbers))]
    hops = graph_hops(heavy_adjacency, sorted(seeds))
    members = set(np.flatnonzero((hops >= 0) & (hops <= radius)).tolist())
    members |= {j for i in list(members) for j in adjacency[i] if not heavy[j]}
    members |= {int(a) for a in site_atoms}
    return np.asarray(sorted(members), dtype=np.int64)


def _block(deltas: list[np.ndarray]) -> dict[str, Any]:
    if not deltas:
        return {"n_atoms": 0}
    stacked = np.concatenate(deltas, axis=0)
    errors = np.linalg.norm(stacked, axis=1)
    return {"n_atoms": int(len(stacked)), "force_rmse": float(np.sqrt(np.mean(stacked**2))),
            "atom_error_p95": float(np.percentile(errors, 95)), "atom_error_p99": float(np.percentile(errors, 99)),
            "atom_error_max": float(np.max(errors)), "atom_error_mean": float(np.mean(errors))}


def evaluate_frames(frames: Sequence[GTFrame], config: EvaluationConfig) -> dict[str, Any]:
    """Overall / core-region / per-family metrics on frames with model and DFT forces."""

    if not frames:
        return {"status": "no_data", "reason": "no final_eval configuration with a valid full-system DFT label"}
    overall, core = [], []
    per_family: dict[str, list[np.ndarray]] = {}
    per_family_core: dict[str, list[np.ndarray]] = {}
    for item in frames:
        if item.dft_forces is None:
            raise ValueError(f"{item.parent_id} has no DFT reference")
        delta = item.model_forces - item.dft_forces
        overall.append(delta)
        per_family.setdefault(item.family, []).append(delta)
        if item.perturbation_atoms:
            members = site_region(item.numbers, item.positions, item.perturbation_atoms, radius=config.core_radius_bonds, bond_scale=config.bond_scale)
            core.append(delta[members])
            per_family_core.setdefault(item.family, []).append(delta[members])
    families = {}
    for family, deltas in sorted(per_family.items()):
        families[family] = {"n_configurations": len(deltas), **_block(deltas), "core": _block(per_family_core.get(family, []))}
    return {"status": "executed", "n_configurations": len(frames), "overall": _block(overall),
            "core": {**_block(core), "n_configurations": len(core),
                     "definition": f"perturbation-site region, {config.core_radius_bonds} bond hop(s)"},
            "per_family": families, "parent_ids": [item.parent_id for item in frames]}


# ---------------------------------------------------------------------------------------------- curves


def value_at(points: Sequence[Mapping[str, Any]], x_key: str, y_key: str, x: float) -> dict[str, Any]:
    """Metric at cost ``x`` by interpolation between bracketing measured points (never extrapolated)."""

    pts = sorted(((float(p[x_key]), float(p[y_key])) for p in points if p.get(y_key) is not None and p.get(x_key) is not None),
                 key=lambda item: item[0])
    if not pts:
        return {"value": None, "reason": "no measured points"}
    xs = [p[0] for p in pts]
    if x < xs[0] or x > xs[-1]:
        return {"value": None, "reason": f"budget {x:g} outside the measured range [{xs[0]:g}, {xs[-1]:g}]; not extrapolated"}
    exact = [y for px, y in pts if px == x]
    if exact:
        return {"value": exact[-1], "interpolated": False}
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 < x < x1:
            return {"value": y0 + (x - x0) * (y1 - y0) / (x1 - x0), "interpolated": True, "between": [x0, x1]}
    return {"value": None, "reason": "no bracketing points"}  # pragma: no cover - unreachable for sorted points


def cost_to_reach(points: Sequence[Mapping[str, Any]], x_key: str, y_key: str, target: float) -> dict[str, Any]:
    """Cost at which the metric first reaches ``<= target`` (interpolated within measured points only)."""

    pts = sorted(((float(p[x_key]), float(p[y_key])) for p in points if p.get(y_key) is not None and p.get(x_key) is not None),
                 key=lambda item: item[0])
    if not pts:
        return {"cost": None, "reason": "no measured points"}
    for index, (x, y) in enumerate(pts):
        if y <= target:
            if index == 0:
                return {"cost": x, "interpolated": False, "reached_at_first_point": True}
            x0, y0 = pts[index - 1]
            if y0 == y:
                return {"cost": x, "interpolated": False}
            return {"cost": x0 + (target - y0) * (x - x0) / (y - y0), "interpolated": True, "between": [x0, x]}
    return {"cost": None, "reason": f"target {target:g} not reached within the measured budget (best {min(y for _, y in pts):g})"}


def seed_statistics(values: Sequence[float | None]) -> dict[str, Any]:
    """Variation across independent repetitions (None entries = undefined for that seed)."""

    defined = np.asarray([v for v in values if v is not None], dtype=float)
    out: dict[str, Any] = {"n": len(values), "n_defined": int(len(defined)), "values": list(values)}
    if len(defined):
        out.update(mean=float(defined.mean()), min=float(defined.min()), max=float(defined.max()),
                   std=float(defined.std(ddof=1)) if len(defined) >= 2 else None)
    else:
        out.update(mean=None, min=None, max=None, std=None)
    return out


def curve_analysis(curves: Mapping[str, Sequence[Sequence[Mapping[str, Any]]]], config: EvaluationConfig) -> dict[str, Any]:
    """``curves[arm][rep] = [points]`` with keys cumulative_cpu_core_s, n_labels and the metric."""

    metric = config.metric
    out: dict[str, Any] = {}
    for arm, reps in curves.items():
        arm_out: dict[str, Any] = {"n_repetitions": len(reps), "at_budget_cpu_core_s": {}, "at_budget_labels": {}, "cost_to_reach": {}}
        for budget in config.budgets_cpu_core_s:
            per = [value_at(rep, "cumulative_cpu_core_s", metric, float(budget)) for rep in reps]
            arm_out["at_budget_cpu_core_s"][f"{budget:g}"] = {**seed_statistics([p["value"] for p in per]), "per_rep": per}
        for budget in config.budgets_labels:
            per = [value_at(rep, "n_labels", metric, float(budget)) for rep in reps]
            arm_out["at_budget_labels"][str(budget)] = {**seed_statistics([p["value"] for p in per]), "per_rep": per}
        for target in config.accuracy_targets:
            per_cost = [cost_to_reach(rep, "cumulative_cpu_core_s", metric, float(target)) for rep in reps]
            per_labels = [cost_to_reach(rep, "n_labels", metric, float(target)) for rep in reps]
            arm_out["cost_to_reach"][f"{target:g}"] = {
                "cpu_core_s": {**seed_statistics([p["cost"] for p in per_cost]), "per_rep": per_cost},
                "labels": {**seed_statistics([p["cost"] for p in per_labels]), "per_rep": per_labels},
            }
        finals = [rep[-1].get(metric) if rep else None for rep in reps]
        arm_out["final"] = seed_statistics(finals)
        arm_out["final_cumulative_cpu_core_s"] = seed_statistics([rep[-1].get("cumulative_cpu_core_s") if rep else None for rep in reps])
        out[arm] = arm_out
    return out


EvalFn = Callable[[Any], dict[str, Any]]

__all__ = ["EvaluationConfig", "cost_to_reach", "curve_analysis", "evaluate_frames", "seed_statistics", "site_region", "value_at"]
