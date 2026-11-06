"""Empirical receptive field of the demo's GMD_SGT model: how a core atom's force depends on far atoms.

Fragment extraction removes every atom outside the fragment. Whether that can leave the core force
unchanged depends on the model's receptive field, which is measured here rather than assumed.

Nominal values (``nominal_receptive_field``):

* message passing builds edges within ``local_cutoff`` (r_c). After ``n_blocks`` (T) blocks an atom's
  energy E_j depends on atoms at most T hops away in the r_c graph, i.e. within T * r_c (Euclidean).
* the force F_i = -dE/dr_i = -sum_j dE_j/dr_i sums over every j whose E_j depends on r_i, so F_i
  depends on atoms at most **2 T hops** away, i.e. within **2 T r_c**. The often-quoted T * r_c is the
  receptive field of the per-atom energy, not of the force.
* any ``long_range_type`` other than ``none`` (attention, electrostatics) is global within a structure:
  the nominal receptive field is infinite and every atom can influence every force.

Measurements, all on the core atom's force F_c (eV/A), with the model in float64 and eval mode:

* ``perturb``: displace one atom k by ``displacement`` A along a seeded random unit vector;
  influence_k = |F_c(perturbed) - F_c|.
* ``delete``: remove atom k; influence_k = |F_c(without k) - F_c|.
* radii are reported for three thresholds: ``tolerance`` (absolute, eV/A), ``relative_tolerance`` x the
  largest single-atom influence (scale-free; use it for random-init models, whose force magnitudes are
  arbitrary) and ``structural_tolerance`` (just above float64 round-off: any dependence at all).
* ``truncate``: keep only atoms within Euclidean radius R of the core (no caps); error(R) = |F_c(R) - F_c|.
  This is what an uncapped extraction does to the model's own prediction.

``influence_radius`` (per measurement) = the largest distance of an atom whose influence is >=
``tolerance``: every atom farther away changes F_c by less than the tolerance. For truncation it is
the smallest measured R from which on every larger measured R has error < tolerance (``null`` when no
measured radius qualifies). Distances are reported as Euclidean (A), covalent-bond hops and r_c-graph
hops.

A freshly initialised GMD_SGT model has a zero-initialised readout layer (all forces exactly 0). For a
random-init measurement, ``random_init.reinit_zero_weights_std`` re-draws every all-zero weight matrix
from N(0, std) with the given seed (recorded); this measures the architecture's dependency structure,
not a trained model's magnitudes. A trained checkpoint is used as is.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from gmd_active_learning.core.exceptions import ConfigurationError

from .ground_truth import bond_adjacency, graph_hops

LOGGER = logging.getLogger(__name__)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RandomInit(_Strict):
    seed: int = 0
    reinit_zero_weights_std: float | None = Field(0.5, gt=0.0, description="null keeps zero-initialised weights (forces may all be 0)")


class StructureSelector(_Strict):
    parent_id: str | None = None
    split: str = "calibration"
    family: str = "normal"
    index: int = Field(0, ge=0, description="n-th matching parent (file order) when parent_id is null")


class ReceptiveFieldConfig(_Strict):
    enabled: bool = True
    model_source: Literal["random_init", "checkpoint"] = "random_init"
    checkpoint: str | None = Field(None, description="GMD_SGT checkpoint (relative to the artifacts root) for model_source=checkpoint")
    random_init: RandomInit = Field(default_factory=RandomInit)
    structure: StructureSelector = Field(default_factory=StructureSelector)
    core_atoms: Literal["central_carbon"] | list[int] = "central_carbon"
    measurements: list[Literal["perturb", "delete", "truncate"]] = Field(default_factory=lambda: ["perturb", "delete", "truncate"])
    displacement: float = Field(0.05, gt=0.0, description="A")
    seed: int = 0
    tolerance: float = Field(1.0e-4, gt=0.0, description="eV/A on |dF_core|")
    relative_tolerance: float = Field(1.0e-3, gt=0.0, description="fraction of the largest single-atom influence")
    structural_tolerance: float = Field(1.0e-12, gt=0.0, description="eV/A; above float64 round-off: any dependence at all")
    truncation_radii: list[float] = Field(default_factory=lambda: [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0])
    bond_scale: float = Field(1.2, gt=0.0)


# ---------------------------------------------------------------------------------------------- model


def nominal_receptive_field(model_config: Mapping[str, Any]) -> dict[str, Any]:
    n_blocks = int(model_config.get("n_blocks", 4))
    cutoff = float(model_config.get("local_cutoff", 5.0))
    long_range = str(model_config.get("long_range_type", "invariant_attention"))
    global_term = long_range != "none"
    return {
        "n_blocks": n_blocks,
        "local_cutoff": cutoff,
        "long_range_type": long_range,
        "energy_receptive_field_A": n_blocks * cutoff,
        "force_receptive_field_local_A": 2 * n_blocks * cutoff,
        "force_receptive_field_hops": 2 * n_blocks,
        "global": global_term,
        "force_receptive_field_A": math.inf if global_term else 2 * n_blocks * cutoff,
        "note": "force depends on atoms within 2 x n_blocks x local_cutoff (per-atom energy: n_blocks x local_cutoff); "
                "a long-range term is global",
    }


def build_model(model_config: Mapping[str, Any] | None, config: ReceptiveFieldConfig, *, checkpoint: Path | None = None,
                model_type: str = "UnifiedEquivariantMLIP") -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """(float64 eval-mode model, its model config, provenance) through GMD_SGT's own factory."""

    import torch
    from gmd_sgt.models.factory import instantiate_model, load_model_from_checkpoint

    if config.model_source == "checkpoint":
        if checkpoint is None or not Path(checkpoint).is_file():
            raise ConfigurationError(f"receptive_field.model_source=checkpoint needs an existing checkpoint, got {checkpoint}")
        state, model = load_model_from_checkpoint(checkpoint)
        resolved = dict(state["model_config"])
        provenance = {"source": "checkpoint", "checkpoint": str(checkpoint)}
    else:
        if not model_config:
            raise ConfigurationError("receptive_field.model_source=random_init needs the demo's GMD_SGT model section")
        resolved = {key: value for key, value in dict(model_config).items() if key != "type"}
        torch.manual_seed(config.random_init.seed)
        model = instantiate_model(model_type, resolved)
        reinitialised = []
        std = config.random_init.reinit_zero_weights_std
        if std is not None:
            generator = torch.Generator().manual_seed(config.random_init.seed + 1)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if parameter.ndim >= 2 and float(parameter.detach().abs().max()) == 0.0:
                        parameter.copy_(torch.randn(parameter.shape, generator=generator) * std)
                        reinitialised.append(name)
        provenance = {"source": "random_init", "seed": config.random_init.seed, "reinit_zero_weights_std": std,
                      "reinitialised_parameters": reinitialised}
    model = model.double().eval()
    return model, resolved, provenance


def model_forces(model: Any, numbers: np.ndarray, positions: np.ndarray) -> np.ndarray:
    import torch

    n = len(numbers)
    out = model(species=torch.as_tensor(np.asarray(numbers), dtype=torch.long),
                positions=torch.as_tensor(np.asarray(positions, dtype=float), dtype=torch.float64),
                batch=torch.zeros(n, dtype=torch.long), compute_forces=True)
    return out["forces"].detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------------------------- measurement


def central_carbon(numbers: np.ndarray, positions: np.ndarray) -> int:
    carbons = np.flatnonzero(np.asarray(numbers) == 6)
    if len(carbons) == 0:
        raise ConfigurationError("core_atoms=central_carbon but the structure has no carbon")
    centroid = positions[carbons].mean(axis=0)
    return int(carbons[np.argmin(np.linalg.norm(positions[carbons] - centroid, axis=1))])


def cutoff_adjacency(positions: np.ndarray, cutoff: float) -> list[list[int]]:
    distance = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    return [np.flatnonzero((row < cutoff) & (row > 0)).tolist() for row in distance]


def _influence_radius(distances: np.ndarray, influence: np.ndarray, tolerance: float) -> float:
    above = influence >= tolerance
    return float(np.max(distances[above])) if np.any(above) else 0.0


def measure_core(model: Any, numbers: np.ndarray, positions: np.ndarray, core: int, config: ReceptiveFieldConfig,
                 nominal: Mapping[str, Any]) -> dict[str, Any]:
    numbers = np.asarray(numbers, dtype=np.int64)
    positions = np.asarray(positions, dtype=float)
    n = len(numbers)
    reference = model_forces(model, numbers, positions)[core]
    distance = np.linalg.norm(positions - positions[core], axis=1)
    bond_hops = graph_hops(bond_adjacency(numbers, positions, scale=config.bond_scale), [core])
    cutoff_hops = graph_hops(cutoff_adjacency(positions, float(nominal["local_cutoff"])), [core])
    others = np.asarray([k for k in range(n) if k != core], dtype=np.int64)
    rng = np.random.default_rng(config.seed)
    rows = [{"atom": int(k), "Z": int(numbers[k]), "distance_A": float(distance[k]), "bond_hops": int(bond_hops[k]),
             "cutoff_graph_hops": int(cutoff_hops[k])} for k in others]
    summary: dict[str, Any] = {"core_atom": int(core), "core_force": reference.tolist(), "core_force_norm": float(np.linalg.norm(reference))}
    if "perturb" in config.measurements:
        influence = []
        for row, k in zip(rows, others):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            moved = positions.copy()
            moved[k] += config.displacement * direction
            value = float(np.linalg.norm(model_forces(model, numbers, moved)[core] - reference))
            row["perturb"] = value
            influence.append(value)
        summary.update(_radii("perturb", distance[others], np.asarray(influence), config))
        summary["perturb_max_influence_beyond_nominal"] = _beyond(distance[others], np.asarray(influence), nominal)
    if "delete" in config.measurements:
        influence = []
        for row, k in zip(rows, others):
            keep = np.r_[np.arange(k), np.arange(k + 1, n)]
            new_core = core if core < k else core - 1
            value = float(np.linalg.norm(model_forces(model, numbers[keep], positions[keep])[new_core] - reference))
            row["delete"] = value
            influence.append(value)
        summary.update(_radii("delete", distance[others], np.asarray(influence), config))
        summary["delete_max_influence_beyond_nominal"] = _beyond(distance[others], np.asarray(influence), nominal)
    if "truncate" in config.measurements:
        truncation = []
        for radius in sorted(config.truncation_radii):
            keep = np.flatnonzero(distance <= radius)
            new_core = int(np.flatnonzero(keep == core)[0])
            error = float(np.linalg.norm(model_forces(model, numbers[keep], positions[keep])[new_core] - reference))
            truncation.append({"radius_A": float(radius), "n_atoms_kept": int(len(keep)), "error": error})
        summary["truncation"] = truncation
        converged = None
        for index in range(len(truncation)):
            if all(item["error"] < config.tolerance for item in truncation[index:]):
                converged = truncation[index]["radius_A"]
                break
        summary["truncation_converged_radius_A"] = converged
    summary["atoms"] = rows
    return summary


def _radii(kind: str, distances: np.ndarray, influence: np.ndarray, config: ReceptiveFieldConfig) -> dict[str, float]:
    peak = float(np.max(influence)) if len(influence) else 0.0
    return {f"{kind}_influence_radius_A": _influence_radius(distances, influence, config.tolerance),
            f"{kind}_relative_influence_radius_A": _influence_radius(distances, influence, config.relative_tolerance * peak) if peak > 0 else 0.0,
            f"{kind}_structural_radius_A": _influence_radius(distances, influence, config.structural_tolerance),
            f"{kind}_max_influence": peak}


def _beyond(distances: np.ndarray, influence: np.ndarray, nominal: Mapping[str, Any]) -> float | None:
    limit = float(nominal["force_receptive_field_local_A"])
    mask = distances > limit
    return float(np.max(influence[mask])) if np.any(mask) else None


def run_receptive_field(model: Any, model_config: Mapping[str, Any], numbers: np.ndarray, positions: np.ndarray,
                        config: ReceptiveFieldConfig, *, provenance: Mapping[str, Any] | None = None,
                        structure: Mapping[str, Any] | None = None) -> dict[str, Any]:
    nominal = nominal_receptive_field(model_config)
    numbers = np.asarray(numbers)
    positions = np.asarray(positions, dtype=float)
    cores = [central_carbon(numbers, positions)] if config.core_atoms == "central_carbon" else list(config.core_atoms)
    for core in cores:
        if not 0 <= core < len(numbers):
            raise ConfigurationError(f"core atom {core} outside the structure ({len(numbers)} atoms)")
    results = [measure_core(model, numbers, positions, core, config, nominal) for core in cores]
    keys = [f"{kind}_{name}" for kind in ("perturb", "delete") for name in ("influence_radius_A", "relative_influence_radius_A",
                                                                             "structural_radius_A") if f"{kind}_{name}" in results[0]]
    measured = {key: max(item[key] for item in results) for key in keys}
    conv = [item.get("truncation_converged_radius_A") for item in results if "truncation" in item]
    if conv:
        measured["truncation_converged_radius_A"] = None if any(c is None for c in conv) else max(conv)
    span = float(np.max(np.linalg.norm(positions[:, None] - positions[None], axis=2)))
    return {
        "status": "executed",
        "tolerance_eV_per_A": config.tolerance,
        "relative_tolerance": config.relative_tolerance,
        "structural_tolerance_eV_per_A": config.structural_tolerance,
        "displacement_A": config.displacement,
        "nominal": {key: (None if isinstance(value, float) and math.isinf(value) else value) for key, value in nominal.items()},
        "measured": measured,
        "structure": {**dict(structure or {}), "n_atoms": int(len(numbers)), "max_interatomic_distance_A": span},
        "model": {"config": dict(model_config), **dict(provenance or {})},
        "interpretation": _interpret(nominal, measured, span),
        "cores": results,
    }


def _interpret(nominal: Mapping[str, Any], measured: Mapping[str, Any], span: float) -> str:
    radius = max([value for key, value in measured.items() if key.endswith("structural_radius_A") and value is not None] or [0.0])
    if nominal["global"]:
        return (f"long-range term {nominal['long_range_type']!r} is global: atoms up to {radius:.2f} A (structure span {span:.2f} A) "
                "change the core force at all (structural radius); extraction cannot reproduce the full-system model force exactly")
    return (f"local model: structural influence radius {radius:.2f} A (any dependence above round-off) vs nominal force receptive field "
            f"{nominal['force_receptive_field_local_A']:.2f} A (2 x n_blocks x local_cutoff)")


__all__ = ["ReceptiveFieldConfig", "build_model", "measure_core", "model_forces", "nominal_receptive_field", "run_receptive_field"]
