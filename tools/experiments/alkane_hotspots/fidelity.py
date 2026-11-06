"""Fragment-force fidelity gate: is a capped fragment's force on its core atoms the parent's force?

Pipeline (calibration split first, final_eval once afterwards):

1. ``prepare``: for every parent of the split and every configured site (the perturbation site, and a
   normal-site control far from it) build one fragment per sweep setting with the existing
   extraction (graph hops / radial cluster, ``terminal_atoms: follow_partner``) and H capping (every
   cut C-C bond capped). Parent atom IDs, roles and caps are preserved. The first calibration prepare
   writes ``fidelity/preregistration.json`` (targets and sweep); later changes are refused.
2. ``run``: full parent labels and fragment labels through :mod:`labels` (one settings hash).
3. ``report``: per fragment, ``core_force_rmse = sqrt(mean((F_frag_core - F_parent_core)^2))`` over
   the same original core atoms and components, the max core force-vector error, per-core-atom
   error vs graph/Euclidean distance to the nearest cut bond / cap / excluded atom, sizes, failures
   and measured cost. On calibration, a setting passes when it is complete, has at most
   ``max_failures`` failures, and its pooled RMSE and max error meet the pre-registered targets; the
   cheapest passing setting (measured mean cost) is frozen in ``fidelity/selection.json``. If none
   passes, ``fidelity/failure.json`` records the diagnosis. Thresholds are never changed. final_eval
   uses only the frozen setting and never feeds back into the selection.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from gmd_active_learning.contracts import ExtractedRegion, Frame, RegionRole, content_hash, stable_id
from gmd_active_learning.extraction.config import CappingConfig, ExtractionConfig
from gmd_active_learning.extraction.dispatch import extract_regions, prepare_region
from gmd_active_learning.utils.atomic_io import write_json_atomic
from tools.experiments.alkane_hotspots.checks import pair_distances
from tools.experiments.alkane_hotspots.config import AlkaneConfig, LoadedConfig
from tools.experiments.alkane_hotspots.families import load_parents, load_positions, location_index, torsion_site
from tools.experiments.alkane_hotspots.geometry import AlkaneTopology
from tools.experiments.alkane_hotspots.labels import LabelRequest, LabelStore, label_id_for, parent_region
from tools.experiments.alkane_hotspots.manifest import utc_now

LOGGER = logging.getLogger(__name__)

FIDELITY_DIR = "fidelity"
FRAGMENTS_FILE = "fragments.jsonl"
PREREGISTRATION = "preregistration.json"
SELECTION = "selection.json"
FAILURE = "failure.json"


class FidelityError(RuntimeError):
    """A rule of the fidelity gate would be violated (changed targets, final_eval before selection)."""


# -- settings and sites ------------------------------------------------------------------------------
def sweep_settings(config: AlkaneConfig) -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = []
    for hops in config.fragments.graph_hops:
        settings.append({"setting_id": f"graph-h{hops}", "policy": "graph", "hops": int(hops), "buffer": f"{hops} C-C bonds"})
    for core in config.fragments.cluster.core_radii:
        for buffer in config.fragments.cluster.buffer_radii:
            settings.append(
                {"setting_id": f"cluster-c{core:g}-b{buffer:g}", "policy": "cluster", "core_radius": float(core), "buffer_radius": float(buffer),
                 "buffer": f"{buffer:g} A"}
            )
    return settings


def control_site(parent: dict[str, Any], positions: np.ndarray, config: AlkaneConfig) -> tuple[list[int], dict[str, Any]]:
    """Normal-site control: the torsion-like site (carbons k-1..k+2 + H) farthest from the perturbation.

    For a normal parent it sits at ``fragments.normal_location``. Otherwise k maximizes the minimum
    graph distance to the perturbation atoms (ties: larger Euclidean distance, then smaller k).
    """

    n = int(parent["chain_length"])
    topology = AlkaneTopology(n)
    perturbed = list(parent["perturbation"]["atoms"])
    if not perturbed:
        k = location_index(config.fragments.normal_location, 1, n - 3, n)
        return torsion_site(topology, k), {"location": k, "rule": "normal_location"}
    distances = topology.graph_distances
    euclid = pair_distances(positions)
    best: tuple[Any, ...] | None = None
    for k in range(1, n - 2):
        site = torsion_site(topology, k)
        graph_min = int(distances[np.ix_(site, perturbed)].min())
        euclid_min = float(euclid[np.ix_(site, perturbed)].min())
        key = (graph_min, round(euclid_min, 6), -k)
        if best is None or key > best[0]:
            best = (key, k, site, graph_min, euclid_min)
    assert best is not None
    _, k, site, graph_min, euclid_min = best
    return site, {"location": k, "rule": "farthest_from_perturbation", "graph_distance": graph_min, "euclidean_distance": euclid_min}


def parent_frame(parent: dict[str, Any], positions: np.ndarray) -> Frame:
    topology = AlkaneTopology(int(parent["chain_length"]))
    return Frame(parent["parent_id"], 0, np.arange(topology.n_atoms), topology.numbers, positions, None, (False, False, False))


# -- fragment construction ---------------------------------------------------------------------------
def extraction_config(setting: dict[str, Any], config: AlkaneConfig) -> ExtractionConfig:
    common = {"terminal_atoms": "follow_partner", "vacuum_padding": config.fragments.vacuum_padding, "freeze_boundary": False}
    if setting["policy"] == "graph":
        return ExtractionConfig.model_validate(
            {**common, "mode": "graph", "graph": {"hops": setting["hops"], "radial_shell": False, "covalent_scale": config.geometry.covalent_scale}}
        )
    core, buffer = setting["core_radius"], setting["buffer_radius"]
    return ExtractionConfig.model_validate(
        {**common, "mode": "cluster", "core_radius": core, "extract_radius": core + buffer, "boundary_thickness": min(1.0, buffer),
         "graph": {"covalent_scale": config.geometry.covalent_scale}}
    )


def capping_config(config: AlkaneConfig) -> CappingConfig:
    # Every cut bond is capped whatever the role of its atom; caps are H at the C-H length along the bond.
    return CappingConfig.model_validate(
        {"enabled": True, "allow_core_capping": True, "cap_core_atoms": True, "covalent_scale": config.geometry.covalent_scale,
         "bond_lengths": {"C": config.geometry.ch_bond}, "uncapped_policy": "error", "partial_policy": "error"}
    )


def build_fragment(frame: Frame, site_atoms: list[int], setting: dict[str, Any], config: AlkaneConfig) -> tuple[ExtractedRegion, list[int]]:
    """Extract + cap one fragment; returns (region, core parent IDs). Neutral singlet by construction."""

    region = extract_regions(frame, [site_atoms], extraction_config(setting, config))[0]
    capped = prepare_region(region, frame, capping=capping_config(config))
    data = capped.to_dict()
    if setting["policy"] == "graph":
        core = sorted(int(i) for i in site_atoms)
        boundary = {entry["parent_atom_id"] for entry in capped.metadata.get("caps", [])}
        roles = []
        for parent_id, role in zip(data["parent_atom_ids"], data["roles"]):
            if role == RegionRole.H_CAP.value:
                roles.append(role)
            elif parent_id in core:
                roles.append(RegionRole.CORE.value)
            elif parent_id in boundary:
                roles.append(RegionRole.BOUNDARY.value)
            else:
                roles.append(RegionRole.INNER_BUFFER.value)
        weights = ExtractionConfig().force_weights.as_dict()
        data["roles"] = roles
        data["force_weights"] = [float(weights[role]) for role in roles]
    else:
        core = sorted(int(pid) for pid, role in zip(data["parent_atom_ids"], data["roles"]) if role == RegionRole.CORE.value)
    data["charge"] = 0
    data["multiplicity"] = 1
    data["metadata"] = {**data["metadata"], "fragment_setting": setting, "core_parent_ids": core}
    return ExtractedRegion.from_dict(data), core


def fragment_geometry_diagnostics(region: ExtractedRegion, frame: Frame, core: list[int], config: AlkaneConfig) -> dict[str, Any]:
    """Distances from every core atom to the artificial boundary and to the missing environment."""

    topology = AlkaneTopology(frame.n_atoms // 3)
    real = [i for i, role in enumerate(region.roles) if role is not RegionRole.H_CAP]
    caps = [i for i, role in enumerate(region.roles) if role is RegionRole.H_CAP]
    included = set(int(region.parent_atom_ids[i]) for i in real)
    cut_atoms = sorted({int(entry["parent_atom_id"]) for entry in region.metadata.get("caps", [])})
    cut_bonds = sorted({(int(e["parent_atom_id"]), int(e["broken_partner_atom_id"])) for e in region.metadata.get("caps", [])})
    excluded = sorted(set(range(frame.n_atoms)) - included)
    distances = topology.graph_distances
    parent_xyz = frame.positions
    # Cap positions in the parent frame: region positions minus the rigid origin translation.
    translation = np.asarray(region.metadata.get("origin_translation", [0.0, 0.0, 0.0]))
    cap_xyz = region.positions[caps] - translation if caps else np.zeros((0, 3))
    per_atom = []
    for atom in core:
        graph_cut = int(distances[atom, cut_atoms].min()) if cut_atoms else None
        cap_distance = float(np.linalg.norm(cap_xyz - parent_xyz[atom], axis=1).min()) if caps else None
        missing = np.linalg.norm(parent_xyz[excluded] - parent_xyz[atom], axis=1) if excluded else np.zeros(0)
        per_atom.append(
            {
                "parent_atom_id": int(atom),
                "graph_distance_to_cut": graph_cut,
                "euclidean_distance_to_cap": cap_distance,
                "euclidean_distance_to_excluded": float(missing.min()) if missing.size else None,
                "n_excluded_within_environment_radius": int(np.sum(missing <= config.fidelity.environment_radius)),
            }
        )
    # Cap clashes: any cap closer than cap_clash_distance to an atom other than its own carbon.
    clashes = []
    all_xyz = region.positions
    for local, entry in zip(caps, region.metadata.get("caps", [])):
        own = int(entry["parent_local_index"])
        d = np.linalg.norm(all_xyz - all_xyz[local], axis=1)
        d[[local, own]] = np.inf
        if float(d.min()) < config.fragments.cap_clash_distance:
            clashes.append({"cap_local_index": int(local), "min_distance": float(d.min())})
    return {
        "cut_bonds": [list(b) for b in cut_bonds],
        "n_excluded_atoms": len(excluded),
        "per_core_atom": per_atom,
        "cap_clashes": clashes,
    }


def _spec_id(parent_id: str, site_kind: str, setting_id: str) -> str:
    return stable_id("frag", parent_id, site_kind, setting_id)


def fidelity_dir(loaded: LoadedConfig) -> Path:
    return loaded.artifacts_root / FIDELITY_DIR


def load_specs(loaded: LoadedConfig) -> dict[str, dict[str, Any]]:
    path = fidelity_dir(loaded) / FRAGMENTS_FILE
    if not path.exists():
        return {}
    return {r["spec_id"]: r for r in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def _save_specs(loaded: LoadedConfig, specs: dict[str, dict[str, Any]]) -> None:
    path = fidelity_dir(loaded) / FRAGMENTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(specs[key], sort_keys=True) + "\n" for key in sorted(specs)), encoding="utf-8")
    temporary.replace(path)


# -- pre-registration ---------------------------------------------------------------------------------
def preregistration_payload(config: AlkaneConfig) -> dict[str, Any]:
    return {
        "fidelity": config.fidelity.model_dump(mode="json"),
        "fragments": config.fragments.model_dump(mode="json"),
        "cp2k": config.cp2k.model_dump(mode="json", exclude={"executable", "data_dir", "launcher", "omp_threads", "mpi_ranks"}),
    }


def ensure_preregistration(loaded: LoadedConfig, calibration_parents: list[str]) -> dict[str, Any]:
    """Write the pre-registration once; afterwards refuse any change of targets, sweep or settings."""

    path = fidelity_dir(loaded) / PREREGISTRATION
    payload = preregistration_payload(loaded.config)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing["payload_hash"] != content_hash(payload):
            raise FidelityError(
                f"{path}: the fidelity targets, fragment sweep or CP2K settings differ from the pre-registered ones. "
                "Thresholds are fixed before running and are never changed; use a new artifacts_root for a new study."
            )
        return existing
    record = {
        "created_at": utc_now(),
        "config_hash": loaded.config.config_hash(),
        "payload": payload,
        "payload_hash": content_hash(payload),
        "calibration_parents": sorted(calibration_parents),
    }
    write_json_atomic(path, record)
    return record


def load_selection(loaded: LoadedConfig) -> dict[str, Any] | None:
    path = fidelity_dir(loaded) / SELECTION
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# -- prepare --------------------------------------------------------------------------------------------
def prepare(loaded: LoadedConfig, split: str, settings_hash: str) -> list[dict[str, Any]]:
    """Fragment specs (and region files) for one split; idempotent."""

    config = loaded.config
    root = loaded.artifacts_root
    parents = [p for p in load_parents(root) if p["split"] == split]
    if split == "calibration":
        ensure_preregistration(loaded, [p["parent_id"] for p in parents])
        settings = sweep_settings(config)
    elif split == "final_eval":
        selection = load_selection(loaded)
        if selection is None or selection.get("status") != "selected":
            raise FidelityError("final_eval fragments need a frozen calibration selection (fidelity/selection.json) first")
        ensure_preregistration(loaded, [])
        settings = [selection["setting"]]
    else:
        raise FidelityError(f"fragment fidelity runs on calibration or final_eval, not {split!r}")
    specs = load_specs(loaded)
    out: list[dict[str, Any]] = []
    for parent in parents:
        positions = load_positions(root, parent)
        frame = parent_frame(parent, positions)
        full_label = label_id_for(parent_region(parent, positions), settings_hash)
        sites: list[tuple[str, list[int], dict[str, Any]]] = []
        for site_kind in config.fragments.sites:
            if site_kind == "perturbation_site":
                if parent["perturbation"]["atoms"]:
                    sites.append((site_kind, sorted(parent["perturbation"]["atoms"]), {"location": parent["perturbation"]["location"]}))
            else:
                atoms, info = control_site(parent, positions, config)
                sites.append((site_kind, atoms, info))
        for site_kind, site_atoms, site_info in sites:
            for setting in settings:
                spec_id = _spec_id(parent["parent_id"], site_kind, setting["setting_id"])
                spec: dict[str, Any] = {
                    "spec_id": spec_id,
                    "parent_id": parent["parent_id"],
                    "chain_length": parent["chain_length"],
                    "family": parent["family"],
                    "split": split,
                    "site_kind": site_kind,
                    "site_atoms": site_atoms,
                    "site_info": site_info,
                    "setting": setting,
                    "setting_id": setting["setting_id"],
                    "parent_label_id": full_label,
                }
                try:
                    region, core = build_fragment(frame, site_atoms, setting, config)
                except Exception as exc:  # noqa: BLE001 - recorded as a failed fragment of this setting
                    spec.update({"status": "rejected", "reason": f"extraction failed: {type(exc).__name__}: {exc}", "label_id": None})
                    specs[spec_id] = spec
                    out.append(spec)
                    continue
                diagnostics = fragment_geometry_diagnostics(region, frame, core, config)
                n_caps = len(region.indices_with(RegionRole.H_CAP))
                covers = n_caps == 0 and region.n_atoms == frame.n_atoms
                region_path = fidelity_dir(loaded) / "regions" / f"{spec_id}.json"
                write_json_atomic(region_path, region.to_dict())
                spec.update(
                    {
                        "region_path": str(region_path.relative_to(root)),
                        "core_parent_ids": core,
                        "n_atoms": region.n_atoms,
                        "n_core": len(core),
                        "n_caps": n_caps,
                        "covers_parent": covers,
                        "diagnostics": diagnostics,
                        "label_id": full_label if covers else label_id_for(region, settings_hash),
                        "status": "ready",
                        "reason": None,
                    }
                )
                if diagnostics["cap_clashes"]:
                    spec.update({"status": "rejected", "reason": f"cap clash ({len(diagnostics['cap_clashes'])} caps closer than "
                                 f"{config.fragments.cap_clash_distance} A to another atom)"})
                specs[spec_id] = spec
                out.append(spec)
    _save_specs(loaded, specs)
    return out


def fragment_dict(spec: dict[str, Any]) -> dict[str, Any]:
    """The ``fragment`` field of a label record (shared data model)."""

    setting = spec["setting"]
    size = {"hops": setting["hops"]} if setting["policy"] == "graph" else {"core_radius": setting["core_radius"], "buffer_radius": setting["buffer_radius"]}
    return {
        "policy": setting["policy"],
        **size,
        "buffer": setting["buffer"],
        "setting_id": setting["setting_id"],
        "site_kind": spec["site_kind"],
        "core_parent_ids": spec["core_parent_ids"],
        "n_atoms": spec["n_atoms"],
        "n_core": spec["n_core"],
        "n_caps": spec["n_caps"],
    }


def label_requests(loaded: LoadedConfig, split: str) -> list[LabelRequest]:
    """Full-parent requests for the split's parents, then fragment requests for its ready specs."""

    root = loaded.artifacts_root
    parents = {p["parent_id"]: p for p in load_parents(root) if p["split"] == split}
    specs = [s for s in load_specs(loaded).values() if s["split"] == split]
    selection = load_selection(loaded) if split == "final_eval" else None
    if selection is not None:
        specs = [s for s in specs if s["setting_id"] == selection["setting"]["setting_id"]]
    requests = [
        LabelRequest("full", pid, int(p["chain_length"]), parent_region(p, load_positions(root, p))) for pid, p in sorted(parents.items())
    ]
    for spec in sorted(specs, key=lambda s: (s["parent_id"], s["site_kind"], s["setting_id"])):
        if spec["status"] != "ready" or spec["covers_parent"]:
            continue
        region = ExtractedRegion.from_dict(json.loads((root / spec["region_path"]).read_text(encoding="utf-8")))
        requests.append(LabelRequest("fragment", spec["parent_id"], int(spec["chain_length"]), region, fragment_dict(spec), [spec["spec_id"]]))
    return requests


# -- metrics --------------------------------------------------------------------------------------------
def core_force_errors(
    parent_forces: np.ndarray,
    fragment_forces: np.ndarray,
    fragment_parent_ids: np.ndarray,
    core_parent_ids: list[int],
) -> dict[str, Any]:
    """Errors on the same original core atoms. Parent forces are indexed by parent atom ID."""

    parent_forces = np.asarray(parent_forces, dtype=float)
    fragment_forces = np.asarray(fragment_forces, dtype=float)
    lookup = {int(pid): row for row, pid in enumerate(np.asarray(fragment_parent_ids).tolist()) if int(pid) >= 0}
    missing = [pid for pid in core_parent_ids if pid not in lookup]
    if missing:
        raise ValueError(f"core atoms {missing} are not in the fragment")
    rows = [lookup[pid] for pid in core_parent_ids]
    diff = fragment_forces[rows] - parent_forces[list(core_parent_ids)]
    norms = np.linalg.norm(diff, axis=1)
    return {
        "core_force_rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_core_force_error": float(norms.max()),
        "per_atom_error": norms.tolist(),
        "sum_squared": float(np.sum(diff * diff)),
        "n_components": int(diff.size),
        "parent_core_force_rms": float(np.sqrt(np.mean(parent_forces[list(core_parent_ids)] ** 2))),
    }


def _load_forces(root: Path, record: dict[str, Any] | None) -> np.ndarray | None:
    if record is None or record.get("status") != "valid" or not record.get("forces_path"):
        return None
    return np.load(root / record["forces_path"])


def evaluate_specs(loaded: LoadedConfig, split: str) -> list[dict[str, Any]]:
    """Per-spec outcome: evaluated (with metrics), failed (with reason) or pending."""

    root = loaded.artifacts_root
    store = LabelStore(root)
    rows = []
    for spec in sorted((s for s in load_specs(loaded).values() if s["split"] == split), key=lambda s: s["spec_id"]):
        row = {k: spec.get(k) for k in ("spec_id", "parent_id", "chain_length", "family", "site_kind", "setting_id", "n_atoms", "n_core", "n_caps", "covers_parent")}
        parent_record = store.get(spec["parent_label_id"])
        fragment_record = store.get(spec["label_id"]) if spec.get("label_id") else None
        row["parent_cost_core_s"] = ((parent_record or {}).get("cost") or {}).get("cpu_core_s")
        row["fragment_cost_core_s"] = 0.0 if spec.get("covers_parent") else ((fragment_record or {}).get("cost") or {}).get("cpu_core_s")
        if spec["status"] != "ready":
            row.update({"outcome": "failed", "reason": spec["reason"]})
        elif fragment_record is None:
            row.update({"outcome": "pending", "reason": "fragment not labeled yet"})
        elif fragment_record["status"] == "budget_skipped":
            row.update({"outcome": "pending", "reason": "fragment label budget_skipped: " + "; ".join(fragment_record.get("reasons", []))})
        elif fragment_record["status"] != "valid":
            row.update({"outcome": "failed", "reason": f"fragment label {fragment_record['status']}: " + "; ".join(fragment_record.get("reasons", []))[:500]})
        elif parent_record is None or parent_record["status"] != "valid":
            status = "missing" if parent_record is None else parent_record["status"]
            row.update({"outcome": "pending", "reason": f"parent label {status}"})
        else:
            parent_forces = _load_forces(root, parent_record)
            fragment_forces = _load_forces(root, fragment_record)
            if spec.get("covers_parent"):
                # The fragment is the whole parent: its label is the parent label (parent atom order).
                fragment_ids = np.arange(len(parent_forces))
            else:
                region = json.loads((root / fragment_record["region_path"]).read_text(encoding="utf-8"))
                fragment_ids = np.asarray(region["parent_atom_ids"])
            metrics = core_force_errors(parent_forces, fragment_forces, fragment_ids, spec["core_parent_ids"])
            per_atom = []
            for info, error in zip(spec["diagnostics"]["per_core_atom"], metrics["per_atom_error"]):
                per_atom.append({**info, "force_error": error})
            metrics["per_atom"] = per_atom
            metrics.pop("per_atom_error")
            row.update({"outcome": "evaluated", "reason": None, **metrics})
        rows.append(row)
    return rows


def _bin_label(value: float | None, bins: list[float]) -> str:
    if value is None:
        return "no_cut"
    for low, high in zip(bins[:-1], bins[1:]):
        if low <= value < high:
            return f"[{low:g},{high:g})"
    return f">={bins[-1]:g}"


def aggregate(rows: list[dict[str, Any]], config: AlkaneConfig) -> list[dict[str, Any]]:
    targets = config.fidelity
    by_setting: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_setting[row["setting_id"]].append(row)
    summaries = []
    for setting_id, items in sorted(by_setting.items()):
        evaluated = [r for r in items if r["outcome"] == "evaluated"]
        failed = [r for r in items if r["outcome"] == "failed"]
        pending = [r for r in items if r["outcome"] == "pending"]
        sum_sq = sum(r["sum_squared"] for r in evaluated)
        n_comp = sum(r["n_components"] for r in evaluated)
        pooled = float(np.sqrt(sum_sq / n_comp)) if n_comp else None
        max_error = max((r["max_core_force_error"] for r in evaluated), default=None)
        costs = [r["fragment_cost_core_s"] for r in evaluated if r["fragment_cost_core_s"] is not None]
        parent_costs = [r["parent_cost_core_s"] for r in evaluated if r["parent_cost_core_s"] is not None]
        bins: dict[str, list[float]] = defaultdict(list)
        for r in evaluated:
            for atom in r["per_atom"]:
                bins[_bin_label(atom["graph_distance_to_cut"], targets.distance_bins)].append(atom["force_error"])
        by_site: dict[str, list[float]] = defaultdict(list)
        for r in evaluated:
            by_site[r["site_kind"]].append(r["core_force_rmse"])
        complete = not pending and bool(items)
        passes = (
            complete
            and len(failed) <= targets.max_failures
            and pooled is not None
            and pooled <= targets.core_force_rmse
            and max_error is not None
            and max_error <= targets.max_core_force_error
        )
        summaries.append(
            {
                "setting_id": setting_id,
                "n_specs": len(items),
                "n_evaluated": len(evaluated),
                "n_failed": len(failed),
                "n_pending": len(pending),
                "complete": complete,
                "pooled_core_force_rmse": pooled,
                "max_core_force_error": max_error,
                "per_spec_rmse_max": max((r["core_force_rmse"] for r in evaluated), default=None),
                "rmse_by_site_kind": {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in by_site.items()},
                "error_vs_graph_distance_to_cut": {k: {"n": len(v), "mean": float(np.mean(v)), "max": float(np.max(v))} for k, v in sorted(bins.items())},
                "mean_n_atoms": float(np.mean([r["n_atoms"] for r in items if r["n_atoms"]])) if any(r["n_atoms"] for r in items) else None,
                "mean_n_caps": float(np.mean([r["n_caps"] for r in items if r["n_caps"] is not None])) if any(r["n_caps"] is not None for r in items) else None,
                "mean_fragment_cost_core_s": float(np.mean(costs)) if costs else None,
                "mean_parent_cost_core_s": float(np.mean(parent_costs)) if parent_costs else None,
                "failures": [{"spec_id": r["spec_id"], "reason": r["reason"]} for r in failed],
                "passes": bool(passes),
            }
        )
    return summaries


def diagnose_failure(rows: list[dict[str, Any]], summaries: list[dict[str, Any]], config: AlkaneConfig) -> dict[str, Any]:
    """Boundary-distance trend, missing environment and SCF/label failures when no setting passes."""

    evaluated = [r for r in rows if r["outcome"] == "evaluated"]
    trend: dict[str, list[float]] = defaultdict(list)
    environment = []
    for r in evaluated:
        for atom in r["per_atom"]:
            trend[_bin_label(atom["graph_distance_to_cut"], config.fidelity.distance_bins)].append(atom["force_error"])
            environment.append((atom["n_excluded_within_environment_radius"], atom["force_error"]))
    env = np.asarray(environment, dtype=float) if environment else np.zeros((0, 2))
    correlation = None
    if len(env) > 2 and np.std(env[:, 0]) > 0 and np.std(env[:, 1]) > 0:
        correlation = float(np.corrcoef(env[:, 0], env[:, 1])[0, 1])
    worst = sorted(evaluated, key=lambda r: -r["max_core_force_error"])[:10]
    label_failures = [{"spec_id": r["spec_id"], "setting_id": r["setting_id"], "reason": r["reason"]} for r in rows if r["outcome"] == "failed"]
    scf = [f for f in label_failures if "SCF" in (f["reason"] or "") or "scf" in (f["reason"] or "")]
    return {
        "boundary_distance_trend": {k: {"n": len(v), "mean": float(np.mean(v)), "max": float(np.max(v))} for k, v in sorted(trend.items())},
        "missing_environment": {
            "environment_radius": config.fidelity.environment_radius,
            "correlation_excluded_count_vs_error": correlation,
            "worst_specs": [
                {"spec_id": r["spec_id"], "setting_id": r["setting_id"], "family": r["family"], "site_kind": r["site_kind"],
                 "max_core_force_error": r["max_core_force_error"],
                 "max_excluded_within_radius": max(a["n_excluded_within_environment_radius"] for a in r["per_atom"])}
                for r in worst
            ],
        },
        "label_failures": label_failures,
        "scf_failures": scf,
        "closest_setting": min(
            (s for s in summaries if s["pooled_core_force_rmse"] is not None), key=lambda s: s["pooled_core_force_rmse"], default=None
        ),
    }


def _write_per_atom_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["spec_id", "parent_id", "family", "site_kind", "setting_id", "parent_atom_id", "force_error", "graph_distance_to_cut",
              "euclidean_distance_to_cap", "euclidean_distance_to_excluded", "n_excluded_within_environment_radius", "n_atoms", "n_caps"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row["outcome"] != "evaluated":
                continue
            for atom in row["per_atom"]:
                writer.writerow({**{k: row[k] for k in ("spec_id", "parent_id", "family", "site_kind", "setting_id", "n_atoms", "n_caps")}, **atom})


def select(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Cheapest passing setting by measured mean fragment cost (ties: fewer atoms, then id)."""

    passing = [s for s in summaries if s["passes"]]
    if not passing:
        return None
    return min(passing, key=lambda s: (s["mean_fragment_cost_core_s"] if s["mean_fragment_cost_core_s"] is not None else float("inf"),
                                       s["mean_n_atoms"] or 0.0, s["setting_id"]))


def report(loaded: LoadedConfig, split: str) -> dict[str, Any]:
    config = loaded.config
    directory = fidelity_dir(loaded)
    preregistration = ensure_preregistration(loaded, [])
    rows = evaluate_specs(loaded, split)
    summaries = aggregate(rows, config)
    _write_per_atom_csv(directory / f"per_atom_{split}.csv", rows)
    with (directory / f"metrics_{split}.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    result: dict[str, Any] = {
        "split": split,
        "created_at": utc_now(),
        "targets": config.fidelity.model_dump(mode="json"),
        "preregistration_hash": preregistration["payload_hash"],
        "settings": summaries,
        "n_specs": len(rows),
    }
    if split == "calibration":
        frozen = load_selection(loaded)
        if frozen is not None:
            result["decision"] = {"status": "frozen", "selection": frozen}
        else:
            chosen = select(summaries)
            incomplete = [s["setting_id"] for s in summaries if not s["complete"]]
            if chosen is not None:
                selection = {
                    "status": "selected",
                    "created_at": utc_now(),
                    "setting": next(s["setting"] for s in load_specs(loaded).values() if s["setting_id"] == chosen["setting_id"]),
                    "summary": chosen,
                    "objective": config.fidelity.selection_objective,
                    "targets": config.fidelity.model_dump(mode="json"),
                    "preregistration_hash": preregistration["payload_hash"],
                    "incomplete_settings_at_selection": incomplete,
                    "calibration_spec_ids": sorted(r["spec_id"] for r in rows),
                }
                write_json_atomic(directory / SELECTION, selection)
                result["decision"] = {"status": "selected", "selection": selection}
            elif incomplete:
                result["decision"] = {"status": "undecided", "reason": "no complete setting passes yet", "incomplete_settings": incomplete}
            else:
                failure = {
                    "status": "no_setting_passes",
                    "created_at": utc_now(),
                    "targets": config.fidelity.model_dump(mode="json"),
                    "summaries": summaries,
                    "diagnosis": diagnose_failure(rows, summaries, config),
                    "note": "Targets were pre-registered and are not relaxed. The fragment approach fails the gate for this sweep.",
                }
                write_json_atomic(directory / FAILURE, failure)
                result["decision"] = {"status": "failed", "failure": failure}
    else:
        selection = load_selection(loaded)
        result["selection_setting_id"] = None if selection is None else selection["setting"]["setting_id"]
        result["evaluation"] = summaries
    write_json_atomic(directory / f"report_{split}.json", result)
    return result


__all__ = [
    "FidelityError",
    "aggregate",
    "build_fragment",
    "control_site",
    "core_force_errors",
    "diagnose_failure",
    "ensure_preregistration",
    "evaluate_specs",
    "fragment_dict",
    "label_requests",
    "load_selection",
    "load_specs",
    "prepare",
    "report",
    "select",
    "sweep_settings",
]
