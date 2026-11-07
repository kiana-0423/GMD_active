"""Protocol freezing, reference (DFT) labels per stage, and the fragment-fidelity audit.

* ``freeze_protocol``: hashes the protocol document and every configuration file together with the
  generated structure sets; every later command refuses to run when a hashed file changed
  (``require_frozen``). A changed protocol is a new benchmark (new name and artifacts root).
* ``label_stage``: full-system CP2K labels of a stage's structures through alpha's ``LabelRunner``
  (one CP2K configuration, one settings hash, resumable, measured costs). Each stage has its own
  measured CPU-hour budget; a job whose estimate would exceed it is not started. final_test labels
  are produced in the sealed root and are never read outside :mod:`evaluate`.
* ``fidelity_audit``: the frozen fragment setting (copied with provenance from the pilot's gate) is
  re-checked on the development challenge structures: capped fragments around the challenged site,
  DFT core forces vs the full parent's. A family whose audit fails may not use local labels.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gmd_active_learning.core.exceptions import ConfigurationError, WorkflowError

from tools.experiments.alkane_hotspots.cost import BudgetLedger
from tools.experiments.alkane_hotspots.families import load_positions
from tools.experiments.alkane_hotspots.geometry import AlkaneTopology
from tools.experiments.alkane_hotspots.manifest import utc_now

from .common import read_json, write_json
from .config import LoadedBenchmark
from .generate import load_records

LOGGER = logging.getLogger(__name__)

FREEZE_FILE = "protocol_freeze.json"
STAGES = ("initial_model", "development", "repair_val", "final_test")


# ---------------------------------------------------------------------------------------------- protocol freeze


def structure_hashes(loaded: LoadedBenchmark) -> dict[str, Any]:
    from gmd_active_learning.contracts.ids import file_sha256

    out = {}
    for name, root in (("public", loaded.root), ("sealed", loaded.sealed_root)):
        path = root / "parents.jsonl"
        out[name] = None if not path.is_file() else file_sha256(path)
    return out


def freeze_protocol(loaded: LoadedBenchmark, *, note: str = "") -> dict[str, Any]:
    """Write ``protocol_freeze.json`` once; an identical re-freeze is a no-op, a different one is refused."""

    current = {"protocol": loaded.protocol_hash(), "structures": structure_hashes(loaded), "benchmark": loaded.config.name}
    if current["structures"]["public"] is None:
        raise WorkflowError("generate the structures before freezing the protocol (the structure sets are part of the freeze)")
    path = loaded.root / FREEZE_FILE
    if path.is_file():
        frozen = read_json(path)
        if (frozen["protocol"]["sha256"], frozen["structures"]) != (current["protocol"]["sha256"], current["structures"]):
            raise WorkflowError(f"{path} records a different protocol or structure set; a changed protocol is a new benchmark "
                                "(new name and artifacts root), never an in-place revision")
        return frozen
    record = {**current, "frozen_at": utc_now(), "note": note,
              "statement": "Frozen before any confirmatory experiment of this benchmark. All pilot results (alkane_hotspots v1, v2, v2b, "
                           "v2c) predate this protocol and are exploratory evidence; this protocol was not registered before them."}
    write_json(path, record)
    return record


def require_frozen(loaded: LoadedBenchmark) -> dict[str, Any]:
    path = loaded.root / FREEZE_FILE
    if not path.is_file():
        raise WorkflowError(f"the protocol is not frozen ({path} missing): run `freeze-protocol` after `generate`")
    frozen = read_json(path)
    current = loaded.protocol_hash()
    if frozen["protocol"]["sha256"] != current["sha256"]:
        changed = [name for name, item in current["files"].items() if frozen["protocol"]["files"].get(name, {}).get("sha256") != item["sha256"]]
        raise WorkflowError(f"protocol files changed since the freeze: {changed}; a changed protocol is a new benchmark")
    if frozen["structures"] != structure_hashes(loaded):
        raise WorkflowError("the structure sets changed since the freeze")
    return frozen


# ---------------------------------------------------------------------------------------------- stage labels


def stage_structures(loaded: LoadedBenchmark, stage: str, *, qualification_stage: int | None = None) -> tuple[Any, list[dict[str, Any]]]:
    """(alpha loaded config for the stage's root, structures) of a labeling stage."""

    config = loaded.config
    if stage == "final_test":
        return loaded.sealed_alpha(), load_records(loaded.sealed_root)
    records = load_records(loaded.root)
    if stage == "initial_model":
        stages = config.initial_model.stages
        if qualification_stage is None:
            raise ConfigurationError("initial_model labels need an explicit qualification stage (--qualification-stage); "
                                     "a default would silently select the expanded training set")
        index = int(qualification_stage)
        if not 0 <= index < len(stages):
            raise ConfigurationError(f"qualification stage {index} does not exist ({len(stages)} stages)")
        bases = sorted({r["base_id"] for r in records if r["partition"] == "id_train"})[: stages[index].id_train_bases]
        chosen = [r for r in records if (r["partition"] == "id_train" and r["base_id"] in bases) or r["partition"] == "id_val"]
    elif stage == "development":
        chosen = [r for r in records if r["partition"] == "development"]
    elif stage == "repair_val":
        chosen = [r for r in records if r["partition"] == "repair_val"]
    else:
        raise ConfigurationError(f"unknown labeling stage {stage!r} (one of {STAGES})")
    return loaded.alpha, chosen


def cost_estimate(loaded: LoadedBenchmark, store: Any, alpha: Any, kind: str, chain_length: int | None) -> tuple[float, str]:
    """Measured mean of this store; without a measurement yet, the measured mean of ``plan.timing_sources``
    (never an assumed number); only if neither exists, alpha's conservative upper bound."""

    estimate, source = BudgetLedger(store.all(), alpha.config.budgets, alpha.config.cp2k.cores).estimate(kind, chain_length)
    if "no measurement" not in source:
        return estimate, source
    from .plan import measured_timings

    timings = measured_timings(loaded)
    key = "full_label_core_s" if kind == "full" else "fragment_label_core_s"
    if timings[key].get("n"):
        return float(timings[key]["mean"]), f"measured mean of {timings[key]['n']} {kind} label(s) in plan.timing_sources"
    return estimate, source


def _full_records(store: Any, parent_ids: set[str]) -> list[dict[str, Any]]:
    return [r for r in store.all() if r.get("kind") == "full" and r.get("parent_id") in parent_ids]


def stage_spent_core_s(store: Any, parent_ids: set[str], kinds: Sequence[str] = ("full",)) -> float:
    return float(sum((r.get("cost") or {}).get("cpu_core_s") or 0.0 for r in store.all()
                     if r.get("kind") in kinds and r.get("parent_id") in parent_ids and r.get("cost")))


def label_stage(loaded: LoadedBenchmark, stage: str, *, qualification_stage: int | None = None, max_wall_s: float | None = None,
                echo: Any = print) -> dict[str, Any]:
    """Full-system labels of a stage, serially, within the stage's measured CPU-hour budget."""

    from tools.experiments.alkane_hotspots.labels import LabelRequest, LabelRunner, LabelStore, build_service, cp2k_provenance, label_id_for, parent_region

    from .budget import BudgetRefused, GuardedService, build_guard

    require_frozen(loaded)
    alpha, structures = stage_structures(loaded, stage, qualification_stage=qualification_stage)
    root = Path(alpha.artifacts_root)
    budget_core_s = float(getattr(loaded.config.budgets, stage)) * 3600.0
    guard = build_guard(loaded)
    raw_service = build_service(alpha)
    service = GuardedService(raw_service, guard, kind_of=lambda work: "full")
    store = LabelStore(root)
    runner = LabelRunner(service, store, alpha, cp2k_provenance(raw_service, smoke_only=alpha.config.cp2k.smoke_only), echo=echo)
    ids = {r["parent_id"] for r in structures}
    if stage == "initial_model":  # every qualification stage draws on one budget
        _, everything = stage_structures(loaded, stage, qualification_stage=len(loaded.config.initial_model.stages) - 1)
        ids_budget = {r["parent_id"] for r in everything}
    else:
        ids_budget = ids
    if stage == "initial_model":
        q0_bases = guard.ledger.q0_bases()
    outcome: dict[str, Any] = {"stage": stage, "qualification_stage": qualification_stage, "root": "sealed" if stage == "final_test" else "public",
                               "n_structures": len(structures), "completed": [], "reused": [], "failed": [], "pending": [], "not_started": []}
    for record in sorted(structures, key=lambda r: r["parent_id"]):
        region = parent_region(record, load_positions(root, record))
        label_id = label_id_for(region, service.settings_hash)
        existing = store.get(label_id)
        if existing is not None and existing["status"] == "valid":
            outcome["reused"].append(record["parent_id"])
            continue
        scope: dict[str, Any] = {"stage": stage}
        if stage == "initial_model":
            scope["q"] = 0 if record["partition"] == "id_val" or record["base_id"] in q0_bases else 1
        try:
            guard.require_room(scope, "full", label=record["parent_id"])
        except BudgetRefused as exc:
            outcome["not_started"].append({"parent_id": record["parent_id"], "reason": str(exc)})
            continue
        summary = runner.run([LabelRequest("full", record["parent_id"], int(record["chain_length"]), region)], max_wall_s=max_wall_s)
        store = runner.store
        for key in ("completed", "failed", "pending"):
            if getattr(summary, key):
                outcome[key].append(record["parent_id"])
        if summary.skipped:
            outcome["not_started"].append({"parent_id": record["parent_id"], "reason": "alpha budget (labels/budget_skipped record)"})
        if summary.pending:
            break
    outcome["measured_stage_core_h"] = stage_spent_core_s(LabelStore(root), ids_budget) / 3600.0
    outcome["ledger"] = guard.ledger.snapshot()
    outcome["budget_core_h"] = budget_core_s / 3600.0
    outcome["valid"] = sorted(r["parent_id"] for r in _full_records(LabelStore(root), ids) if r["status"] == "valid")
    outcome["missing"] = sorted(ids - set(outcome["valid"]))
    outcome["finished_at"] = utc_now()
    name = stage if qualification_stage is None else f"{stage}_q{qualification_stage}"
    write_json(loaded.root / "reference_labels" / f"{name}.json", outcome)
    return outcome


def full_label_for(store: Any, record: dict[str, Any]) -> dict[str, Any] | None:
    valid = [r for r in store.all() if r.get("kind") == "full" and r.get("parent_id") == record["parent_id"] and r.get("status") == "valid"]
    return valid[0] if valid else None


def label_forces(root: Path, label: dict[str, Any]) -> np.ndarray:
    return np.load(Path(root) / label["forces_path"])


# ---------------------------------------------------------------------------------------------- fidelity audit


def install_fidelity_selection(loaded: LoadedBenchmark) -> dict[str, Any]:
    """Copy the frozen fragment setting into ``<root>/fidelity`` with provenance (identical bytes or refuse)."""

    source = loaded.repo_path(loaded.config.fidelity.selection_dir)
    target = loaded.root / "fidelity"
    target.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in ("selection.json", "preregistration.json"):
        if not (source / name).is_file():
            raise ConfigurationError(f"fidelity source {source / name} missing: the frozen setting must exist before the benchmark uses it")
        if (target / name).is_file() and (target / name).read_bytes() != (source / name).read_bytes():
            raise WorkflowError(f"{target / name} differs from {source / name}")
        shutil.copyfile(source / name, target / name)
        copied[name] = str(source / name)
    selection = read_json(target / "selection.json")
    if selection.get("status") not in ("selected", "frozen"):
        raise WorkflowError(f"{source}: the fidelity gate did not select a setting ({selection.get('status')})")
    write_json(target / "TRANSFER_PROVENANCE.json", {"source": copied, "setting": selection["setting"],
                                                     "note": "frozen setting of the pilot's C16 calibration gate, reused unchanged; its coverage of "
                                                             "this benchmark's challenge geometries is audited on development structures"})
    return selection


def audit_specs(loaded: LoadedBenchmark, selection: dict[str, Any]) -> list[dict[str, Any]]:
    """Fragments around each development challenge's site: the gate's site shape and the closed loop's."""

    from tools.experiments.alkane_hotspots.closed_loop import hotspot_site
    from tools.experiments.alkane_hotspots.fidelity import build_fragment, parent_frame

    setting = selection["setting"]
    per_challenge = loaded.config.plan.fidelity_audit_fragments_per_challenge
    specs = []
    for record in load_records(loaded.root):
        if record["partition"] != "development" or record["kind"] != "challenge":
            continue
        topology = AlkaneTopology(int(record["chain_length"]))
        frame = parent_frame(record, load_positions(loaded.root, record))
        site = [int(a) for a in record["perturbation"]["atoms"]]
        _, loop_site = hotspot_site(topology, site, None)
        for shape, atoms in (("gate_site", site), ("closed_loop_site", loop_site))[:per_challenge]:
            region, core = build_fragment(frame, atoms, setting, loaded.alpha.config)
            covers = region.n_atoms == topology.n_atoms and not any(r.value == "h_cap" for r in region.roles)
            specs.append({"parent_id": record["parent_id"], "family": record["family"], "shape": shape, "region": region, "core": core,
                          "covers_parent": covers})
    return specs


def fidelity_audit(loaded: LoadedBenchmark, *, run_labels: bool = True, max_wall_s: float | None = None, echo: Any = print) -> dict[str, Any]:
    from tools.experiments.alkane_hotspots.labels import LabelRequest, LabelRunner, LabelStore, build_service, cp2k_provenance, label_id_for

    require_frozen(loaded)
    selection = install_fidelity_selection(loaded)
    setting = selection["setting"]
    from .budget import BudgetRefused, GuardedService, build_guard

    guard = build_guard(loaded)
    raw_service = build_service(loaded.alpha)
    service = GuardedService(raw_service, guard, kind_of=lambda work: "fragment")
    store = LabelStore(loaded.root)
    runner = LabelRunner(service, store, loaded.alpha, cp2k_provenance(raw_service, smoke_only=loaded.alpha.config.cp2k.smoke_only), echo=echo)
    budget_core_s = loaded.config.budgets.fidelity_audit * 3600.0
    specs = audit_specs(loaded, selection)
    dev_ids = {s["parent_id"] for s in specs}
    rows = []
    for spec in specs:
        label_id = label_id_for(spec["region"], service.settings_hash)
        record = store.get(label_id)
        if (record is None or record["status"] != "valid") and run_labels:
            try:
                guard.require_room({"stage": "fidelity_audit"}, "fragment", label=f"audit:{spec['parent_id']}:{spec['shape']}")
                room = True
            except BudgetRefused as exc:
                room, refusal = False, str(exc)
            if room:
                fragment = {"policy": setting["policy"], "setting_id": setting["setting_id"], "site_kind": f"audit:{spec['shape']}",
                            "core_parent_ids": spec["core"], "n_atoms": spec["region"].n_atoms, "n_core": len(spec["core"]),
                            "buffer": setting.get("buffer")}
                runner.run([LabelRequest("fragment", spec["parent_id"], loaded.config.chain_length, spec["region"], fragment, [f"audit:{spec['shape']}"])],
                           max_wall_s=max_wall_s)
                store = runner.store
                record = store.get(label_id)
            else:
                rows.append({**_row(spec), "status": "not_started", "reason": refusal})
                continue
        rows.append(_audit_row(loaded, store, spec, record))
    targets = loaded.alpha.config.fidelity
    families = summarize_audit(rows, loaded.config.challenges.enabled(), targets.core_force_rmse, targets.max_core_force_error)
    report = {"created_at": utc_now(), "setting": setting, "targets": {"core_force_rmse": targets.core_force_rmse,
                                                                         "max_core_force_error": targets.max_core_force_error},
              "families": families, "fragments": rows,
              "statement": "DFT fidelity of capped fragments on development challenge structures (static). It does not establish the "
                           "trained model's fragment-vs-parent consistency, which is measured separately."}
    write_json(loaded.root / "fidelity" / "audit.json", report)
    return report


def summarize_audit(rows: list[dict[str, Any]], families: Sequence[str], rmse_target: float, max_target: float) -> dict[str, Any]:
    """Per family: pooled core RMSE and maximum over the evaluated, non-trivial fragments; passes only if complete."""

    out: dict[str, Any] = {}
    for family in families:
        mine = [r for r in rows if r["family"] == family]
        trivial = [r for r in mine if r.get("covers_parent")]
        mine = [r for r in mine if not r.get("covers_parent")]  # a fragment equal to the parent is exact by construction: no evidence
        evaluated = [r for r in mine if r["status"] == "evaluated"]
        complete = bool(mine) and len(evaluated) == len(mine)
        if evaluated:
            sq = sum(r["sum_squared"] for r in evaluated)
            n = sum(r["n_components"] for r in evaluated)
            rmse, worst = float(np.sqrt(sq / n)), max(r["max_core_force_error"] for r in evaluated)
        else:
            rmse = worst = None
        passes = complete and rmse is not None and rmse <= rmse_target and worst <= max_target
        out[family] = {"n_fragments": len(mine), "n_trivial_whole_parent": len(trivial), "n_evaluated": len(evaluated), "complete": complete,
                       "pooled_core_force_rmse": rmse, "max_core_force_error": worst, "passes": bool(passes)}
    return out


def _row(spec: dict[str, Any]) -> dict[str, Any]:
    return {"parent_id": spec["parent_id"], "family": spec["family"], "shape": spec["shape"], "n_atoms": spec["region"].n_atoms,
            "n_core": len(spec["core"]), "covers_parent": bool(spec.get("covers_parent"))}


def _audit_row(loaded: LoadedBenchmark, store: Any, spec: dict[str, Any], record: dict[str, Any] | None) -> dict[str, Any]:
    from gmd_active_learning.contracts import ExtractedRegion

    from tools.experiments.alkane_hotspots.fidelity import core_force_errors

    row = _row(spec)
    if record is None or record.get("status") != "valid":
        return {**row, "status": "pending" if record is None else record.get("status"), "reasons": (record or {}).get("reasons")}
    parent_label = full_label_for(store, {"parent_id": spec["parent_id"]})
    if parent_label is None:
        return {**row, "status": "pending", "reasons": ["no valid full label of the parent (label the development stage first)"]}
    labeled = ExtractedRegion.from_dict(read_json(loaded.root / record["region_path"]))
    errors = core_force_errors(label_forces(loaded.root, parent_label), label_forces(loaded.root, record), np.asarray(labeled.parent_atom_ids), spec["core"])
    return {**row, "status": "evaluated", "label_id": record["label_id"], "cpu_core_s": (record.get("cost") or {}).get("cpu_core_s"),
            **{k: errors[k] for k in ("core_force_rmse", "max_core_force_error", "sum_squared", "n_components", "parent_core_force_rms")}}


def audit_status(loaded: LoadedBenchmark) -> dict[str, Any] | None:
    path = loaded.root / "fidelity" / "audit.json"
    return read_json(path) if path.is_file() else None


__all__ = ["FREEZE_FILE", "STAGES", "audit_status", "fidelity_audit", "freeze_protocol", "full_label_for", "install_fidelity_selection", "label_forces",
           "label_stage", "require_frozen", "stage_structures"]
