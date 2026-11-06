"""Fragment fidelity: fragment construction, metrics with known answers, pre-registration, selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ._helpers import tiny_config
from gmd_active_learning.contracts import RegionRole
from gmd_active_learning.utils.bonds import bonded_neighbors
from tools.experiments.alkane_hotspots.families import generate, load_parents, load_positions, write_generation
from tools.experiments.alkane_hotspots.fidelity import (
    FidelityError,
    aggregate,
    build_fragment,
    control_site,
    core_force_errors,
    load_specs,
    parent_frame,
    prepare,
    report,
    select,
    sweep_settings,
)
from tools.experiments.alkane_hotspots.labels import LabelStore, label_id_for, parent_region
from tools.experiments.alkane_hotspots.manifest import utc_now

SETTINGS_HASH = "synthetic-settings"


def test_core_force_errors_known_answer() -> None:
    parent = np.zeros((5, 3))
    fragment = np.zeros((4, 3))
    fragment_ids = np.array([3, 1, 4, -1])  # last row is a cap
    fragment[0] = [0.03, 0.04, 0.0]  # parent atom 3
    fragment[1] = [0.0, 0.0, 0.05]  # parent atom 1
    fragment[3] = [9.0, 9.0, 9.0]  # cap force is ignored
    result = core_force_errors(parent, fragment, fragment_ids, [1, 3])
    assert result["core_force_rmse"] == pytest.approx(np.sqrt((0.03**2 + 0.04**2 + 0.05**2) / 6))
    assert result["max_core_force_error"] == pytest.approx(0.05)
    assert result["per_atom_error"] == pytest.approx([0.05, 0.05])
    with pytest.raises(ValueError, match="not in the fragment"):
        core_force_errors(parent, fragment, fragment_ids, [0])


@pytest.fixture()
def tiny(tmp_path: Path):
    loaded = tiny_config(tmp_path)
    write_generation(generate(loaded.config), loaded.artifacts_root)
    return loaded


def _no_lone_terminal_atoms(region) -> bool:
    neighbours = bonded_neighbors(region.positions, region.numbers)
    return all(len(neighbours.get(i, [])) >= 1 for i in range(region.n_atoms))


@pytest.mark.parametrize("setting_index", [0, 1, 2])
def test_fragments_are_capped_saturated_and_keep_parent_ids(tiny, setting_index: int) -> None:
    config = tiny.config
    setting = sweep_settings(config)[setting_index]
    parent = next(p for p in load_parents(tiny.artifacts_root) if p["family"] == "torsion")
    positions = load_positions(tiny.artifacts_root, parent)
    frame = parent_frame(parent, positions)
    region, core = build_fragment(frame, parent["perturbation"]["atoms"], setting, config)
    caps = region.indices_with(RegionRole.H_CAP)
    real = [i for i in range(region.n_atoms) if i not in set(caps.tolist())]
    assert len(caps) == region.metadata["h_capping"]["n_cut_bonds"] >= 1 and not region.metadata["h_capping"]["partial"]
    n_c = int(np.sum(region.numbers == 6))
    assert int(np.sum(region.numbers == 1)) == 2 * n_c + 2 or len(set(region.parent_atom_ids[real]) & set(range(n_c))) < n_c  # CnH2n+2 per piece
    assert _no_lone_terminal_atoms(region)
    assert set(core) <= set(region.parent_atom_ids[real].tolist())
    translation = np.asarray(region.metadata["origin_translation"])
    np.testing.assert_allclose(region.positions[real] - translation, positions[region.parent_atom_ids[real]], atol=1e-9)
    assert region.charge == 0 and region.multiplicity == 1
    core_roles = {int(pid) for pid, role in zip(region.parent_atom_ids, region.roles) if role is RegionRole.CORE}
    assert core_roles == set(core)
    if setting["policy"] == "graph":
        assert core == sorted(parent["perturbation"]["atoms"])


def test_control_site_is_far_from_the_perturbation(tiny) -> None:
    for parent in load_parents(tiny.artifacts_root):
        atoms, info = control_site(parent, load_positions(tiny.artifacts_root, parent), tiny.config)
        if parent["family"] in {"torsion", "bend"}:
            assert info["graph_distance"] >= 2 and not set(atoms) & set(parent["perturbation"]["atoms"])
        elif parent["family"] == "contact":  # the contact spans the chain; the control is as far as the chain allows
            assert info["graph_distance"] >= 1
        else:
            assert info["rule"] == "normal_location"


def test_prepare_writes_specs_and_preregistration(tiny) -> None:
    specs = prepare(tiny, "calibration", SETTINGS_HASH)
    calibration = [p for p in load_parents(tiny.artifacts_root) if p["split"] == "calibration"]
    n_sites = sum(2 if p["perturbation"]["atoms"] else 1 for p in calibration)
    assert len(specs) == n_sites * len(sweep_settings(tiny.config))
    assert prepare(tiny, "calibration", SETTINGS_HASH) == specs  # idempotent
    prereg = json.loads((tiny.artifacts_root / "fidelity" / "preregistration.json").read_text())
    assert prereg["payload"]["fidelity"]["core_force_rmse"] == tiny.config.fidelity.core_force_rmse
    for spec in specs:
        assert spec["status"] in {"ready", "rejected"}
        if spec["status"] == "ready":
            assert spec["n_atoms"] <= 3 * spec["chain_length"] + 2 + spec["n_caps"]
            assert spec["diagnostics"]["per_core_atom"] and spec["label_id"].startswith("label-")


def test_changed_targets_are_refused_after_preregistration(tiny, tmp_path: Path) -> None:
    prepare(tiny, "calibration", SETTINGS_HASH)
    looser = tiny_config(tmp_path, fidelity={"core_force_rmse": 0.5})
    looser.artifacts_root = tiny.artifacts_root
    with pytest.raises(FidelityError, match="pre-registered"):
        prepare(looser, "calibration", SETTINGS_HASH)


def test_final_eval_needs_a_frozen_selection(tiny) -> None:
    with pytest.raises(FidelityError, match="selection"):
        prepare(tiny, "final_eval", SETTINGS_HASH)
    with pytest.raises(FidelityError):
        prepare(tiny, "validation", SETTINGS_HASH)


def _write_labels(loaded, split: str, error: float, cost_per_atom: float = 10.0) -> None:
    """Synthetic labels: parent force on an atom = a fixed function of its position (so identical fragments
    shared by two parents get identical forces), fragment forces = parent forces + ``error`` per component."""

    root = loaded.artifacts_root
    store = LabelStore(root)
    parents = {p["parent_id"]: p for p in load_parents(root)}
    parent_forces = {}
    for pid, parent in parents.items():
        if parent["split"] != split:
            continue
        region = parent_region(parent, load_positions(root, parent))
        label_id = label_id_for(region, SETTINGS_HASH)
        forces = np.sin(3.0 * region.positions)
        parent_forces[pid] = forces
        directory = store.label_dir(label_id)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "forces.npy", forces)
        store.put({"label_id": label_id, "kind": "full", "parent_id": pid, "chain_length": parent["chain_length"], "status": "valid",
                   "forces_path": str((directory / "forces.npy").relative_to(root)), "region_path": None, "cost": {"cpu_core_s": 1000.0},
                   "task_id": "t", "fragment": None, "created_at": utc_now()})
    for spec in load_specs(loaded).values():
        if spec["split"] != split or spec["status"] != "ready" or spec["covers_parent"] or store.get(spec["label_id"]):
            continue
        region = json.loads((root / spec["region_path"]).read_text())
        ids = np.asarray(region["parent_atom_ids"])
        forces = np.zeros((len(ids), 3))
        real = ids >= 0
        forces[real] = parent_forces[spec["parent_id"]][ids[real]] + error
        directory = store.label_dir(spec["label_id"])
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "forces.npy", forces)
        (directory / "region.json").write_text(json.dumps(region))
        store.put({"label_id": spec["label_id"], "kind": "fragment", "parent_id": spec["parent_id"], "status": "valid",
                   "forces_path": str((directory / "forces.npy").relative_to(root)), "region_path": str((directory / "region.json").relative_to(root)),
                   "cost": {"cpu_core_s": cost_per_atom * spec["n_atoms"]}, "task_id": "t", "fragment": {}, "created_at": utc_now()})


def test_report_selects_the_cheapest_passing_setting_and_freezes_it(tiny) -> None:
    prepare(tiny, "calibration", SETTINGS_HASH)
    _write_labels(tiny, "calibration", error=0.01)  # rmse 0.01 < 0.03 target everywhere
    result = report(tiny, "calibration")
    assert result["decision"]["status"] == "selected"
    passing = [s for s in result["settings"] if s["passes"]]
    assert len(passing) == len(sweep_settings(tiny.config))
    for summary in result["settings"]:  # a fragment covering the whole parent has zero error
        assert 0.0 < summary["pooled_core_force_rmse"] <= 0.01 + 1e-12 and summary["max_core_force_error"] == pytest.approx(0.01 * np.sqrt(3))
    chosen = result["decision"]["selection"]["setting"]["setting_id"]
    cheapest = min(passing, key=lambda s: s["mean_fragment_cost_core_s"])["setting_id"]
    assert chosen == cheapest
    again = report(tiny, "calibration")
    assert again["decision"]["status"] == "frozen"
    assert (tiny.artifacts_root / "fidelity" / "per_atom_calibration.csv").exists()
    # final_eval now prepares only the frozen setting.
    final = prepare(tiny, "final_eval", SETTINGS_HASH)
    assert final and {s["setting_id"] for s in final} == {chosen}
    _write_labels(tiny, "final_eval", error=0.02)
    evaluation = report(tiny, "final_eval")
    assert evaluation["selection_setting_id"] == chosen and 0.0 < evaluation["evaluation"][0]["pooled_core_force_rmse"] <= 0.02 + 1e-12
    assert json.loads((tiny.artifacts_root / "fidelity" / "selection.json").read_text())["setting"]["setting_id"] == chosen


def test_no_passing_setting_writes_a_failure_record(tiny) -> None:
    prepare(tiny, "calibration", SETTINGS_HASH)
    _write_labels(tiny, "calibration", error=0.2)
    result = report(tiny, "calibration")
    assert result["decision"]["status"] == "failed"
    failure = json.loads((tiny.artifacts_root / "fidelity" / "failure.json").read_text())
    assert failure["targets"]["core_force_rmse"] == tiny.config.fidelity.core_force_rmse  # never relaxed
    diagnosis = failure["diagnosis"]
    assert diagnosis["boundary_distance_trend"] and "missing_environment" in diagnosis and "scf_failures" in diagnosis
    assert not (tiny.artifacts_root / "fidelity" / "selection.json").exists()


def test_incomplete_settings_stay_undecided(tiny) -> None:
    prepare(tiny, "calibration", SETTINGS_HASH)
    result = report(tiny, "calibration")  # nothing labeled yet
    assert result["decision"]["status"] == "undecided"
    assert not (tiny.artifacts_root / "fidelity" / "failure.json").exists()


def test_aggregate_and_select_on_synthetic_rows(tiny) -> None:
    def row(setting, rmse, max_error, cost, outcome="evaluated"):
        return {"spec_id": f"{setting}-{rmse}", "setting_id": setting, "outcome": outcome, "reason": None if outcome == "evaluated" else "x",
                "n_atoms": 30, "n_caps": 1, "site_kind": "perturbation_site", "core_force_rmse": rmse, "max_core_force_error": max_error,
                "sum_squared": rmse**2 * 30, "n_components": 30, "fragment_cost_core_s": cost, "parent_cost_core_s": 1000.0,
                "per_atom": [{"graph_distance_to_cut": 2, "force_error": max_error, "n_excluded_within_environment_radius": 0}]}

    rows = [row("a", 0.01, 0.05, 300.0), row("b", 0.02, 0.05, 100.0), row("c", 0.01, 0.2, 10.0), row("d", 0.01, 0.01, 1.0, outcome="failed")]
    summaries = {s["setting_id"]: s for s in aggregate(rows, tiny.config)}
    assert summaries["a"]["passes"] and summaries["b"]["passes"]
    assert not summaries["c"]["passes"] and not summaries["d"]["passes"] and summaries["d"]["n_failed"] == 1
    assert select(list(summaries.values()))["setting_id"] == "b"
