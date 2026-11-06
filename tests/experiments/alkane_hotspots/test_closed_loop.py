"""Closed loop (part gamma): labeling-mode gate, extraction, budget, held-out validation, energy-jump
bookkeeping, provenance chain, observables and configuration. Offline; the real run is ``closed-loop-run``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ._helpers import CONFIGS, ROOT, tiny_config
from gmd_active_learning.contracts import DatasetVersion, EventRecord, ExtractedRegion, Frame, LabelTask, RegionRole
from gmd_active_learning.core.exceptions import ConfigurationError, DataContractError
from gmd_active_learning.datasets.config import RoleWeights
from gmd_active_learning.workflow.store import WorkflowStore
from tools.experiments.alkane_hotspots.closed_loop import (
    AlkaneRegionExtractor,
    BudgetedLabeler,
    BudgetExceededError,
    ClosedLoopConfig,
    FrozenEvalSection,
    HeldOutValidationStore,
    LabelingDecision,
    MDSection,
    chain_observables,
    decide_labeling_mode,
    hotspot_site,
    isolated_frame,
    load_closed_loop,
    model_change_jumps,
    provenance_chain,
    summarize_observables,
)
from tools.experiments.alkane_hotspots.geometry import AlkaneTopology, apply_torsion, build_all_trans


# ---------------------------------------------------------------------------------------------- labeling mode


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


SETTING = {"setting_id": "graph-h2", "policy": "graph", "hops": 2, "buffer": "2 C-C bonds"}


def test_local_labels_need_a_passed_gate(tmp_path: Path) -> None:
    fidelity = tmp_path / "fidelity"
    undecided = decide_labeling_mode(fidelity, "auto")
    assert undecided.mode == "full" and undecided.gate_status == "undecided"
    with pytest.raises(ConfigurationError, match="local refused.*undecided"):
        decide_labeling_mode(fidelity, "local")

    _write(fidelity / "failure.json", {"status": "no_setting_passes"})
    failed = decide_labeling_mode(fidelity, "auto")
    assert failed.mode == "full" and failed.gate_status == "failed" and failed.setting is None
    with pytest.raises(ConfigurationError, match="local refused.*failed"):
        decide_labeling_mode(fidelity, "local")

    _write(fidelity / "selection.json", {"status": "selected", "setting": SETTING})
    with pytest.raises(ConfigurationError, match="inconsistent"):
        decide_labeling_mode(fidelity, "auto")
    (fidelity / "failure.json").unlink()
    passed = decide_labeling_mode(fidelity, "auto")
    assert passed.mode == "local" and passed.setting == SETTING
    assert decide_labeling_mode(fidelity, "local").mode == "local"
    assert decide_labeling_mode(fidelity, "full").mode == "full"  # full labels may always be requested


def test_selection_without_setting_is_not_a_pass(tmp_path: Path) -> None:
    _write(tmp_path / "selection.json", {"status": "selected"})
    assert decide_labeling_mode(tmp_path, "auto").mode == "full"


# ---------------------------------------------------------------------------------------------- extraction


def _event(frame: Frame, hotspots: list[int], scores: np.ndarray | None = None) -> EventRecord:
    metadata = {"trigger_hotspot_atom_ids": hotspots}
    if scores is not None:
        metadata["trigger_atom_scores"] = scores.tolist()
    return EventRecord(event_id="evt-1", run_id="run", trigger_step=frame.step, trigger_steps=[frame.step], pre_frames=[], trigger_frame=frame,
                       post_frames=[], post_frames_requested=0, complete=True, hotspot_atom_ids=hotspots, trigger_reasons=["committee"],
                       max_score=7.5, truncation_reason=None, model_version="model-a", metadata=metadata)


def _periodic_frame(topology: AlkaneTopology, positions: np.ndarray, length: float = 40.0, step: int = 30) -> Frame:
    shifted = positions - positions.mean(0) + np.array([1.0, length / 2, length / 2])  # crosses x = 0
    return Frame(run_id="run", step=step, atom_ids=np.arange(topology.n_atoms), numbers=topology.numbers, positions=np.mod(shifted, length),
                 cell=np.eye(3) * length, pbc=(True, True, True), model_version="model-a")


def test_full_mode_labels_the_whole_isolated_molecule() -> None:
    topology = AlkaneTopology(12)
    positions = build_all_trans(12)
    frame = _periodic_frame(topology, positions)
    extractor = AlkaneRegionExtractor(decision=LabelingDecision("full", "auto", "failed", "gate failed", None), alpha_config=None,
                                      topology=topology, force_weights=RoleWeights(), full_force_weight=1.0, merge_radius=3.0, parent_id="P")
    [(region, score)] = extractor.extract(_event(frame, [5, 6]))
    assert score == 7.5 and region.mode == "full_system" and region.pbc == (False, False, False) and region.cell is None
    assert region.event_id == "evt-1" and region.parent_step == 30 and region.model_version == "model-a"
    assert region.metadata["md_parent_id"] == "P" and region.metadata["kind"] == "full"
    distances = np.linalg.norm(region.positions[:, None] - region.positions[None], axis=-1)
    reference = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    np.testing.assert_allclose(distances, reference, atol=1e-8)  # whole molecule, not split by the boundary
    assert all(role is RegionRole.CORE for role in region.roles) and np.all(region.force_weights == 1.0)


def test_local_mode_cuts_the_gate_fragment_around_each_hotspot(tmp_path: Path) -> None:
    loaded = tiny_config(tmp_path)
    topology = AlkaneTopology(12)
    positions, _ = apply_torsion(build_all_trans(12), topology, (5, 6), 120.0)
    frame = _periodic_frame(topology, positions)
    scores = np.zeros(topology.n_atoms)
    scores[6] = 9.0
    decision = LabelingDecision("local", "auto", "passed", "gate passed", SETTING)
    extractor = AlkaneRegionExtractor(decision=decision, alpha_config=loaded.config, topology=topology,
                                      force_weights=RoleWeights(core=1.0, inner_buffer=0.3), full_force_weight=1.0, merge_radius=3.0, parent_id="P")
    regions = extractor.extract(_event(frame, [6, 7], scores))
    assert len(regions) == 1  # one hotspot group -> one fragment
    region = regions[0][0]
    assert region.metadata["site_carbon"] == 6 and region.event_id == "evt-1" and region.pbc == (False, False, False)
    core = set(region.metadata["core_parent_ids"])
    assert {5, 6, 7, 8} <= core  # carbons k-1..k+2 around the top-scoring carbon
    caps = [i for i, role in enumerate(region.roles) if role is RegionRole.H_CAP]
    assert caps and all(region.force_weights[i] == 0.0 for i in caps)
    with pytest.raises(ConfigurationError):
        AlkaneRegionExtractor(decision=LabelingDecision("local", "local", "passed", "", None), alpha_config=None, topology=topology,
                              force_weights=RoleWeights(), full_force_weight=1.0, merge_radius=3.0, parent_id="P")


def test_hotspot_site_maps_hydrogens_to_carbons_and_clamps() -> None:
    topology = AlkaneTopology(8)
    hydrogen_of_c7 = topology.hydrogens_of[7][0]
    k, site = hotspot_site(topology, [hydrogen_of_c7], None)
    assert k == 5 and {4, 5, 6, 7} <= set(site)  # clamped so that k+2 is the last carbon
    k, _ = hotspot_site(topology, [0], None)
    assert k == 1


def test_isolated_frame_rejects_wrong_species() -> None:
    topology = AlkaneTopology(6)
    frame = _periodic_frame(topology, build_all_trans(6))
    frame.numbers = np.where(frame.numbers == 6, 7, frame.numbers)
    with pytest.raises(DataContractError, match="species"):
        isolated_frame(frame, topology)


# ---------------------------------------------------------------------------------------------- budget


class _Service:
    def __init__(self, root: Path) -> None:
        self.task_root = root


class _Inner:
    def __init__(self, root: Path) -> None:
        self.service = _Service(root)
        self.submitted: list[str] = []

    def submit(self, task: LabelTask) -> LabelTask:
        self.submitted.append(task.task_id)
        return task


def _task(root: Path, task_id: str, region: ExtractedRegion, *, done: bool, wall: float | None = None) -> LabelTask:
    work = root / task_id
    work.mkdir(parents=True)
    (work / "region.json").write_text(json.dumps(region.to_dict()))
    attempts = []
    if done:
        attempts = [{"index": 1, "stage": "sp", "kind": "initial", "dir": "attempts/01-sp", "submitted_at": "2026-11-07T00:00:00+00:00",
                     "finished_at": f"2026-11-07T00:00:{int(wall or 0):02d}+00:00"}]
    task = LabelTask(task_id, region.region_id, "cp2k", "h", str(work), metadata={"attempts": attempts, "stage": "done" if done else "sp"})
    from gmd_active_learning.labeling.task_store import save_task

    save_task(task)
    return task


def _full_region(n: int = 8, region_id: str = "r") -> ExtractedRegion:
    topology = AlkaneTopology(n)
    count = topology.n_atoms
    return ExtractedRegion(region_id, "full_system", "run", 0, "evt", topology.numbers, build_all_trans(n), None, (False,) * 3, np.arange(count),
                           np.zeros((count, 3), int), [RegionRole.CORE] * count, np.ones(count), np.zeros(count, bool), [0])


def test_budget_refuses_jobs_beyond_the_measured_cpu_budget(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    inner = _Inner(root)
    labeler = BudgetedLabeler(inner, max_core_s=100.0, cores=2, max_atoms={"full": 30, "fragment": 30},
                              estimate=lambda kind: (40.0, "measured mean"), ledger_path=tmp_path / "ledger.json")
    _task(root, "done", _full_region(region_id="a"), done=True, wall=20.0)  # 20 s x 2 cores = 40 core-s spent
    allowed = _task(root, "new", _full_region(region_id="b"), done=False)
    labeler.submit(allowed)  # 40 spent + 40 in flight? no: "new" has no attempts yet -> 40 + 40 = 80 <= 100
    assert inner.submitted == ["new"]
    running = _task(root, "running", _full_region(region_id="c"), done=False)
    running.metadata["attempts"] = [{"index": 1, "stage": "sp", "kind": "initial", "dir": "attempts/01-sp"}]
    from gmd_active_learning.labeling.task_store import save_task

    save_task(running)
    refused = _task(root, "refused", _full_region(region_id="d"), done=False)
    with pytest.raises(BudgetExceededError, match="max_cpu_core_hours"):
        labeler.submit(refused)  # 40 spent + 40 running + 40 estimate = 120 > 100
    assert inner.submitted == ["new"]
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert [entry["allowed"] for entry in ledger] == [True, False] and ledger[1]["spent_core_s"] == pytest.approx(40.0)
    small = BudgetedLabeler(inner, max_core_s=1e9, cores=2, max_atoms={"full": 10, "fragment": 10}, estimate=lambda kind: (1.0, "x"),
                            ledger_path=tmp_path / "l2.json")
    with pytest.raises(BudgetExceededError, match="atoms exceeds"):
        small.submit(refused)


# ---------------------------------------------------------------------------------------------- held-out validation


class _SplitConfig:
    def __init__(self, enabled: bool) -> None:
        self.split = type("S", (), {"enabled": enabled})()


class _Store:
    def __init__(self, enabled: bool = False) -> None:
        self.config = _SplitConfig(enabled)
        self.verified = 0

    def get(self, version: str) -> DatasetVersion:
        return DatasetVersion(version, "/x", "sha", 3, 3, 0, None, ("s1", "s2", "s3"), "t", {"train_file": "/x/train.extxyz", "all_file": "/x/all"})

    def verify(self, record: DatasetVersion) -> None:
        self.verified += 1


def test_held_out_validation_is_the_fixed_validation_version() -> None:
    validation = DatasetVersion("val-v", "/v", "vsha", 2, 2, 0, None, ("v1", "v2"), "t", {"all_file": "/v/all.extxyz"})
    vstore = _Store()
    store = HeldOutValidationStore(_Store(), vstore, validation)
    version = store.get("train-v")
    assert version.metadata["val_file"] == "/v/all.extxyz" and version.metadata["validation_version"] == "val-v"
    assert version.n_val == 2 and version.n_train == 3 and vstore.verified == 1
    leaking = HeldOutValidationStore(_Store(), vstore, DatasetVersion("val-v", "/v", "s", 1, 1, 0, None, ("s2",), "t", {"all_file": "/v"}))
    with pytest.raises(DataContractError, match="validation samples"):
        leaking.get("train-v")
    with pytest.raises(ConfigurationError, match="split.enabled"):
        HeldOutValidationStore(_Store(enabled=True), vstore, validation)


# ---------------------------------------------------------------------------------------------- energy jumps


def _frame(step: int, positions: np.ndarray, energy: float | None) -> Frame:
    return Frame(run_id="r", step=step, atom_ids=np.arange(len(positions)), numbers=np.ones(len(positions), int), positions=positions,
                 cell=np.eye(3) * 20, pbc=(True,) * 3, energy=energy)


def test_model_change_jump_at_fixed_coordinates() -> None:
    positions = np.random.default_rng(0).normal(size=(4, 3))
    segments = [
        {"segment_index": 0, "model_id": "A", "status": "completed", "end_step": 100},
        {"segment_index": 1, "model_id": "B", "status": "completed", "end_step": 200},
        {"segment_index": 2, "model_id": "B", "status": "completed", "end_step": 300},
        {"segment_index": 3, "model_id": "C", "status": "completed", "end_step": 400},
    ]
    frames = {
        0: [_frame(90, positions + 1, -9.0), _frame(100, positions, -10.0)],
        1: [_frame(100, positions, -10.6), _frame(110, positions + 0.1, -10.5), _frame(200, positions + 0.2, -10.4)],
        2: [_frame(200, positions + 0.2, -10.4), _frame(300, positions + 0.3, -10.3)],
        3: [_frame(300, positions + 0.35, -11.0)],  # coordinates differ: no jump is claimed
    }
    calls = []

    def reevaluate(model_id: str, frame: Frame) -> float:
        calls.append((model_id, frame.step))
        return {"A": -10.0, "B": -10.6, "C": -11.0}[model_id]

    records = model_change_jumps(segments, lambda seg: frames[seg["segment_index"]], reevaluate)
    assert [(r["old_model_id"], r["new_model_id"]) for r in records] == [("A", "B"), ("B", "C")]
    first, second = records
    assert first["kind"] == "model_change" and "not integration drift" in first["note"]
    assert first["same_coordinates"] and first["boundary_step"] == 100 and first["jump_ev"] == pytest.approx(-0.6)
    assert first["reevaluated"]["jump_ev"] == pytest.approx(-0.6) and calls[:2] == [("A", 100), ("B", 100)]
    assert second["same_coordinates"] is False and second["jump_ev"] is None and second["max_position_difference"] == pytest.approx(0.05)


def test_no_frames_gives_an_explicit_empty_record() -> None:
    segments = [{"segment_index": 0, "model_id": "A", "status": "completed", "end_step": 10},
                {"segment_index": 1, "model_id": "B", "status": "failed", "end_step": None},
                {"segment_index": 2, "model_id": "B", "status": "completed", "end_step": 20}]
    [record] = model_change_jumps(segments, lambda seg: [])
    assert record["jump_ev"] is None and record["same_coordinates"] is None and record["new_segment"] == 2


# ---------------------------------------------------------------------------------------------- provenance


def test_provenance_chain_reaches_the_parent_structure(tmp_path: Path) -> None:
    store = WorkflowStore(tmp_path / "workflow.sqlite")
    store.ensure_run("run", config_hash="h", initial_model_id=None, stage="sampling")
    region = _full_region(region_id="region-1")
    region.event_id, region.parent_step, region.model_version = "evt-1", 130, "model-0"
    task_dir = tmp_path / "cp2k_tasks" / "task-1"
    task_dir.mkdir(parents=True)
    (task_dir / "region.json").write_text(json.dumps(region.to_dict()))
    summary = {"event_id": "evt-1", "run_id": "run", "trigger_step": 130, "trigger_steps": [130], "pre_frame_steps": [120], "post_frame_steps": [140],
               "post_frames_requested": 1, "pre_frames_requested": 1, "complete": True, "truncation_reason": None, "hotspot_atom_ids": [3],
               "trigger_reasons": ["committee"], "max_score": 8.0, "model_version": "model-0", "metadata": {}}
    store.record_event("run", 0, summary, path=str(tmp_path / "events" / "evt-1"))
    store.upsert_task("task-1", candidate_id="region-1", run_id="run", cycle=0, state="ingested", work_dir=str(task_dir), job={"job_id": "4242"})
    store.record_sample(sample_id="sample-new", task_id="task-1", candidate_id="region-1", run_id="run", cycle=0, path="/s.npz", split_group="evt-1")
    store.record_model({"model_id": "model-1", "dataset_version": "dsv-1", "parent_model_id": "model-0"}, state="promoted")
    initial = {"train": [{"sample_id": "sample-init", "label_id": "label-x", "parent_id": "C16-b01-normal0", "split": "initial_train",
                          "split_group": "C16-b01"}]}
    chain = provenance_chain(store, "run", "model-1", dataset_reader=lambda v: ["sample-init", "sample-new", "mystery"],
                             initial_samples_record=initial, md_parent={"parent_id": "C16-b01-normal0", "structure_path": "structures/x.extxyz",
                                                                       "split": "initial_train"})
    assert chain["dataset_version"] == "dsv-1" and chain["parent_model_id"] == "model-0" and chain["n_samples"] == 3
    by_id = {item["sample_id"]: item for item in chain["samples"]}
    new = by_id["sample-new"]
    assert new["task_id"] == "task-1" and new["job_id"] == "4242" and new["region_id"] == "region-1" and new["event_id"] == "evt-1"
    assert new["parent_frame_steps"] == [120, 130, 140] and new["parent_step"] == 130 and new["model_version_at_event"] == "model-0"
    assert new["parent_structure"]["parent_id"] == "C16-b01-normal0"
    assert by_id["sample-init"]["origin"] == "initial_train_full_dft" and by_id["sample-init"]["label_id"] == "label-x"
    assert chain["unresolved_samples"] == ["mystery"]  # never silently dropped


# ---------------------------------------------------------------------------------------------- observables


def test_all_trans_observables() -> None:
    topology = AlkaneTopology(8)
    positions = build_all_trans(8)
    frame = Frame(run_id="r", step=5, atom_ids=np.arange(topology.n_atoms), numbers=topology.numbers, positions=positions + 3.0,
                  cell=np.eye(3) * 50, pbc=(True,) * 3, energy=-1.0)
    row = chain_observables(frame, topology)
    assert len(row["dihedrals_deg"]) == 5 and all(abs(abs(d) - 180.0) < 1e-6 for d in row["dihedrals_deg"])
    assert row["end_to_end"] == pytest.approx(np.linalg.norm(positions[7] - positions[0]))
    gauche, _ = apply_torsion(positions, topology, (3, 4), 120.0)
    rows = [row, chain_observables(Frame(run_id="r", step=10, atom_ids=np.arange(topology.n_atoms), numbers=topology.numbers, positions=gauche,
                                         cell=None, pbc=(False,) * 3), topology)]
    summary = summarize_observables(rows, FrozenEvalSection(dihedral_bins=12))
    assert summary["n_frames"] == 2 and summary["dihedral_gauche_fraction"] == pytest.approx(1 / 10)
    assert sum(summary["dihedral_histogram"]["counts"]) == 10
    assert summarize_observables([], FrozenEvalSection()) == {"n_frames": 0}


def test_observables_exclude_frames_whose_bond_topology_broke() -> None:
    topology = AlkaneTopology(8)
    positions = build_all_trans(8)

    def frame(step: int, xyz: np.ndarray) -> Frame:
        return Frame(run_id="r", step=step, atom_ids=np.arange(topology.n_atoms), numbers=topology.numbers, positions=xyz,
                     cell=None, pbc=(False,) * 3)

    broken = positions.copy()
    broken[7] += np.array([0.0, 0.0, 30.0])  # move the last carbon away: C6-C7 and C7's C-H bonds are gone
    intact = chain_observables(frame(0, positions), topology, covalent_scale=1.3)
    lost = chain_observables(frame(10, broken), topology, covalent_scale=1.3)
    assert intact["topology_intact"] is True and lost["topology_intact"] is False
    assert chain_observables(frame(0, positions), topology)["topology_intact"] is None  # not checked without a scale
    summary = summarize_observables([intact, lost], FrozenEvalSection(dihedral_bins=12))
    assert summary["topology"] == {"checked": True, "n_intact": 1, "intact_fraction": 0.5, "first_broken_step": 10}
    assert summary["n_frames"] == 2 and summary["n_frames_used"] == 1
    assert summary["end_to_end"]["max"] == pytest.approx(intact["end_to_end"])  # the broken frame is not averaged in
    none_left = summarize_observables([lost], FrozenEvalSection())
    assert none_left["topology"]["intact_fraction"] == 0.0 and "undefined" in none_left["reason"]


# ---------------------------------------------------------------------------------------------- configuration


@pytest.mark.parametrize("name", ["smoke", "pilot"])
def test_closed_loop_configs_validate(name: str) -> None:
    cl = load_closed_loop(CONFIGS / f"closed_loop_{name}.yaml")
    assert cl.config.base_config == f"{name}.yaml" and cl.out_dir.name == name
    assert cl.config.recovery is not None and cl.config.budget.max_cpu_core_hours > 0
    assert "final_eval" not in cl.config.initial_model.train_splits + cl.config.initial_model.validation_splits


def test_md_parent_never_from_validation_or_final_eval() -> None:
    with pytest.raises(ValueError, match="allowed_splits"):
        MDSection(allowed_splits=["validation"])
    with pytest.raises(ValueError, match="allowed_splits"):
        MDSection(allowed_splits=["final_eval"])
    data = json.loads(json.dumps(__import__("yaml").safe_load((CONFIGS / "closed_loop_pilot.yaml").read_text())))
    data["initial_model"]["validation_splits"] = ["final_eval"]
    with pytest.raises(ValueError, match="final_eval"):
        ClosedLoopConfig.model_validate(data)


def test_wrong_base_config_is_refused(tmp_path: Path) -> None:
    from tools.experiments.alkane_hotspots.config import load_config

    pilot = load_config(CONFIGS / "pilot.yaml")
    with pytest.raises(ConfigurationError, match="not the closed-loop base_config"):
        load_closed_loop(CONFIGS / "closed_loop_smoke.yaml", alpha=pilot)


def test_cli_registers_the_closed_loop_commands() -> None:
    from tools.experiments.alkane_hotspots.cli import COMMANDS, load_extensions

    load_extensions()
    assert {"closed-loop-run", "frozen-eval", "recovery-demo"} <= set(COMMANDS)
    assert (ROOT / "examples" / "alkane_hotspot_demo" / "closed_loop.py").is_file()


def test_collapsed_event_geometry_is_never_labeled(tmp_path: Path) -> None:
    from tools.experiments.alkane_hotspots.closed_loop import LabelGeometrySection

    topology = AlkaneTopology(8)
    positions = build_all_trans(8)
    positions[topology.hydrogens_of[3][0]] = positions[3] + np.array([0.3, 0.0, 0.0])  # H pushed into its carbon
    extractor = AlkaneRegionExtractor(decision=LabelingDecision("full", "auto", "failed", "", None), alpha_config=None, topology=topology,
                                      force_weights=RoleWeights(), full_force_weight=1.0, merge_radius=3.0, parent_id="P",
                                      geometry=LabelGeometrySection(min_distance=0.7), rejected_path=tmp_path / "rejected.jsonl")
    with pytest.raises(DataContractError, match="not labeled.*min interatomic distance"):
        extractor.extract(_event(_periodic_frame(topology, positions), [3]))
    stretched = build_all_trans(8)
    stretched[4:] += np.array([0.0, 0.0, 3.0])  # C3-C4 bond broken
    with pytest.raises(DataContractError, match="bond topology changed"):
        extractor.extract(_event(_periodic_frame(topology, stretched), [3]))
    rejected = [json.loads(line) for line in (tmp_path / "rejected.jsonl").read_text().splitlines()]
    assert len(rejected) == 2 and rejected[0]["event_id"] == "evt-1"


@pytest.mark.integration
@pytest.mark.cp2k
@pytest.mark.slow
def test_real_smoke_closed_loop(tmp_path: Path) -> None:
    """Two real cycles (GMD, GMD_SGT, CP2K at smoke-only settings) on a copy of the smoke artifacts (~3-5 min)."""

    import shutil

    from tools.experiments.alkane_hotspots.closed_loop import run_closed_loop
    from tools.experiments.alkane_hotspots.config import load_config

    smoke = load_config(CONFIGS / "smoke.yaml")
    source = smoke.artifacts_root
    gmd = ROOT.parent / "GMD" / "build-torch" / "gmd"
    if shutil.which("cp2k.psmp") is None or not gmd.is_file():
        pytest.skip("cp2k.psmp or the GMD Torch build is missing")
    if not (source / "labels.jsonl").is_file():
        pytest.skip(f"smoke labels missing in {source} (run fidelity-run with smoke.yaml)")
    target = tmp_path / "artifacts"
    target.mkdir()
    for name in ("parents.jsonl", "labels.jsonl", "structures", "labels", "fidelity"):
        if (source / name).is_dir():
            shutil.copytree(source / name, target / name)
        elif (source / name).is_file():
            shutil.copy2(source / name, target / name)
    cl = load_closed_loop(CONFIGS / "closed_loop_smoke.yaml", alpha=load_config(CONFIGS / "smoke.yaml", artifacts_root=target))
    report = run_closed_loop(cl)
    assert report["final_stage"] == "done" and report["labeling"]["mode"] == "full"  # the smoke gate failed
    assert report["image_check"]["min_image_distance"] > report["md_box"]["receptive_field"]["force_radius"]
    assert report["summary"]["samples"] == report["summary"]["tasks"]["ingested"]
    for change in report["model_changes"]:
        assert change["kind"] == "model_change" and change["same_coordinates"] and change["jump_ev"] is not None
    for chain in report["provenance"].values():
        assert not chain["unresolved_samples"]
