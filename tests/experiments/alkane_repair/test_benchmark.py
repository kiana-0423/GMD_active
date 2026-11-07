"""Scientific invariants of the repair benchmark (offline: no CP2K, GMD or training)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from .conftest import CONFIGS, copy_profile

from gmd_active_learning.core.exceptions import WorkflowError
from tools.experiments.alkane_hotspots.geometry import AlkaneTopology
from tools.experiments.alkane_repair.config import PARTITION_SPLIT, RepairBenchmarkConfig, load_benchmark
from tools.experiments.alkane_repair.generate import (
    GenerationError,
    check_partition_isolation,
    coverage_descriptors,
    generate,
    load_records,
    write_structures,
)


# ---------------------------------------------------------------------------------------------- configuration


@pytest.mark.parametrize("name", ["repair_c16", "repair_smoke"])
def test_committed_profiles_load(name: str) -> None:
    loaded = load_benchmark(CONFIGS / f"{name}.yaml")
    arms = {a.name: a for a in loaded.config.arms}
    assert arms["A_frozen"].kind == "frozen" and arms["B_continued"].kind == "continued_training"
    assert loaded.config.smoke_only == (name == "repair_smoke")
    assert Path(loaded.model.artifacts_root) == loaded.root


def test_confirmatory_profile_keeps_the_protocol_settings() -> None:
    config = load_benchmark(CONFIGS / "repair_c16.yaml").config
    assert config.fidelity.require_audit and not config.smoke_only
    assert len(config.repair.seeds) >= 5 and config.repair.promotion.max_force_rmse < 1.0  # not the pilot's permissive 1.0 eV/A gate
    assert config.evaluation.success.require_stability
    c = config.arm("C_committee").monitor_overrides["metrics"]
    assert all(c[name] == {"enabled": False} for name in ("force", "delta_force", "displacement", "delta_q", "lj_residual"))
    assert config.arm("D_phal").monitor_overrides == {}  # D is the pre-registered staged monitor, unchanged
    assert config.arm("D_noLJ").monitor_overrides == {"metrics": {"lj_residual": {"enabled": False}}}


def test_config_rules(tmp_path: Path) -> None:
    raw = yaml.safe_load((CONFIGS / "repair_smoke.yaml").read_text())
    no_frozen = {**raw, "arms": [a for a in raw["arms"] if a["kind"] != "frozen"]}
    with pytest.raises(ValueError, match="frozen arm"):
        RepairBenchmarkConfig.model_validate(no_frozen)
    with pytest.raises(ValueError, match="smoke-only profiles only"):
        RepairBenchmarkConfig.model_validate({**raw, "smoke_only": False})
    shrinking = {**raw, "initial_model": {**raw["initial_model"], "stages": [{"id_train_bases": 2}, {"id_train_bases": 1}]}}
    with pytest.raises(ValueError, match="grow strictly"):
        RepairBenchmarkConfig.model_validate(shrinking)
    leaky = {**raw, "partitions": {**raw["partitions"], "id_val": {"bases": 1, "ordinary_per_base": 1, "challenges_per_family": 1}}}
    with pytest.raises(ValueError, match="in-domain only"):
        RepairBenchmarkConfig.model_validate(leaky)
    with pytest.raises(ValueError, match="monitor/events settings apply to closed_loop arms only"):
        RepairBenchmarkConfig.model_validate({**raw, "arms": [*raw["arms"], {"name": "X", "kind": "frozen", "required_warm_metrics": ["force"]}]})


# ---------------------------------------------------------------------------------------------- structures


def test_generation_is_deterministic_and_isolates_bases(smoke) -> None:
    first, second = generate(smoke), generate(smoke)
    assert [r["parent_id"] for r in first.records] == [r["parent_id"] for r in second.records]
    assert all(np.array_equal(first.positions[k], second.positions[k]) for k in first.positions)
    check_partition_isolation(first.records)
    partitions = {}
    for record in first.records:
        partitions.setdefault(record["base_id"], set()).add(record["partition"])
        assert record["split"] == PARTITION_SPLIT[record["partition"]] and record["split_group"] == record["base_id"]
        if record["kind"] == "ordinary":
            assert record["coverage"]["in_domain"]
        else:
            assert record["coverage"]["challenged"]["outside"] and record["partition"] not in ("id_train", "id_val", "calibration")
    assert all(len(p) == 1 for p in partitions.values())
    spec = smoke.config.partitions
    assert len(partitions) == spec.total_bases()


def test_isolation_check_rejects_leaks(smoke) -> None:
    records = generate(smoke).records
    straddle = [dict(records[0]), {**records[1], "base_id": records[0]["base_id"], "partition": "final_test", "split": "final_eval"}]
    if straddle[0]["partition"] != "final_test":
        with pytest.raises(ValueError, match="straddles"):
            check_partition_isolation(straddle)
    challenge = next(r for r in records if r["kind"] == "challenge")
    with pytest.raises(ValueError, match="in-domain partition"):
        check_partition_isolation([{**challenge, "partition": "id_train", "split": "initial_train"}])
    with pytest.raises(ValueError, match="does not match partition"):
        check_partition_isolation([{**challenge, "split": "initial_train"}])


def test_final_test_structures_live_only_in_the_sealed_root(smoke) -> None:
    result = generate(smoke)
    write_structures(smoke, result)
    public, sealed = load_records(smoke.root), load_records(smoke.sealed_root)
    assert sealed and all(r["partition"] == "final_test" for r in sealed)
    assert all(r["partition"] != "final_test" for r in public)
    assert {r["base_id"] for r in public}.isdisjoint({r["base_id"] for r in sealed})
    assert not any("final_test" in line for line in (smoke.root / "parents.jsonl").read_text().splitlines())
    write_structures(smoke, generate(smoke))  # identical regeneration is a no-op
    tampered = generate(smoke)
    tampered.positions[tampered.records[0]["parent_id"]] = tampered.positions[tampered.records[0]["parent_id"]] + 0.01
    with pytest.raises(GenerationError, match="different content"):
        write_structures(smoke, tampered)


def test_coverage_descriptors_of_an_all_trans_chain(smoke) -> None:
    from tools.experiments.alkane_hotspots.geometry import build_all_trans

    topology = AlkaneTopology(8)
    cov = coverage_descriptors(build_all_trans(8), topology, smoke.config, 109.47122063449069)
    assert cov["in_domain"] and cov["max_torsion_deviation_deg"] < 1e-6 and cov["max_angle_deviation_deg"] < 1e-6


# ---------------------------------------------------------------------------------------------- protocol freeze


def test_protocol_freeze_refuses_changes(smoke, tmp_path: Path) -> None:
    from tools.experiments.alkane_repair.references import freeze_protocol, require_frozen

    with pytest.raises(WorkflowError, match="generate the structures"):
        freeze_protocol(smoke)
    write_structures(smoke, generate(smoke))
    with pytest.raises(WorkflowError, match="not frozen"):
        require_frozen(smoke)
    frozen = freeze_protocol(smoke, note="test")
    assert "not registered before them" in frozen["statement"] and require_frozen(smoke)["protocol"]["sha256"] == frozen["protocol"]["sha256"]
    assert freeze_protocol(smoke)["frozen_at"] == frozen["frozen_at"]  # idempotent
    smoke.protocol_path.write_text(smoke.protocol_path.read_text() + "\nedited\n")
    with pytest.raises(WorkflowError, match="protocol files changed"):
        require_frozen(smoke)
    with pytest.raises(WorkflowError, match="new benchmark"):
        freeze_protocol(smoke)


def test_stage_structures(smoke) -> None:
    from tools.experiments.alkane_repair.references import stage_structures

    write_structures(smoke, generate(smoke))
    _, initial = stage_structures(smoke, "initial_model", qualification_stage=0)
    assert {r["partition"] for r in initial} == {"id_train", "id_val"} and all(r["kind"] == "ordinary" for r in initial)
    alpha, final = stage_structures(smoke, "final_test")
    assert Path(alpha.artifacts_root) == smoke.sealed_root and {r["partition"] for r in final} == {"final_test"}
    _, repair_val = stage_structures(smoke, "repair_val")
    assert {r["partition"] for r in repair_val} == {"repair_val"}


# ---------------------------------------------------------------------------------------------- regions and errors


def test_challenged_region_and_force_errors() -> None:
    from tools.experiments.alkane_repair.common import challenged_region, force_errors, scan_directives

    topology = AlkaneTopology(8)
    record = {"parent_id": "p", "perturbation": {"atoms": topology.site_atoms([3, 4])}}
    region = challenged_region(record, topology, 1)
    assert set(topology.site_atoms([2, 3, 4, 5])) == set(region)
    with pytest.raises(ValueError, match="no perturbation site"):
        challenged_region({"parent_id": "o", "perturbation": {"atoms": []}}, topology, 1)
    reference = np.zeros((4, 3))
    model = np.zeros((4, 3))
    model[0] = [0.3, 0.0, 0.4]
    stats = force_errors(model, reference)
    assert stats["atom_error_max"] == pytest.approx(0.5) and stats["force_rmse"] == pytest.approx(np.sqrt(0.25 / 12))
    assert force_errors(model, reference, [1, 2])["force_rmse"] == 0.0
    assert scan_directives({"time_step": 0.5, "force_cap": 3.0, "fix_com": True, "thermostat": "nh"}, ["force_cap", "fix"]) == ["fix_com", "force_cap"]


# ---------------------------------------------------------------------------------------------- runs


def test_run_configs_pair_seeds_and_respect_splits(smoke) -> None:
    from tools.experiments.alkane_hotspots.closed_loop import ClosedLoopConfig
    from tools.experiments.alkane_repair.repair import closed_loop_raw, promotion_section, run_matrix

    write_structures(smoke, generate(smoke))
    specs = run_matrix(smoke)
    assert all(s.kind != "frozen" for s in specs)
    assert len(specs) == 3 * 1 * 2  # B, C, D x 1 case x 2 seeds
    manifest = smoke.root / "initial_parents.json"
    manifest.write_text(json.dumps({"train": [], "validation": []}))
    by_seed: dict[int, list[dict]] = {}
    for spec in [s for s in specs if s.kind == "closed_loop"]:
        raw = closed_loop_raw(smoke, spec, manifest)
        ClosedLoopConfig.model_validate(raw)
        assert raw["md"]["allowed_splits"] == ["calibration"] and raw["md"]["parent"]["parent_id"] == spec.case
        assert raw["initial_model"]["validation_splits"] == ["validation"] and "final_eval" not in json.dumps(raw["initial_model"])
        by_seed.setdefault(spec.seed, []).append(raw)
    for seed, raws in by_seed.items():  # paired across arms
        assert len({r["md"]["directives"]["velocity_seed"] for r in raws}) == 1 and len({r["finetune"]["seed"] for r in raws}) == 1
        assert len({json.dumps(r["monitor_overrides"], sort_keys=True) for r in raws}) == 2  # only the monitor differs
    gates = promotion_section(smoke, "torsion")["metric_gates"]
    assert {g["metric"] for g in gates} == {"force_rmse.repair_val.torsion", "force_rmse.id_val"}


def test_subset_samples_refuse_non_validation_structures(smoke) -> None:
    from tools.experiments.alkane_repair.repair import subset_samples

    write_structures(smoke, generate(smoke))
    records = load_records(smoke.root)
    val = [r for r in records if r["partition"] in ("id_val", "repair_val")]
    groups = subset_samples(smoke, ["s"] * len(val), [{"parent_id": r["parent_id"]} for r in val])
    assert set(groups) == {"id_val", "repair_val.torsion"}
    train = next(r for r in records if r["partition"] == "id_train")
    with pytest.raises(WorkflowError, match="neither id_val nor repair_val"):
        subset_samples(smoke, ["s"], [{"parent_id": train["parent_id"]}])


def test_repair_budget_stops_further_runs(smoke) -> None:
    from tools.experiments.alkane_repair.repair import RUNS_SUBDIR, require_budget, run_matrix

    write_structures(smoke, generate(smoke))
    spec = next(s for s in run_matrix(smoke) if s.kind == "closed_loop")
    assert require_budget(smoke, spec)["checked"]
    done = smoke.root / RUNS_SUBDIR / "earlier" / "repair_run.json"
    done.parent.mkdir(parents=True)
    done.write_text(json.dumps({"measured_dft_cpu_core_hours": smoke.config.budgets.repair_runs}))
    with pytest.raises(WorkflowError, match="repair-run budget"):
        require_budget(smoke, spec)


# ---------------------------------------------------------------------------------------------- outcomes


def test_success_requires_every_criterion(smoke) -> None:
    from tools.experiments.alkane_repair.evaluate import run_outcome

    config = load_benchmark(CONFIGS / "repair_c16.yaml").config
    stable = {"all_stable": True}
    good = run_outcome(config, {"force_rmse": 0.5}, {"force_rmse": 0.15}, {"force_rmse": 0.08}, {"force_rmse": 0.085}, stable)
    assert good["success"] and good["relative_improvement"] == pytest.approx(0.7)
    assert not run_outcome(config, {"force_rmse": 0.5}, {"force_rmse": 0.25}, {"force_rmse": 0.08}, {"force_rmse": 0.08}, stable)["success"]  # > 0.20
    assert not run_outcome(config, {"force_rmse": 0.22}, {"force_rmse": 0.18}, {"force_rmse": 0.08}, {"force_rmse": 0.08}, stable)["success"]  # 18 % < 30 %
    regress = run_outcome(config, {"force_rmse": 0.5}, {"force_rmse": 0.1}, {"force_rmse": 0.08}, {"force_rmse": 0.1}, stable)
    assert not regress["success"] and not regress["checks"]["retention"]
    unstable = run_outcome(config, {"force_rmse": 0.5}, {"force_rmse": 0.1}, {"force_rmse": 0.08}, {"force_rmse": 0.08}, {"all_stable": False})
    assert not unstable["success"] and not unstable["checks"]["stability"]


def test_bootstrap_and_paired_differences_report_raw_values() -> None:
    from tools.experiments.alkane_repair.evaluate import bootstrap_mean, paired_differences

    one = bootstrap_mean({"b1": [0.2, 0.4]}, {"n": 100, "seed": 1})
    assert one["mean"] == pytest.approx(0.3) and one["ci"] is None and "fewer than 2 bases" in one["ci_note"]
    two = bootstrap_mean({"b1": [0.2], "b2": [0.6]}, {"n": 200, "seed": 1, "confidence": 0.9})
    assert two["n_bases"] == 2 and 0.2 <= two["ci"][0] <= two["ci"][1] <= 0.6
    rows = [{"arm": arm, "case": "c", "seed": s, "outcome": {"region_rmse_after": v}} for arm, s, v in
            (("C", 1, 0.3), ("D", 1, 0.2), ("C", 2, 0.25), ("D", 2, None))]
    diff = paired_differences(rows)
    assert diff[0]["n_pairs"] == 1 and diff[0]["pairs"][0]["a_minus_b"] == pytest.approx(0.1) and "no equivalence" in diff[0]["note"]


def test_evaluation_refuses_before_runs_finish(smoke, tmp_path: Path) -> None:
    from gmd_active_learning.contracts.ids import file_sha256

    from tools.experiments.alkane_repair.evaluate import evaluate
    from tools.experiments.alkane_repair.qualify import FROZEN_FILE
    from tools.experiments.alkane_repair.references import freeze_protocol

    write_structures(smoke, generate(smoke))
    freeze_protocol(smoke)
    dummy = tmp_path / "ckpt.pt"
    dummy.write_bytes(b"x")
    (smoke.root / FROZEN_FILE).parent.mkdir(parents=True, exist_ok=True)
    (smoke.root / FROZEN_FILE).write_text(json.dumps({"checkpoints": [{"path": str(dummy), "sha256": file_sha256(dummy)}],
                                                      "export": {"path": str(dummy), "sha256": file_sha256(dummy)}, "id_val_force_rmse": 0.1}))
    with pytest.raises(WorkflowError, match="planned runs are missing"):
        evaluate(smoke)
    assert not (smoke.root / "sealed_access.jsonl").exists()  # a refused evaluation never touched the sealed data


def test_assessment_answers_only_from_evidence(smoke) -> None:
    from tools.experiments.alkane_repair.report import collect

    write_structures(smoke, generate(smoke))
    data = collect(smoke)
    answers = data["assessment"]["answers"]
    assert answers["1_initial_model_reliable_in_domain"].startswith("not established")
    assert answers["2_reference_confirmed_failure"].startswith("not yet tested")
    assert answers["4_local_labels_improved_independent_predictions"].startswith("not yet tested")
    assert data["assessment"]["scientific_status"] == "no confirmatory result yet"
    assert data["assessment"]["implementation"]["planned_runs"] == 6
