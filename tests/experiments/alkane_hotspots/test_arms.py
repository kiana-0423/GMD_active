"""Acquisition arms on a fixed pool: unsupported combinations, pool/label matching, budgets, cost
charging, the four arms end to end with a stub model (known answers), ablations and resumability."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from .beta_synthetic import StubWorld, make_labels, make_parents

from tools.experiments.alkane_hotspots.arms import (  # noqa: E402
    ArmContext,
    ArmsConfig,
    ArmsRunner,
    ArmVariant,
    BetaConfig,
    BudgetConfig,
    PoolConfig,
    RecordedLabelProvider,
    acquire,
    arm_monitor_config,
    build_pool,
    collect_progress,
    learning_curves,
)
from gmd_active_learning.core.exceptions import ConfigurationError, DataContractError  # noqa: E402

MONITOR = {"metrics": {"delta_force": {"enabled": False}, "displacement": {"enabled": False}, "delta_q": {"enabled": False}},
           "running_stats": {"warmup_frames": 0},
           "stages": {"physics_interval": "always", "full_interval": "always"}}
POOL = {"source": "explicit", "splits": ["validation"], "stride": 1, "extraction": {"mode": "cluster", "terminal_atoms": "follow_partner", "core_radius": 1.2, "buffer_radius": 2.0, "boundary_thickness": 1.0, "extract_radius": 3.2}}


def arms_cfg(**overrides) -> dict:
    data = {"arms": ["random_local", "committee_local", "phal_local", "phal_full_system"], "repetitions": 2, "n_rounds": 2,
            "budget": {"labels": 2}, "pool": POOL, "fidelity_gate": {"required": True},
            "events": {"pre_trigger_frames": 0, "post_trigger_frames": 1, "cooldown_steps": 0}, "hotspots": {"merge_radius": 2.0},
            "calibration_splits": ["calibration"], "calibration_families": ["normal"]}
    data.update(overrides)
    return data


@pytest.fixture()
def world(tmp_path: Path):
    root = tmp_path / "art"
    parents = make_parents(root)
    pool = build_pool(parents, PoolConfig.model_validate(POOL), [root])
    labels = make_labels(root, parents, pool, invalid_every=7)
    return root, parents, labels, pool


def runner_for(world, arms: dict, *, gate: bool | None = True) -> tuple[ArmsRunner, StubWorld]:
    root, parents, labels, _ = world
    beta = BetaConfig.model_validate({"name": "t", "monitor": MONITOR, "arms": arms,
                                      "training": {"model": {"n_blocks": 1}, "initial_splits": ["initial_train"],
                                                   "validation_splits": ["calibration"]},
                                      "evaluation": {"splits": ["final_eval"]}})
    stub = StubWorld(parents, labels, root)
    context = ArmContext(parents=parents, labels=labels, bases=(root,), provider=RecordedLabelProvider(labels, (root,)), trainer=stub.train,
                         predictor_factory=stub.predictor, out_dir=root / "study", config_hash="h", gate_passed=gate)
    return ArmsRunner(beta, context), stub


# ---------------------------------------------------------------------------------------------- configuration


@pytest.mark.parametrize("variant, match", [
    ({"name": "x", "kind": "random_local", "lj": False}, "apply to PHAL arms only"),
    ({"name": "x", "kind": "committee_local", "event_buffering": False}, "apply to PHAL arms only"),
    ({"name": "x", "kind": "phal_full_system", "force_weights": {"core": 1.0}}, "local labels"),
    ({"name": "x", "kind": "phal_local", "physical_signals": False, "lj": False}, "redundant"),
])
def test_unsupported_variant_combinations_raise(variant: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ArmVariant.model_validate(variant)


def test_config_rejects_final_eval_pool_missing_budget_and_bad_seeds() -> None:
    with pytest.raises(ValueError, match="final_eval"):
        ArmsConfig.model_validate(arms_cfg(pool={**POOL, "splits": ["final_eval"]}))
    with pytest.raises(ValueError, match="cpu_core_s and/or labels"):
        ArmsConfig.model_validate(arms_cfg(budget={}))
    with pytest.raises(ValueError, match="distinct"):
        ArmsConfig.model_validate(arms_cfg(seeds=[1, 1]))
    with pytest.raises(ValueError, match="duplicate"):
        ArmsConfig.model_validate(arms_cfg(ablations=[{"name": "phal_local", "kind": "phal_local", "lj": False}]))
    cfg = ArmsConfig.model_validate(arms_cfg(repetitions=3))
    assert len({cfg.seed_for(k) for k in range(3)}) == 3


def test_temporal_metrics_are_refused_on_static_pools() -> None:
    with pytest.raises(ConfigurationError, match="leak"):
        arm_monitor_config({}, {}, ArmVariant(name="c", kind="phal_local"))
    physical_off = arm_monitor_config(MONITOR, {}, ArmVariant(name="c", kind="phal_local", physical_signals=False))
    assert not physical_off.metrics.lj_residual.enabled and not physical_off.metrics.force.enabled and physical_off.metrics.committee.enabled
    assert not arm_monitor_config(MONITOR, {}, ArmVariant(name="c", kind="phal_local", lj=False)).metrics.lj_residual.enabled


# ---------------------------------------------------------------------------------------------- pool and labels


def test_pool_is_fixed_and_labels_match_by_region_atoms(world) -> None:
    root, parents, labels, pool = world
    assert len(pool) == 3 * 8  # three validation parents x one candidate per carbon
    again = build_pool(parents, PoolConfig.model_validate(POOL), [root])
    assert [c.candidate_id for c in again] == [c.candidate_id for c in pool]
    provider = RecordedLabelProvider(labels, (root,))
    for candidate in pool:
        record = provider.fragment(candidate)
        assert record is not None and record["label_id"] == f"frag-{candidate.candidate_id}"
    assert provider.full(pool[0].parent_id)["kind"] == "full"


def test_budget_by_labels_and_core_seconds_with_overshoot_policies(world) -> None:
    root, _, labels, pool = world
    provider = RecordedLabelProvider(labels, (root,))
    kwargs = dict(kind="random_local", round_index=1, labeled_parents=set(), label_mode="recorded", missing_policy="error")
    got, acc = acquire(pool, provider, BudgetConfig(labels=3), charged_labels=set(), **kwargs)
    assert len(got) == 3 and acc["n_charged"] == 3
    first = provider.fragment(pool[0])["cost"]["cpu_core_s"]
    got, acc = acquire(pool, provider, BudgetConfig(cpu_core_s=first + 1.0), charged_labels=set(), **kwargs)
    assert len(got) == 2 and acc["overshoot_cpu_core_s"] > 0  # allow: the crossing calculation is kept and recorded
    got, acc = acquire(pool, provider, BudgetConfig(cpu_core_s=first + 1.0, overshoot="forbid"), charged_labels=set(), **kwargs)
    assert [a.cpu_core_s for a in got] == [first] and acc["overshoot_cpu_core_s"] == 0.0
    charged = {f"frag-{pool[0].candidate_id}"}
    got, acc = acquire(pool[:2], provider, BudgetConfig(labels=5), charged_labels=charged, **kwargs)
    assert [a.candidate_id for a in got] == [pool[1].candidate_id]
    assert acc["skipped"][pool[0].candidate_id] == "label already acquired by this arm"


def test_missing_labels_raise_or_are_skipped(world) -> None:
    root, _, labels, pool = world
    provider = RecordedLabelProvider([r for r in labels if r["kind"] == "full"], (root,))
    kwargs = dict(kind="random_local", round_index=1, labeled_parents=set(), label_mode="recorded", charged_labels=set())
    with pytest.raises(DataContractError, match="no label record"):
        acquire(pool, provider, BudgetConfig(labels=1), missing_policy="error", **kwargs)
    got, acc = acquire(pool, provider, BudgetConfig(labels=1), missing_policy="skip", **kwargs)
    assert got == [] and set(acc["skipped"].values()) == {"no label record"}


def test_unmeasured_cost_is_refused(world) -> None:
    root, _, labels, pool = world
    broken = [dict(r, cost={"wall_s": 1.0}) if r["kind"] == "fragment" else r for r in labels]
    provider = RecordedLabelProvider(broken, (root,))
    with pytest.raises(DataContractError, match="measured cpu_core_s"):
        acquire(pool, provider, BudgetConfig(labels=1), kind="random_local", round_index=1, labeled_parents=set(), label_mode="recorded",
                charged_labels=set(), missing_policy="error")


# ---------------------------------------------------------------------------------------------- end to end


def test_four_arms_ablations_and_resume(world) -> None:
    ablations = [{"name": "no_lj", "kind": "phal_local", "lj": False},
                 {"name": "no_buffering", "kind": "phal_local", "event_buffering": False},
                 {"name": "committee_signals_only", "kind": "phal_local", "physical_signals": False},
                 {"name": "core_only_weights", "kind": "phal_local", "force_weights": {"core": 1.0, "inner_buffer": 0.0}}]
    runner, stub = runner_for(world, arms_cfg(ablations=ablations))
    results = runner.run()
    assert set(results) == {"random_local", "committee_local", "phal_local", "phal_full_system", *[a["name"] for a in ablations]}
    root, parents, labels, pool = world
    site_candidates = {c.candidate_id for c in pool if set(c.core_ids) & set(next(p for p in parents if p["parent_id"] == c.parent_id)
                                                                               ["perturbation"]["atoms"])}
    for name, reps in results.items():
        assert [rep["seed"] for rep in reps] == [runner.cfg.seed_for(0), runner.cfg.seed_for(1)]
        for rep in reps:
            assert rep["status"] == "complete" and len(rep["rounds"]) == 3  # round 0 + two rounds
            costs = [r["cumulative_cpu_core_s"] for r in rep["rounds"]]
            assert costs[0] == 0.0 and costs == sorted(costs)
            for r in rep["rounds"][1:]:
                assert len(r["acquisitions"]) <= 2
                charged = sum(a["cpu_core_s"] for a in r["acquisitions"])
                assert r["diagnostics"]["accounting"]["spent_cpu_core_s"] == pytest.approx(charged)
                expected = {a["label_id"]: labels_by_id(labels)[a["label_id"]]["cost"]["cpu_core_s"] for a in r["acquisitions"]}
                assert all(a["cpu_core_s"] == expected[a["label_id"]] for a in r["acquisitions"])  # measured value
    # committee-local ranks perturbation-site candidates first (the stub committee spreads only there)
    first = results["committee_local"][0]["rounds"][1]["acquisitions"]
    assert all(a["candidate_id"] in site_candidates for a in first)
    # PHAL with committee signals only (events + hotspot grouping + diversity) also finds the sites; the full PHAL
    # score additionally carries the LJ residual, which is ~1 on every atom of these random (non-LJ) forces, so
    # it is not a known answer here -- only that it produced events and acquisitions
    assert all(a["candidate_id"] in site_candidates for a in results["committee_signals_only"][0]["rounds"][1]["acquisitions"])
    diagnostics = results["phal_local"][0]["rounds"][1]["diagnostics"]
    assert diagnostics["events"] and diagnostics["n_event_candidates"] >= 1
    assert results["phal_local"][0]["rounds"][1]["acquisitions"]
    # D labels whole parents with C's acquisition
    full = results["phal_full_system"][0]["rounds"]
    assert all(a["kind"] == "full" for r in full[1:] for a in r["acquisitions"])
    parents_labeled = [a["parent_id"] for r in full[1:] for a in r["acquisitions"]]
    assert len(parents_labeled) == len(set(parents_labeled))  # one charge per parent
    # the stub learns from site samples: committee-ranked acquisition beats random at equal label count on final_eval
    for rep_c, rep_a in zip(results["committee_local"], results["random_local"]):
        assert rep_c["rounds"][-1]["force_rmse"] <= rep_a["rounds"][-1]["force_rmse"] + 1e-12
    # identical round-0 model across arms of a repetition (shared initial data and seed)
    keys = {name: reps[0]["rounds"][0]["model"]["key"] for name, reps in results.items()}
    assert len(set(keys.values())) == 1
    curves = learning_curves(collect_progress(root / "study"))
    assert len(curves["phal_local"]) == 2 and curves["phal_local"][0][0]["n_labels"] == 0

    calls = stub.calls
    runner2, stub2 = runner_for(world, arms_cfg(ablations=ablations))
    again = runner2.run()
    assert stub2.calls == 0  # every repetition is complete: nothing retrained
    assert json.dumps(again["phal_local"][0]["rounds"], sort_keys=True) == json.dumps(results["phal_local"][0]["rounds"], sort_keys=True)
    assert calls > 0


def test_resume_after_interruption_continues_from_last_round(world) -> None:
    runner, _ = runner_for(world, arms_cfg(arms=["random_local"], repetitions=1, n_rounds=1))
    runner.run()
    path = world[0] / "study" / "arms" / "random_local" / "rep0" / "progress.json"
    progress = json.loads(path.read_text())
    progress["rounds"] = progress["rounds"][:1]
    progress["status"] = "running"
    path.write_text(json.dumps(progress))
    runner2, stub2 = runner_for(world, arms_cfg(arms=["random_local"], repetitions=1, n_rounds=1))
    result = runner2.run()["random_local"][0]
    assert result["status"] == "complete" and len(result["rounds"]) == 2 and stub2.calls == 1
    runner3, _ = runner_for(world, arms_cfg(arms=["random_local"], repetitions=1, n_rounds=1, budget={"labels": 3}))
    with pytest.raises(ConfigurationError, match="another configuration"):
        runner3.run()


def test_local_arms_require_a_passed_fidelity_gate(world) -> None:
    runner, _ = runner_for(world, arms_cfg(arms=["phal_local"], repetitions=1, n_rounds=1), gate=None)
    with pytest.raises(ConfigurationError, match="not been run"):
        runner.run()
    runner, _ = runner_for(world, arms_cfg(arms=["random_local"], repetitions=1, n_rounds=1), gate=False)
    with pytest.raises(ConfigurationError, match="failed"):
        runner.run()
    runner, _ = runner_for(world, arms_cfg(arms=["phal_full_system"], repetitions=1, n_rounds=1), gate=False)
    assert runner.run()["phal_full_system"][0]["status"] == "complete"  # full-system labels need no fragment gate


def test_invalid_labels_are_charged_but_not_trained_on(world) -> None:
    runner, _ = runner_for(world, arms_cfg(arms=["random_local"], repetitions=1, n_rounds=2, budget={"labels": 6}))
    rep = runner.run()["random_local"][0]
    acquired = [a for r in rep["rounds"][1:] for a in r["acquisitions"]]
    invalid = [a for a in acquired if a["status"] != "valid"]
    assert invalid, "fixture marks every 7th fragment label invalid"
    assert rep["rounds"][-1]["n_valid_labels"] == len(acquired) - len(invalid)
    manifest = json.loads((Path(runner.ctx.out_dir) / "models" / rep["rounds"][-1]["model"]["key"] / "dataset" / "manifest.json").read_text())
    trained = {item["label_id"] for item in manifest["train"]}
    assert not trained & {a["label_id"] for a in invalid}
    capped = [item for item in manifest["train"] if item["kind"] == "fragment"]
    assert capped and all(item["energy_weight"] == 0.0 and "capped fragment" in item["energy_weight_reason"] for item in capped)


def labels_by_id(labels):
    return {r["label_id"]: r for r in labels}


def test_rng_streams_differ_between_repetitions(world) -> None:
    runner, _ = runner_for(world, arms_cfg(arms=["random_local"], repetitions=2, n_rounds=1, budget={"labels": 3}))
    reps = runner.run()["random_local"]
    picks = [[a["candidate_id"] for a in rep["rounds"][1]["acquisitions"]] for rep in reps]
    assert picks[0] != picks[1]
    assert np.isfinite([rep["rounds"][-1]["force_rmse"] for rep in reps]).all()


def test_gate_files_and_alphas_fragment_builder(tmp_path: Path) -> None:
    """The pool reuses alpha's build_fragment with the gate-selected setting, so pool fragments are alpha's label requests."""

    from tools.experiments.alkane_hotspots.arms import beta_from_mapping, gate_builder, read_gate
    from tools.experiments.alkane_hotspots.config import AlkaneConfig, LoadedConfig
    from tools.experiments.alkane_hotspots.fidelity import build_fragment, parent_frame
    from tools.experiments.alkane_hotspots.families import load_positions

    root = tmp_path / "art"
    parents = make_parents(root)
    alpha = LoadedConfig(AlkaneConfig.model_validate({"name": "unit"}), None, artifacts_override=root)
    loaded = beta_from_mapping({"name": "unit", "arms": arms_cfg()}, path=tmp_path / "x.yaml", alpha=alpha)
    assert loaded.artifacts_root == root and read_gate(loaded) == (None, None)
    (root / "fidelity").mkdir()
    (root / "fidelity" / "failure.json").write_text(json.dumps({"status": "no_setting_passes"}))
    assert read_gate(loaded) == (False, None)
    setting = {"setting_id": "graph-h2", "policy": "graph", "hops": 2, "buffer": "2 C-C bonds"}
    (root / "fidelity" / "selection.json").write_text(json.dumps({"status": "selected", "setting": setting}))
    assert read_gate(loaded) == (True, setting)
    pool = build_pool(parents, PoolConfig.model_validate({"splits": ["validation"]}), [root], builder=gate_builder(alpha, setting))
    parent = next(p for p in parents if p["parent_id"] == pool[3].parent_id)
    reference, core = build_fragment(parent_frame(parent, load_positions(root, parent)), [pool[3].center], setting, alpha.config)
    assert np.allclose(reference.positions, pool[3].region.positions) and sorted(pool[3].core_ids) == core
    with pytest.raises(ConfigurationError, match="gate_selection"):
        build_pool(parents, PoolConfig.model_validate({"splits": ["validation"]}), [root])


def test_phal_arms_refuse_a_monitor_that_cannot_leave_warm_up(world) -> None:
    root, parents, labels, _ = world
    monitor = {**MONITOR, "running_stats": {"warmup_frames": 99}}
    beta = BetaConfig.model_validate({"name": "t", "monitor": monitor, "arms": arms_cfg(),
                                      "training": {"model": {"n_blocks": 1}, "initial_splits": ["initial_train"],
                                                   "validation_splits": ["calibration"]},
                                      "evaluation": {"splits": ["final_eval"]}})
    stub = StubWorld(parents, labels, root)
    context = ArmContext(parents=parents, labels=labels, bases=(root,), provider=RecordedLabelProvider(labels, (root,)), trainer=stub.train,
                         predictor_factory=stub.predictor, out_dir=root / "study", config_hash="h", gate_passed=True)
    runner = ArmsRunner(beta, context)
    with pytest.raises(ConfigurationError, match="warmup_frames=99"):
        runner.run(only=["random_local", "phal_local"])
    assert not (root / "study" / "arms").exists()  # refused before any arm (or training) started
    assert set(runner.run(only=["random_local"], repetitions=[0])) == {"random_local"}  # non-PHAL arms are unaffected
