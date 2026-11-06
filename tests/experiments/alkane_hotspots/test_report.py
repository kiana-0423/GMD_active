"""report.json / report.md: executed vs planned, traceable numbers, negative results kept; the CLI chain
(evaluate -> figures -> report) runs from a study YAML on synthetic artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml

from .beta_synthetic import make_labels, make_parents

from tools.experiments.alkane_hotspots.arms import load_beta_config, write_json_atomic  # noqa: E402
from tools.experiments.alkane_hotspots.report import (  # noqa: E402
    build_report,
    cmd_evaluate,
    cmd_figures,
    cmd_report,
    label_accounting,
    register_cli,
    render_markdown,
)


def _study(tmp_path: Path, root: Path) -> Path:
    study = {"name": "unit", "artifacts_root": str(root), "arms": {"arms": ["random_local", "phal_local"], "repetitions": 2,
                                                                   "budget": {"labels": 1}, "pool": {"source": "explicit"}},
             "evaluation": {"budgets_cpu_core_s": [50.0], "accuracy_targets": [0.5]}}
    path = tmp_path / "arms_unit.yaml"
    path.write_text(yaml.safe_dump(study))
    return path


def test_label_accounting_counts_measured_costs(tmp_path: Path) -> None:
    root = tmp_path / "art"
    labels = make_labels(root, make_parents(root))
    accounting = label_accounting(labels + [{"kind": "fragment", "status": "budget_skipped", "cost": None}])
    assert accounting["n_records"] == len(labels) + 1
    assert accounting["by_kind_status"]["fragment/budget_skipped"] == 1
    assert accounting["measured_cost"]["full"]["total_cpu_core_s"] == pytest.approx(sum(r["cost"]["cpu_core_s"] for r in labels))


def test_nothing_executed_is_all_planned(tmp_path: Path) -> None:
    root = tmp_path / "art"
    make_parents(root)
    loaded = load_beta_config(_study(tmp_path, root))
    report = build_report(loaded)
    assert report["executed"] == {}
    assert set(report["planned_not_executed"]) >= {"receptive_field", "ground_truth", "arms", "evaluation", "figures"}
    assert set(report["planned_not_executed"]["arms"]) == {"random_local", "phal_local"}
    text = render_markdown(report)
    assert "Nothing executed yet." in text and "## Planned / not executed" in text


def test_executed_sections_and_negative_findings(tmp_path: Path) -> None:
    root = tmp_path / "art"
    parents = make_parents(root)
    make_labels(root, parents)
    loaded = load_beta_config(_study(tmp_path, root))
    out = loaded.output_dir
    gt = {"status": "executed", "n_configurations": 6, "n_regions": 48, "model": {"key": "k"}, "localisation": {},
          "per_threshold": {"0.2": {"n_positive": 5, "signals": {"committee": {"average_precision": 0.8}, "lj_residual": {
              "average_precision": None, "normal": {"fpr_normal": 0.4, "alarm_rate_normal": 1.0}}},
              "comparisons": [{"a": "committee_plus_lj", "b": "committee_monitor", "delta": -0.01,
                               "delta_ci": {"low": -0.05, "high": 0.03}}]}}}
    write_json_atomic(out / "ground_truth.json", gt)
    for rep in range(2):
        write_json_atomic(out / "arms" / "random_local" / f"rep{rep}" / "progress.json", {
            "status": "complete", "seed": rep, "rounds": [
                {"round": 0, "n_labels": 0, "n_valid_labels": 0, "cumulative_cpu_core_s": 0.0, "force_rmse": 0.9},
                {"round": 1, "n_labels": 1, "n_valid_labels": 1, "cumulative_cpu_core_s": 100.0, "force_rmse": 0.4 + 0.1 * rep}]})
    args = argparse.Namespace(publish=False)
    cmd_evaluate(loaded, args)
    cmd_figures(loaded, args)
    path = cmd_report(loaded, args)
    report = json.loads(path.read_text())
    executed, planned = report["executed"], report["planned_not_executed"]
    assert set(executed) == {"ground_truth", "arms", "evaluation", "figures"}
    assert "phal_local" in planned["arms"] and "receptive_field" in planned
    arms = executed["arms"]["random_local"]
    assert arms["repetitions_complete"] == 2 and arms["cumulative_cpu_core_s"] == [100.0, 100.0]
    assert all(item["sha256"] for item in arms["progress_files"])
    assert executed["ground_truth"]["artifact"]["sha256"]
    findings = executed["ground_truth"]["findings"]
    assert any("no detectable difference" in f for f in findings)  # a null result is stated, not hidden
    assert any("LJ residual on normal configurations: FPR 0.400" in f for f in findings)
    analysis = json.loads((out / "evaluation.json").read_text())["analysis"]["random_local"]
    assert analysis["cost_to_reach"]["0.5"]["cpu_core_s"]["values"] == [pytest.approx(80.0), pytest.approx(100.0)]  # rep1 reaches 0.5 exactly at its last point
    figures = executed["figures"]["figures"]
    assert set(figures) == {"learning_curves", "learning_curves_labels", "ablations"}  # only figures with executed data
    assert all(Path(p.replace("<artifacts_root>", str(loaded.artifacts_root))).is_file() for p in figures["ablations"])
    without = planned["figures"]["rendered_without_data (panels say NOT EXECUTED)"]
    assert {"hotspots", "timeline", "receptive_field"} <= set(without)
    assert str(loaded.artifacts_root) not in json.dumps(report)  # artifact paths are portable
    markdown = (out / "report.md").read_text()
    assert "### Acquisition arms" in markdown and "undefined" in markdown and report["config"]["config_hash"][:16] in markdown


def test_commands_register_on_alphas_cli() -> None:
    registered = {}
    register_cli(lambda name, help, add, handler: registered.setdefault(name, (add, handler)))
    assert set(registered) == {"receptive-field", "ground-truth", "arms-run", "evaluate", "figures", "report"}
    parser = argparse.ArgumentParser()
    registered["arms-run"][0](parser)
    args = parser.parse_args(["--study", "x.yaml", "--only", "phal_local", "--repetitions", "0"])
    assert args.only == ["phal_local"] and args.repetitions == [0]


@pytest.mark.parametrize("profile", ["smoke", "pilot", "pilot_v2", "campaign"])
def test_committed_study_configs_validate(profile: str, tmp_path: Path) -> None:
    from .beta_synthetic import REPO_ROOT

    path = REPO_ROOT / "configs" / "demos" / "alkane_hotspots" / f"arms_{profile}.yaml"
    loaded = load_beta_config(path)
    assert loaded.config.name == profile and loaded.alpha is not None
    assert loaded.artifacts_root == loaded.alpha.artifacts_root
    assert loaded.config.arms is not None and loaded.config.training.model["long_range_type"] == "none"
    assert "final_eval" not in loaded.config.arms.pool.splits + loaded.config.training.validation_splits


@pytest.mark.integration
def test_cli_chain_with_real_gmd_sgt(tmp_path: Path) -> None:
    """ground-truth -> arms-run -> evaluate -> figures -> report with real (tiny) GMD_SGT training on synthetic labels."""

    from .beta_synthetic import SGT_PATH, import_sgt
    from tools.experiments.alkane_hotspots.arms import PoolConfig, build_pool
    from tools.experiments.alkane_hotspots.report import cmd_arms_run, cmd_ground_truth

    import_sgt()
    root = tmp_path / "art"
    parents = make_parents(root)
    pool_cfg = {"source": "explicit", "splits": ["validation"], "stride": 2,
                "extraction": {"mode": "cluster", "terminal_atoms": "follow_partner", "core_radius": 1.2, "buffer_radius": 2.0,
                               "boundary_thickness": 1.0, "extract_radius": 3.2}}
    make_labels(root, parents, build_pool(parents, PoolConfig.model_validate(pool_cfg), [root]))
    monitor = {"metrics": {"delta_force": {"enabled": False}, "displacement": {"enabled": False}, "delta_q": {"enabled": False},
                           "mlip_force_deviation": {"enabled": False}}, "running_stats": {"warmup_frames": 1}}
    study = {"name": "chain", "artifacts_root": str(root), "monitor": monitor,
             "training": {"sgt": {"source_path": str(SGT_PATH), "committee_size": 1},
                          "model": {"n_species": 10, "n_blocks": 1, "scalar_dim": 8, "irreps": "8x0e", "n_basis": 4, "l_max": 0,
                                    "local_cutoff": 3.0, "lr_cutoff": 3.0, "long_range_type": "none", "n_heads": 1, "avg_neighbors": 6.0},
                          "train": {"n_epochs": 2, "batch_size": 2, "warmup_steps": 1, "patience": 0},
                          "validation_splits": ["calibration"]},
             "ground_truth": {"splits": ["validation"], "bootstrap": {"n": 10}},
             "arms": {"arms": ["committee_local", "phal_full_system"], "repetitions": 1, "n_rounds": 1, "budget": {"labels": 1},
                      "pool": pool_cfg, "fidelity_gate": {"required": False}},
             "evaluation": {"budgets_labels": [1]}}
    path = tmp_path / "arms_chain.yaml"
    path.write_text(yaml.safe_dump(study))
    loaded = load_beta_config(path)
    args = argparse.Namespace(checkpoints=None, only=None, repetitions=None, publish=False)
    gt = json.loads(cmd_ground_truth(loaded, args).read_text())
    assert gt["status"] == "executed" and gt["n_configurations"] == 3 and len(gt["model"]["checkpoints"]) == 2
    status = json.loads(cmd_arms_run(loaded, args).read_text())
    assert status == {"committee_local": [{"status": "complete", "rounds": 1}], "phal_full_system": [{"status": "complete", "rounds": 1}]}
    cmd_evaluate(loaded, args)
    cmd_figures(loaded, args)
    report = json.loads(cmd_report(loaded, args).read_text())
    assert {"ground_truth", "arms", "evaluation", "figures"} <= set(report["executed"])
    progress = json.loads((loaded.output_dir / "arms" / "committee_local" / "rep0" / "progress.json").read_text())
    assert progress["rounds"][0]["model"]["key"] == gt["model"]["key"]  # ground truth analysed the arms' round-0 model


def test_trained_checkpoint_receptive_field_is_a_separate_executed_section(tmp_path: Path) -> None:
    registered = {}
    register_cli(lambda name, help, add, handler: registered.setdefault(name, (add, handler)))
    parser = argparse.ArgumentParser()
    registered["receptive-field"][0](parser)
    assert parser.parse_args(["--checkpoint", "model_study/models/m/train/member_0/best.pt"]).checkpoint.endswith("best.pt")
    assert parser.parse_args([]).checkpoint is None
    root = tmp_path / "art"
    make_parents(root)
    loaded = load_beta_config(_study(tmp_path, root))
    assert {"receptive_field", "receptive_field_checkpoint"} <= set(build_report(loaded)["planned_not_executed"])
    result = {"nominal": {"force_receptive_field_local_A": 16.0}, "measured": {"delete": 3.1}, "model": {"source": "checkpoint"},
              "interpretation": "trained"}
    write_json_atomic(loaded.output_dir / "receptive_field_checkpoint.json", result)
    report = build_report(loaded)
    assert report["executed"]["receptive_field_checkpoint"]["model_source"] == "checkpoint"
    assert "receptive_field" in report["planned_not_executed"]  # the random-init measurement is still separate
    assert "### Receptive field, trained checkpoint" in render_markdown(report)
