from __future__ import annotations

import subprocess

from gmd_active_learning.active_learning.workflow import ActiveLearningWorkflow
from gmd_active_learning.adapters.gmd_adapter import GMDAdapter


def test_workflow_dry_run(tmp_path) -> None:
    active_config = {
        "max_iterations": 1,
        "max_candidates_per_iteration": 5,
        "candidate_dir": str(tmp_path / "candidates"),
        "labeled_data_dir": str(tmp_path / "labeled"),
        "model_registry_dir": str(tmp_path / "models"),
        "workflow_state_path": str(tmp_path / "workflow_state.json"),
        "dry_run": True,
        "md": {"n_steps": 1},
    }
    monitor_config = {
        "window_size": 4,
        "cutoff": 6.0,
        "ridge_lambda": 1.0e-8,
        "thresholds": {
            "lj_residual_candidate": 0.01,
            "lj_residual_stop": 100.0,
            "max_force_stop": 20.0,
            "min_distance_stop": 0.6,
            "ensemble_deviation_candidate": 0.15,
            "ensemble_deviation_stop": 0.3,
            "parameter_jump_ratio": 5.0,
        },
        "parameter_bounds": {
            "epsilon_min": 1.0e-6,
            "epsilon_max": 10.0,
            "sigma_min": 0.5,
            "sigma_max": 8.0,
        },
        "weights": {
            "lj_residual": 1.0,
            "param_anomaly": 1.0,
            "param_jump": 0.5,
            "ensemble_deviation": 1.0,
            "max_force": 0.5,
            "min_distance": 0.5,
            "reliability_model": 0.5,
        },
    }
    labeling_config = {"labeler": "cp2k", "work_dir": str(tmp_path / "jobs"), "scheduler": "slurm", "cp2k_command": "cp2k.popt"}
    retraining_config = {
        "call_mode": "subprocess",
        "train_command": "python train.py --config {config} --data {data} --output {output}",
        "export_command": "python export.py --model {model} --output {output}",
    }
    workflow = ActiveLearningWorkflow(active_config, monitor_config, labeling_config, retraining_config)
    state = workflow.run()
    assert state["iteration"] == 1
    assert (tmp_path / "workflow_state.json").exists()
    assert (tmp_path / "models" / "model_v000" / "metadata.json").exists()


def test_workflow_passes_md_command_template(tmp_path) -> None:
    active_config = {
        "max_iterations": 1,
        "max_candidates_per_iteration": 5,
        "candidate_dir": str(tmp_path / "candidates"),
        "labeled_data_dir": str(tmp_path / "labeled"),
        "model_registry_dir": str(tmp_path / "models"),
        "workflow_state_path": str(tmp_path / "workflow_state.json"),
        "dry_run": True,
        "md": {
            "n_steps": 1,
            "command_template": "python /path/to/gmd/run.py --model {model} --config {config}",
            "config": "configs/gmd.yaml",
        },
    }
    monitor_config = {
        "window_size": 4,
        "cutoff": 6.0,
        "ridge_lambda": 1.0e-8,
        "thresholds": {
            "lj_residual_candidate": 0.01,
            "lj_residual_stop": 100.0,
            "max_force_stop": 20.0,
            "min_distance_stop": 0.6,
            "ensemble_deviation_candidate": 0.15,
            "ensemble_deviation_stop": 0.3,
            "parameter_jump_ratio": 5.0,
        },
        "parameter_bounds": {
            "epsilon_min": 1.0e-6,
            "epsilon_max": 10.0,
            "sigma_min": 0.5,
            "sigma_max": 8.0,
        },
        "weights": {
            "lj_residual": 1.0,
            "param_anomaly": 1.0,
            "param_jump": 0.5,
            "ensemble_deviation": 1.0,
            "max_force": 0.5,
            "min_distance": 0.5,
            "reliability_model": 0.5,
        },
    }
    labeling_config = {"labeler": "cp2k", "work_dir": str(tmp_path / "jobs"), "scheduler": "slurm", "cp2k_command": "cp2k.popt"}
    retraining_config = {
        "call_mode": "subprocess",
        "train_command": "python train.py --config {config} --data {data} --output {output}",
        "export_command": "python export.py --model {model} --output {output}",
    }

    workflow = ActiveLearningWorkflow(active_config, monitor_config, labeling_config, retraining_config)

    assert workflow.md_adapter.command_template == active_config["md"]["command_template"]


def test_gmd_adapter_runs_subprocess_command(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], check: bool) -> None:
        captured["command"] = command
        captured["check"] = check

    monkeypatch.setattr(subprocess, "run", fake_run)

    adapter = GMDAdapter(
        command_template="python /tmp/gmd/run.py --model {model} --config {config}",
        dry_run=False,
    )

    result = adapter.run_md(tmp_path / "model_v000", {"config": "configs/gmd.yaml"})

    assert result.frames == []
    assert result.metadata["mode"] == "subprocess"
    assert captured["command"] == [
        "python",
        "/tmp/gmd/run.py",
        "--model",
        str(tmp_path / "model_v000"),
        "--config",
        "configs/gmd.yaml",
    ]
    assert captured["check"] is True
