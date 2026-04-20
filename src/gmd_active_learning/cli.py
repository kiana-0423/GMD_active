from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gmd_active_learning.active_learning.candidate_selector import CandidateSelector
from gmd_active_learning.active_learning.workflow import ActiveLearningWorkflow
from gmd_active_learning.monitors.reliability_monitor import ReliabilityMonitor
from gmd_active_learning.reliability_model.dataset import ReliabilityDataset
from gmd_active_learning.reliability_model.train import train_reliability_model
from gmd_active_learning.utils.logging import configure_logging
from gmd_active_learning.utils.yaml import load_yaml


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_configs(root: Path) -> tuple[dict, dict, dict, dict]:
    return (
        load_yaml(root / "configs" / "active_learning.yaml"),
        load_yaml(root / "configs" / "monitor.yaml"),
        load_yaml(root / "configs" / "dft_labeling.yaml"),
        load_yaml(root / "configs" / "retraining.yaml"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gmd-al")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", default="configs/active_learning.yaml")
    subparsers.add_parser("monitor-example")
    subparsers.add_parser("select-candidates")
    subparsers.add_parser("generate-dft-jobs")
    subparsers.add_parser("import-labels")
    subparsers.add_parser("retrain")
    train_parser = subparsers.add_parser("train-reliability-model")
    train_parser.add_argument("--dataset", required=False)
    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    root = _default_root()
    active_config, monitor_config, labeling_config, retraining_config = _load_configs(root)
    if args.command == "init":
        print("Project already initialized.")
        return
    if args.command == "run":
        workflow = ActiveLearningWorkflow(active_config, monitor_config, labeling_config, retraining_config)
        print(workflow.run())
        return
    if args.command == "monitor-example":
        monitor = ReliabilityMonitor(monitor_config)
        result = monitor.evaluate(
            step=0,
            positions=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=float),
            symbols=["Ar", "Ar"],
            ml_forces=np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=float),
        )
        print(result.to_json_dict())
        return
    if args.command == "select-candidates":
        selector = CandidateSelector(active_config["candidate_dir"])
        print([str(path) for path in selector.select(active_config.get("max_candidates_per_iteration", 10))])
        return
    if args.command == "generate-dft-jobs":
        workflow = ActiveLearningWorkflow(active_config, monitor_config, labeling_config, retraining_config)
        selector = workflow.selector
        selected = selector.select(active_config.get("max_candidates_per_iteration", 10))
        print([str(path) for path in selected])
        return
    if args.command == "import-labels":
        print("Import hook placeholder. Add parser for your DFT results here.")
        return
    if args.command == "retrain":
        workflow = ActiveLearningWorkflow(active_config, monitor_config, labeling_config, retraining_config)
        print(workflow.run())
        return
    if args.command == "train-reliability-model":
        if args.dataset:
            dataset = ReliabilityDataset.from_file(args.dataset)
        else:
            features = np.random.rand(8, 10)
            labels = np.random.rand(8, 2)
            dataset = ReliabilityDataset(features, labels)
        checkpoint = train_reliability_model(dataset, input_dim=dataset.features.shape[1], output_dir=root / "artifacts")
        print(checkpoint)
