from __future__ import annotations

from pathlib import Path
from typing import Any

from gmd_active_learning.active_learning.candidate_queue import CandidateQueue
from gmd_active_learning.active_learning.candidate_selector import CandidateSelector
from gmd_active_learning.active_learning.state_manager import StateManager
from gmd_active_learning.adapters.gmd_adapter import GMDAdapter
from gmd_active_learning.adapters.gmd_se3gnn_adapter import GMDSE3GNNAdapter
from gmd_active_learning.labeling.cp2k_labeler import CP2KLabeler
from gmd_active_learning.labeling.vasp_labeler import VASPLabeler
from gmd_active_learning.monitors.reliability_monitor import ReliabilityMonitor
from gmd_active_learning.retraining.dataset_merger import merge_datasets
from gmd_active_learning.retraining.export_model import export_model_for_gmd
from gmd_active_learning.retraining.model_registry import ModelRegistry
from gmd_active_learning.retraining.retrain_runner import RetrainRunner
from gmd_active_learning.utils.io import ensure_dir
from gmd_active_learning.utils.logging import get_logger


LOGGER = get_logger(__name__)


class ActiveLearningWorkflow:
    def __init__(
        self,
        active_config: dict[str, Any],
        monitor_config: dict[str, Any],
        labeling_config: dict[str, Any],
        retraining_config: dict[str, Any],
    ) -> None:
        self.active_config = active_config
        self.monitor_config = monitor_config
        self.labeling_config = labeling_config
        self.retraining_config = retraining_config
        dry_run = bool(active_config.get("dry_run", True))
        md_config = active_config.get("md", {})
        self.monitor = ReliabilityMonitor(monitor_config)
        self.candidate_queue = CandidateQueue(active_config["candidate_dir"])
        self.selector = CandidateSelector(active_config["candidate_dir"])
        self.md_adapter = GMDAdapter(
            command_template=md_config.get("command_template"),
            dry_run=dry_run,
        )
        self.mlip_adapter = GMDSE3GNNAdapter(
            call_mode=str(retraining_config.get("call_mode", "subprocess")),
            train_command=retraining_config.get("train_command"),
            export_command=retraining_config.get("export_command"),
            dry_run=dry_run,
        )
        self.retrain_runner = RetrainRunner(self.mlip_adapter)
        self.registry = ModelRegistry(active_config["model_registry_dir"])
        self.state_manager = StateManager(active_config.get("workflow_state_path", "workflow_state.json"))

    def run(self) -> dict[str, Any]:
        state = self.state_manager.load()
        current_model = self.active_config.get("initial_model", "model_v000")
        max_iterations = int(self.active_config.get("max_iterations", 1))
        for iteration in range(state.get("iteration", 0), max_iterations):
            LOGGER.info("Starting iteration %s", iteration)
            run_result = self.md_adapter.run_md(current_model, self.active_config.get("md", {}))
            for frame in run_result.frames:
                result = self.monitor.evaluate(
                    step=frame.step,
                    positions=frame.positions,
                    symbols=frame.symbols,
                    ml_forces=frame.forces,
                    cell=frame.cell,
                    pbc=frame.pbc,
                    energy=frame.energy,
                    ensemble_forces=frame.ensemble_forces,
                )
                if result.should_save_frame:
                    self.candidate_queue.save_candidate(
                        step=frame.step,
                        positions=frame.positions,
                        symbols=frame.symbols,
                        ml_forces=frame.forces,
                        result=result,
                        cell=frame.cell,
                        pbc=frame.pbc,
                        metadata={"iteration": iteration, "model_version": str(current_model)},
                    )
            selected = self.selector.select(max_candidates=int(self.active_config.get("max_candidates_per_iteration", 10)))
            labeler_name = str(self.labeling_config.get("labeler", "cp2k")).lower()
            labeler = CP2KLabeler() if labeler_name == "cp2k" else VASPLabeler()
            labeler.generate(selected, self.labeling_config)
            labeled_dir = ensure_dir(self.active_config["labeled_data_dir"])
            dataset_manifest = merge_datasets("existing_dataset", labeled_dir, labeled_dir / f"merged_iter_{iteration:03d}")
            trained_model = self.retrain_runner.run(
                dataset_manifest,
                {"output_dir": str(Path(self.active_config["model_registry_dir"]) / "_staging"), "config": "retraining.yaml"},
            )
            registered_dir = self.registry.register(
                model_path=trained_model,
                training_config=self.retraining_config,
                dataset_version=dataset_manifest.name,
                metrics={"selected_candidates": len(selected)},
                parent_model=str(current_model),
            )
            export_model_for_gmd(self.mlip_adapter, trained_model, registered_dir / "exported", self.retraining_config)
            current_model = registered_dir.name
            state = {"iteration": iteration + 1, "selected_candidates": [path.name for path in selected], "current_model": current_model}
            self.state_manager.save(state)
        return state
