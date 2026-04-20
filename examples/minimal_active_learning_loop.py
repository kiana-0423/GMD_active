from __future__ import annotations

from pathlib import Path

from gmd_active_learning.active_learning.workflow import ActiveLearningWorkflow
from gmd_active_learning.utils.yaml import load_yaml


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = ActiveLearningWorkflow(
        load_yaml(root / "configs" / "active_learning.yaml"),
        load_yaml(root / "configs" / "monitor.yaml"),
        load_yaml(root / "configs" / "dft_labeling.yaml"),
        load_yaml(root / "configs" / "retraining.yaml"),
    )
    print(workflow.run())


if __name__ == "__main__":
    main()
