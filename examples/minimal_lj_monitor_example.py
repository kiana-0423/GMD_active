from __future__ import annotations

from pathlib import Path

import numpy as np

from gmd_active_learning.monitors.reliability_monitor import ReliabilityMonitor
from gmd_active_learning.utils.yaml import load_yaml


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_yaml(root / "configs" / "monitor.yaml")
    monitor = ReliabilityMonitor(config)
    result = monitor.evaluate(
        step=0,
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=float),
        symbols=["Ar", "Ar"],
        ml_forces=np.array([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]], dtype=float),
    )
    print(result.to_json_dict())


if __name__ == "__main__":
    main()
