from __future__ import annotations

from pathlib import Path

from gmd_active_learning.adapters.ase_adapter import ASEAdapter
from gmd_active_learning.monitors.reliability_monitor import ReliabilityMonitor
from gmd_active_learning.utils.yaml import load_yaml


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    monitor = ReliabilityMonitor(load_yaml(root / "configs" / "monitor.yaml"))
    adapter = ASEAdapter()

    def callback(frame: object) -> None:
        print(frame)

    result = adapter.run_md("dummy_model", {"n_steps": 3}, monitor_callback=callback)
    print(result.metadata)
    print(monitor.evaluate(step=0, positions=result.frames[0].positions, symbols=result.frames[0].symbols, ml_forces=result.frames[0].forces))


if __name__ == "__main__":
    main()
