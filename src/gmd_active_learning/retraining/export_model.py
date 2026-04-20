from __future__ import annotations

from pathlib import Path

from gmd_active_learning.adapters.gmd_se3gnn_adapter import GMDSE3GNNAdapter


def export_model_for_gmd(
    adapter: GMDSE3GNNAdapter,
    model_path: str | Path,
    output_dir: str | Path,
    export_config: dict[str, object],
) -> str:
    return adapter.export_model(model_path, output_dir, export_config)
