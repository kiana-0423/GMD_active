from __future__ import annotations

from pathlib import Path

from gmd_active_learning.utils.io import ensure_dir


def merge_datasets(existing_dataset: str | Path, labeled_data_dir: str | Path, output_dir: str | Path) -> Path:
    output = ensure_dir(output_dir)
    manifest = output / "dataset_manifest.txt"
    lines = [str(existing_dataset), str(labeled_data_dir)]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest
