from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class ReliabilityDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]

    @classmethod
    def from_file(cls, path: str | Path) -> "ReliabilityDataset":
        file_path = Path(path)
        if file_path.suffix == ".npz":
            data = np.load(file_path)
            return cls(data["features"], data["labels"])
        if file_path.suffix == ".json":
            records = json.loads(file_path.read_text(encoding="utf-8"))
            features = np.asarray([record["features"] for record in records], dtype=float)
            labels = np.asarray([record["labels"] for record in records], dtype=float)
            return cls(features, labels)
        if file_path.suffix == ".csv":
            with file_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows: list[dict[str, Any]] = list(reader)
            feature_keys = sorted(key for key in rows[0] if key.startswith("feature_"))
            label_keys = sorted(key for key in rows[0] if key.startswith("label_"))
            features = np.asarray([[float(row[key]) for key in feature_keys] for row in rows], dtype=float)
            labels = np.asarray([[float(row[key]) for key in label_keys] for row in rows], dtype=float)
            return cls(features, labels)
        raise ValueError(f"Unsupported dataset format: {file_path.suffix}")
