from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from gmd_active_learning.reliability_model.dataset import ReliabilityDataset
from gmd_active_learning.reliability_model.model import ReliabilityMLP
from gmd_active_learning.utils.torch_utils import get_torch_device


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1.0e-3
    val_fraction: float = 0.2
    patience: int = 5


def train_reliability_model(
    dataset: ReliabilityDataset,
    input_dim: int,
    output_dir: str | Path,
    config: TrainConfig | None = None,
) -> Path:
    cfg = config or TrainConfig()
    device = get_torch_device()
    model = ReliabilityMLP(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    val_size = max(1, int(len(dataset) * cfg.val_fraction)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size]) if val_size > 0 else (dataset, None)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size) if val_dataset is not None else None
    best_val = float("inf")
    patience = 0
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint = output_path / "reliability_model.pt"
    for _epoch in tqdm(range(cfg.epochs), desc="train_reliability_model"):
        model.train()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            predictions = model(features)
            loss = mse_loss(predictions[:, :1], labels[:, :1]) + bce_loss(predictions[:, 1:], labels[:, 1:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                losses = []
                for features, labels in val_loader:
                    features = features.to(device)
                    labels = labels.to(device)
                    predictions = model(features)
                    losses.append(
                        float(mse_loss(predictions[:, :1], labels[:, :1]) + bce_loss(predictions[:, 1:], labels[:, 1:]))
                    )
                val_loss = sum(losses) / max(len(losses), 1)
        if val_loader is None or val_loss < best_val:
            torch.save({"model_state": model.state_dict(), "input_dim": input_dim}, checkpoint)
            best_val = val_loss
            patience = 0
        else:
            patience += 1
        if patience >= cfg.patience:
            break
    return checkpoint
