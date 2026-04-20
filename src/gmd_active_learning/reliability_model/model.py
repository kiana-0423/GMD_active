from __future__ import annotations

import torch
from torch import nn


class ReliabilityMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        dims = [input_dim] + (hidden_dims or [64, 64])
        layers: list[nn.Module] = []
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        last_dim = dims[-1]
        self.force_head = nn.Linear(last_dim, 1)
        self.unsafe_head = nn.Linear(last_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(features)
        predicted_force_error = self.force_head(hidden)
        unsafe_logit = self.unsafe_head(hidden)
        return torch.cat([predicted_force_error, unsafe_logit], dim=-1)
