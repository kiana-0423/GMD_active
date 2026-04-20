from __future__ import annotations

import torch

from gmd_active_learning.reliability_model.model import ReliabilityMLP


def test_reliability_model_forward() -> None:
    model = ReliabilityMLP(input_dim=12, hidden_dims=[16, 8])
    output = model(torch.randn(4, 12))
    assert output.shape == (4, 2)
