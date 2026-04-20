from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gmd_active_learning.reliability_model.model import ReliabilityMLP
from gmd_active_learning.utils.torch_utils import get_torch_device


def predict_unsafe_probability(checkpoint_path: str | Path, features: np.ndarray) -> tuple[float, float]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=get_torch_device())
    model = ReliabilityMLP(input_dim=int(checkpoint["input_dim"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
        output = model(tensor)[0]
    predicted_force_error = float(output[0].item())
    unsafe_probability = float(torch.sigmoid(output[1]).item())
    return predicted_force_error, unsafe_probability
