from __future__ import annotations

from pathlib import Path

import numpy as np

from gmd_active_learning.utils.geometry import pair_distance_histogram
from gmd_active_learning.utils.io import read_json
from gmd_active_learning.utils.structure_io import read_extxyz


def candidate_feature(candidate_path: str | Path, cutoff: float = 6.0, bins: int = 16) -> np.ndarray:
    atoms = read_extxyz(Path(candidate_path) / "structure.extxyz")
    return pair_distance_histogram(atoms.positions, cell=atoms.cell, pbc=atoms.pbc, cutoff=cutoff, bins=bins)


def distance(feature_a: np.ndarray, feature_b: np.ndarray, metric: str = "cosine") -> float:
    if metric == "l2":
        return float(np.linalg.norm(feature_a - feature_b))
    denom = max(np.linalg.norm(feature_a) * np.linalg.norm(feature_b), 1.0e-12)
    cosine_similarity = float(np.dot(feature_a, feature_b) / denom)
    return 1.0 - cosine_similarity


def deduplicate_candidates(candidate_paths: list[Path], threshold: float = 0.1, metric: str = "cosine") -> list[Path]:
    selected: list[Path] = []
    selected_features: list[np.ndarray] = []
    for candidate in candidate_paths:
        feature = candidate_feature(candidate)
        if all(distance(feature, other, metric=metric) > threshold for other in selected_features):
            selected.append(candidate)
            selected_features.append(feature)
    return selected


def load_risk_score(candidate_path: Path) -> float:
    data = read_json(candidate_path / "monitor_metrics.json")
    return float(data["risk_score"])
