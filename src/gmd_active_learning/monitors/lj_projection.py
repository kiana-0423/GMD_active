from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from gmd_active_learning.core.data_types import LJFitResult
from gmd_active_learning.utils.geometry import minimum_image


def canonical_pair(symbol_a: str, symbol_b: str) -> str:
    ordered = tuple(sorted((symbol_a, symbol_b)))
    return f"{ordered[0]}-{ordered[1]}"


@dataclass(slots=True)
class LJProjectionFitter:
    cutoff: float = 6.0
    ridge_lambda: float = 1.0e-6
    parameter_bounds: dict[str, float] | None = None

    def fit_single_frame(
        self,
        positions: np.ndarray,
        symbols: list[str],
        ml_forces: np.ndarray,
        cell: np.ndarray | None = None,
        pbc: tuple[bool, bool, bool] = (False, False, False),
        theta0: dict[str, dict[str, float]] | None = None,
    ) -> LJFitResult:
        frame = {
            "positions": positions,
            "symbols": symbols,
            "ml_forces": ml_forces,
            "cell": cell,
            "pbc": pbc,
        }
        return self.fit_window([frame], theta0=theta0)

    def fit_window(
        self,
        frames: list[dict[str, object]],
        theta0: dict[str, dict[str, float]] | None = None,
    ) -> LJFitResult:
        pair_types = sorted(
            {
                canonical_pair(symbol_i, symbol_j)
                for frame in frames
                for symbol_i, symbol_j in combinations(frame["symbols"], 2)  # type: ignore[index]
            }
        )
        if not pair_types:
            return LJFitResult({}, {}, 0.0, np.array([]), True, ["no_pairs"], 0, 0.0)
        pair_to_col = {pair: index * 2 for index, pair in enumerate(pair_types)}
        total_rows = sum(len(frame["symbols"]) * 3 for frame in frames)  # type: ignore[index]
        X = np.zeros((total_rows, len(pair_types) * 2), dtype=float)
        y = np.zeros(total_rows, dtype=float)
        row_offset = 0
        per_atom_residuals: list[float] = []

        for frame in frames:
            positions = np.asarray(frame["positions"], dtype=float)
            symbols = list(frame["symbols"])  # type: ignore[arg-type]
            ml_forces = np.asarray(frame["ml_forces"], dtype=float)
            cell = frame.get("cell")
            pbc = frame.get("pbc", (False, False, False))
            n_atoms = len(symbols)
            for atom_index in range(n_atoms):
                y[row_offset + atom_index * 3 : row_offset + atom_index * 3 + 3] = ml_forces[atom_index]
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    displacement = minimum_image(positions[j] - positions[i], cell, pbc)  # type: ignore[arg-type]
                    distance = float(np.linalg.norm(displacement))
                    if distance <= 0.0 or distance > self.cutoff:
                        continue
                    pair = canonical_pair(symbols[i], symbols[j])
                    column_a = pair_to_col[pair]
                    column_b = column_a + 1
                    basis_a = 12.0 * displacement / (distance ** 14)
                    basis_b = -6.0 * displacement / (distance ** 8)
                    X[row_offset + i * 3 : row_offset + i * 3 + 3, column_a] += basis_a
                    X[row_offset + i * 3 : row_offset + i * 3 + 3, column_b] += basis_b
                    X[row_offset + j * 3 : row_offset + j * 3 + 3, column_a] -= basis_a
                    X[row_offset + j * 3 : row_offset + j * 3 + 3, column_b] -= basis_b
            row_offset += n_atoms * 3

        theta0_vector = np.zeros(X.shape[1], dtype=float)
        if theta0:
            for pair, params in theta0.items():
                if pair in pair_to_col:
                    theta0_vector[pair_to_col[pair]] = float(params.get("A", 0.0))
                    theta0_vector[pair_to_col[pair] + 1] = float(params.get("B", 0.0))
        ridge_sqrt = np.sqrt(self.ridge_lambda)
        augmented_X = np.vstack([X, ridge_sqrt * np.eye(X.shape[1], dtype=float)])
        augmented_y = np.concatenate([y, ridge_sqrt * theta0_vector])
        theta, *_ = np.linalg.lstsq(augmented_X, augmented_y, rcond=None)
        prediction = X @ theta
        residual = prediction - y
        row_offset = 0
        for frame in frames:
            n_atoms = len(frame["symbols"])  # type: ignore[index]
            frame_residual = residual[row_offset : row_offset + n_atoms * 3].reshape(n_atoms, 3)
            per_atom_residuals.extend(np.linalg.norm(frame_residual, axis=1).tolist())
            row_offset += n_atoms * 3
        relative_residual = float(np.linalg.norm(residual) / max(np.linalg.norm(y), 1.0e-12))
        warnings: list[str] = []
        pair_params_ab: dict[str, dict[str, float]] = {}
        pair_params_eps_sigma: dict[str, dict[str, float | None]] = {}
        is_physical = True
        for pair, column_a in pair_to_col.items():
            A = float(theta[column_a])
            B = float(theta[column_a + 1])
            pair_params_ab[pair] = {"A": A, "B": B}
            sigma = None
            epsilon = None
            if A <= 0.0 or B <= 0.0:
                is_physical = False
                warnings.append(f"{pair}:nonphysical_ab")
            else:
                sigma = float((A / B) ** (1.0 / 6.0))
                epsilon = float((B * B) / (4.0 * A))
                bounds = self.parameter_bounds or {}
                if sigma < float(bounds.get("sigma_min", -np.inf)) or sigma > float(bounds.get("sigma_max", np.inf)):
                    warnings.append(f"{pair}:sigma_out_of_bounds")
                if epsilon < float(bounds.get("epsilon_min", -np.inf)) or epsilon > float(bounds.get("epsilon_max", np.inf)):
                    warnings.append(f"{pair}:epsilon_out_of_bounds")
            pair_params_eps_sigma[pair] = {"epsilon": epsilon, "sigma": sigma}
        rank = int(np.linalg.matrix_rank(X))
        condition_number = float(np.linalg.cond(X)) if X.size else 0.0
        return LJFitResult(
            pair_params_ab=pair_params_ab,
            pair_params_epsilon_sigma=pair_params_eps_sigma,
            relative_force_residual=relative_residual,
            per_atom_residual=np.asarray(per_atom_residuals, dtype=float),
            is_physical=is_physical,
            warnings=sorted(set(warnings)),
            design_matrix_rank=rank,
            condition_number=condition_number,
        )


def compute_parameter_jump_ratio(
    current: LJFitResult,
    previous: dict[str, dict[str, float]] | None,
) -> float:
    if not previous:
        return 0.0
    ratios: list[float] = []
    for pair, params in current.pair_params_epsilon_sigma.items():
        prev = previous.get(pair)
        sigma = params.get("sigma")
        epsilon = params.get("epsilon")
        if prev is None or sigma is None or epsilon is None:
            continue
        prev_sigma = float(prev.get("sigma", sigma))
        prev_epsilon = float(prev.get("epsilon", epsilon))
        ratios.append(abs(sigma - prev_sigma) / max(abs(prev_sigma), 1.0e-12))
        ratios.append(abs(epsilon - prev_epsilon) / max(abs(prev_epsilon), 1.0e-12))
    return float(max(ratios, default=0.0))
