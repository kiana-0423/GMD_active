from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gmd_active_learning.core.data_types import CandidateRecord, ReliabilityResult
from gmd_active_learning.utils.io import ensure_dir, write_json
from gmd_active_learning.utils.structure_io import write_extxyz


class CandidateQueue:
    def __init__(self, candidate_dir: str | Path) -> None:
        self.candidate_dir = ensure_dir(candidate_dir)

    def save_candidate(
        self,
        *,
        step: int,
        positions: np.ndarray,
        symbols: list[str],
        ml_forces: np.ndarray,
        result: ReliabilityResult,
        cell: np.ndarray | None = None,
        pbc: tuple[bool, bool, bool] = (False, False, False),
        metadata: dict[str, Any] | None = None,
    ) -> CandidateRecord:
        next_index = len([path for path in self.candidate_dir.iterdir() if path.is_dir()]) + 1
        candidate_id = f"candidate_{next_index:06d}"
        candidate_path = ensure_dir(self.candidate_dir / candidate_id)
        write_extxyz(candidate_path / "structure.extxyz", symbols=symbols, positions=positions, cell=cell, pbc=pbc)
        np.save(candidate_path / "ml_forces.npy", ml_forces)
        write_json(candidate_path / "monitor_metrics.json", result.to_json_dict())
        if result.lj_fit_result is not None:
            write_json(candidate_path / "lj_fit_params.json", result.lj_fit_result.to_json_dict())
        (candidate_path / "trigger_reason.txt").write_text("\n".join(result.reasons), encoding="utf-8")
        candidate_metadata = {
            "step": step,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }
        write_json(candidate_path / "metadata.json", candidate_metadata)
        return CandidateRecord(
            candidate_id=candidate_id,
            path=str(candidate_path),
            risk_score=float(result.risk_score),
            reasons=result.reasons,
            metadata=candidate_metadata,
        )
