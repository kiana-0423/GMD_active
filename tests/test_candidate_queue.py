from __future__ import annotations

import numpy as np

from gmd_active_learning.active_learning.candidate_queue import CandidateQueue
from gmd_active_learning.core.constants import CANDIDATE
from gmd_active_learning.core.data_types import LJFitResult, ReliabilityResult


def test_candidate_queue_write(tmp_path) -> None:
    queue = CandidateQueue(tmp_path / "active_learning_candidates")
    result = ReliabilityResult(
        status=CANDIDATE,
        risk_score=1.23,
        metrics={"lj_residual": 0.4},
        reasons=["lj_residual_candidate"],
        lj_fit_result=LJFitResult(
            pair_params_ab={"Ar-Ar": {"A": 1.0, "B": 1.0}},
            pair_params_epsilon_sigma={"Ar-Ar": {"epsilon": 0.25, "sigma": 1.0}},
            relative_force_residual=0.4,
            per_atom_residual=np.array([0.1, 0.1], dtype=float),
            is_physical=True,
            warnings=[],
            design_matrix_rank=2,
            condition_number=1.0,
        ),
        should_save_frame=True,
        should_stop_md=False,
    )
    record = queue.save_candidate(
        step=12,
        positions=np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=float),
        symbols=["Ar", "Ar"],
        ml_forces=np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=float),
        result=result,
        metadata={"model_version": "model_v000"},
    )
    candidate_path = tmp_path / "active_learning_candidates" / record.candidate_id
    assert candidate_path.exists()
    assert (candidate_path / "structure.extxyz").exists()
    assert (candidate_path / "ml_forces.npy").exists()
    assert (candidate_path / "monitor_metrics.json").exists()
    assert (candidate_path / "lj_fit_params.json").exists()
    assert (candidate_path / "trigger_reason.txt").read_text(encoding="utf-8").strip() == "lj_residual_candidate"
