"""Held-out metrics and budget analysis: hand-computed values, interpolation only inside measured points,
seed statistics across repetitions."""

from __future__ import annotations

import numpy as np
import pytest

from .beta_synthetic import alkane  # noqa: F401  (puts the repository root on sys.path)

from tools.experiments.alkane_hotspots.evaluation import (  # noqa: E402
    EvaluationConfig,
    cost_to_reach,
    curve_analysis,
    evaluate_frames,
    seed_statistics,
    site_region,
    value_at,
)
from tools.experiments.alkane_hotspots.geometry import AlkaneTopology, build_all_trans  # noqa: E402
from tools.experiments.alkane_hotspots.ground_truth import GTFrame  # noqa: E402


def _frame(family: str, site: list[int], delta: np.ndarray, pid: str) -> GTFrame:
    topology = AlkaneTopology(6)
    dft = np.zeros((20, 3))
    return GTFrame(parent_id=pid, family=family, split="final_eval", split_group=pid, numbers=topology.numbers,
                   positions=build_all_trans(6), model_forces=dft + delta, dft_forces=dft, perturbation_atoms=site)


def test_evaluate_frames_hand_values() -> None:
    delta_normal = np.zeros((20, 3))
    delta_normal[0] = [0.3, 0.0, 0.4]  # |e| = 0.5
    delta_bend = np.zeros((20, 3))
    delta_bend[2] = [0.0, 1.0, 0.0]
    result = evaluate_frames([_frame("normal", [], delta_normal, "a"), _frame("bend", [2], delta_bend, "b")], EvaluationConfig())
    overall = result["overall"]
    assert overall["n_atoms"] == 40
    assert overall["force_rmse"] == pytest.approx(np.sqrt((0.25 + 1.0) / 120))
    assert overall["atom_error_max"] == pytest.approx(1.0)
    core = result["core"]
    region = site_region(AlkaneTopology(6).numbers, build_all_trans(6), [2], radius=1, bond_scale=1.2)
    assert core["n_configurations"] == 1 and core["n_atoms"] == len(region) == 3 + 6
    assert core["force_rmse"] == pytest.approx(np.sqrt(1.0 / (3 * len(region))))
    assert result["per_family"]["normal"]["core"] == {"n_atoms": 0}
    assert result["per_family"]["bend"]["n_configurations"] == 1
    assert evaluate_frames([], EvaluationConfig())["status"] == "no_data"


def test_interpolation_never_extrapolates() -> None:
    points = [{"c": 0.0, "m": 1.0}, {"c": 100.0, "m": 0.6}, {"c": 300.0, "m": 0.2}]
    assert value_at(points, "c", "m", 50.0)["value"] == pytest.approx(0.8)
    assert value_at(points, "c", "m", 100.0) == {"value": 0.6, "interpolated": False}
    outside = value_at(points, "c", "m", 400.0)
    assert outside["value"] is None and "not extrapolated" in outside["reason"]
    assert cost_to_reach(points, "c", "m", 0.4)["cost"] == pytest.approx(200.0)
    assert cost_to_reach(points, "c", "m", 1.5) == {"cost": 0.0, "interpolated": False, "reached_at_first_point": True}
    never = cost_to_reach(points, "c", "m", 0.1)
    assert never["cost"] is None and "not reached" in never["reason"]


def test_seed_statistics_are_across_repetitions() -> None:
    stats = seed_statistics([1.0, 3.0, None])
    assert stats["n"] == 3 and stats["n_defined"] == 2 and stats["mean"] == 2.0 and stats["std"] == pytest.approx(np.sqrt(2.0))
    assert seed_statistics([None])["mean"] is None and seed_statistics([2.0])["std"] is None
    curves = {"A": [[{"cumulative_cpu_core_s": 0.0, "n_labels": 0, "force_rmse": 1.0},
                     {"cumulative_cpu_core_s": 10.0, "n_labels": 2, "force_rmse": 0.5}],
                    [{"cumulative_cpu_core_s": 0.0, "n_labels": 0, "force_rmse": 0.9},
                     {"cumulative_cpu_core_s": 30.0, "n_labels": 2, "force_rmse": 0.6}]]}
    config = EvaluationConfig(budgets_cpu_core_s=[20.0], budgets_labels=[1], accuracy_targets=[0.55])
    analysis = curve_analysis(curves, config)["A"]
    at = analysis["at_budget_cpu_core_s"]["20"]
    assert at["values"][0] is None and at["values"][1] == pytest.approx(0.9 - 0.3 * 20 / 30) and at["n_defined"] == 1
    assert analysis["at_budget_labels"]["1"]["values"] == [pytest.approx(0.75), pytest.approx(0.75)]
    reach = analysis["cost_to_reach"]["0.55"]["cpu_core_s"]
    assert reach["values"][0] == pytest.approx(9.0) and reach["values"][1] is None
    assert analysis["final"]["mean"] == pytest.approx(0.55)
