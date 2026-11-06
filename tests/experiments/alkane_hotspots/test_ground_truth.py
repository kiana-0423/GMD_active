"""Hotspot ground truth: metric definitions on hand-computed cases and signals evaluated with the real
monitor code on synthetic configurations whose high-error regions are known by construction."""

from __future__ import annotations

import numpy as np
import pytest

from .beta_synthetic import stub_errors

from tools.experiments.alkane_hotspots.geometry import AlkaneTopology, apply_bend, build_all_trans  # noqa: E402
from tools.experiments.alkane_hotspots.ground_truth import (  # noqa: E402
    GroundTruthConfig,
    GTFrame,
    RegionDefinition,
    average_precision,
    confusion_at,
    define_regions,
    graph_hops,
    roc_auc,
    run_ground_truth,
    threshold_sweep,
)

MONITOR = {"metrics": {"delta_force": {"enabled": False}, "displacement": {"enabled": False}, "delta_q": {"enabled": False}},
           "running_stats": {"warmup_frames": 2}}


def test_average_precision_and_roc_auc_hand_values() -> None:
    scores, labels = np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 0, 1, 0], bool)
    assert average_precision(scores, labels) == pytest.approx(0.5 * 1.0 + 0.5 * 2.0 / 3.0)
    assert roc_auc(scores, labels) == pytest.approx(0.75)
    assert average_precision(np.array([1.0, 1.0]), np.array([1, 0], bool)) == pytest.approx(0.5)  # a tie is one step
    assert roc_auc(np.array([1.0, 1.0]), np.array([1, 0], bool)) == pytest.approx(0.5)
    assert average_precision(scores, np.zeros(4, bool)) is None and roc_auc(scores, np.ones(4, bool)) is None
    point = confusion_at(scores, labels, 0.75)
    assert (point["tp"], point["fp"], point["fn"], point["tn"]) == (1, 1, 1, 1) and point["fpr"] == 0.5 and point["f1"] == 0.5
    sweep = threshold_sweep(scores, labels, 5)
    recalls = [p["recall"] for p in sweep]
    assert recalls == sorted(recalls, reverse=True)


def test_regions_are_bond_graph_neighbourhoods() -> None:
    topology = AlkaneTopology(8)
    positions = build_all_trans(8)
    regions = define_regions(topology.numbers, positions, RegionDefinition(radius_bonds=1))
    assert len(regions) == 8
    assert set(regions[3].tolist()) == {2, 3, 4, *topology.hydrogens_of[2], *topology.hydrogens_of[3], *topology.hydrogens_of[4]}
    assert len(regions[0]) == 2 + 3 + 2
    no_h = define_regions(topology.numbers, positions, RegionDefinition(radius_bonds=0, include_hydrogens=False))
    assert [r.tolist() for r in no_h] == [[k] for k in range(8)]
    adjacency = [[1], [0, 2], [1]]
    assert graph_hops(adjacency, [0]).tolist() == [0, 1, 2]


def _frames(seed: int = 0):
    """Calibration: 4 normal configs; evaluation: 2 normal + 4 bent configs with an error bump at the bend."""

    topology = AlkaneTopology(8)
    rng = np.random.default_rng(seed)
    calibration, evaluation = [], []
    for k in range(10):
        positions = build_all_trans(8) + rng.normal(0, 0.02, (26, 3))
        family, site = "normal", []
        if k >= 6:
            positions, _ = apply_bend(positions, topology, 2 + (k % 4), 20.0)
            family, site = "bend", [2 + (k % 4)]
        dft = rng.normal(0.0, 0.1, (26, 3))
        error = stub_errors(topology.numbers, positions, site, 1.0)
        model = dft + error[:, None] * np.array([1.0, 0.0, 0.0])
        committee = np.stack([model + (m - 1) * 0.5 * error[:, None] * np.array([0.0, 1.0, 0.0]) + rng.normal(0, 1e-5, (26, 3))
                              for m in range(3)])
        frame = GTFrame(parent_id=f"p{k}", family=family, split="calibration" if k < 4 else "validation", split_group=f"g{k // 2}",
                        numbers=topology.numbers, positions=positions, model_forces=model, dft_forces=dft, committee_forces=committee,
                        perturbation_atoms=site)
        (calibration if k < 4 else evaluation).append(frame)
    return calibration, evaluation


def _config(**overrides) -> GroundTruthConfig:
    data = {"reference_thresholds": [0.3], "signals": {"committee": {"threshold": 0.1}, "lj_residual": {"threshold": 0.5},
                                                       "phal_staged": {"threshold": 6.0}},
            "bootstrap": {"n": 50, "seed": 3}}
    data.update(overrides)
    return GroundTruthConfig.model_validate(data)


def test_signals_against_known_hotspots() -> None:
    calibration, evaluation = _frames()
    config = _config()
    report, tables = run_ground_truth(evaluation, calibration, MONITOR, config)
    assert report["status"] == "executed" and report["n_configurations"] == 6
    assert report["calibration"]["n_frames"] == 4 and report["calibration"]["warmup_satisfied"]
    block = report["per_threshold"]["0.3"]
    assert block["n_positive"] > 0
    committee = block["signals"]["committee"]
    assert committee["average_precision"] == pytest.approx(1.0)  # the committee spread follows the error exactly
    assert committee["roc_auc"] == pytest.approx(1.0)
    assert committee["operating_point"]["recall"] == 1.0
    normal = committee["normal"]
    assert normal["n_configurations"] == 2 and normal["fpr_normal"] == 0.0 and normal["alarm_rate_normal"] == 0.0
    ci = committee["average_precision_ci"]
    assert ci["n_resamples"] == 50 and ci["low"] <= 1.0 <= ci["high"] + 1e-12
    # the LJ residual of non-LJ (random) forces carries no information about the error
    lj = block["signals"]["lj_residual"]
    assert lj["average_precision"] < committee["average_precision"]
    assert lj["normal"]["alarm_rate_normal"] > 0.0  # it fires on normal configurations: a recorded false-positive rate
    # temporal metrics are unavailable on static configurations (no leakage of the perturbation site)
    assert block["signals"]["delta_force"]["reason"] == "signal unavailable on every region"
    assert any(item.startswith("delta_force:") for item in report["signals_unavailable"])
    # incremental value of LJ beyond the committee, and LJ enabled vs disabled, are reported with intervals
    names = {(c["a"], c["b"]) for c in block["comparisons"]}
    assert names == {("committee_plus_lj", "committee_monitor"), ("phal_staged", "phal_staged_no_lj")}
    for comparison in block["comparisons"]:
        assert comparison["delta"] is not None and comparison["delta_ci"]["n_resamples"] == 50
    local = report["localisation"]
    assert local["n_perturbed"] == 4 and local["fraction_site_is_max_error"] == 1.0
    assert local["top1_hit_rate"]["committee"] == 1.0
    assert set(tables[0].values) >= {"committee", "lj_residual", "phal_staged", "committee_plus_lj"}
    again, _ = run_ground_truth(evaluation, calibration, MONITOR, config)
    assert again["per_threshold"] == report["per_threshold"]  # deterministic (seeded bootstrap)


def test_no_positives_and_no_frames_are_reported_not_zeroed() -> None:
    calibration, evaluation = _frames()
    report, _ = run_ground_truth(evaluation, calibration, MONITOR, _config(reference_thresholds=[100.0]))
    committee = report["per_threshold"]["100"]["signals"]["committee"]
    assert committee["average_precision"] is None and committee["average_precision_reason"] == "no positive regions"
    empty, _ = run_ground_truth([], calibration, MONITOR, _config())
    assert empty["status"] == "no_data"


def test_config_guards() -> None:
    with pytest.raises(ValueError, match="final_eval"):
        GroundTruthConfig.model_validate({"calibration": {"splits": ["final_eval"]}})
    with pytest.raises(ValueError, match="unknown names"):
        GroundTruthConfig.model_validate({"signals": {"nonsense": {"threshold": 1.0}}})
    with pytest.raises(ValueError, match="unknown signal"):
        GroundTruthConfig.model_validate({"monitor_variants": {}, "comparisons": [["committee_plus_lj", "committee"]]})
