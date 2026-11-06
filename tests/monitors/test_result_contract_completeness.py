"""MonitorResult carries per-metric scores (CORE-002) and events their requested context (EVT-002).

CORE-002 acceptance: the contract also carries ReliabilityResult's frame-level decision fields
(global_lj engine), and PHAL's golden OODFrameResults are rebuilt from serialised MonitorResults alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from gmd_active_learning.contracts import Frame, FrameAction, MetricStatus
from gmd_active_learning.monitors import build_frame_monitor
from gmd_active_learning.monitors.online import OnlineMonitorDriver

from .phal_replay import SCENARIOS, load_scenario
from .test_phal_parity import ATOL, RTOL, _replay

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "phal_reference" / "pt111_co_slab" / "input_frames.extxyz"


def test_metric_scores_and_requested_context_are_populated() -> None:
    driver = OnlineMonitorDriver({"stages": {"light_interval": 1}}, {"pre_trigger_frames": 2, "post_trigger_frames": 2})
    events = []
    results = []
    for atoms in read(FIXTURE, index=":"):
        frame = Frame.from_atoms(atoms, run_id="slab")
        result, completed = driver.process(frame)
        results.append(result)
        events.extend(completed)
    tail = driver.finish("end_of_stream")
    if tail is not None:
        events.append(tail)
    scored = [r for r in results if r.metric_scores]
    assert scored, "staged results must expose per-metric scores"
    for result in scored:
        for name, values in result.metric_scores.items():
            assert values.shape == result.atom_scores.shape
            if result.metrics[name].status is not MetricStatus.MEASURED:
                assert not np.any(np.isfinite(values) & (values != 0)), name
    assert events and all(event.pre_frames_requested == 2 and event.post_frames_requested == 2 for event in events)
    assert all(event.summary()["pre_frames_requested"] == 2 for event in events)


# -- CORE-002: frame-level decisions of ReliabilityResult and lossless PHAL mapping ---------------

def test_reliability_result_decision_fields_are_carried_by_monitor_result() -> None:
    """ReliabilityResult's status, risk_score, reasons, metrics, should_save_frame and should_stop_md
    are all recoverable from the MonitorResult the global_lj engine returns."""

    from ase import Atoms
    from ase.calculators.lj import LennardJones

    monitor = build_frame_monitor({"engine": "global_lj"})
    captured = []
    original = monitor.monitor.evaluate

    def capture(**kwargs):  # noqa: ANN003, ANN202
        captured.append(original(**kwargs))
        return captured[-1]

    monitor.monitor.evaluate = capture  # type: ignore[method-assign]
    positions = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [1.9, 3.3, 0.0], [1.9, 1.1, 3.1]])
    atoms = Atoms("Ar4", positions=positions)
    atoms.calc = LennardJones(epsilon=0.0104, sigma=3.40, rc=50.0, smooth=False)
    frames = [
        Frame("ar", 0, np.arange(4), atoms.numbers, positions, None, (False,) * 3, forces=atoms.get_forces()),
        Frame("ar", 1, np.arange(4), atoms.numbers, positions * 0.2, None, (False,) * 3, forces=atoms.get_forces() * 40),
        Frame("ar", 2, np.arange(4), atoms.numbers, positions * 0.1, None, (False,) * 3, forces=atoms.get_forces() * 40),
    ]
    results = [monitor.evaluate(frame) for frame in frames]
    statuses = set()
    for reliability, result in zip(captured, results, strict=True):
        statuses.add(reliability.status)
        restored = json.loads(json.dumps(result.to_dict()))  # survives the JSON form too
        assert restored["metadata"]["status"] == reliability.status
        assert restored["max_score"] == pytest.approx(reliability.risk_score)
        assert restored["trigger_reasons"] == reliability.reasons
        assert restored["metadata"]["frame_metrics"] == json.loads(json.dumps(reliability.metrics))
        assert (result.action is FrameAction.STOP) == reliability.should_stop_md
        assert (result.action in {FrameAction.SAVE, FrameAction.STOP}) == reliability.should_save_frame
        assert result.triggered == (reliability.status != "SAFE")
    assert statuses == {"SAFE", "CANDIDATE", "STOP"}


def _phal_view(result: dict, lj_lazy_threshold: float) -> dict:
    """PHAL's OODFrameResult rebuilt from a serialised MonitorResult alone (atom IDs == indices here)."""

    metadata = result["metadata"]
    phal_metadata = {key: metadata[key] for key in ("label_threshold", "physics_threshold", "screen_threshold")}
    phal_metadata["lj_fit_count"] = metadata.get("lj_fit_count", 0)  # PHAL wrote 0 when no fit ran
    phal_metadata["lj_lazy_threshold"] = lj_lazy_threshold  # a config constant PHAL echoed per frame
    for key in ("screen_max_score", "physics_max_score"):
        # GMD also records the light score of scheduled full frames; PHAL's score_full did not.
        if key in metadata and not metadata.get("scheduled_full"):
            phal_metadata[key] = metadata[key]
    return {
        "step": result["step"],
        "stage": "light" if result["stage"] == "skipped" else result["stage"],  # PHAL: light + skipped flag
        "triggered": result["triggered"],
        "max_score": result["max_score"],
        "hotspot_indices": result["hotspot_atom_ids"],
        "trigger_reason": result["trigger_reasons"],
        "atom_scores": np.asarray(result["atom_scores"], dtype=float),
        "metric_scores": {name: np.asarray(values, dtype=float) for name, values in result["metric_scores"].items()},
        "metadata": phal_metadata,
    }


@pytest.mark.parametrize("name", SCENARIOS)
def test_phal_golden_results_map_losslessly_onto_monitor_result(name: str) -> None:
    scenario = load_scenario(name)
    evaluations, _events, driver = _replay(scenario, negate_lj=True)
    lazy = driver.monitor.config.metrics.lj_residual.lazy_threshold
    assert lazy == scenario.config["ood_score"]["lj_lazy_threshold"]
    for index, (evaluation, record) in enumerate(zip(evaluations, scenario.results, strict=True)):
        view = _phal_view(json.loads(json.dumps(evaluation.result.to_dict())), lazy)
        context = f"{name} frame {index}"
        for key in ("step", "stage", "triggered", "hotspot_indices", "trigger_reason"):
            assert view[key] == record[key], f"{context} {key}"
        assert view["max_score"] == pytest.approx(record["max_score"], rel=RTOL, abs=ATOL), context
        np.testing.assert_allclose(view["atom_scores"], scenario.arrays[f"frame{index:03d}/atom_scores"], rtol=RTOL, atol=ATOL)
        assert sorted(view["metric_scores"]) == record["evaluated_metrics"], context
        for metric, values in view["metric_scores"].items():
            np.testing.assert_allclose(values, scenario.arrays[f"frame{index:03d}/metric/{metric}"], rtol=RTOL, atol=ATOL,
                                       err_msg=f"{context} {metric}")
        assert sorted(view["metadata"]) == sorted(record["metadata"]), context
        for key, value in record["metadata"].items():
            assert view["metadata"][key] == pytest.approx(value, rel=RTOL, abs=ATOL), f"{context} metadata {key}"
