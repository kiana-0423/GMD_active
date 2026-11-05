"""End to end: a historical LAMMPS trajectory, imported as Frames, runs through the GMD_active monitor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gmd_active_learning.contracts import FrameAction, MonitorResult, Stage
from gmd_active_learning.importers import iter_lammps_dump, read_lammps_dump
from gmd_active_learning.monitors import MonitorConfig, OnlineMonitorDriver, StagedOODMonitor

TOY = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "lammps" / "toy_hotspot.lammpstrj"
TYPE_MAP = {1: "C", 3: "H"}


def _trajectory_text(n_frames: int, *, kick_frame: int) -> str:
    """The PHAL toy ethane geometry, gently vibrating, with a large force kick on atom id 1 at ``kick_frame``.

    Rows are written in shuffled id order with a non-zero box origin, so the test also proves the
    importer's id sorting and origin shift feed the monitor consistently.
    """

    base = read_lammps_dump(TOY, type_map=TYPE_MAP)[0]
    rng = np.random.default_rng(11)
    origin = np.array([-6.0, -6.0, -6.0])
    blocks = []
    for index in range(n_frames):
        positions = base.positions + origin + 0.01 * rng.standard_normal(base.positions.shape)
        forces = 0.05 * rng.standard_normal(base.positions.shape)
        if index == kick_frame:
            forces[0] = [25.0, -18.0, 12.0]
        order = rng.permutation(base.n_atoms)
        rows = [
            f"{base.atom_ids[i]} {base.metadata['types'][i]} "
            + " ".join(f"{value:.8f}" for value in (*positions[i], *forces[i]))
            for i in order
        ]
        blocks.append(
            f"ITEM: TIMESTEP\n{index * 10}\nITEM: NUMBER OF ATOMS\n{base.n_atoms}\n"
            "ITEM: BOX BOUNDS pp pp pp\n-6.0 6.0\n-6.0 6.0\n-6.0 6.0\n"
            "ITEM: ATOMS id type x y z fx fy fz\n" + "\n".join(rows) + "\n"
        )
    return "".join(blocks)


@pytest.fixture()
def trajectory(tmp_path: Path) -> Path:
    path = tmp_path / "legacy.lammpstrj"
    path.write_text(_trajectory_text(24, kick_frame=20), encoding="utf-8")
    return path


def test_imported_frames_run_through_the_online_driver_with_default_config(trajectory: Path) -> None:
    driver = OnlineMonitorDriver(MonitorConfig())  # default monitor/events/hotspots configuration

    summary = driver.run(iter_lammps_dump(trajectory, type_map=TYPE_MAP, timestep_fs=1.0, run_id="legacy"))

    assert summary.stop_reason == "end_of_stream"
    assert summary.errors == 0
    assert [result.step for result in summary.results] == [10 * index for index in range(24)]
    assert all(isinstance(result, MonitorResult) for result in summary.results)
    assert all(result.run_id == "legacy" for result in summary.results)
    assert all(result.atom_scores.shape == (8,) for result in summary.results)
    assert all(result.action in set(FrameAction) for result in summary.results)
    # rmin is a light metric: measured on every imported frame, from the origin-shifted positions.
    for result in summary.results:
        rmin = result.metrics["rmin"]
        assert rmin.evaluated.all()
        assert 1.0 < float(np.min(rmin.values)) < 1.2  # C-H bonds of the toy ethane (~1.09 A)
    assert driver.stats["processed_frames"] == 24


def test_the_force_kick_on_atom_id_1_is_localised(trajectory: Path) -> None:
    monitor = StagedOODMonitor(MonitorConfig())
    results = [monitor.evaluate(frame) for frame in iter_lammps_dump(trajectory, type_map=TYPE_MAP)]

    kicked = results[20]
    assert kicked.triggered
    assert kicked.stage is not Stage.SKIPPED
    assert 1 in kicked.hotspot_atom_ids  # LAMMPS id 1, carried through as the stable atom id
    assert {"force_large", "force_burst"} & set(kicked.trigger_reasons)
    assert int(np.argmax(kicked.atom_scores)) == 0
