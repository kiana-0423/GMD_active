"""Offline end-to-end protocol (ledger TST-002, FIX-001): a frame written by the real GMD ->
monitoring -> hotspot -> cluster extraction -> H capping / masks -> CP2K input -> (fake) CP2K output
-> parse -> GMD_SGT-ready dataset, read back by GMD_SGT's own reader when available.

PHAL's version read a LAMMPS dump with a force burst on one atom; here the burst is added to atom 3
of the recorded GMD frame (tests/fixtures/gmd). Forces must equal PHAL's conversion of the CP2K
output (Hartree/Bohr -> eV/A with ASE units, cp2k_force_parser.py:32) and the per-atom force
weights must equal PHAL's default training_mask (core 1.0, inner_buffer 0.3, outer_buffer /
boundary / h_cap 0.0).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from ase.units import Bohr, Hartree

from e2e.helpers import fake_cp2k_class
from gmd_active_learning.adapters.dft.cp2k.config import CP2KConfig
from gmd_active_learning.adapters.dft.cp2k.input import write_cp2k_inputs
from gmd_active_learning.adapters.dft.cp2k.parse import parse_cp2k_output_file
from gmd_active_learning.adapters.gmd.engine import iter_frames
from gmd_active_learning.contracts import LabelResult, RegionRole, ValidationStatus
from gmd_active_learning.datasets import DatasetsConfig, build_sample, read_samples_extxyz
from gmd_active_learning.datasets.sgt_export import export_for_gmd_sgt
from gmd_active_learning.events.config import EventsConfig
from gmd_active_learning.extraction.config import CappingConfig, EmbeddingConfig, ExtractionConfig
from gmd_active_learning.extraction.dispatch import EventRegionExtractor
from gmd_active_learning.hotspots.config import HotspotsConfig
from gmd_active_learning.monitors.config import MonitorConfig
from gmd_active_learning.monitors.online import OnlineMonitorDriver

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "gmd" / "frames.extxyz"
PHAL_MASK = {RegionRole.CORE: 1.0, RegionRole.INNER_BUFFER: 0.3, RegionRole.OUTER_BUFFER: 0.0, RegionRole.BOUNDARY: 0.0, RegionRole.H_CAP: 0.0}
HOT_ATOM = 3


def _monitor() -> OnlineMonitorDriver:
    # Four recorded frames are too few for running z-scores, so raw magnitudes are scored (PHAL's
    # e2e test used the same shortcut through its fake force metric).
    monitor = MonitorConfig.model_validate({"running_stats": {"enabled": False}})
    events = EventsConfig.model_validate({"pre_trigger_frames": 1, "post_trigger_frames": 1})
    return OnlineMonitorDriver(monitor, events, HotspotsConfig())


def test_gmd_frame_to_gmd_sgt_dataset(tmp_path: Path) -> None:
    frames = list(iter_frames(FIXTURE, run_id="offline"))
    frames[2].forces[HOT_ATOM] += np.array([30.0, 0.0, 0.0])  # force burst on one H (PHAL: large force on atom 1)
    summary = _monitor().run(frames)
    triggered = [result for result in summary.results if result.triggered]
    assert [result.step for result in triggered] == [4, 6]  # the burst, then its disappearance (delta_force)
    assert all(result.hotspot_atom_ids == [HOT_ATOM] for result in triggered)  # the same hotspot atom is detected
    assert len(summary.events) == 1
    event = summary.events[0]
    assert event.hotspot_atom_ids == [HOT_ATOM] and event.trigger_frame.step == 4

    extraction = ExtractionConfig.model_validate({"mode": "cluster", "core_radius": 1.2, "buffer_radius": 1.0, "extract_radius": 3.6,
                                                  "boundary_thickness": 0.5})
    [(region, score)] = EventRegionExtractor(extraction, CappingConfig(), EmbeddingConfig()).extract(event)
    assert score == event.max_score and HOT_ATOM in region.parent_atom_ids.tolist()
    assert region.event_id == event.event_id and region.parent_step == 4
    # Masks: PHAL's default training_mask weights per role.
    np.testing.assert_array_equal(region.force_weights, [PHAL_MASK[role] for role in region.roles])
    assert region.force_weights[region.parent_atom_ids.tolist().index(HOT_ATOM)] == 1.0

    FakeCP2K = fake_cp2k_class()  # noqa: N806
    fake = FakeCP2K(tmp_path / "fake_cp2k")
    cp2k = CP2KConfig.model_validate({"executable": str(fake.executable)})
    inputs = write_cp2k_inputs(region, tmp_path / "cp2k", cp2k, job_name="offline")
    single_point = inputs["input"] if "input" in inputs else inputs[sorted(key for key in inputs if key != "hopt_input")[0]]
    text = single_point.read_text()
    assert "RUN_TYPE ENERGY_FORCE" in text.replace("  ", " ") or re.search(r"RUN_TYPE\s+ENERGY_FORCE", text)
    assert len(re.search(r"&COORD\n(.*?)\n\s*&END COORD", text, re.S).group(1).splitlines()) == region.n_atoms
    output = tmp_path / "cp2k" / "offline.out"
    subprocess.run([str(fake.executable), "-i", str(single_point), "-o", str(output)], check=True, env={**os.environ, "FAKE_CP2K_DIR": str(fake.state)})
    parsed = parse_cp2k_output_file(output)
    hartree_bohr = np.array([[0.01 * i, -0.02 * i, 0.005] for i in range(1, region.n_atoms + 1)])
    np.testing.assert_allclose(parsed.forces, hartree_bohr * (Hartree / Bohr), rtol=1e-12, atol=0.0)  # PHAL's conversion

    label = LabelResult("task-offline", region.region_id, parsed.forces, parsed.energy, True, ValidationStatus.VALID, engine="cp2k")
    sample = build_sample(region, label, DatasetsConfig())
    files = export_for_gmd_sgt([sample], tmp_path / "dataset", DatasetsConfig())
    written = [files["all"]]
    back = read_samples_extxyz(files["all"])
    assert len(back) == 1 and {files["train"].name, files["val"].name, files["npz"].name} == {"train.extxyz", "val.extxyz", "sgt.npz"}
    np.testing.assert_allclose(back[0].forces, parsed.forces, rtol=1e-12)
    np.testing.assert_array_equal(back[0].region.force_weights, region.force_weights)
    assert back[0].region.parent_atom_ids.tolist() == region.parent_atom_ids.tolist()

    sgt_path = os.environ.get("GMD_SGT_PATH", str(REPO.parent / "GMD_SGT"))
    try:
        import sys

        if sgt_path not in sys.path:
            sys.path.insert(0, sgt_path)
        from gmd_sgt.data.reader import read_extxyz  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("GMD_SGT not importable; dataset written and round-tripped through GMD_active only")
    items = read_extxyz(str(written[0]))
    assert len(items) == 1
    np.testing.assert_allclose(np.asarray(items[0]["forces"]), parsed.forces, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(items[0]["force_weight"]), region.force_weights, rtol=0, atol=1e-7)  # float32 in GMD_SGT
