"""LAB-008: the per-label dataset files carry every PHAL ``write_dataset_entry`` field with PHAL's values.

PHAL's ``CP2KTaskSubmitter.poll_job`` wrote ``<task>.extxyz``, ``<task>_labels.npz`` and
``<task>_metadata.json`` with ``datasets.write_dataset_entry(region, forces, mask)``. Its inputs are in
the golden fixtures for every capped region: ``capped_region`` (indices per role, original indices),
``mask`` (``training.mask_generator.build_atom_mask``), ``region_labels`` and ``region_codes``. That is
enough to rebuild exactly what PHAL wrote. GMD's ``datasets.write_dataset_entry`` on GMD's own capped
region with the same forces must contain all of it. Deliberate deviations, asserted here:

* the extxyz uses GMD_SGT's ``force_weight`` key; PHAL's ``mask_weights`` is the same array;
* GMD adds ``force_weight``/``region_code``/``image_shift``/``energy``/``stress`` arrays, the energy
  and stress weights, and provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from extraction.phal_fixtures import SCENARIOS, capping_config, events, extraction_config, frame_at, hotspots_for, region_records
from gmd_active_learning.contracts import LabelResult, ValidationStatus
from gmd_active_learning.datasets import DatasetsConfig, build_sample
from gmd_active_learning.datasets.writer import write_dataset_entry
from gmd_active_learning.extraction.cluster import extract_cluster
from gmd_active_learning.extraction.graph import extract_graph
from gmd_active_learning.extraction.h_capping import add_h_caps
from gmd_active_learning.extraction.slab import extract_slab

EXTRACTORS = {"cluster": extract_cluster, "graph": extract_graph, "slab": extract_slab}
ROLE_KEYS = ("core_indices", "inner_buffer_indices", "outer_buffer_indices", "boundary_indices", "h_cap_indices")


def _cases() -> list[tuple[str, str, str, int]]:
    cases = []
    for scenario in SCENARIOS:
        for event in events(scenario):
            for mode in EXTRACTORS:
                for index, record in enumerate(region_records(scenario, event, mode)):
                    if record.get("capped_region") is not None:
                        cases.append((scenario, event, mode, index))
    return cases


CASES = _cases()


def _gmd_capped(scenario: str, event: str, mode: str, index: int):  # noqa: ANN202
    raw = region_records(scenario, event, mode)[index]["raw_region"]
    frame = frame_at(scenario, int(raw["metadata"]["original_frame_id"]))
    config = extraction_config(scenario, mode, periodic_guard={"policy": "flag"})
    region = EXTRACTORS[mode](frame, hotspots_for(scenario, event, mode, index), config, event_id=event)
    return region, add_h_caps(region, frame, capping_config(scenario))


def test_cases_cover_every_scenario_and_hotspot_mode() -> None:
    assert {(scenario, mode) for scenario, _event, mode, _index in CASES} == {(s, m) for s in SCENARIOS for m in EXTRACTORS}


@pytest.mark.parametrize(("scenario", "event", "mode", "index"), CASES)
def test_dataset_entry_contains_phal_fields(tmp_path: Path, scenario: str, event: str, mode: str, index: int) -> None:
    record = region_records(scenario, event, mode)[index]
    phal = record["capped_region"]
    raw, capped = _gmd_capped(scenario, event, mode, index)
    if capped.n_atoms != len(phal["atoms"]["symbols"]):
        # Documented torn-bond capping fix (tests/extraction/test_extraction_phal_parity.py): different cap count.
        assert raw.metadata["periodic_guard"]["torn_pairs"] > 0
        pytest.skip("GMD caps a torn-bond region differently from PHAL (documented deviation)")
    forces = np.random.default_rng(index).normal(size=(capped.n_atoms, 3))
    label = LabelResult(task_id="task", region_id=capped.region_id, forces=forces, energy=-1.0, converged=True,
                        validation=ValidationStatus.VALID)
    sample = build_sample(capped, label, DatasetsConfig())
    paths = write_dataset_entry(sample, output_dir=tmp_path, prefix="task")
    assert {path.name for path in paths.values()} == {"task.extxyz", "task_labels.npz", "task_metadata.json"}
    mask = np.asarray(record["mask"], dtype=float)
    n_parent = raw.n_atoms

    with np.load(paths["labels"]) as data:
        np.testing.assert_array_equal(data["forces"], forces)
        np.testing.assert_array_equal(data["mask"], mask)
        for key in ROLE_KEYS:
            assert data[key].tolist() == phal[key], key
        assert data["original_indices"].tolist() == phal["original_indices"]  # caps are -1 in both
        assert np.all(data["original_indices"][n_parent:] == -1)

    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["region_labels"] == metadata["atom_role"] == record["region_labels"]
    assert metadata["force_weight"] == mask.tolist()
    assert metadata["energy_weight"] == 0.0  # PHAL's default; GMD writes it explicitly (F-12)
    assert metadata["core_atom_indices"] == phal["core_indices"]
    assert metadata["masked_atom_indices"] == np.where(mask <= 0.0)[0].tolist()

    atoms = read(paths["structure"], format="extxyz")
    np.testing.assert_allclose(atoms.positions, np.asarray(phal["atoms"]["positions"]), atol=5.0e-8, rtol=0)
    assert atoms.get_chemical_symbols() == list(phal["atoms"]["symbols"])
    np.testing.assert_array_equal(atoms.arrays["force_weight"], mask)  # PHAL: mask_weights
    np.testing.assert_array_equal(atoms.arrays["region_code"], np.asarray(record["region_codes"]))
    np.testing.assert_allclose(atoms.get_forces(), forces, rtol=0, atol=0)
