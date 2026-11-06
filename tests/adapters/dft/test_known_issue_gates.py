"""EXT-012: PHAL's carried-forward known issues fail closed end to end (extraction -> CP2K writer).

Issue 1: an uncapped isolated metal or oxide fragment is rejected by default. The extraction pipeline
tags the region ``h_capping_skipped`` because capping is skipped for metals and oxides by default. The
built-in CP2K writer then refuses it as a non-periodic cluster unless
``cp2k.uncapped_fragment_policy`` is explicitly relaxed.
Issue 2: enabled point-charge embedding is rejected by the built-in CP2K writer, because it emits pure
Quickstep and cannot consume the charges.
Issue 3 (explicit GMD_SGT species/type mapping) is covered by tests/adapters/sgt/test_sgt_inference.py
and tests/adapters/gmd/test_engine.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ase import Atoms
from ase.build import molecule

from gmd_active_learning.adapters.dft.cp2k.backend import CP2KBackend
from gmd_active_learning.adapters.dft.cp2k.config import CP2KConfig
from gmd_active_learning.adapters.dft.cp2k.input import build_cp2k_input, write_cp2k_inputs
from gmd_active_learning.contracts import Frame
from gmd_active_learning.core.exceptions import UnsupportedCombinationError
from gmd_active_learning.extraction import CappingConfig, EmbeddingConfig, ExtractionConfig
from gmd_active_learning.extraction.dispatch import extract_and_prepare


def _frame(atoms: Atoms) -> Frame:
    return Frame.from_atoms(atoms, run_id="gates", step=0)


METAL = Atoms("Cu4", positions=[[5, 5, 5], [7.5, 5, 5], [10, 5, 5], [12.5, 5, 5]], cell=[20, 20, 20])
OXIDE = Atoms("SiOSiO", positions=[[5, 5, 5], [6.6, 5, 5], [8.2, 5, 5], [9.8, 5, 5]], cell=[20, 20, 20])


@pytest.mark.parametrize(("atoms", "reason"), [(METAL, "metal_system"), (OXIDE, "oxide_system")])
def test_uncapped_isolated_metal_or_oxide_is_rejected_by_default(atoms: Atoms, reason: str, tmp_path: Path) -> None:
    frame = _frame(atoms)
    (region,) = extract_and_prepare(frame, [[0]], extraction=ExtractionConfig(extract_radius=2.6, core_radius=0.1))
    assert region.metadata["h_capping_skipped"] is True and region.metadata["h_capping_skipped_reason"] == reason
    assert not any(region.pbc)  # an isolated cluster: CP2K would compute it non-periodic
    assert CP2KConfig().uncapped_fragment_policy == "error"
    with pytest.raises(UnsupportedCombinationError, match="uncapped broken bonds"):
        build_cp2k_input(region, CP2KConfig())
    with pytest.raises(UnsupportedCombinationError, match="uncapped broken bonds"):
        CP2KBackend.from_config({"cp2k": {}}).prepare_inputs(region, tmp_path / "job", task_id="job")
    with pytest.raises(UnsupportedCombinationError, match="uncapped broken bonds"):
        write_cp2k_inputs(region, tmp_path / "job", CP2KConfig(), job_name="job")
    assert not list(tmp_path.rglob("*.inp"))  # nothing runnable is left behind
    # Only an explicit expert override lets it through.
    assert "&QS" in build_cp2k_input(region, CP2KConfig(uncapped_fragment_policy="allow")).text


def test_capped_organic_fragment_passes_the_uncapped_gate() -> None:
    atoms = molecule("CH3CH2OH")
    atoms.positions += 10.0
    atoms.cell = [20, 20, 20]
    (region,) = extract_and_prepare(_frame(atoms), [[0]], extraction=ExtractionConfig(extract_radius=1.2, core_radius=0.1),
                                    capping=CappingConfig())
    assert not region.metadata.get("h_capping_skipped")
    build_cp2k_input(region, CP2KConfig())


def test_enabled_embedding_is_rejected_by_the_cp2k_writer() -> None:
    atoms = molecule("CH3CH2OH")
    atoms.positions += 10.0
    atoms.cell = [20, 20, 20]
    frame = _frame(atoms)
    embedding = EmbeddingConfig(enabled=True, charges={"H": 0.1, "O": -0.8, "C": -0.2}, cutoff=2.5)
    (region,) = extract_and_prepare(frame, [[0]], extraction=ExtractionConfig(extract_radius=1.2, core_radius=0.1), embedding=embedding)
    assert region.metadata["embedding"]["status"] == "ok" and region.metadata["embedding"]["n_point_charges"] > 0
    with pytest.raises(UnsupportedCombinationError, match="point-charge embedding"):
        build_cp2k_input(region, CP2KConfig())
    (plain,) = extract_and_prepare(frame, [[0]], extraction=ExtractionConfig(extract_radius=1.2, core_radius=0.1))
    with pytest.raises(UnsupportedCombinationError, match="point-charge embedding"):  # the config switch alone refuses too
        build_cp2k_input(plain, CP2KConfig(), embedding_enabled=True)
    build_cp2k_input(plain, CP2KConfig())  # disabled by default: a pure-QM input is written
