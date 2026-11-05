"""H capping, uncapped-fragment gate, electronic state, embedding data and its fail-closed gate."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from gmd_active_learning.contracts import Frame, LabelResult, RegionRole, ValidationStatus
from gmd_active_learning.core.exceptions import CapabilityUnavailableError, ConfigurationError, ExtractionError
from gmd_active_learning.datasets import DatasetsConfig, build_sample, read_samples_extxyz, write_samples_extxyz
from gmd_active_learning.extraction import (
    CappingConfig,
    EmbeddingConfig,
    ExtractionConfig,
    add_h_caps,
    assess_capping,
    build_embedding,
    check_labelable,
    extract_block_regions,
    extract_cluster,
    require_embedding_backend,
    resolve_region_electronic_state,
)

from extraction.test_extraction_modes import chain_frame, frame_of


def ethane_frame() -> Frame:
    mol = molecule("C2H6")
    mol.center(vacuum=6.0)
    return frame_of(mol)


def periodic_chain_frame() -> Frame:
    """An infinite zig-zag CH2 chain along x (period 2.54 A x 4), so extraction must cut C-C bonds."""

    symbols, positions = [], []
    for i in range(8):
        x, y = 1.27 * i, 0.0 if i % 2 == 0 else 0.87
        symbols += ["C", "H", "H"]
        dy = -0.63 if i % 2 == 0 else 0.63
        positions += [[x, 5.0 + y, 5.0], [x, 5.0 + y + dy, 5.89], [x, 5.0 + y + dy, 4.11]]
    atoms = Atoms(symbols, positions=positions, cell=[10.16, 10.0, 10.0], pbc=(True, False, False))
    return frame_of(atoms)


def ethane_h_region(frame: Frame):
    """Hotspot on one H: the region is that H (core) and its carbon (boundary); C keeps 3 cut bonds."""

    carbons = [i for i, s in enumerate(frame.symbols) if s == "C"]
    hydrogen = next(i for i, s in enumerate(frame.symbols) if s == "H" and np.linalg.norm(frame.positions[i] - frame.positions[carbons[0]]) < 1.2)
    config = ExtractionConfig(extract_radius=1.2, core_radius=0.1, boundary_thickness=1.2)
    return extract_cluster(frame, [hydrogen], config), carbons


def test_cut_ethane_gets_one_cap_along_each_broken_bond() -> None:
    frame = ethane_frame()
    region, carbons = ethane_h_region(frame)
    assert region.n_atoms == 2 and carbons[1] not in region.parent_atom_ids
    capped = add_h_caps(region, frame)
    assert capped.roles.count(RegionRole.H_CAP) == 3
    to_c1 = [k for k, cap in enumerate(capped.metadata["caps"]) if cap["broken_partner_atom_id"] == carbons[1]]
    assert len(to_c1) == 1
    cap_meta = capped.metadata["caps"][to_c1[0]]
    direction = frame.positions[carbons[1]] - frame.positions[carbons[0]]
    expected = capped.positions[cap_meta["parent_local_index"]] + 1.09 * direction / np.linalg.norm(direction)
    np.testing.assert_allclose(capped.positions[region.n_atoms + to_c1[0]], expected, atol=1e-10)
    assert np.all(capped.parent_atom_ids[region.n_atoms:] == -1) and np.all(capped.force_weights[region.n_atoms:] == 0.0)
    assert capped.metadata["h_capping"]["partial"] is False and capped.metadata["h_capping"]["n_caps"] == 3


def test_capping_disabled_metal_and_oxide_are_recorded_skips() -> None:
    frame = ethane_frame()
    region = extract_cluster(frame, [0], ExtractionConfig(extract_radius=1.2, core_radius=0.1))
    off = add_h_caps(region, frame, CappingConfig(enabled=False))
    assert off.n_atoms == region.n_atoms and off.metadata["h_capping_skipped_reason"] == "disabled_by_config"
    metal = frame_of(Atoms("Cu4", positions=[[0, 0, 0], [2.5, 0, 0], [5, 0, 0], [7.5, 0, 0]], cell=[20, 20, 20]))
    region = extract_cluster(metal, [0], ExtractionConfig(extract_radius=2.6, core_radius=0.1))
    assert add_h_caps(region, metal).metadata["h_capping_skipped_reason"] == "metal_system"
    sio = frame_of(Atoms("SiOSi", positions=[[0, 0, 0], [1.6, 0, 0], [3.2, 0, 0]], cell=[20, 20, 20]))
    region = extract_cluster(sio, [0], ExtractionConfig(extract_radius=1.0, core_radius=0.1))
    skipped = add_h_caps(region, sio)
    assert skipped.metadata["h_capping_skipped_reason"] == "oxide_system"
    # Fail-closed gate: an isolated uncapped fragment with dangling bonds is refused by default.
    with pytest.raises(ExtractionError, match="uncapped"):
        check_labelable(skipped, periodic_labeling=False)
    assert check_labelable(skipped, CappingConfig(uncapped_policy="allow"), periodic_labeling=False).n_dangling == 1
    assert check_labelable(skipped, periodic_labeling=True).has_dangling_bonds


def test_allow_core_capping_means_inner_buffer_and_cap_core_atoms_caps_core() -> None:
    frame = chain_frame()
    # core <= 1.5 A (C3 and its H), inner (1.5, 1.55] (C2, C4), nothing beyond 1.6 A.
    region = extract_cluster(frame, [3], ExtractionConfig(extract_radius=1.6, core_radius=1.5, boundary_thickness=0.0))
    assert sorted(region.parent_atom_ids[region.indices_with(RegionRole.INNER_BUFFER)].tolist()) == [2, 4]
    default = add_h_caps(region, frame)
    uncapped = default.metadata["h_capping"]["uncapped_cut_bonds"]
    assert default.roles.count(RegionRole.H_CAP) == 0 and default.metadata["h_capping"]["partial"] is True
    assert {entry["role"] for entry in uncapped} == {"inner_buffer"}
    inner = add_h_caps(region, frame, CappingConfig(allow_core_capping=True))
    assert inner.roles.count(RegionRole.H_CAP) == len(uncapped) and inner.metadata["h_capping"]["partial"] is False
    core_only = extract_cluster(frame, [3], ExtractionConfig(extract_radius=0.5, core_radius=0.5))
    assert add_h_caps(core_only, frame, CappingConfig(allow_core_capping=True)).roles.count(RegionRole.H_CAP) == 0
    assert add_h_caps(core_only, frame, CappingConfig(cap_core_atoms=True)).roles.count(RegionRole.H_CAP) == 4
    with pytest.raises(ExtractionError, match="partially"):
        check_labelable(default, CappingConfig(partial_policy="error"), periodic_labeling=False)


def test_block_region_with_caps_writes_a_dataset(tmp_path) -> None:
    """F-04 regression on a cut organic chain: roles/weights/codes have n_atoms entries incl. caps."""

    frame = periodic_chain_frame()
    config = ExtractionConfig(block={"size": [2.54, 10.0, 10.0], "halo": 2.0, "buffer": {"inner": 1.0, "outer": 2.0}, "frozen": {"thickness": 0.5}, "min_atoms": 1})
    region = extract_block_regions(frame, [0], config)[0]
    capped = add_h_caps(region, frame)
    assert capped.roles.count(RegionRole.H_CAP) >= 2
    assert capped.region_codes.shape == (capped.n_atoms,)
    label = LabelResult(task_id="t", region_id=capped.region_id, forces=np.zeros((capped.n_atoms, 3)), energy=-5.0, converged=True, validation=ValidationStatus.VALID)
    path = write_samples_extxyz([build_sample(capped, label, DatasetsConfig())], tmp_path / "block.extxyz")
    back = read_samples_extxyz(path)[0]
    assert back.region.roles == capped.roles and back.region.n_atoms == capped.n_atoms


def test_electronic_state_of_a_capped_radical() -> None:
    frame = ethane_frame()
    region, _ = ethane_h_region(frame)  # C + H: 7 electrons, a radical
    radical, state = resolve_region_electronic_state(region, {"charge": 0, "multiplicity": 1})
    assert state.n_electrons == 7 and radical.multiplicity == 2 and state.corrected
    with pytest.raises(ConfigurationError):
        resolve_region_electronic_state(region, {"multiplicity": 1, "spin_policy": "strict"})
    capped, state = resolve_region_electronic_state(add_h_caps(region, frame), {})  # CH4 after capping
    assert state.n_electrons == 10 and capped.multiplicity == 1


def test_embedding_data_and_fail_closed_gate() -> None:
    frame = frame_of(molecule("CH3CH2OH"), ids=None)
    frame = frame_of(Atoms(frame.symbols, positions=frame.positions + 10.0, cell=[20, 20, 20]))
    region = extract_cluster(frame, [0], ExtractionConfig(extract_radius=1.2, core_radius=0.1))
    disabled = build_embedding(region, frame)
    assert disabled.metadata["embedding"]["status"] == "disabled"
    require_embedding_backend(disabled)  # disabled embedding passes the gate
    config = EmbeddingConfig(enabled=True, charges={"H": 0.1, "O": -0.8}, cutoff=2.5)
    embedded = build_embedding(region, frame, config)
    data = embedded.metadata["embedding"]
    assert data["status"] == "ok"
    region_ids = set(region.parent_atom_ids.tolist())
    assert all(charge["source_atom_id"] not in region_ids for charge in data["point_charges"])  # no charge on a QM atom
    wide = build_embedding(region, frame, config.model_copy(update={"cutoff": 10.0}))
    assert wide.metadata["embedding"]["n_point_charges"] >= data["n_point_charges"]
    for charge in data["point_charges"]:
        source = frame.positions[charge["source_atom_id"]] + np.asarray(region.metadata["origin_translation"])
        np.testing.assert_allclose(charge["position"], source, atol=1e-9)
    moved = region.to_dict()
    moved["positions"] = (region.positions * 1.1).tolist()
    from gmd_active_learning.contracts import ExtractedRegion

    assert build_embedding(ExtractedRegion.from_dict(moved), frame, config).metadata["embedding"]["status"] == "unavailable"
    assert build_embedding(region, frame, EmbeddingConfig(enabled=True, method="qmmm")).metadata["embedding"]["status"] == "unsupported"
    with pytest.raises(CapabilityUnavailableError, match="unsupported"):
        require_embedding_backend(embedded, "cp2k")


def test_assess_requires_the_capping_step() -> None:
    frame = ethane_frame()
    region = extract_cluster(frame, [0], ExtractionConfig(extract_radius=1.2))
    with pytest.raises(ExtractionError):
        assess_capping(region)
    with pytest.raises(ExtractionError, match="already capped"):
        add_h_caps(add_h_caps(ethane_h_region(frame)[0], frame), frame)
