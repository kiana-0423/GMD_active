"""Parent generation: determinism, checks, perturbation metadata and extxyz output."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from ._helpers import tiny_config
from tools.experiments.alkane_hotspots.checks import segment_contact
from tools.experiments.alkane_hotspots.families import generate, load_parents, load_positions, write_generation
from tools.experiments.alkane_hotspots.geometry import AlkaneTopology, dihedral_deg, wrap_deg


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gen")
    loaded = tiny_config(tmp)
    result = generate(loaded.config)
    write_generation(result, loaded.artifacts_root)
    return loaded, result


def test_same_config_gives_byte_identical_outputs(generated, tmp_path: Path) -> None:
    loaded, _ = generated
    other = tiny_config(tmp_path)
    write_generation(generate(other.config), other.artifacts_root)
    for name in ["parents.jsonl", "rejected.jsonl", *[f"structures/{p.name}" for p in (loaded.artifacts_root / "structures").iterdir()]]:
        assert (loaded.artifacts_root / name).read_bytes() == (other.artifacts_root / name).read_bytes(), name


def test_parents_follow_the_shared_schema_and_pass_checks(generated) -> None:
    loaded, result = generated
    parents = load_parents(loaded.artifacts_root)
    assert len(parents) == loaded.config.generation.n_parents == 16
    required = {"parent_id", "chain_length", "formula", "family", "seed", "perturbation", "split", "split_group", "geometry_checks",
                "structure_path", "config_hash"}
    for parent in parents:
        assert required <= set(parent)
        assert parent["geometry_checks"]["passed"] and parent["geometry_checks"]["topology"]["same_as_parent"]
        assert set(parent["perturbation"]) >= {"kind", "atoms", "location", "magnitude", "units"}
        assert parent["config_hash"] == loaded.config.generation_hash()
    assert {p["family"] for p in parents} == {"normal", "torsion", "bend", "contact"}
    assert len({p["parent_id"] for p in parents}) == len(parents)


def test_extxyz_is_readable_with_atom_ids(generated) -> None:
    loaded, _ = generated
    parent = load_parents(loaded.artifacts_root)[0]
    atoms = read(loaded.artifacts_root / parent["structure_path"])
    assert list(atoms.arrays["atom_id"]) == list(range(len(atoms)))
    assert atoms.get_chemical_formula(mode="hill") == parent["formula"]
    np.testing.assert_allclose(atoms.positions, load_positions(loaded.artifacts_root, parent), atol=1e-11)


def test_perturbations_reach_their_targets(generated) -> None:
    loaded, result = generated
    for parent in result.parents:
        p = parent["perturbation"]
        positions = result.positions[parent["parent_id"]]
        topology = AlkaneTopology(parent["chain_length"])
        if p["kind"] == "torsion":
            assert wrap_deg(p["dihedral_change_deg"] - p["magnitude"]) == pytest.approx(0.0, abs=1e-6)
            assert parent["geometry_checks"]["intended_change"]["error_deg"] < 1e-6
            # The final (noisy) dihedral is still near the requested one.
            final = dihedral_deg(positions, *p["dihedral_atoms"])
            assert abs(wrap_deg(final - p["dihedral_after_deg"])) < 10.0
        elif p["kind"] == "bend":
            assert p["angle_after_deg"] - p["angle_before_deg"] == pytest.approx(p["magnitude"], abs=1e-6)
        elif p["kind"] == "contact":
            assert abs(p["contact_distance_noise_free"] - p["magnitude"]) <= p["target_tolerance"]
            value, _ = segment_contact(positions, topology, loaded.config.families.contact.min_segment_separation)
            low, high = loaded.config.checks.contact_window
            assert low <= value <= high
        if p["kind"] != "none":
            assert p["atoms"] and all(0 <= a < topology.n_atoms for a in p["atoms"])


def test_parents_of_one_base_share_the_thermal_field(generated) -> None:
    _, result = generated
    by_group: dict[str, set[int]] = {}
    for parent in result.parents:
        by_group.setdefault(parent["split_group"], set()).add(parent["seed"])
    assert all(len(seeds) == 1 for seeds in by_group.values())


def test_rejections_are_recorded(tmp_path: Path) -> None:
    # An impossible clash limit rejects every perturbed candidate; generation then fails explicitly.
    loaded = tiny_config(tmp_path, checks={"bond_tolerance": 0.25, "min_nonbonded_hh": 2.6})
    with pytest.raises(RuntimeError, match="passed the checks"):
        generate(loaded.config)


def test_rejected_jsonl_lines_carry_their_check_record(generated) -> None:
    loaded, result = generated
    lines = (loaded.artifacts_root / "rejected.jsonl").read_text().splitlines()
    assert len(lines) == len(result.rejected)
    for line in lines:
        record = json.loads(line)
        assert record["geometry_checks"]["failures"] and record["parent_id"] is None


def test_summary_is_small_and_has_no_coordinates(generated) -> None:
    loaded, _ = generated
    summary = json.loads((loaded.artifacts_root / "parents_summary.json").read_text())
    assert summary["n_parents"] == 16 and len(summary["parents"]) == 16
    assert summary["config_hash"] == loaded.config.generation_hash()
    assert "positions" not in json.dumps(summary)
