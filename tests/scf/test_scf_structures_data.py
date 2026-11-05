"""The regenerated SCF structure collections (finding F-11, docs/scf/structure_provenance.md)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.scf.generate_structures import COLLECTIONS, check_collections
from tools.scf.structures import descriptors, read_structures

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data" / "structures"
INVENTORY = REPO / "docs" / "migration" / "phal_inventory.json"


def test_the_committed_files_are_what_the_generator_produces() -> None:
    assert check_collections(ROOT) == []


def test_every_file_the_sampling_plan_names_exists() -> None:
    plan = json.loads((REPO / "configs/cp2k/pulay_sampling_plan.json").read_text(encoding="utf-8"))
    for group in plan["groups"]:
        for path in group["structures"]:
            assert (REPO / path).is_file(), path
        assert (REPO / group["template"]).is_file()


def test_si_diamond_reproduces_the_descriptors_of_the_phal_fixture() -> None:
    sample = json.loads((REPO / "tests/fixtures/scf/pulay_sample_periodic/sample.json").read_text())
    recorded = sample["structure"]
    structures = read_structures(REPO / recorded["source"])

    regenerated = descriptors(structures[recorded["source_index"]])

    assert regenerated["label"] == recorded["label"] == "si-diamond"
    assert regenerated["formula"] == recorded["formula"]
    for key in ("cell_volume_A3", "min_distance_A", "extent_A", "number_density_A-3"):
        assert regenerated[key] == pytest.approx(recorded[key], rel=1e-12), key
    assert regenerated["family"] == recorded["family"]


def test_the_collections_have_the_documented_membership() -> None:
    sizes = {relative: len(read_structures(ROOT / relative)) for relative in COLLECTIONS}

    assert sizes["pulay_sampling/small_molecules.extxyz"] == 17
    labels = [item.label for item in read_structures(ROOT / "pulay_sampling/periodic.extxyz")]
    labels += [item.label for item in read_structures(ROOT / "pulay_sampling/transition_metal_oxides.extxyz")]
    assert sorted(labels) == sorted(
        [
            "aln-wurtzite", "c-diamond", "cu2o-cuprite", "lih-rocksalt", "mgo-rocksalt", "mno-rocksalt",
            "nacl-rocksalt", "nio-rocksalt", "si-diamond", "si-diamond-strained", "tio2-rutile", "zno-wurtzite",
        ]
    )


def test_the_anatase_cif_is_the_original_byte_for_byte() -> None:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))["files"]
    entries = inventory.items() if isinstance(inventory, dict) else [(item["path"], item) for item in inventory]
    expected = dict(entries)["data/structures/pulay_sampling/tio2_anatase.cif"]["sha256"]
    path = ROOT / "pulay_sampling/tio2_anatase.cif"

    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_the_scf_fixtures_are_byte_identical_to_phal() -> None:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))["files"]
    entries = inventory.items() if isinstance(inventory, dict) else [(item["path"], item) for item in inventory]
    checked = 0
    for relative, entry in entries:
        if relative.startswith("tests/fixtures/pulay") or relative.endswith("_scf_trajectory.npz"):
            path = REPO / "tests/fixtures/scf" / relative.removeprefix("tests/fixtures/")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], relative
            checked += 1
    assert checked == 72
