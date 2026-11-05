# Ported from PHAL (tests.scf.test_structure_tasks) at 96263b6; MIT License, see NOTICE.
"""Tests for turning a structure collection into CP2K dataset tasks.

The failures worth catching here are the quiet ones: a molecule silently made
periodic, a basis paired with the wrong pseudopotential valence, or a template
typo that reaches CP2K verbatim and aborts an hour into a queued job.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase", reason="structure collections are read through ASE")

from ase import Atoms  # noqa: E402
from ase.io import write as ase_write  # noqa: E402

from tools.scf.structures import (  # noqa: E402
    KindSpec,
    StructureError,
    descriptors,
    read_collection,
    read_structures,
    render_input,
    resolve_kinds,
    tasks_from_collection,
    tasks_from_plan,
)

TEMPLATE = """&GLOBAL
  PROJECT {{PROJECT}}
&END GLOBAL
&FORCE_EVAL
  &SUBSYS
    &CELL
{{CELL}}
    &END CELL
    &COORD
{{COORD}}
    &END COORD
{{KIND}}
  &END SUBSYS
&END FORCE_EVAL
"""

KINDS = {
    "H": KindSpec("H", "DZVP-MOLOPT-SR-GTH", "GTH-PBE-q1"),
    "O": KindSpec("O", "DZVP-MOLOPT-SR-GTH", "GTH-PBE-q6"),
}


@pytest.fixture
def water(tmp_path: Path) -> Path:
    atoms = Atoms("OH2", positions=[(0, 0, 0.119), (0, 0.763, -0.477), (0, -0.763, -0.477)])
    atoms.info["label"] = "water"
    path = tmp_path / "water.xyz"
    ase_write(str(path), atoms, format="extxyz")
    return path


@pytest.fixture
def crystal(tmp_path: Path) -> Path:
    atoms = Atoms("H2", positions=[(0, 0, 0), (1.5, 1.5, 1.5)], cell=np.eye(3) * 3.0, pbc=True)
    atoms.info["label"] = "cubic"
    path = tmp_path / "cubic.extxyz"
    ase_write(str(path), atoms, format="extxyz")
    return path


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A stand-in for CP2K's data directory, in its real format."""

    directory = tmp_path / "data"
    directory.mkdir()
    (directory / "BASIS_MOLOPT").write_text(
        "# comment\n"
        " H  DZVP-MOLOPT-GTH DZVP-MOLOPT-GTH-q1\n"
        " H  DZVP-MOLOPT-SR-GTH DZVP-MOLOPT-SR-GTH-q1\n"
        " O  DZVP-MOLOPT-SR-GTH DZVP-MOLOPT-SR-GTH-q6\n"
        " Ti DZVP-MOLOPT-SR-GTH DZVP-MOLOPT-SR-GTH-q12\n"
        " Xx DZVP-MOLOPT-SR-GTH DZVP-MOLOPT-SR-GTH-q3\n",
        encoding="utf-8",
    )
    (directory / "POTENTIAL").write_text(
        " H GTH-PBE-q1 GTH-PBE\n O GTH-PBE-q6 GTH-PBE\n Ti GTH-PBE-q12 GTH-PBE\n",
        encoding="utf-8",
    )
    return directory


# -- reading ------------------------------------------------------------


def test_a_molecule_is_read_as_non_periodic(water: Path):
    (structure,) = read_structures(water)

    assert structure.label == "water"
    assert structure.symbols == ("O", "H", "H")
    assert structure.periodic is False


def test_a_cell_in_the_file_makes_it_periodic(crystal: Path):
    (structure,) = read_structures(crystal)

    assert structure.periodic is True
    assert structure.cell is not None


def test_a_missing_file_is_refused(tmp_path: Path):
    with pytest.raises(StructureError, match="no such structure file"):
        read_structures(tmp_path / "absent.xyz")


def test_duplicate_labels_across_files_are_refused(water: Path, tmp_path: Path):
    copy = tmp_path / "again.xyz"
    copy.write_text(water.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(StructureError, match="duplicate structure label"):
        read_collection([water, copy])


def test_frames_in_one_file_get_distinct_labels(tmp_path: Path):
    frames = [Atoms("H2", positions=[(0, 0, 0), (0, 0, d)]) for d in (0.7, 0.8)]
    path = tmp_path / "scan.extxyz"
    ase_write(str(path), frames, format="extxyz")

    structures = read_structures(path)

    assert [item.label for item in structures] == ["scan-000", "scan-001"]


# -- descriptors --------------------------------------------------------


def test_descriptors_describe_the_geometry(water: Path):
    (structure,) = read_structures(water)

    detail = descriptors(structure)

    assert detail["formula"] == "H2O1"
    assert detail["elements"] == ["H", "O"]
    assert detail["n_atoms"] == 3
    assert detail["periodic"] is False
    assert detail["min_distance_A"] == pytest.approx(0.96, abs=0.02)


def test_periodic_min_distance_uses_the_minimum_image(crystal: Path):
    """Across the boundary the two atoms are closer than inside the cell."""

    (structure,) = read_structures(crystal)

    detail = descriptors(structure)

    assert detail["cell_volume_A3"] == pytest.approx(27.0)
    assert detail["min_distance_A"] == pytest.approx(np.sqrt(3) * 1.5)


# -- kinds --------------------------------------------------------------


def test_kinds_are_read_from_cp2k_data_files(data_dir: Path):
    kinds = resolve_kinds(["H", "O", "Ti"], data_dir=data_dir)

    assert kinds["Ti"].basis_set == "DZVP-MOLOPT-SR-GTH"
    # The valence comes from the basis alias, so the pair cannot disagree.
    assert kinds["Ti"].potential == "GTH-PBE-q12"


def test_an_element_without_a_matching_potential_is_refused(data_dir: Path):
    """Xx has a basis but no potential: better to fail here than in the queue."""

    with pytest.raises(StructureError, match="Xx"):
        resolve_kinds(["Xx"], data_dir=data_dir)


def test_an_unknown_element_is_refused(data_dir: Path):
    with pytest.raises(StructureError, match="Zz"):
        resolve_kinds(["Zz"], data_dir=data_dir)


def test_an_override_wins(data_dir: Path):
    kinds = resolve_kinds(
        ["H"],
        data_dir=data_dir,
        overrides={"H": {"basis_set": "TZV2P-MOLOPT-GTH", "potential": "GTH-PBE-q1"}},
    )

    assert kinds["H"].basis_set == "TZV2P-MOLOPT-GTH"


def test_a_missing_data_directory_says_so(tmp_path: Path):
    with pytest.raises(StructureError, match="BASIS_MOLOPT"):
        resolve_kinds(["H"], data_dir=tmp_path / "nowhere")


# -- rendering ----------------------------------------------------------


def test_a_molecule_gets_a_vacuum_box(water: Path):
    (structure,) = read_structures(water)

    text = render_input(TEMPLATE, structure, KINDS, vacuum=5.0)

    assert "PERIODIC NONE" in text
    assert "ABC" in text
    assert "PROJECT water" in text
    assert text.count("&KIND") == 2


def test_a_crystal_keeps_its_own_cell(crystal: Path):
    (structure,) = read_structures(crystal)

    text = render_input(TEMPLATE, structure, {"H": KINDS["H"]})

    assert "PERIODIC XYZ" in text
    assert "A     3.00000000     0.00000000     0.00000000" in text


def test_an_unknown_placeholder_is_refused(water: Path):
    (structure,) = read_structures(water)

    with pytest.raises(StructureError, match="CUTOFF"):
        render_input(TEMPLATE + "  {{CUTOFF}}\n", structure, KINDS)


def test_a_template_without_coordinates_is_refused(water: Path):
    (structure,) = read_structures(water)

    with pytest.raises(StructureError, match="COORD"):
        render_input("&GLOBAL\n  PROJECT {{PROJECT}}\n{{KIND}}\n&END GLOBAL\n", structure, KINDS)


def test_a_structure_with_no_kind_is_refused(water: Path):
    (structure,) = read_structures(water)

    with pytest.raises(StructureError, match=r"\['O'\]"):
        render_input(TEMPLATE, structure, {"H": KINDS["H"]})


# -- tasks --------------------------------------------------------------


def test_tasks_carry_the_rendered_input_and_the_descriptors(
    water: Path, data_dir: Path, tmp_path: Path
):
    template = tmp_path / "template.inp"
    template.write_text(TEMPLATE, encoding="utf-8")

    (task,) = tasks_from_collection(
        [water], template_path=template, input_dir=tmp_path / "inputs", data_dir=data_dir
    )

    assert task.task_id == "water"
    assert task.input_path.is_file()
    assert task.input_path.is_absolute()
    assert task.structure["formula"] == "H2O1"
    assert task.structure["kinds"]["O"]["potential"] == "GTH-PBE-q6"


def test_a_plan_records_the_group_on_every_task(water: Path, data_dir: Path, tmp_path: Path):
    template = tmp_path / "template.inp"
    template.write_text(TEMPLATE, encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "groups": [
                    {"name": "molecules", "structures": [str(water)], "template": str(template)}
                ]
            }
        ),
        encoding="utf-8",
    )

    (task,) = tasks_from_plan(
        plan, input_dir=tmp_path / "inputs", data_dir=data_dir, root=tmp_path
    )

    assert task.structure["group"] == "molecules"


def test_selecting_an_absent_group_is_an_error(water: Path, data_dir: Path, tmp_path: Path):
    template = tmp_path / "template.inp"
    template.write_text(TEMPLATE, encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {"groups": [{"name": "molecules", "structures": [str(water)], "template": str(template)}]}
        ),
        encoding="utf-8",
    )

    with pytest.raises(StructureError, match="no groups selected"):
        tasks_from_plan(
            plan, input_dir=tmp_path / "inputs", data_dir=data_dir, root=tmp_path, groups=["oxides"]
        )


def test_the_shipped_plan_and_templates_agree(tmp_path: Path, data_dir: Path):
    """The plan in configs/ must name templates and collections that exist."""

    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "configs/cp2k/pulay_sampling_plan.json").read_text(encoding="utf-8"))

    for group in payload["groups"]:
        assert (root / group["template"]).is_file(), group["name"]
        for item in group["structures"]:
            assert (root / item).is_file(), item
