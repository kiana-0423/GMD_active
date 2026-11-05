# Ported from PHAL (hotspot_al tests/unit/test_lammps_reader.py, test_lammps_input_full.py,
# test_lammps_controller.py) at 96263b6; MIT License, see NOTICE.
"""Legacy LAMMPS dump importer: PHAL parsing semantics on the Frame contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read as ase_read
from ase.units import kcal, mol

from gmd_active_learning.contracts import Frame
from gmd_active_learning.core.exceptions import DataContractError
from gmd_active_learning.importers import iter_lammps_dump, parse_lammps_dump_frame, read_lammps_dump

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "lammps"
TOY = FIXTURES / "toy_hotspot.lammpstrj"
PHAL_SRC = REPO_ROOT / "ref" / "lammps_monitor" / "src"
FIXTURE_TYPE_MAPS = {"toy_hotspot.lammpstrj": {1: "C", 3: "H"}}


def _dump(
    *,
    step: int = 0,
    bounds: str = "pp pp pp",
    box: tuple[str, str, str] = ("0.0 10.0", "0.0 10.0", "0.0 30.0"),
    fields: str = "id type x y z",
    rows: list[str] | None = None,
) -> str:
    rows = rows if rows is not None else ["1 1 1.0 1.0 1.0", "2 1 2.0 2.0 2.0"]
    return (
        f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{len(rows)}\n"
        f"ITEM: BOX BOUNDS {bounds}\n" + "\n".join(box) + f"\nITEM: ATOMS {fields}\n" + "\n".join(rows) + "\n"
    )


def _write(tmp_path: Path, text: str, name: str = "dump.lammpstrj") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# -- PHAL test_lammps_reader.py ----------------------------------------------------------------


def test_lammps_dump_reader_parses_positions_forces_cell_and_step() -> None:
    frames = read_lammps_dump(TOY, type_map={1: "C", 3: "H"}, timestep_fs=0.5)
    assert len(frames) == 1

    frame = frames[0]
    assert isinstance(frame, Frame)
    assert frame.step == 100
    assert frame.time_fs == 50.0
    assert frame.n_atoms == 8
    assert frame.symbols == ["C", "C", "H", "H", "H", "H", "H", "H"]
    assert frame.forces is not None and frame.forces.shape == (8, 3)
    assert np.allclose(frame.positions[0], [4.0, 6.0, 6.0])
    assert np.allclose(frame.forces[0], [9.0, 0.0, 0.0])
    assert np.allclose(frame.cell, np.diag([12.0, 12.0, 12.0]))
    assert frame.pbc == (True, True, True)
    assert frame.atom_ids.tolist() == list(range(1, 9))
    assert frame.run_id == "lammps-import"
    assert frame.metadata["types"] == [1, 1, 3, 3, 3, 3, 3, 3]
    assert frame.metadata["origin"] == [0.0, 0.0, 0.0]
    assert frame.metadata["source_path"] == str(TOY)
    assert frame.energy is None and frame.velocities is None


def test_lammps_dump_reader_converts_scaled_coordinates_without_origin_shift(tmp_path: Path) -> None:
    path = _write(tmp_path, _dump(box=("10 20", "-5 5", "2 12"), fields="id type xs ys zs", rows=["1 1 0.25 0.50 0.75", "2 1 0.00 0.00 0.00"]))

    frame = read_lammps_dump(path, type_map={1: "H"})[0]

    assert np.allclose(frame.cell, np.diag([10.0, 10.0, 10.0]))
    assert np.allclose(frame.positions[0], [2.5, 5.0, 7.5])
    assert np.allclose(frame.positions[1], [0.0, 0.0, 0.0])
    assert frame.metadata["coordinate_style"] == "xs"
    assert frame.metadata["origin"] == [10.0, -5.0, 2.0]


def test_truncated_dump_file_raises(tmp_path: Path) -> None:
    text = _dump(rows=["1 1 0.0 0.0 0.0", "2 1 1.0 1.0 1.0"]).splitlines()[:-1]
    path = _write(tmp_path, "\n".join(text))

    with pytest.raises(DataContractError, match="truncated atom section"):
        read_lammps_dump(path, type_map={1: "H"})


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ("pp pp pp", (True, True, True)),
        ("pp pp ff", (True, True, False)),
        ("pp pp ss", (True, True, False)),
        # 'm' (minimum shrink-wrapped) is the usual free-surface slab setup; PHAL once read it as periodic.
        ("pp pp mm", (True, True, False)),
        ("pp mm mm", (True, False, False)),
        ("pp ss sm", (True, False, False)),
        ("ff ff ff", (False, False, False)),
        ("mm mm mm", (False, False, False)),
        ("fs sf ms", (False, False, False)),
        ("pp fm pp", (True, False, True)),
    ],
)
def test_all_lammps_boundary_styles_map_to_pbc(tmp_path: Path, bounds: str, expected: tuple[bool, bool, bool]) -> None:
    frame = read_lammps_dump(_write(tmp_path, _dump(bounds=bounds)), type_map={1: "Cu"})[0]

    assert frame.pbc == expected


def test_triclinic_header_does_not_confuse_boundary_detection(tmp_path: Path) -> None:
    path = _write(tmp_path, _dump(bounds="xy xz yz pp pp mm", box=("0.0 10.0 0.0", "0.0 10.0 0.0", "0.0 30.0 0.0")))

    frame = read_lammps_dump(path, type_map={1: "Cu"})[0]

    assert frame.pbc == (True, True, False)


def test_missing_boundary_styles_default_to_periodic_as_phal(tmp_path: Path) -> None:
    frame = read_lammps_dump(_write(tmp_path, _dump(bounds="")), type_map={1: "Cu"})[0]

    assert frame.pbc == (True, True, True)


# -- triclinic bounds, origin, ordering ---------------------------------------------------------


TRICLINIC = _dump(
    step=5,
    bounds="xy xz yz pp pp ff",
    box=("-1.0 13.5 2.0", "1.0 11.5 -1.5", "2.0 12.0 0.5"),
    fields="id type x y z fx fy fz",
    rows=["2 1 3.0 4.0 5.0 0.1 0.2 0.3", "1 2 1.0 2.0 3.0 -0.1 0.0 0.0", "3 1 6.0 7.0 8.0 0.0 0.0 1.0"],
)


def test_triclinic_bounds_give_the_lammps_cell_and_origin(tmp_path: Path) -> None:
    frame = read_lammps_dump(_write(tmp_path, TRICLINIC), type_map={1: "H", 2: "O"})[0]

    # xlo = xlo_bound - min(0, xy, xz, xy + xz) = -1 - (-1.5) = 0.5; xhi = 13.5 - 2 = 11.5; ylo = 1.0; yhi = 11.5 - 0.5.
    np.testing.assert_allclose(frame.cell, [[11.0, 0.0, 0.0], [2.0, 10.0, 0.0], [-1.5, 0.5, 10.0]], rtol=0, atol=1e-12)
    assert frame.metadata["origin"] == [0.5, 1.0, 2.0]
    assert frame.pbc == (True, True, False)
    # Sorted by id: the row of id 1 comes first, forces travel with their atom.
    assert frame.atom_ids.tolist() == [1, 2, 3]
    assert frame.symbols == ["O", "H", "H"]
    np.testing.assert_allclose(frame.positions, [[0.5, 1.0, 1.0], [2.5, 3.0, 3.0], [5.5, 6.0, 6.0]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(frame.forces, [[-0.1, 0.0, 0.0], [0.1, 0.2, 0.3], [0.0, 0.0, 1.0]], rtol=0, atol=1e-12)


def test_triclinic_cell_agrees_with_ase_independent_reader(tmp_path: Path) -> None:
    path = _write(tmp_path, TRICLINIC)
    reference = ase_read(path, format="lammps-dump-text")
    frame = read_lammps_dump(path, type_map={1: "H", 2: "O"})[0]

    np.testing.assert_allclose(frame.cell, reference.cell.array, rtol=1e-12, atol=1e-12)
    origin = np.asarray(frame.metadata["origin"])
    np.testing.assert_allclose(origin, reference.get_celldisp().ravel(), rtol=1e-12, atol=1e-12)
    # ASE keeps raw LAMMPS coordinates plus celldisp; PHAL (and this importer) shift by -origin.
    np.testing.assert_allclose(frame.positions + origin, reference.positions, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(frame.forces, reference.get_forces(), rtol=1e-12, atol=1e-12)
    assert frame.pbc == tuple(bool(flag) for flag in reference.pbc)


@pytest.mark.parametrize("style", ["x", "xu"])
def test_cartesian_positions_are_shifted_by_the_box_origin(tmp_path: Path, style: str) -> None:
    fields = "id type x y z" if style == "x" else "id type xu yu zu"
    path = _write(tmp_path, _dump(box=("-5 5", "10 20", "0 10"), fields=fields, rows=["1 1 -4.0 11.0 0.5"]))

    frame = read_lammps_dump(path, type_map={1: "Ar"})[0]

    np.testing.assert_allclose(frame.positions, [[1.0, 1.0, 0.5]], rtol=0, atol=1e-12)
    assert frame.metadata["origin"] == [-5.0, 10.0, 0.0]
    assert frame.metadata["coordinate_style"] == style


def test_x_columns_take_precedence_over_unwrapped_and_scaled(tmp_path: Path) -> None:
    path = _write(tmp_path, _dump(fields="id type xs ys zs xu yu zu x y z", rows=["1 1 0.5 0.5 0.5 21.0 1.0 1.0 1.0 2.0 3.0"]))

    frame = read_lammps_dump(path, type_map={1: "H"})[0]

    np.testing.assert_allclose(frame.positions, [[1.0, 2.0, 3.0]])


def test_rows_are_sorted_by_atom_id_with_every_column(tmp_path: Path) -> None:
    rows = [
        "3 2 3.0 0.0 0.0 0.3 0.0 0.0 -0.5 7",
        "1 1 1.0 0.0 0.0 0.1 0.0 0.0 0.25 5",
        "2 1 2.0 0.0 0.0 0.2 0.0 0.0 0.25 5",
    ]
    path = _write(tmp_path, _dump(box=("0 20", "0 20", "0 20"), fields="id type x y z fx fy fz q mol", rows=rows))

    frame = read_lammps_dump(path, type_map={1: "H", 2: "O"})[0]

    assert frame.atom_ids.tolist() == [1, 2, 3]
    assert frame.symbols == ["H", "H", "O"]
    np.testing.assert_allclose(frame.positions[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(frame.forces[:, 0], [0.1, 0.2, 0.3])
    assert frame.metadata["types"] == [1, 1, 2]
    assert frame.metadata["charges"] == [0.25, 0.25, -0.5]
    assert frame.metadata["molecule_ids"] == [5, 5, 7]


def test_noncontiguous_lammps_ids_are_kept_as_stable_atom_ids(tmp_path: Path) -> None:
    rows = ["40 1 1.0 1.0 1.0", "7 1 2.0 2.0 2.0", "1001 1 3.0 3.0 3.0"]
    frame = read_lammps_dump(_write(tmp_path, _dump(rows=rows)), type_map={1: "H"})[0]

    assert frame.atom_ids.tolist() == [7, 40, 1001]
    np.testing.assert_array_equal(frame.index_of([1001, 7]), [2, 0])


def test_element_column_velocities_and_energy(tmp_path: Path) -> None:
    rows = ["1 Cu 1.0 1.0 1.0 1.0 2.0 3.0 -3.5", "2 O 2.0 2.0 2.0 0.0 0.0 -1.0 -1.25"]
    path = _write(tmp_path, _dump(fields="id element x y z vx vy vz c_pe", rows=rows))

    frame = read_lammps_dump(path)[0]  # no type_map needed with an element column

    assert frame.symbols == ["Cu", "O"]
    assert frame.energy == pytest.approx(-4.75)
    # metal units: A/ps -> A/fs
    np.testing.assert_allclose(frame.velocities, [[1.0e-3, 2.0e-3, 3.0e-3], [0.0, 0.0, -1.0e-3]])
    assert "types" not in frame.metadata


def test_real_units_are_converted_to_ev(tmp_path: Path) -> None:
    rows = ["1 1 1.0 1.0 1.0 1.0 0.0 0.0 0.5 2.0 0.0 0.0", "2 1 3.0 1.0 1.0 -1.0 0.0 0.0 0.5 0.0 0.0 0.0"]
    path = _write(tmp_path, _dump(fields="id type x y z fx fy fz c_pe vx vy vz", rows=rows))

    frame = read_lammps_dump(path, type_map={1: "Ar"}, units="real")[0]

    np.testing.assert_allclose(frame.forces[:, 0], [kcal / mol, -kcal / mol])
    assert frame.energy == pytest.approx(kcal / mol)
    np.testing.assert_allclose(frame.velocities[0], [2.0, 0.0, 0.0])  # real: already A/fs
    assert frame.metadata["lammps_units"] == "real"


def test_unsupported_units_are_rejected() -> None:
    with pytest.raises(DataContractError, match="unsupported LAMMPS units"):
        read_lammps_dump(TOY, type_map={1: "C", 3: "H"}, units="lj")  # type: ignore[arg-type]


def test_type_map_may_be_a_sequence_or_have_string_keys(tmp_path: Path) -> None:
    path = _write(tmp_path, _dump(rows=["1 1 1.0 1.0 1.0", "2 2 2.0 2.0 2.0"]))

    assert read_lammps_dump(path, type_map=["C", "H"])[0].symbols == ["C", "H"]
    assert read_lammps_dump(path, type_map={"1": "O", "2": "H"})[0].symbols == ["O", "H"]


def test_run_id_and_time_are_propagated() -> None:
    frame = read_lammps_dump(TOY, type_map={1: "C", 3: "H"}, run_id="legacy-run-7", timestep_fs=2.0)[0]

    assert frame.run_id == "legacy-run-7"
    assert frame.time_fs == 200.0
    assert frame.to_atoms().info["run_id"] == "legacy-run-7"


# -- strict checks ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n", "truncated frame header"),
        (_dump().replace("ITEM: TIMESTEP\n0", "ITEM: TIMESTEP\nzero"), "invalid timestep"),
        (_dump().replace("NUMBER OF ATOMS", "NUMBER OF THINGS"), "missing NUMBER OF ATOMS"),
        (_dump().replace("ITEM: NUMBER OF ATOMS\n2", "ITEM: NUMBER OF ATOMS\ntwo"), "invalid atom count"),
        (_dump().replace("ITEM: ATOMS", "ITEM: STUFF"), "missing ATOMS section"),
        (_dump(box=("0 10", "0 ten", "0 10")), "non-numeric BOX BOUNDS"),
        (_dump(box=("0 10", "0 10 0", "0 10")), "2 \\(orthogonal\\) or 3 \\(triclinic\\)"),
        (_dump(bounds="abc origin pp pp pp", box=("10 0 0 0", "0 10 0 0", "0 0 10 0")), "abc origin"),
        (_dump(rows=["1 1 1.0 1.0", "2 1 2.0 2.0 2.0"]), "fewer columns"),
        (_dump(rows=["1 1 1.0 x 1.0", "2 1 2.0 2.0 2.0"]), "non-numeric values in column 'y'"),
        (_dump(rows=["1 1 1.0 1.0 1.0", "1 1 2.0 2.0 2.0"]), "duplicate atom ids"),
        (_dump(fields="id type a b c"), "x/y/z, xu/yu/zu, or xs/ys/zs"),
        (_dump(fields="id mass x y z", rows=["1 1.0 1.0 1.0 1.0"]), "either 'element' or 'type'"),
        (_dump(rows=["1 2 1.0 1.0 1.0"]), "LAMMPS type 2 is missing from type_map"),
        (_dump(fields="id element x y z", rows=["1 Qq 1.0 1.0 1.0"]), "unknown species 'Qq'"),
        (_dump(bounds="pp pp pp", box=("0 0", "0 10", "0 10")), "non-degenerate cell"),
    ],
)
def test_malformed_dumps_raise_data_contract_error(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(DataContractError, match=message):
        read_lammps_dump(_write(tmp_path, text), type_map={1: "H"})


def test_type_map_is_required_without_element_column() -> None:
    with pytest.raises(DataContractError, match="type_map is required"):
        read_lammps_dump(TOY)


def test_unknown_species_in_type_map_fails_before_reading(tmp_path: Path) -> None:
    missing = tmp_path / "never_opened.lammpstrj"
    with pytest.raises(DataContractError, match="unknown species 'Xx'"):
        iter_lammps_dump(missing, type_map={1: "Xx"})
    with pytest.raises(DataContractError, match="does not exist"):
        iter_lammps_dump(missing, type_map={1: "H"})


def test_a_frame_cut_by_the_next_frame_header_is_truncated(tmp_path: Path) -> None:
    first = _dump(step=0).replace("NUMBER OF ATOMS\n2", "NUMBER OF ATOMS\n3")
    with pytest.raises(DataContractError, match="truncated atom section"):
        read_lammps_dump(_write(tmp_path, first + _dump(step=1)), type_map={1: "H"})


# -- PHAL test_lammps_controller.py (parsing parts) ---------------------------------------------


def _controller_frame(step: int, n_atoms: int = 3) -> str:
    header = (
        f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{n_atoms}\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\nITEM: ATOMS id type x y z\n"
    )
    return header + "".join(f"{index + 1} 1 {index}.0 2.0 3.0\n" for index in range(n_atoms))


def test_concatenated_frames_stream_in_order_exactly_once(tmp_path: Path) -> None:
    path = _write(tmp_path, "".join(_controller_frame(step, n_atoms=20) for step in range(12)))

    iterator = iter_lammps_dump(path, type_map={1: "Cu"})
    first = next(iterator)  # lazy: frames are produced one at a time
    rest = list(iterator)

    assert [frame.step for frame in [first, *rest]] == list(range(12))
    assert all(frame.n_atoms == 20 for frame in rest)


def test_streaming_frame_with_forces_matches_phal_controller_expectation(tmp_path: Path) -> None:
    frame0 = "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS pp pp pp\n0 5\n0 5\n0 5\nITEM: ATOMS id type x y z fx fy fz\n1 1 0 0 0 1 0 0\n"
    frame1 = frame0.replace("TIMESTEP\n0", "TIMESTEP\n1").replace("1 1 0 0 0 1 0 0", "1 1 0.1 0 0 0.5 0 0")

    frames = read_lammps_dump(_write(tmp_path, frame0 + frame1), type_map={1: "H"}, timestep_fs=0.5)

    assert [frame.step for frame in frames] == [0, 1]
    assert frames[0].forces[0, 0] == 1.0 and frames[1].forces[0, 0] == 0.5
    assert frames[1].time_fs == 0.5


def test_text_between_frames_is_skipped(tmp_path: Path) -> None:
    path = _write(tmp_path, "LAMMPS banner\n" + _dump(step=3) + "garbage line\n\n" + _dump(step=4))

    assert [frame.step for frame in read_lammps_dump(path, type_map={1: "H"})] == [3, 4]


def test_supports_read_text_shim() -> None:
    class Shim:
        def read_text(self, encoding: str = "utf-8") -> str:
            return _dump(step=9)

    frames = read_lammps_dump(Shim(), type_map={1: "H"})

    assert [frame.step for frame in frames] == [9]
    assert "source_path" not in frames[0].metadata


def test_parse_single_frame_from_text_or_lines() -> None:
    text = TOY.read_text(encoding="utf-8")

    from_text = parse_lammps_dump_frame(text, type_map={1: "C", 3: "H"}, source_path="toy")
    from_lines = parse_lammps_dump_frame(text.splitlines(), type_map={1: "C", 3: "H"})

    np.testing.assert_array_equal(from_text.positions, from_lines.positions)
    assert from_text.metadata["source_path"] == "toy"
    with pytest.raises(DataContractError, match="exactly one"):
        parse_lammps_dump_frame(text + text, type_map={1: "C", 3: "H"})
    with pytest.raises(DataContractError, match="found 0"):
        parse_lammps_dump_frame("", type_map={1: "C", 3: "H"})


# -- PHAL test_lammps_input_full.py (type-map round trip) ---------------------------------------


def _round_trip_dump(original: Atoms, type_map: dict[int, str]) -> str:
    symbol_to_type = {symbol: lammps_type for lammps_type, symbol in type_map.items()}
    rows = [
        f"{index + 1} {symbol_to_type[symbol]} {x} {y} {z}"
        for index, (symbol, (x, y, z)) in enumerate(zip(original.get_chemical_symbols(), original.positions))
    ]
    return _dump(box=("0.0 20.0", "0.0 20.0", "0.0 20.0"), rows=rows)


def test_element_identity_survives_the_type_map_round_trip(tmp_path: Path) -> None:
    type_map = {1: "C", 2: "O", 3: "H"}
    original = Atoms("COHHOC", positions=[[i * 1.4, 0.0, 0.0] for i in range(6)], cell=[20.0] * 3, pbc=True)

    decoded = read_lammps_dump(_write(tmp_path, _round_trip_dump(original, type_map)), type_map=type_map)[0]

    assert decoded.symbols == original.get_chemical_symbols()
    np.testing.assert_allclose(decoded.positions, original.positions, atol=1e-6)


def test_round_trip_is_order_sensitive_so_the_test_can_actually_fail(tmp_path: Path) -> None:
    original = Atoms("COHHOC", positions=[[i * 1.4, 0.0, 0.0] for i in range(6)], cell=[20.0] * 3, pbc=True)
    path = _write(tmp_path, _round_trip_dump(original, {1: "C", 2: "O", 3: "H"}))

    decoded = read_lammps_dump(path, type_map={1: "C", 2: "H", 3: "O"})[0]

    assert decoded.symbols != original.get_chemical_symbols()


# -- acceptance: every fixture dump, hard-coded and against PHAL --------------------------------


def _fixture_dumps() -> list[Path]:
    return sorted(path for path in FIXTURES.iterdir() if path.suffix in {".lammpstrj", ".dump"})


def test_fixture_directory_has_the_phal_dumps() -> None:
    assert [path.name for path in _fixture_dumps()] == sorted(FIXTURE_TYPE_MAPS)


def test_toy_fixture_hard_coded_expectations() -> None:
    frame = read_lammps_dump(TOY, type_map=FIXTURE_TYPE_MAPS[TOY.name])[0]
    expected_positions = np.array(
        [
            [4.00, 6.00, 6.00], [5.54, 6.00, 6.00], [3.37, 6.90, 6.00], [3.37, 5.55, 6.78],
            [3.37, 5.55, 5.22], [6.17, 6.90, 6.00], [6.17, 5.55, 6.78], [6.17, 5.55, 5.22],
        ]
    )
    expected_forces = np.zeros((8, 3))
    expected_forces[0, 0], expected_forces[1, 0] = 9.0, 0.1

    np.testing.assert_allclose(frame.positions, expected_positions, rtol=1e-12, atol=0)
    np.testing.assert_allclose(frame.forces, expected_forces, rtol=1e-12, atol=0)
    np.testing.assert_allclose(frame.cell, np.diag([12.0, 12.0, 12.0]), rtol=1e-12, atol=0)
    assert frame.pbc == (True, True, True)


@pytest.mark.skipif(not PHAL_SRC.is_dir(), reason="PHAL reference tree ref/lammps_monitor is absent")
@pytest.mark.parametrize("dump_path", _fixture_dumps(), ids=lambda path: path.name)
def test_fixture_dumps_match_phal_parser(dump_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """MD-002 acceptance: identical positions, cell, pbc and forces (rtol 1e-12) to PHAL's parser."""

    monkeypatch.syspath_prepend(str(PHAL_SRC))
    for name in [module for module in sys.modules if module == "hotspot_al" or module.startswith("hotspot_al.")]:
        monkeypatch.delitem(sys.modules, name)
    from hotspot_al.lammps.dump_parser import read_lammps_dump as phal_read_lammps_dump  # type: ignore[import-not-found]

    type_map = FIXTURE_TYPE_MAPS[dump_path.name]
    ours = read_lammps_dump(dump_path, type_map=type_map, timestep_fs=0.5)
    theirs = phal_read_lammps_dump(dump_path, type_map=type_map, timestep_fs=0.5)

    assert len(ours) == len(theirs) > 0
    for frame, reference in zip(ours, theirs):
        assert frame.step == reference.step
        assert frame.time_fs == reference.time
        np.testing.assert_allclose(frame.positions, reference.atoms.get_positions(), rtol=1e-12, atol=0)
        np.testing.assert_allclose(frame.cell, reference.atoms.cell.array, rtol=1e-12, atol=0)
        assert frame.pbc == tuple(bool(flag) for flag in reference.atoms.pbc)
        np.testing.assert_allclose(frame.forces, reference.forces, rtol=1e-12, atol=0)
        assert frame.symbols == reference.atoms.get_chemical_symbols()
        assert frame.atom_ids.tolist() == reference.metadata["ids"]
        assert frame.metadata["origin"] == reference.metadata["origin"]
        assert frame.metadata.get("types") == reference.metadata.get("types")
