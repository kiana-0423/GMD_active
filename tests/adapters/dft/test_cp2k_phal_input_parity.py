"""IO-004: CP2K inputs are byte-identical to PHAL's for fixture regions (without needing ref/).

``phal_cp2k_inputs.json`` was captured from PHAL 96263b6 (``hotspot_al.cp2k.cp2k_input.build_cp2k_input``
with PHAL's ``load_config()`` defaults, ``cp2k.periodic`` set to the region's class and
``poisson_solver`` WAVELET or PERIODIC). Both programs receive the same atoms, cell and H-cap indices.
Isolated regions first get GMD's cubic padded cell. The regions are three hand-made fixture regions plus
the golden ethane event evt-80, extracted by GMD in cluster mode and capped with PHAL's capping
settings. GMD's equivalent settings are ``CP2KConfig(stress="never")``, because PHAL never requested
stress. When ``ref/lammps_monitor`` exists, the capture is re-generated from PHAL and compared too.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from extraction.phal_fixtures import capping_config, extraction_config, frame_at, hotspots_for, region_records
from gmd_active_learning.adapters.dft.cp2k.backend import CP2KBackend
from gmd_active_learning.adapters.dft.cp2k.config import CP2KConfig
from gmd_active_learning.adapters.dft.cp2k.input import build_cp2k_input, prepare_geometry
from gmd_active_learning.contracts.regions import ExtractedRegion, RegionRole
from gmd_active_learning.extraction.cluster import extract_cluster
from gmd_active_learning.extraction.h_capping import add_h_caps

HERE = Path(__file__).resolve().parent
CAPTURE = HERE / "phal_cp2k_inputs.json"
REF = Path(__file__).resolve().parents[3] / "ref" / "lammps_monitor"
REGIONS = Path(__file__).resolve().parents[2] / "fixtures" / "structures" / "regions"
CASES = [("ch3_capped", "NONE"), ("ch3_radical", "NONE"), ("h2_periodic", "XYZ"), ("ethane_evt80_cluster", "NONE")]


def _region(name: str) -> ExtractedRegion:
    if name == "ethane_evt80_cluster":
        scenario, event = "ethane_box", "evt-80-00000001"
        raw = region_records(scenario, event, "cluster")[0]["raw_region"]
        frame = frame_at(scenario, int(raw["metadata"]["original_frame_id"]))
        config = extraction_config(scenario, "cluster", periodic_guard={"policy": "flag"})
        region = extract_cluster(frame, hotspots_for(scenario, event, "cluster", 0), config, event_id=event)
        region = add_h_caps(region, frame, capping_config(scenario))
    else:
        region = ExtractedRegion.from_dict(json.loads((REGIONS / f"{name}.json").read_text()))
    if not any(region.pbc):  # give PHAL and GMD the same cubic, padded cell
        geometry = prepare_geometry(region, CP2KConfig())
        region = replace(region, positions=geometry.positions, cell=geometry.cell)
    return region


def _caps(region: ExtractedRegion) -> list[int]:
    return region.indices_with(RegionRole.H_CAP).tolist()


def _run_types(region: ExtractedRegion) -> list[tuple[str, list[int] | None]]:
    caps = _caps(region)
    runs: list[tuple[str, list[int] | None]] = [("ENERGY_FORCE", None)]
    if caps:
        runs.append(("GEO_OPT", [index for index in range(region.n_atoms) if index not in caps]))
    return runs


def phal_capture() -> dict[str, str]:
    """PHAL's inputs for every case (needs ref/lammps_monitor on sys.path)."""

    from ase import Atoms
    from hotspot_al.config import load_config
    from hotspot_al.cp2k.cp2k_input import build_cp2k_input as phal_build
    from hotspot_al.cp2k.cp2k_input import write_cp2k_inputs as phal_write
    from hotspot_al.models import ExtractedRegion as PhalRegion

    texts = {}
    for name, periodic in CASES:
        region = _region(name)
        atoms = Atoms(numbers=region.numbers, positions=region.positions, cell=region.cell, pbc=region.pbc)
        phal_region = PhalRegion(atoms=atoms, original_indices=list(range(region.n_atoms)), core_indices=[0],
                                 inner_buffer_indices=[], outer_buffer_indices=[], boundary_indices=[],
                                 h_cap_indices=_caps(region), hotspot_indices=[0], metadata={})
        config = load_config()
        config["cp2k"]["periodic"] = periodic
        config["cp2k"]["poisson_solver"] = "WAVELET" if periodic == "NONE" else "PERIODIC"
        for run_type, fixed in _run_types(region):
            texts[f"{name}/{run_type}"] = phal_build(phal_region, config=config, run_type=run_type,
                                                     fixed_atom_indices=fixed, project_name="job")
        # The dispatcher (write_cp2k_inputs), given the region's real boundary atoms.
        phal_region.boundary_indices = region.indices_with(RegionRole.BOUNDARY).tolist()
        with tempfile.TemporaryDirectory() as directory:
            for path in phal_write(phal_region, directory, config=config, job_name="job").values():
                texts[f"dispatch/{name}/{path.name}"] = path.read_text(encoding="utf-8")
    return texts


@pytest.fixture(scope="module")
def captured() -> dict[str, str]:
    return json.loads(CAPTURE.read_text(encoding="utf-8"))


def test_capture_covers_every_case_and_run_type(captured: dict[str, str]) -> None:
    expected = {f"{name}/{run}" for name, _ in CASES for run, _fixed in _run_types(_region(name))}
    assert {key for key in captured if not key.startswith("dispatch/")} == expected
    assert {key.split("/")[1] for key in captured if key.startswith("dispatch/")} == {name for name, _ in CASES}
    assert any(key.endswith("GEO_OPT") for key in captured)  # H-only optimisation inputs are covered
    assert _caps(_region("ethane_evt80_cluster"))  # the golden region really carries H caps


@pytest.mark.parametrize(("name", "periodic"), CASES)
def test_cp2k_inputs_are_byte_identical_to_phal(captured: dict[str, str], name: str, periodic: str) -> None:
    region = _region(name)
    assert ("NONE" if not any(region.pbc) else "XYZ") == periodic
    for run_type, fixed in _run_types(region):
        got = build_cp2k_input(region, CP2KConfig(stress="never"), run_type=run_type, project_name="job", fixed_atom_indices=fixed).text
        assert got.encode() == captured[f"{name}/{run_type}"].encode(), f"{name} {run_type}"


@pytest.mark.parametrize(("name", "periodic"), CASES)
def test_cp2k_dispatch_writes_phal_files_byte_identically(tmp_path: Path, captured: dict[str, str], name: str, periodic: str) -> None:
    """CP2KBackend.prepare_inputs (what the labeling service calls) vs PHAL's write_cp2k_inputs."""

    region = _region(name)
    written = CP2KBackend(CP2KConfig(stress="never")).prepare_inputs(region, tmp_path, task_id="job")
    expected = {key.split("/", 2)[2]: text for key, text in captured.items() if key.startswith(f"dispatch/{name}/")}
    assert {path.name for path in written.values()} == set(expected)
    for path in written.values():
        assert path.read_bytes() == expected[path.name].encode(), f"{name} {path.name}"


def test_unknown_dft_engine_raises() -> None:
    from gmd_active_learning.adapters.registry import BackendRole, create_backend
    from gmd_active_learning.core.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="gaussian"):
        create_backend({"backends": {"dft": {"engine": "gaussian"}}}, BackendRole.DFT)


@pytest.mark.skipif(not (REF / "src" / "hotspot_al").is_dir(), reason="ref/lammps_monitor not present")
def test_capture_is_what_phal_writes_today(monkeypatch: pytest.MonkeyPatch, captured: dict[str, str]) -> None:
    monkeypatch.syspath_prepend(str(REF / "src"))
    assert phal_capture() == captured
