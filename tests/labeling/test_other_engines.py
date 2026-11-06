"""Optional, UNVALIDATED Gaussian/ORCA writers and the point-charge table (LAB-003, LAB-010)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gmd_active_learning.adapters.dft.cp2k.point_charges import write_point_charges
from gmd_active_learning.contracts.regions import ExtractedRegion
from gmd_active_learning.labeling.other_engines import (
    VALIDATED,
    OtherEngineSettings,
    build_gaussian_input,
    build_orca_input,
    write_gaussian_inputs,
    write_orca_inputs,
)

REGIONS = Path(__file__).resolve().parents[1] / "fixtures" / "structures" / "regions"


def _region(name: str) -> ExtractedRegion:
    return ExtractedRegion.from_dict(json.loads((REGIONS / f"{name}.json").read_text()))


def test_gaussian_and_orca_inputs(tmp_path: Path) -> None:
    assert VALIDATED is False
    radical = _region("ch3_radical")
    gaussian = build_gaussian_input(radical, OtherEngineSettings(basis="def2SVP"))
    assert gaussian.startswith("#p UPBE/def2SVP force\n\nHotspot region\n\n0 2\nC ")
    orca = build_orca_input(radical, OtherEngineSettings(basis="def2-SVP"))
    assert orca.startswith("! UKS PBE def2-SVP EnGrad\n\n* xyz 0 2\n") and orca.endswith("*")
    assert write_gaussian_inputs(radical, tmp_path, OtherEngineSettings(basis="x"), job_name="j")["single_point_input"].name == "j.gjf"
    assert write_orca_inputs(radical, tmp_path, OtherEngineSettings(basis="x"), job_name="j")["single_point_input"].name == "j.inp"


def test_gaussian_h_only_optimisation_freezes_non_cap_atoms() -> None:
    text = build_gaussian_input(_region("ch3_capped"), OtherEngineSettings(basis="def2SVP"), optimize_h_only=True)
    atoms = text.splitlines()[5:10]
    assert text.startswith("#p PBE/def2SVP opt\n")
    assert [line.split()[1] for line in atoms] == ["-1", "-1", "-1", "-1", "0"]


def test_point_charge_table(tmp_path: Path) -> None:
    region = _region("h2_cluster")
    assert write_point_charges(region, tmp_path, job_name="j") is None
    region.metadata["embedding"] = {"status": "ok", "point_charges": [{"position": [1.0, 2.0, 3.0], "charge": -0.5, "element": "O"}]}
    path = write_point_charges(region, tmp_path, job_name="j")
    assert path.name == "j_pointcharges.dat"
    assert path.read_text().splitlines()[-1] == "      1.00000000       2.00000000       3.00000000    -0.500000  O"


# -- LAB-010: identical text to PHAL 96263b6 for fixture regions, with configurable settings ----------

CAPTURE = Path(__file__).resolve().parent / "phal_other_engine_inputs.json"
REF = Path(__file__).resolve().parents[2] / "ref" / "lammps_monitor"
#: (region, functional, Gaussian basis, ORCA basis, charge); charge None = the region's own (0)
ENGINE_CASES = [
    ("ch3_radical", "B3LYP", "def2SVP", "def2-SVP", None),
    ("ch3_capped", "PBE0", "6-31G*", "def2-TZVP", 1),
    ("h2_cluster", "PBE", "cc-pVDZ", "cc-pVDZ", 0),
]


def phal_engine_capture() -> dict[str, str]:
    """PHAL's Gaussian/ORCA text for ENGINE_CASES (needs ref/lammps_monitor on sys.path).

    PHAL read ``config["dft"]``; the 'dft' section was not in its schema, so the dict is passed directly.
    """

    from hotspot_al.labeling.gaussian_input import build_gaussian_input as phal_gaussian
    from hotspot_al.labeling.orca_input import build_orca_input as phal_orca
    from hotspot_al.models import ExtractedRegion as PhalRegion

    texts = {}
    for name, functional, g_basis, o_basis, charge in ENGINE_CASES:
        region = _region(name)
        caps = [index for index, role in enumerate(region.roles) if role.value == "h_cap"]
        phal_region = PhalRegion(atoms=region.to_atoms(), original_indices=list(range(region.n_atoms)), core_indices=[0],
                                 inner_buffer_indices=[], outer_buffer_indices=[], boundary_indices=[], h_cap_indices=caps,
                                 hotspot_indices=[0], metadata={})
        dft = {"functional": functional, "charge": 0 if charge is None else charge}
        texts[f"{name}/gaussian"] = phal_gaussian(phal_region, config={"dft": {**dft, "basis": g_basis}})
        texts[f"{name}/gaussian_h_only"] = phal_gaussian(phal_region, config={"dft": {**dft, "basis": g_basis}}, optimize_h_only=True)
        texts[f"{name}/orca"] = phal_orca(phal_region, config={"dft": {**dft, "basis": o_basis}})
    return texts


def _settings(functional: str, basis: str, charge: int | None) -> OtherEngineSettings:
    # A validated settings object built from plain config data (extra keys are rejected).
    return OtherEngineSettings.model_validate({"functional": functional, "basis": basis, "charge": charge})


def test_gaussian_and_orca_text_is_identical_to_phal() -> None:
    captured = json.loads(CAPTURE.read_text(encoding="utf-8"))
    assert set(captured) == {f"{name}/{kind}" for name, *_ in ENGINE_CASES for kind in ("gaussian", "gaussian_h_only", "orca")}
    for name, functional, g_basis, o_basis, charge in ENGINE_CASES:
        region = _region(name)
        assert build_gaussian_input(region, _settings(functional, g_basis, charge)) == captured[f"{name}/gaussian"], name
        assert build_orca_input(region, _settings(functional, o_basis, charge)) == captured[f"{name}/orca"], name


def test_functional_basis_and_charge_reach_the_text() -> None:
    region = _region("ch3_capped")
    gaussian = build_gaussian_input(region, _settings("PBE0", "6-31G*", 1))
    orca = build_orca_input(region, _settings("PBE0", "def2-TZVP", 1))
    assert gaussian.splitlines()[0] == "#p UPBE0/6-31G* force" and gaussian.splitlines()[4] == "1 2"
    assert orca.splitlines()[0] == "! UKS PBE0 def2-TZVP EnGrad" and orca.splitlines()[2] == "* xyz 1 2"
    assert build_orca_input(region, _settings("PBE0", "def2-TZVP", None)).splitlines()[2] == "* xyz 0 1"
    with pytest.raises(ValidationError):
        OtherEngineSettings.model_validate({"functional": "PBE"})  # basis is required: no non-Gaussian default
    with pytest.raises(ValidationError):
        OtherEngineSettings.model_validate({"basis": "x", "charges": 1})


def test_gaussian_h_only_text_differs_from_phal_only_by_the_freeze_fix() -> None:
    """PHAL's opt=modredundant had no frozen-atom section; GMD freezes non-cap atoms (deliberate)."""

    captured = json.loads(CAPTURE.read_text(encoding="utf-8"))
    region = _region("ch3_capped")
    ours = build_gaussian_input(region, _settings("PBE0", "6-31G*", 1), optimize_h_only=True).splitlines()
    phal = captured["ch3_capped/gaussian_h_only"].splitlines()
    assert phal[0] == "#p UPBE0/6-31G* opt=modredundant" and ours[0] == "#p UPBE0/6-31G* opt"
    assert ours[1:5] == phal[1:5]
    for mine, theirs in zip(ours[5:], phal[5:], strict=True):
        if not theirs:
            assert mine == theirs
            continue
        symbol, freeze, *coords = mine.split()
        assert [symbol, *coords] == theirs.split() and freeze in {"0", "-1"}


def test_phal_engine_capture_is_current(monkeypatch: pytest.MonkeyPatch) -> None:
    if not (REF / "src" / "hotspot_al").is_dir():
        pytest.skip("ref/lammps_monitor not present")
    monkeypatch.syspath_prepend(str(REF / "src"))
    assert phal_engine_capture() == json.loads(CAPTURE.read_text(encoding="utf-8"))
