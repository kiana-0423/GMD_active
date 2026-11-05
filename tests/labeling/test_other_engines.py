"""Optional, UNVALIDATED Gaussian/ORCA writers and the point-charge table (LAB-003, LAB-010)."""

from __future__ import annotations

import json
from pathlib import Path

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
