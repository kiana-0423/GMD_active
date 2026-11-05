"""Regenerate the real CP2K fixtures in ``tests/fixtures/cp2k/real_2026.2/`` and the region fixtures.

Run from the repository root with a CP2K binary on PATH (or ``CP2K_BIN``)::

    python tests/fixtures/cp2k/generate_real_cp2k_fixtures.py

Inputs are produced by the GMD CP2K adapter (``build_cp2k_input``) from the region fixtures in
``tests/fixtures/structures/regions/``; outputs are CP2K's own files, except that the host name, user
name and process id in the ``PROGRAM STARTED`` banner are replaced by placeholders (same column
width). ``manifest.json`` records the CP2K version, settings and SHA-256 of every file.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from gmd_active_learning.adapters.dft.cp2k.config import CP2KConfig  # noqa: E402
from gmd_active_learning.adapters.dft.cp2k.input import build_cp2k_input, h_opt_fixed_atoms  # noqa: E402
from gmd_active_learning.contracts.ids import file_sha256  # noqa: E402
from gmd_active_learning.contracts.regions import ExtractedRegion, RegionRole  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
REGIONS = FIXTURES / "structures" / "regions"
OUT = FIXTURES / "cp2k" / "real_2026.2"

# Small, fast settings (a few seconds per run); validity rules do not depend on accuracy.
SETTINGS = {"cutoff": 150, "rel_cutoff": 40, "max_scf": 30, "scf_eps": 1.0e-5, "environment": {"OMP_NUM_THREADS": "1"}}


def _region(region_id: str, numbers, positions, *, cell=None, pbc=(False, False, False), roles=None, metadata=None, charge=0):
    n = len(numbers)
    roles = roles or [RegionRole.CORE] * n
    parent = []
    next_id = 0
    for role in roles:
        parent.append(-1 if role is RegionRole.H_CAP else next_id)
        next_id += role is not RegionRole.H_CAP
    return ExtractedRegion(
        region_id=region_id,
        mode="cluster" if not any(pbc) else ("slab" if not all(pbc) else "block"),
        run_id="fixture",
        parent_step=0,
        event_id="evt-fixture",
        numbers=numbers,
        positions=np.asarray(positions, dtype=float),
        cell=cell,
        pbc=pbc,
        parent_atom_ids=parent,
        image_shifts=np.zeros((n, 3), dtype=int),
        roles=roles,
        force_weights=[0.0 if role is RegionRole.H_CAP else 1.0 for role in roles],
        frozen=[False] * n,
        hotspot_atom_ids=[0],
        charge=charge,
        metadata=metadata or {},
    )


def build_regions() -> dict[str, ExtractedRegion]:
    # Methyl group cut from ethane (C-C 1.53 A, C-H 1.09 A, H-C-C 111 deg); the C-C bond is capped
    # with H at 1.09 A along the bond, as extraction does.
    theta = np.radians(111.0)
    hydrogens = [
        [1.09 * np.sin(theta) * np.cos(phi), 1.09 * np.sin(theta) * np.sin(phi), 1.09 * np.cos(theta)]
        for phi in np.radians([0.0, 120.0, 240.0])
    ]
    methyl = [[0.0, 0.0, 0.0], *hydrogens]
    return {
        "h2_cluster": _region("h2_cluster", [1, 1], [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
        "ch3_capped": _region(
            "ch3_capped",
            [6, 1, 1, 1, 1],
            [*methyl, [0.0, 0.0, 1.09]],
            roles=[RegionRole.CORE, *[RegionRole.INNER_BUFFER] * 3, RegionRole.H_CAP],
            metadata={"caps": [{"parent_atom_id": 0, "broken_partner_atom_id": 4, "bond_length": 1.09}]},
        ),
        "ch3_radical": _region("ch3_radical", [6, 1, 1, 1], methyl, metadata={"h_capping_skipped": False}),
        "h2_periodic": _region(
            "h2_periodic", [1, 1], [[2.13, 2.5, 2.5], [2.87, 2.5, 2.5]], cell=np.diag([5.0, 5.0, 5.0]), pbc=(True, True, True)
        ),
        "h2_slab": _region(
            "h2_slab", [1, 1], [[2.13, 2.5, 5.0], [2.87, 2.5, 5.0]], cell=np.diag([5.0, 5.0, 10.0]), pbc=(True, True, False)
        ),
    }


def _sanitize(text: str) -> str:
    def blank(match: re.Match[str]) -> str:
        return match.group(1) + "X" * len(match.group(2))

    for label in ("PROGRAM STARTED ON", "PROGRAM STARTED BY", "PROGRAM RAN ON", "PROGRAM RAN BY", "PROGRAM PROCESS ID"):
        text = re.sub(rf"({label}\s+)(\S.*)", blank, text)
    return text


def _run(cp2k: str, case: str, text: str, project: str) -> None:
    with tempfile.TemporaryDirectory(prefix="gmdfx", dir="/tmp") as scratch:
        work = Path(scratch)
        (work / f"{project}.inp").write_text(text, encoding="utf-8")
        env = {**os.environ, "OMP_NUM_THREADS": "1"}
        result = subprocess.run([cp2k, "-i", f"{project}.inp", "-o", f"{project}.out"], cwd=work, env=env, capture_output=True, text=True, timeout=900)
        target = OUT / case
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        (target / f"{project}.inp").write_text(text, encoding="utf-8")
        (target / f"{project}.out").write_text(_sanitize((work / f"{project}.out").read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
        trajectory = work / f"{project}-pos-1.xyz"
        if trajectory.exists():
            shutil.copy(trajectory, target / trajectory.name)
        (target / "exit_code").write_text(f"{result.returncode}\n", encoding="utf-8")
        print(f"{case}: exit {result.returncode}")


def main() -> None:
    cp2k = os.environ.get("CP2K_BIN") or shutil.which("cp2k.psmp") or shutil.which("cp2k.popt")
    if not cp2k:
        raise SystemExit("CP2K not found (set CP2K_BIN)")
    regions = build_regions()
    REGIONS.mkdir(parents=True, exist_ok=True)
    for name, region in regions.items():
        (REGIONS / f"{name}.json").write_text(json.dumps(region.to_dict(), indent=1) + "\n", encoding="utf-8")
    base = CP2KConfig.model_validate(SETTINGS)
    cases = {
        "h2_sp": (regions["h2_cluster"], base, "ENERGY_FORCE"),
        "h2_periodic_sp": (regions["h2_periodic"], base, "ENERGY_FORCE"),
        "h2_slab_sp": (regions["h2_slab"], base, "ENERGY_FORCE"),
        "h2_unconverged": (regions["h2_cluster"], base.model_copy(update={"max_scf": 2}), "ENERGY_FORCE"),
        "ch3_capped_hopt": (regions["ch3_capped"], base, "GEO_OPT"),
        "ch3_capped_hopt_maxiter": (
            regions["ch3_capped"],
            CP2KConfig.model_validate({**SETTINGS, "h_only_opt": {"max_iter": 1}}),
            "GEO_OPT",
        ),
        "ch3_capped_sp": (regions["ch3_capped"], base, "ENERGY_FORCE"),
    }
    manifest: dict[str, object] = {"cp2k": cp2k, "settings": SETTINGS, "cases": {}}
    for case, (region, config, run_type) in cases.items():
        project = "hopt" if run_type == "GEO_OPT" else "sp"
        fixed = h_opt_fixed_atoms(region) if run_type == "GEO_OPT" else None
        cp2k_input = build_cp2k_input(region, config, run_type=run_type, project_name=project, fixed_atom_indices=fixed)
        _run(cp2k, case, cp2k_input.text, project)
        manifest["cases"][case] = {  # type: ignore[index]
            "region": f"structures/regions/{region.region_id}.json",
            "run_type": run_type,
            "offset": cp2k_input.geometry.offset.tolist(),
            "files": {path.name: file_sha256(path) for path in sorted((OUT / case).iterdir())},
        }
    version = subprocess.run([cp2k, "--version"], capture_output=True, text=True).stdout.splitlines()[:2]
    manifest["cp2k_version"] = [line.strip() for line in version]
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
