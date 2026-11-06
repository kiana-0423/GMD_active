"""Regenerate ``legacy_ethane_kick.lammpstrj``: a deterministic 24-frame LAMMPS custom dump.

The toy PHAL ethane geometry (``toy_ethane.lammpstrj``) jiggles by 0.01 A with 0.05 eV/A force
noise; at frame 20 (step 200) atom id 1 receives a 33 eV/A force kick. Rows are written in shuffled
id order with a non-zero box origin, as real LAMMPS dumps may be. Synthetic, no physical meaning.

    python examples/data/make_legacy_ethane_dump.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gmd_active_learning.importers import read_lammps_dump

HERE = Path(__file__).resolve().parent


def build(n_frames: int = 24, kick_frame: int = 20, seed: int = 11) -> str:
    base = read_lammps_dump(HERE / "toy_ethane.lammpstrj", type_map={1: "C", 3: "H"})[0]
    rng = np.random.default_rng(seed)
    origin = np.array([-6.0, -6.0, -6.0])
    blocks = []
    for index in range(n_frames):
        positions = base.positions + origin + 0.01 * rng.standard_normal(base.positions.shape)
        forces = 0.05 * rng.standard_normal(base.positions.shape)
        if index == kick_frame:
            forces[0] = [25.0, -18.0, 12.0]
        order = rng.permutation(base.n_atoms)
        rows = [
            f"{base.atom_ids[i]} {base.metadata['types'][i]} " + " ".join(f"{value:.8f}" for value in (*positions[i], *forces[i]))
            for i in order
        ]
        blocks.append(
            f"ITEM: TIMESTEP\n{index * 10}\nITEM: NUMBER OF ATOMS\n{base.n_atoms}\n"
            "ITEM: BOX BOUNDS pp pp pp\n-6.0 6.0\n-6.0 6.0\n-6.0 6.0\n"
            "ITEM: ATOMS id type x y z fx fy fz\n" + "\n".join(rows) + "\n"
        )
    return "".join(blocks)


if __name__ == "__main__":
    (HERE / "legacy_ethane_kick.lammpstrj").write_text(build(), encoding="utf-8")
