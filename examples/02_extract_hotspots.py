# Ported from PHAL (examples/02_extract_hotspots.py) at 96263b6; MIT License, see NOTICE.
"""Extract hotspot-localised regions from a monitored trajectory.

    python examples/02_extract_hotspots.py                         # first triggered frame of the legacy dump
    python examples/02_extract_hotspots.py traj.extxyz --atom-id 0 --frame 0 --mode graph

By default the trajectory is monitored and the first triggered frame's hotspot atoms are clustered
(``hotspots`` section) and cut out with the configured ``extraction`` mode, then H-capped
(``capping``). With ``--atom-id`` the given atom of frame ``--frame`` is the hotspot instead, as in
PHAL's example (which scored atom 0 with a synthetic result). Prints the number of regions and,
for each, its atom count, roles and capping outcome. Exits 1 when no region results (PHAL's
example indexed ``regions[0]`` unconditionally and crashed).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from _example_utils import DATA_DIR, add_common_arguments, add_hotspot_arguments, extract_from_trajectory, require_file

from gmd_active_learning.utils.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser, default_trajectory=DATA_DIR / "legacy_ethane_kick.lammpstrj")
    add_hotspot_arguments(parser)
    args = parser.parse_args(argv)
    require_file(parser, args.trajectory, "trajectory")
    configure_logging("WARNING")

    frame, regions = extract_from_trajectory(args)
    if frame is None:
        print("no triggered frame (use --atom-id to name a hotspot atom)")
        return 1
    print(f"frame step={frame.step} n_atoms={frame.n_atoms}")
    print("n_regions:", len(regions))
    if not regions:
        return 1
    for region in regions:
        roles = Counter(role.value for role in region.roles)
        capping = "skipped:" + str(region.metadata.get("h_capping_skipped_reason")) if region.metadata.get("h_capping_skipped") else "applied"
        print(f"region {region.region_id} mode={region.mode} n_atoms={region.n_atoms} hotspots={list(region.hotspot_atom_ids)} "
              f"roles={dict(sorted(roles.items()))} capping={capping}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
