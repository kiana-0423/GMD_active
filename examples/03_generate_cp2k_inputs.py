# Ported from PHAL (examples/03_generate_cp2k_inputs.py) at 96263b6; MIT License, see NOTICE.
"""Generate CP2K inputs for the hotspot regions of a trajectory.

    python examples/03_generate_cp2k_inputs.py --output-dir cp2k_jobs
    python examples/03_generate_cp2k_inputs.py traj.extxyz --atom-id 0 --config my_run.yaml

Regions come from the same path as example 02 (monitor -> hotspots -> extraction -> H capping).
Each region then passes the labeling gates the CP2K labeling service applies (uncapped-fragment
policy, embedding fail-closed) and gets ``<region>/hotspot_region_sp.inp`` (plus ``_hopt.inp``
for the cap-only relaxation when ``cp2k.h_only_opt`` is enabled and caps exist). Settings come
from the ``cp2k`` section. PHAL's example skipped H capping, so its uncapped-fragment gate never
fired and an uncapped cluster input was written silently; here capping is part of the path.
Nothing is executed: running CP2K is the labeling service's job (``gmd-al run``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _example_utils import DATA_DIR, add_common_arguments, add_hotspot_arguments, extract_from_trajectory, load_sections, require_file

from gmd_active_learning.adapters.dft.cp2k.config import CP2KConfig
from gmd_active_learning.adapters.dft.cp2k.input import write_cp2k_inputs
from gmd_active_learning.core.exceptions import GMDActiveLearningError
from gmd_active_learning.extraction import CappingConfig, check_labelable
from gmd_active_learning.utils.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_arguments(parser, default_trajectory=DATA_DIR / "legacy_ethane_kick.lammpstrj")
    add_hotspot_arguments(parser)
    parser.add_argument("--output-dir", type=Path, default=Path("cp2k_jobs"), help="directory for the generated inputs")
    args = parser.parse_args(argv)
    require_file(parser, args.trajectory, "trajectory")
    configure_logging("WARNING")

    sections = load_sections(args.config)
    cp2k = CP2KConfig.model_validate(sections.get("cp2k") or {})
    capping = CappingConfig.from_mapping(sections.get("capping"))
    embedding_enabled = bool((sections.get("embedding") or {}).get("enabled", False))
    frame, regions = extract_from_trajectory(args)
    if not regions:
        print("no regions to label")
        return 1
    failures = 0
    for region in regions:
        try:
            check_labelable(region, capping, periodic_labeling=any(region.pbc))
            written = write_cp2k_inputs(region, args.output_dir / region.region_id, cp2k, embedding_enabled=embedding_enabled)
        except GMDActiveLearningError as exc:
            failures += 1
            print(f"region {region.region_id}: refused ({type(exc).__name__}: {exc})")
            continue
        for kind, path in sorted(written.items()):
            print(f"region {region.region_id} n_atoms={region.n_atoms} pbc={''.join('T' if f else 'F' for f in region.pbc)} {kind}: {path}")
    return 1 if failures == len(regions) else 0


if __name__ == "__main__":
    sys.exit(main())
