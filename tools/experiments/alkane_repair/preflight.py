"""Geometry preflight (no DFT, no MD, no training): checks that cost nothing and catch problems early.

* MD boxes: the vacuum-padded periodic box of every repair start and every sealed stability start, sized
  for the model's force receptive field + margin + stretch allowance (``repair.md_box``);
* fragments: the frozen fragment setting cut around every challenge site in both shapes used
  (the fidelity gate's site and the closed loop's hotspot shape): atom counts against the label limit,
  number of caps, and whether capping succeeded;
* coverage: the descriptor distribution of ordinary and challenge structures per partition.

Final-test structures are read for geometry only (positions, never labels).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from tools.experiments.alkane_hotspots.families import load_positions
from tools.experiments.alkane_hotspots.geometry import AlkaneTopology
from tools.experiments.alkane_hotspots.manifest import utc_now

from .common import read_json, write_json
from .config import LoadedBenchmark
from .generate import load_records

LOGGER = logging.getLogger(__name__)


def preflight(loaded: LoadedBenchmark) -> dict[str, Any]:
    from tools.experiments.alkane_hotspots.closed_loop import hotspot_site
    from tools.experiments.alkane_hotspots.fidelity import build_fragment, parent_frame
    from tools.experiments.alkane_hotspots.md_box import MDBoxConfig, build_box, force_receptive_field

    config = loaded.config
    public = load_records(loaded.root)
    sealed = load_records(loaded.sealed_root)
    box_config = MDBoxConfig.model_validate(dict(config.repair.md_box))
    receptive = force_receptive_field(loaded.model.config.training.model, override=box_config.receptive_field_override)
    boxes = []
    for root, records in ((loaded.root, public), (loaded.sealed_root, sealed)):
        for record in records:
            if record["partition"] not in ("repair", "final_test", "calibration", "id_val"):
                continue
            box = build_box(load_positions(root, record), receptive, box_config).to_dict()
            boxes.append({"parent_id": record["parent_id"], "partition": record["partition"], "family": record["family"],
                          "box_length": box["lengths"][0], "diameter": box["diameter"], "min_image_distance": box["min_image_distance"],
                          "required_distance": box["required_distance"]})
    selection_path = loaded.repo_path(config.fidelity.selection_dir) / "selection.json"
    setting = read_json(selection_path)["setting"]
    limit = loaded.alpha.config.budgets.max_atoms_per_fragment
    fragments = []
    for root, records in ((loaded.root, public), (loaded.sealed_root, sealed)):
        for record in records:
            if record["kind"] != "challenge":
                continue
            topology = AlkaneTopology(int(record["chain_length"]))
            frame = parent_frame(record, load_positions(root, record))
            site = [int(a) for a in record["perturbation"]["atoms"]]
            _, loop_site = hotspot_site(topology, site, None)
            for shape, atoms in (("gate_site", site), ("closed_loop_site", loop_site)):
                try:
                    region, core = build_fragment(frame, atoms, setting, loaded.alpha.config)
                    n_caps = int(sum(1 for r in region.roles if r.value == "h_cap"))
                    fragments.append({"parent_id": record["parent_id"], "partition": record["partition"], "family": record["family"], "shape": shape,
                                      "status": "ok", "n_atoms": region.n_atoms, "n_core": len(core), "n_caps": n_caps,
                                      "within_atom_limit": region.n_atoms <= limit})
                except Exception as exc:  # noqa: BLE001 - every failure is recorded, nothing is dropped
                    fragments.append({"parent_id": record["parent_id"], "partition": record["partition"], "family": record["family"], "shape": shape,
                                      "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
    ok = [f for f in fragments if f["status"] == "ok"]
    coverage: dict[str, Any] = {}
    for record in public + sealed:
        key = f"{record['partition']}:{record['family'] if record['kind'] == 'challenge' else 'ordinary'}"
        item = coverage.setdefault(key, {"n": 0, "max_torsion_deviation_deg": [], "max_angle_deviation_deg": [], "segment_contact": []})
        item["n"] += 1
        for name in ("max_torsion_deviation_deg", "max_angle_deviation_deg", "segment_contact"):
            item[name].append(float(record["coverage"][name]))
    for item in coverage.values():
        for name in ("max_torsion_deviation_deg", "max_angle_deviation_deg", "segment_contact"):
            values = np.asarray(item[name])
            item[name] = {"min": float(values.min()), "max": float(values.max())}
    result = {"created_at": utc_now(), "setting": setting, "receptive_field_A": receptive.radius,
              "boxes": {"n": len(boxes), "max_length_A": max(b["box_length"] for b in boxes), "rows": boxes},
              "fragments": {"n": len(fragments), "n_failed": len(fragments) - len(ok), "n_over_atom_limit": sum(1 for f in ok if not f["within_atom_limit"]),
                            "atoms_range": [min(f["n_atoms"] for f in ok), max(f["n_atoms"] for f in ok)] if ok else None,
                            "caps_range": [min(f["n_caps"] for f in ok), max(f["n_caps"] for f in ok)] if ok else None, "rows": fragments},
              "coverage": coverage, "passed": len(fragments) == len(ok) and all(f["within_atom_limit"] for f in ok)}
    write_json(loaded.root / "preflight.json", result)
    return result


__all__ = ["preflight"]
