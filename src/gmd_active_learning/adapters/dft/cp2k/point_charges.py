# Ported from PHAL (hotspot_al.cp2k.cp2k_input:write_point_charges) at 96263b6; MIT License, see NOTICE.
"""Optional tooling: write a region's embedding point charges as a plain table.

Not used by the CP2K input generator, which refuses enabled point-charge embedding (known issue: a
QM/MM ``FORCE_EVAL`` needs a site-specific ``&MM`` template). The table lets such a template read the
charges: ``x y z q element`` in Angstrom and e, in the same frame as the region's coordinates.
"""

from __future__ import annotations

from pathlib import Path

from gmd_active_learning.contracts.regions import ExtractedRegion


def write_point_charges(region: ExtractedRegion, output_dir: str | Path, *, job_name: str) -> Path | None:
    """Write ``<job_name>_pointcharges.dat``; None when the region carries no usable embedding."""

    embedding = region.metadata.get("embedding")
    if not isinstance(embedding, dict) or embedding.get("status") != "ok":
        return None
    point_charges = embedding.get("point_charges") or []
    if not point_charges:
        return None
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{job_name}_pointcharges.dat"
    lines = [
        "# PHAL electrostatic embedding point charges",
        "# Positions are in the same frame as the CP2K &COORD block.",
        "# x[Angstrom] y[Angstrom] z[Angstrom] charge[e] element",
    ]
    for charge in point_charges:
        x, y, z = charge["position"]
        lines.append(f"{x:>16.8f} {y:>16.8f} {z:>16.8f} {float(charge['charge']):>12.6f}  {charge['element']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


__all__ = ["write_point_charges"]
