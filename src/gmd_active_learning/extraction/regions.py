# Ported from PHAL (hotspot_al.extraction.boundary_regions) at 96263b6; MIT License, see NOTICE.
"""Radial partition of extracted atoms into core / inner buffer / outer buffer / boundary."""

from __future__ import annotations

import numpy as np

from gmd_active_learning.contracts import RegionRole


def assign_radial_regions(
    distances: np.ndarray,
    *,
    core_radius: float,
    extract_radius: float,
    boundary_thickness: float,
) -> dict[str, list[int]]:
    """Index lists per radial shell (identical to PHAL).

    ``boundary_start = max(core, extract - thickness)``, ``mid = core + (boundary_start - core)/2``:
    core ``d <= core``; inner ``(core, mid]``; outer ``(mid, boundary_start)``; boundary
    ``[boundary_start, extract]``. Atoms beyond ``extract_radius`` belong to no shell.
    """

    distances = np.asarray(distances, dtype=float)
    boundary_start = max(core_radius, extract_radius - boundary_thickness)
    buffer_midpoint = core_radius + 0.5 * max(boundary_start - core_radius, 0.0)
    core = np.where(distances <= core_radius)[0].tolist()
    inner_buffer = np.where((distances > core_radius) & (distances <= buffer_midpoint))[0].tolist()
    outer_buffer = np.where((distances > buffer_midpoint) & (distances < boundary_start))[0].tolist()
    boundary = np.where((distances >= boundary_start) & (distances <= extract_radius))[0].tolist()
    return {"core": core, "inner_buffer": inner_buffer, "outer_buffer": outer_buffer, "boundary": boundary}


def region_label_vector(n_atoms: int, regions: dict[str, list[int]]) -> list[str]:
    """Per-atom labels with PHAL's priority core > inner > boundary > outer; default unassigned."""

    labels = ["unassigned"] * n_atoms
    for key in ("outer_buffer", "boundary", "inner_buffer", "core"):
        for index in regions.get(key, []):
            labels[index] = key
    return labels


def radial_roles(
    distances: np.ndarray,
    *,
    core_radius: float,
    extract_radius: float,
    boundary_thickness: float,
) -> list[RegionRole]:
    """Contract roles for the radial partition (``unassigned`` for atoms outside every shell)."""

    regions = assign_radial_regions(
        distances, core_radius=core_radius, extract_radius=extract_radius, boundary_thickness=boundary_thickness
    )
    return [RegionRole(label) for label in region_label_vector(len(np.atleast_1d(distances)), regions)]


__all__ = ["assign_radial_regions", "radial_roles", "region_label_vector"]
