# Ported from PHAL (hotspot_al.hotspot.hotspot_cluster) at 96263b6; MIT License, see NOTICE.
"""PBC-aware single-linkage clustering of flagged atoms (union-find over pairs within ``merge_radius``)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from gmd_active_learning.utils.periodic import minimum_image_vectors

_ROW_CHUNK = 256


def _pairs_within(positions: np.ndarray, merge_radius: float, cell: np.ndarray | None, pbc: object) -> list[tuple[int, int]]:
    """All (i, j), i < j, with minimum-image distance <= merge_radius (exact, chunked O(n^2))."""

    pairs: list[tuple[int, int]] = []
    n_points = len(positions)
    for start in range(0, n_points, _ROW_CHUNK):
        stop = min(start + _ROW_CHUNK, n_points)
        vectors = minimum_image_vectors(positions[None, :, :] - positions[start:stop, None, :], cell=cell, pbc=pbc)
        distances = np.linalg.norm(vectors, axis=2)
        rows, cols = np.nonzero(distances <= merge_radius)
        rows = rows + start
        keep = cols > rows
        pairs.extend(zip(rows[keep].tolist(), cols[keep].tolist()))
    pairs.sort()
    return pairs


def cluster_anomalous_atoms(
    flagged_indices: list[int],
    positions: np.ndarray,
    *,
    merge_radius: float,
    cell: np.ndarray | None = None,
    pbc: bool | tuple[bool, bool, bool] | np.ndarray = False,
) -> list[list[int]]:
    """Connected components of flagged atoms; members sorted, clusters in first-appearance order.

    A distance exactly equal to ``merge_radius`` merges. The result depends only on the input order
    of ``flagged_indices`` (callers pass them sorted for a canonical result).
    """

    if not flagged_indices:
        return []
    flagged = [int(index) for index in flagged_indices]
    points = np.asarray(positions, dtype=float)[flagged]
    parent = list(range(len(flagged)))

    def find(index: int) -> int:
        root = index
        while root != parent[root]:
            root = parent[root]
        while index != root:
            parent[index], index = root, parent[index]
        return root

    for left, right in _pairs_within(points, float(merge_radius), cell, pbc):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            # Attach the later root under the earlier one (PHAL), so roots are first appearances.
            if root_right < root_left:
                root_left, root_right = root_right, root_left
            parent[root_right] = root_left

    clusters: dict[int, list[int]] = defaultdict(list)
    for local, index in enumerate(flagged):
        clusters[find(local)].append(index)
    return [sorted(members) for _root, members in sorted(clusters.items())]


__all__ = ["cluster_anomalous_atoms"]
