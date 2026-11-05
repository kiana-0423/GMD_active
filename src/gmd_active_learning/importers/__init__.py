"""Importers that turn legacy (non-GMD) trajectories into :class:`~gmd_active_learning.contracts.Frame` objects.

Optional tooling: nothing in the online workflow depends on it. Importing this package pulls in no
MD/MLIP/DFT adapter.
"""

from gmd_active_learning.importers.lammps_dump import (
    SupportsReadText,
    iter_lammps_dump,
    normalize_type_map,
    parse_lammps_dump_frame,
    read_lammps_dump,
)

__all__ = [
    "SupportsReadText",
    "iter_lammps_dump",
    "normalize_type_map",
    "parse_lammps_dump_frame",
    "read_lammps_dump",
]
