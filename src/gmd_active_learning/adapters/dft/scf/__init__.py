# Ported from PHAL (hotspot_al.cp2k_ml_scf.__init__) at 96263b6; MIT License, see NOTICE.
"""ML-accelerated CP2K SCF support.

This package never performs electronic structure theory.  CP2K remains
responsible for building the Kohn-Sham Hamiltonian, diagonalising it, computing
the density, energy, forces and stress, and for the final convergence
judgement.  Everything here only helps CP2K *start closer* to the answer and
*get there in fewer iterations*:

* Stage 1 (:mod:`.schema`, :mod:`.dataset`, :mod:`.features`,
  :mod:`.validation`, :mod:`.export`) collects SCF trajectories into a dataset.
* Stage 2 (:mod:`.density`) defines the density-prediction contract and a dummy
  predictor that exercises the pipeline before real models exist.
* Stage 3 (:mod:`.training`) trains and evaluates those models.
* Stage 4 (:mod:`.cp2k_capabilities`) detects what the installed CP2K can do.
* Stage 7 (:mod:`.benchmark`) compares native CP2K against the accelerated
  modes.

The submodules are imported lazily so that importing this package does not pull
in numpy-heavy or torch-heavy code paths that a caller may not need.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DEFAULT_DATASET_ROOT",
    "SCHEMA_VERSION",
    "SCFDataset",
    "SCFSample",
    "benchmark",
    "cp2k_capabilities",
    "dataset",
    "density",
    "export",
    "features",
    "schema",
    "training",
    "validation",
]

_LAZY_ATTRIBUTES = {
    "SCFDataset": ("gmd_active_learning.adapters.dft.scf.dataset", "SCFDataset"),
    "DEFAULT_DATASET_ROOT": ("gmd_active_learning.adapters.dft.scf.dataset", "DEFAULT_DATASET_ROOT"),
    "SCFSample": ("gmd_active_learning.adapters.dft.scf.schema", "SCFSample"),
    "SCHEMA_VERSION": ("gmd_active_learning.adapters.dft.scf.schema", "SCHEMA_VERSION"),
}

_LAZY_MODULES = (
    "benchmark",
    "cp2k_capabilities",
    "dataset",
    "density",
    "export",
    "features",
    "schema",
    "training",
    "validation",
)


def __getattr__(name: str) -> Any:
    import importlib

    if name in _LAZY_ATTRIBUTES:
        module_name, attribute = _LAZY_ATTRIBUTES[name]
        return getattr(importlib.import_module(module_name), attribute)
    if name in _LAZY_MODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
