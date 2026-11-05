# Ported from PHAL (hotspot_al.cp2k_ml_scf.training.__init__) at 96263b6; MIT License, see NOTICE.
"""Training tooling for the experimental ML-SCF density models.

Submodules are imported explicitly (``tools.scf.training.dataset_loader``,
``.dataset_adapter``, ``.density_gnn``). Nothing is imported eagerly, so the
torch-backed modules are only loaded when asked for. The paired-run evaluation
that PHAL kept here lives in :mod:`tools.scf.benchmark.evaluate`.
"""
