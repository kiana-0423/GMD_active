"""Experimental SCF / ML-SCF tooling ported from PHAL (offline analysis, benchmarks, training).

Not part of the installed ``gmd_active_learning`` package and never imported by it.
Status (PHAL's own measurements, see ``docs/scf/README.md``): no ML-SCF component
has demonstrated an acceleration; the one positive result -- the model-free
``default -> nbuffer4`` retry ladder -- lives in the runtime package
``gmd_active_learning.adapters.dft.scf.strategy``.

Run scripts as modules from the repository root, e.g.
``python -m tools.scf.scripts.generate_pulay_dataset --help``.
"""
