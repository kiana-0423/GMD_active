# CP2K probe

`h2.inp` is a native Quickstep H2 single point (PBE, DZVP-MOLOPT-GTH,
GTH-PBE-q1, 10 A cell, `MAX_SCF 50`, `EPS_SCF 1.0E-6`), taken from PHAL's
`validate_cp2k_ml.py` `h2_input()` at 96263b6 with the basis/potential paths
changed to bare names resolved through CP2K's data directory. It contains no
ML-SCF section.

`containers/scripts/validate_cp2k_stack.py` copies it into a writable directory,
runs `cp2k.psmp`, and requires exit status 0, `SCF run converged`,
`PROGRAM ENDED` and a finite total energy. It is an integration check of the
labeling engine, not a labeling protocol.
