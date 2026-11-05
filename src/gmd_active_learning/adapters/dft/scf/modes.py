# Ported from PHAL (hotspot_al.cp2k_ml_scf.benchmark: MODE_* constants) at 96263b6; MIT License, see NOTICE.
"""SCF mode labels shared by the ML-SCF backend wrapper and the benchmark tooling.

They live in the runtime package so the backend never imports benchmark tooling
(PHAL's backend imported them from ``benchmark.py``).
"""

from __future__ import annotations

from collections.abc import Mapping

MODE_NATIVE = "native"
MODE_RESTART = "restart"
MODE_ML_GUESS = "ml_guess"
MODE_ML_FULL = "ml_full"
#: A run that was configured for ML-SCF but executed natively (pre-run degradation or retry).
MODE_NATIVE_FALLBACK = "native_fallback"

#: Comparison order; each mode is compared against :data:`MODE_NATIVE`.
BENCHMARK_MODES: tuple[str, ...] = (MODE_NATIVE, MODE_RESTART, MODE_ML_GUESS, MODE_ML_FULL)

#: Every value ``scf_mode`` may take in provenance written by this package.
SCF_MODES: tuple[str, ...] = (MODE_NATIVE, MODE_ML_GUESS, MODE_ML_FULL, MODE_NATIVE_FALLBACK)

MODE_DESCRIPTIONS: Mapping[str, str] = {
    MODE_NATIVE: "SCF_GUESS ATOMIC + Pulay mixing",
    MODE_RESTART: "SCF_GUESS RESTART + Pulay mixing",
    MODE_ML_GUESS: "SCF_GUESS ML + Pulay mixing",
    MODE_ML_FULL: "SCF_GUESS ML + neural mixing (no CP2K build implements this)",
}

__all__ = [
    "BENCHMARK_MODES",
    "MODE_DESCRIPTIONS",
    "MODE_ML_FULL",
    "MODE_ML_GUESS",
    "MODE_NATIVE",
    "MODE_NATIVE_FALLBACK",
    "MODE_RESTART",
    "SCF_MODES",
]
