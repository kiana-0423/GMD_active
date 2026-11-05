# Ported from PHAL (hotspot_al.cp2k_ml_scf.density.__init__) at 96263b6; MIT License, see NOTICE.
"""ML density prediction for CP2K's initial electronic state.

Nothing in this package computes electronic structure.  It produces a *guess*
that CP2K is free to reject, and every guess is validated and recorded before
it is offered.
"""

from __future__ import annotations

from gmd_active_learning.adapters.dft.scf.density.predictor import (
    DummyDensityPredictor,
    PredictionValidation,
    SCFDensityPredictor,
)
from gmd_active_learning.adapters.dft.scf.density.registry import (
    PredictorNotFoundError,
    PredictorSpec,
    available_predictors,
    create_predictor,
    ml_scf_enabled,
    predictor_spec,
    record_prediction,
    register_predictor,
    resolve_predictor_class,
)
from gmd_active_learning.adapters.dft.scf.density.prediction import (
    DENSITY_FORMAT_ATOMIC_POPULATION,
    DENSITY_FORMAT_DENSITY_MATRIX,
    PredictionError,
    PredictionProvenance,
    SCFDensityPrediction,
    read_prediction,
    write_prediction,
)

__all__ = [
    "DENSITY_FORMAT_ATOMIC_POPULATION",
    "DENSITY_FORMAT_DENSITY_MATRIX",
    "DummyDensityPredictor",
    "PredictionError",
    "PredictionProvenance",
    "PredictionValidation",
    "PredictorNotFoundError",
    "PredictorSpec",
    "SCFDensityPrediction",
    "SCFDensityPredictor",
    "available_predictors",
    "create_predictor",
    "ml_scf_enabled",
    "predictor_spec",
    "read_prediction",
    "record_prediction",
    "register_predictor",
    "resolve_predictor_class",
    "write_prediction",
]
