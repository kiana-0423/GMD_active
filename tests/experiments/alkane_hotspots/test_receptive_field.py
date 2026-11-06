"""Receptive field of real GMD_SGT models (tiny, random-init, deterministic): a local model's core force is
exactly independent of atoms beyond 2 x n_blocks x local_cutoff; an attention model is global."""

from __future__ import annotations

import numpy as np
import pytest

from .beta_synthetic import import_sgt

from tools.experiments.alkane_hotspots.geometry import AlkaneTopology, build_all_trans  # noqa: E402
from tools.experiments.alkane_hotspots.receptive_field import (  # noqa: E402
    ReceptiveFieldConfig,
    build_model,
    nominal_receptive_field,
    run_receptive_field,
)
from gmd_active_learning.core.exceptions import ConfigurationError  # noqa: E402

TINY = {"n_species": 10, "n_blocks": 1, "scalar_dim": 8, "irreps": "8x0e", "n_basis": 4, "l_max": 0, "local_cutoff": 2.0,
        "lr_cutoff": 5.0, "long_range_type": "none", "n_heads": 1, "avg_neighbors": 4.0}


def _structure(n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    positions = build_all_trans(n) + np.random.default_rng(0).normal(0.0, 0.02, (3 * n + 2, 3))
    return AlkaneTopology(n).numbers, positions


def _config(**overrides) -> ReceptiveFieldConfig:
    data = {"tolerance": 1.0e-9, "core_atoms": [4], "truncation_radii": [2.0, 3.0, 4.0, 5.0, 8.0, 30.0]}
    data.update(overrides)
    return ReceptiveFieldConfig.model_validate(data)


def test_nominal_values() -> None:
    nominal = nominal_receptive_field({"n_blocks": 3, "local_cutoff": 5.0, "long_range_type": "none"})
    assert nominal["energy_receptive_field_A"] == 15.0 and nominal["force_receptive_field_local_A"] == 30.0 and not nominal["global"]
    assert nominal_receptive_field({"n_blocks": 3, "local_cutoff": 5.0, "long_range_type": "invariant_attention"})["global"]


def test_local_model_influence_vanishes_beyond_nominal_force_receptive_field() -> None:
    import_sgt()
    config = _config()
    model, model_config, provenance = build_model(TINY, config)
    assert provenance["reinitialised_parameters"]  # the zero-initialised readout was re-drawn (recorded)
    numbers, positions = _structure()
    result = run_receptive_field(model, model_config, numbers, positions, config, provenance=provenance)
    core = result["cores"][0]
    limit = result["nominal"]["force_receptive_field_local_A"]
    assert limit == 4.0
    for kind in ("perturb", "delete"):
        far = [row[kind] for row in core["atoms"] if row["distance_A"] > limit]
        near = [row[kind] for row in core["atoms"] if row["distance_A"] < 1.2]
        assert far and max(far) < 1e-12, kind
        assert max(near) > 1e-6, kind
        assert result["measured"][f"{kind}_influence_radius_A"] <= limit
        # the dependence is bounded by 2 x n_blocks hops in the cutoff graph
        assert all(row[kind] < 1e-12 for row in core["atoms"] if row["cutoff_graph_hops"] > 2 * TINY["n_blocks"] or row["cutoff_graph_hops"] < 0)
    assert result["measured"]["truncation_converged_radius_A"] <= limit
    again = run_receptive_field(*build_model(TINY, config)[:2], numbers, positions, config)
    assert again["measured"] == result["measured"] and again["cores"][0]["atoms"] == core["atoms"]  # deterministic


def test_attention_model_is_global() -> None:
    import_sgt()
    config = _config(measurements=["delete"])
    model, model_config, provenance = build_model({**TINY, "long_range_type": "invariant_attention"}, config)
    numbers, positions = _structure()
    result = run_receptive_field(model, model_config, numbers, positions, config, provenance=provenance)
    assert result["nominal"]["global"] and result["nominal"]["force_receptive_field_A"] is None
    core = result["cores"][0]
    far = [row["delete"] for row in core["atoms"] if row["distance_A"] > result["nominal"]["force_receptive_field_local_A"]]
    assert max(far) > config.tolerance  # atoms far outside every cutoff still change the core force
    assert "global" in result["interpretation"]


def test_checkpoint_source_requires_a_file() -> None:
    import_sgt()
    with pytest.raises(ConfigurationError, match="existing checkpoint"):
        build_model(TINY, _config(model_source="checkpoint"), checkpoint=None)
