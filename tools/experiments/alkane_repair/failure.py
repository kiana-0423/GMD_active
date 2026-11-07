"""Reference-confirmed failures: does a challenge expose a model error, not just a large force?

For a challenge structure with a full DFT label, the challenged region (perturbation site plus
``failure.region_bond_hops`` carbons, with hydrogens) is compared between the model and DFT:

* ``region error``: per-component force RMSE of the model against DFT over the region;
* ``reference force``: the DFT force magnitude (RMS) in the region, reported alongside so that a
  physically large force that the model reproduces is never counted as a failure;
* failure: region error >= ``failure.abs_threshold`` and >= ``failure.ratio_to_id_val`` x the qualified
  model's id_val force RMSE.

:func:`analyse_development` runs on development structures (public; used to configure and check the
method). The sealed final-test structures are analysed by :mod:`evaluate` with the same function.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tools.experiments.alkane_hotspots.families import load_positions
from tools.experiments.alkane_hotspots.geometry import AlkaneTopology
from tools.experiments.alkane_hotspots.manifest import utc_now

from .common import challenged_region, force_errors, write_json
from .config import LoadedBenchmark, RepairBenchmarkConfig
from .references import full_label_for, label_forces, require_frozen

LOGGER = logging.getLogger(__name__)

Predict = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, Any]]


def region_rows(config: RepairBenchmarkConfig, root: Path, records: Sequence[dict[str, Any]], store: Any, predict: Predict,
                *, baseline_rmse: float | None) -> list[dict[str, Any]]:
    """One row per structure with a valid full label: whole-structure and challenged-region errors."""

    failure = config.failure
    rows = []
    for record in sorted(records, key=lambda r: r["parent_id"]):
        label = full_label_for(store, record)
        if label is None:
            rows.append({"parent_id": record["parent_id"], "status": "no_reference_label"})
            continue
        topology = AlkaneTopology(int(record["chain_length"]))
        reference = label_forces(root, label)
        forces, _ = predict(topology.numbers, load_positions(root, record))
        row = {"parent_id": record["parent_id"], "base_id": record["base_id"], "partition": record["partition"], "kind": record["kind"],
               "family": record["family"], "status": "evaluated", "whole": force_errors(forces, reference)}
        if record["kind"] == "challenge":
            region = challenged_region(record, topology, failure.region_bond_hops)
            stats = force_errors(forces, reference, region)
            threshold = failure.abs_threshold if baseline_rmse is None else max(failure.abs_threshold, failure.ratio_to_id_val * baseline_rmse)
            row.update(region_atoms=region, region=stats, failure_threshold=threshold, failure=bool(stats["force_rmse"] >= threshold),
                       challenged=record["coverage"].get("challenged"))
        rows.append(row)
    return rows


def summarize_failures(rows: Sequence[dict[str, Any]], families: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family in families:
        mine = [r for r in rows if r.get("family") == family and r.get("status") == "evaluated" and "region" in r]
        out[family] = {"n_structures": len(mine), "n_failures": sum(1 for r in mine if r["failure"]),
                       "n_bases": len({r["base_id"] for r in mine}),
                       "region_force_rmse": [round(r["region"]["force_rmse"], 4) for r in mine],
                       "region_reference_force_rms": [round(r["region"]["reference_force_rms"], 4) for r in mine]}
    return out


def analyse_development(loaded: LoadedBenchmark) -> dict[str, Any]:
    """Frozen initial model on development structures (exploratory; configures nothing automatically)."""

    from tools.experiments.alkane_hotspots.labels import LabelStore

    from .generate import load_records
    from .qualify import require_qualified
    from .repair import frozen_predictor

    require_frozen(loaded)
    frozen = require_qualified(loaded)
    predict = frozen_predictor(loaded, frozen)
    records = [r for r in load_records(loaded.root) if r["partition"] == "development"]
    rows = region_rows(loaded.config, loaded.root, records, LabelStore(loaded.root), predict, baseline_rmse=float(frozen["id_val_force_rmse"]))
    report = {"created_at": utc_now(), "partition": "development", "baseline_id_val_force_rmse": frozen["id_val_force_rmse"],
              "failure_definition": loaded.config.failure.model_dump(mode="json"), "rows": rows,
              "families": summarize_failures(rows, loaded.config.challenges.enabled()),
              "statement": "development structures are used to check the method; they are not confirmatory evidence"}
    write_json(loaded.root / "development" / "failures.json", report)
    return report


__all__ = ["analyse_development", "region_rows", "summarize_failures"]
