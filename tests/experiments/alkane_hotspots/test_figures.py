"""Every figure renders in PDF, SVG and PNG from data that exist, and states missing data explicitly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from .beta_synthetic import alkane

from tools.experiments.alkane_hotspots import figures as fg  # noqa: E402

FORMATS = ("pdf", "svg", "png")


def _saved(fig, tmp_path: Path, stem: str) -> list[Path]:
    paths = [Path(p) for p in fg.save(fig, tmp_path, stem, FORMATS, 60)]
    assert [p.suffix for p in paths] == [".pdf", ".svg", ".png"] and all(p.stat().st_size > 0 for p in paths)
    return paths


def _says_missing(svg: Path) -> bool:
    return "NOT EXECUTED" in svg.read_text()


def test_figures_with_data(tmp_path: Path) -> None:
    numbers, positions, site = alkane(8, "bend", 1)
    rows = [{"parent_id": "p", "family": "bend", "positions": positions.tolist(), "numbers": numbers.tolist(), "perturbation_atoms": site,
             "atom_error": np.linspace(0, 1, len(numbers)).tolist(), "signals": {"committee": np.linspace(1, 0, len(numbers)).tolist()}}]
    assert not _says_missing(_saved(fg.fig_hotspots(rows, signal="committee"), tmp_path, "hot")[1])
    regions = {"error": [0.1, 0.5, 0.9], "family": ["normal", "bend", "bend"], "scores": {"committee": [0.0, 0.3, None]}}
    assert not _says_missing(_saved(fg.fig_risk_vs_error(regions, signals=["committee"], thresholds=[0.3]), tmp_path, "risk")[1])
    fidelity = [{"policy": "graph", "mean_n_atoms": 20, "pooled_core_force_rmse": 0.05}, {"policy": "graph", "mean_n_atoms": 30, "pooled_core_force_rmse": 0.02}]
    assert not _says_missing(_saved(fg.fig_fidelity(fidelity, x_key="mean_n_atoms", y_key="pooled_core_force_rmse", target=0.03), tmp_path, "fid")[1])
    curves = {"random_local": [[{"cumulative_cpu_core_s": 0.0, "force_rmse": 0.3}, {"cumulative_cpu_core_s": 3600.0, "force_rmse": 0.2}]],
              "phal_local": [[{"cumulative_cpu_core_s": 0.0, "force_rmse": 0.3}, {"cumulative_cpu_core_s": 7200.0, "force_rmse": None}]]}
    assert not _says_missing(_saved(fg.fig_learning_curves(curves), tmp_path, "lc")[1])
    assert not _says_missing(_saved(fg.fig_ablations({"phal_local": [0.2, 0.25], "no_lj": [None]}), tmp_path, "abl")[1])
    report = {"segments": [{"segment_index": 0, "model_id": "m0", "end_step": 100, "status": "done"},
                           {"segment_index": 1, "model_id": "m1", "end_step": 250, "status": "done"}],
              "promotions": [{"model_id": "m1", "previous_model_id": "m0", "action": "promote", "reason": "gates"}], "event_steps": [80, 90]}
    assert not _says_missing(_saved(fg.fig_timeline(report), tmp_path, "tl")[1])
    per_atom = [{"setting_id": sid, "force_error": str(err), "graph_distance_to_cut": str(d), "euclidean_distance_to_cap": str(1.5 * d)}
                for sid in ("graph-h2", "cluster-c1.5-b3") for d, err in ((2, 0.1), (4, 0.01), (6, 0.004))]
    final = [{"setting_id": "cluster-c1.5-b3", "force_error": "0.02", "graph_distance_to_cut": "3", "euclidean_distance_to_cap": ""}]
    assert not _says_missing(_saved(fg.fig_fidelity_distance(per_atom, final_eval=final, selected="cluster-c1.5-b3", target=0.1),
                                    tmp_path, "fdist")[1])


@pytest.mark.parametrize("make", [
    lambda: fg.fig_hotspots(None, signal="committee"),
    lambda: fg.fig_risk_vs_error(None, signals=["committee"]),
    lambda: fg.fig_fidelity([], x_key="mean_n_atoms"),
    lambda: fg.fig_fidelity_distance(None),
    lambda: fg.fig_learning_curves(None),
    lambda: fg.fig_learning_curves({"phal_local": [[{"cumulative_cpu_core_s": 0.0, "force_rmse": None}]]}),
    lambda: fg.fig_ablations({}),
    lambda: fg.fig_timeline(None),
])
def test_missing_data_is_labelled_not_fabricated(make, tmp_path: Path) -> None:
    svg = _saved(make(), tmp_path, "missing")[1]
    assert _says_missing(svg)


def test_colour_follows_the_arm() -> None:
    assert fg.colour_for("phal_local", ["phal_local"]) == fg.colour_for("phal_local", ["random_local", "no_lj", "phal_local"])
