"""Publication figures (matplotlib; PDF + SVG + PNG) rendered only from results that exist.

Every figure function takes already-loaded result data. Missing inputs are never replaced by
placeholder numbers: the panel states ``NOT EXECUTED`` / ``no data`` and why, and the figure is still
written so a document that embeds it shows the gap explicitly. Colours follow the entity (an arm keeps
its colour in every figure) in a fixed categorical order; sequential maps are single-hue.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

#: fixed categorical order (validated reference palette, light mode); colour follows the entity
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
ARM_ORDER = ("random_local", "committee_local", "phal_local", "phal_full_system")
ARM_LABELS = {"random_local": "A random local", "committee_local": "B committee local", "phal_local": "C PHAL local",
              "phal_full_system": "D PHAL full system"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SEQUENTIAL = "Blues"


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "lines.linewidth": 2.0, "svg.hashsalt": "alkane-hotspots",
                         "pdf.compression": 6})
    return plt


def colour_for(name: str, names: Sequence[str]) -> str:
    """Arms keep their colour everywhere; other variants follow in fixed order after the four arms."""

    ordered = list(ARM_ORDER) + sorted(n for n in names if n not in ARM_ORDER)
    index = ordered.index(name) if name in ordered else len(ordered)
    return CATEGORICAL[index % len(CATEGORICAL)] if index < len(CATEGORICAL) else MUTED


def mark_missing(ax: Any, reason: str) -> None:
    """Label a panel as having no executed data (the reason is also kept on the figure for the report)."""

    figure = ax.figure
    figure.alkane_missing = [*getattr(figure, "alkane_missing", []), reason]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.text(0.5, 0.5, f"NOT EXECUTED / no data\n{reason}", ha="center", va="center", transform=ax.transAxes, color=MUTED,
            fontsize=9, wrap=True)


def stamp(fig: Any, text: str) -> None:
    """A visible provenance note (e.g. smoke-only settings) in the figure's top-left corner."""

    fig.text(0.01, 0.99, text, ha="left", va="top", fontsize=7, color="#e34948")


def missing_reasons(fig: Any) -> list[str]:
    return list(getattr(fig, "alkane_missing", []))


def save(fig: Any, out_dir: str | Path, stem: str, formats: Sequence[str] = ("pdf", "svg", "png"), dpi: int = 200) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = out / f"{stem}.{fmt}"
        metadata = {"pdf": {"Creator": "alkane_hotspots.figures", "CreationDate": None},
                    "svg": {"Creator": "alkane_hotspots.figures", "Date": None}}.get(fmt)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", metadata=metadata)
        paths.append(str(path))
    _plt().close(fig)
    return paths


# ---------------------------------------------------------------------------------------------- 0 receptive field


def fig_receptive_field(result: Mapping[str, Any] | None) -> Any:
    """Core-force change vs distance of the perturbed / deleted atom, and vs truncation radius (log scale)."""

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
    if not result or not result.get("cores"):
        for ax in axes:
            mark_missing(ax, "no receptive-field result")
        return fig
    core = result["cores"][0]
    floor = float(result.get("structural_tolerance_eV_per_A") or 1e-12) / 10.0
    for index, kind in enumerate(("perturb", "delete")):
        rows = [a for a in core["atoms"] if kind in a]
        if rows:
            axes[0].scatter([a["distance_A"] for a in rows], [max(a[kind], floor) for a in rows], s=12, color=CATEGORICAL[index],
                            label=f"{kind} one atom", linewidths=0)
    truncation = core.get("truncation") or []
    if truncation:
        axes[1].plot([t["radius_A"] for t in truncation], [max(t["error"], floor) for t in truncation], marker="o", markersize=4,
                     color=CATEGORICAL[2], label="keep atoms within R")
    else:
        mark_missing(axes[1], "truncation not measured")
    nominal = result.get("nominal") or {}
    for ax in axes:
        ax.set_yscale("log")
        ax.axhline(result["tolerance_eV_per_A"], color=INK, linestyle=":", linewidth=0.9, label="tolerance")
        if nominal.get("energy_receptive_field_A") is not None:
            ax.axvline(nominal["energy_receptive_field_A"], color=MUTED, linestyle="--", linewidth=0.8, label="n_blocks x r_c")
        if nominal.get("force_receptive_field_local_A") is not None:
            ax.axvline(nominal["force_receptive_field_local_A"], color=INK, linestyle="--", linewidth=0.8, label="2 x n_blocks x r_c")
        ax.legend(fontsize=6, frameon=False)
    for ax in axes:
        ax.text(0.99, 0.02, f"exact zeros drawn at {floor:.0e}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=MUTED)
    axes[0].set_xlabel("distance from core atom (A)")
    axes[0].set_ylabel("|dF_core| (eV/A)")
    axes[1].set_xlabel("truncation radius R (A)")
    source = (result.get("model") or {}).get("source", "?")
    fig.suptitle(f"{(result.get('structure') or {}).get('parent_id', '')}: {nominal.get('long_range_type')} model ({source})", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- 1 hotspots


def _projection(positions: np.ndarray) -> np.ndarray:
    centred = positions - positions.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[:2].T


def fig_hotspots(rows: Sequence[Mapping[str, Any]] | None, *, signal: str, parent_id: str | None = None, bond_scale: float = 1.2) -> Any:
    """Chain projected on its two principal axes; left: reference error, right: the signal."""

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    row = None
    if rows:
        candidates = [r for r in rows if parent_id is None or r["parent_id"] == parent_id]
        if parent_id is None:  # the configuration with the largest reference error
            candidates.sort(key=lambda r: -max(r["atom_error"]))
        row = candidates[0] if candidates else None
    if row is None:
        for ax in axes:
            mark_missing(ax, "no ground-truth per-atom results" if not rows else f"configuration {parent_id} not found")
        return fig
    from .ground_truth import bond_adjacency

    positions = np.asarray(row["positions"], dtype=float)
    numbers = np.asarray(row["numbers"])
    xy = _projection(positions)
    adjacency = bond_adjacency(numbers, positions, scale=bond_scale)
    panels = [("reference error |F_model - F_DFT| (eV/A)", row["atom_error"]), (f"signal: {signal}", (row.get("signals") or {}).get(signal))]
    for ax, (title, values) in zip(axes, panels):
        ax.set_title(f"{row['parent_id']} ({row['family']})\n{title}", fontsize=8, color=INK)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for i, neighbours in enumerate(adjacency):
            for j in neighbours:
                if j > i:
                    ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=GRID, linewidth=1.0, zorder=1)
        if values is None:
            mark_missing(ax, f"signal {signal} unavailable")
            continue
        data = np.asarray([np.nan if v is None else v for v in values], dtype=float)
        sizes = np.where(numbers > 1, 40, 14)
        finite = np.isfinite(data)
        scatter = ax.scatter(xy[finite, 0], xy[finite, 1], c=data[finite], s=sizes[finite], cmap=SEQUENTIAL, edgecolors=INK,
                             linewidths=0.3, zorder=2)
        if (~finite).any():
            ax.scatter(xy[~finite, 0], xy[~finite, 1], s=sizes[~finite], facecolors="none", edgecolors=MUTED, linewidths=0.5, zorder=2)
        site = [int(a) for a in row.get("perturbation_atoms") or []]
        if site:
            ax.scatter(xy[site, 0], xy[site, 1], s=110, facecolors="none", edgecolors="#e34948", linewidths=1.2, zorder=3,
                       label="perturbation atoms")
            ax.legend(loc="lower right", fontsize=7, frameon=False)
        fig.colorbar(scatter, ax=ax, shrink=0.8)
    return fig


# ---------------------------------------------------------------------------------------------- 2 risk vs error


def fig_risk_vs_error(regions: Mapping[str, Any] | None, *, signals: Sequence[str], thresholds: Sequence[float] = ()) -> Any:
    """Region score (x) vs region reference error (y), one panel per signal, normal family in grey."""

    plt = _plt()
    n = max(1, len(signals))
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.0), squeeze=False)
    for ax, name in zip(axes[0], signals or ["(none)"]):
        scores = None if not regions else (regions.get("scores") or {}).get(name)
        if not regions or scores is None:
            mark_missing(ax, f"no region table for {name}")
            continue
        error = np.asarray(regions["error"], dtype=float)
        score = np.asarray([np.nan if v is None else v for v in scores], dtype=float)
        family = np.asarray(regions["family"])
        ok = np.isfinite(score) & np.isfinite(error)
        if not ok.any():
            mark_missing(ax, f"{name} unavailable on every region")
            continue
        normal = family == "normal"
        ax.scatter(score[ok & normal], error[ok & normal], s=9, color=MUTED, alpha=0.6, label="normal", linewidths=0)
        ax.scatter(score[ok & ~normal], error[ok & ~normal], s=9, color=CATEGORICAL[0], alpha=0.8, label="perturbed", linewidths=0)
        for tau in thresholds:
            ax.axhline(tau, color=INK, linewidth=0.8, linestyle=":")
        ax.set_xlabel(f"{name} (region score)")
        ax.set_ylabel("region error (eV/A)")
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- 3 fidelity


def fig_fidelity(rows: Sequence[Mapping[str, Any]] | None, *, x_key: str, y_key: str = "core_force_rmse", group_key: str | None = "policy",
                 target: float | None = None, x_label: str | None = None) -> Any:
    """Core-force error vs fragment/buffer size from alpha's fidelity results (measured settings only)."""

    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    usable = [r for r in rows or [] if r.get(x_key) is not None and r.get(y_key) is not None]
    if not usable:
        mark_missing(ax, f"no fidelity results with {x_key} and {y_key}")
        return fig
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in usable:
        groups.setdefault(str(r.get(group_key, "")) if group_key else "", []).append(r)
    for index, (name, items) in enumerate(sorted(groups.items())):
        items = sorted(items, key=lambda r: float(r[x_key]))
        ax.plot([float(r[x_key]) for r in items], [float(r[y_key]) for r in items], marker="o", markersize=5,
                color=CATEGORICAL[index % len(CATEGORICAL)], label=name or None)
    if target is not None:
        ax.axhline(target, color=INK, linestyle=":", linewidth=0.9, label=f"pre-registered target {target:g}")
    ax.set_xlabel(x_label or x_key)
    ax.set_ylabel(f"{y_key} (eV/A)")
    ax.set_yscale("log")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def fig_fidelity_distance(calibration: Sequence[Mapping[str, Any]] | None, *, final_eval: Sequence[Mapping[str, Any]] | None = None,
                          selected: str | None = None, target: float | None = None) -> Any:
    """Per-core-atom |F_frag - F_parent| vs distance from the artificial boundary (alpha's per_atom_<split>.csv rows).

    (a) bonds to the nearest cut C-C bond, (b) Euclidean distance to the nearest cap; calibration points per
    setting, final_eval points (frozen setting only) as open black markers.
    """

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), sharey=True)
    rows = [r for r in calibration or [] if r.get("force_error") not in (None, "")]
    if not rows:
        for ax in axes:
            mark_missing(ax, "no per-atom fidelity rows")
        return fig
    settings = sorted({str(r["setting_id"]) for r in rows})
    for panel, (ax, key, label) in enumerate(zip(axes, ("graph_distance_to_cut", "euclidean_distance_to_cap"),
                                                ("C-C bonds to the nearest cut bond", "distance to the nearest H cap (A)"))):
        for index, setting in enumerate(settings):
            pts = [(float(r[key]), float(r["force_error"])) for r in rows if str(r["setting_id"]) == setting and r.get(key) not in (None, "")]
            if not pts:
                continue
            x = np.asarray([p[0] for p in pts]) + (0.08 * (index - (len(settings) - 1) / 2) if panel == 0 else 0.0)
            ax.scatter(x, [p[1] for p in pts], s=10, alpha=0.75, color=CATEGORICAL[index % len(CATEGORICAL)],
                       label=f"{setting}{' (selected)' if setting == selected else ''}", linewidths=0)
        final = [r for r in final_eval or [] if r.get(key) not in (None, "") and r.get("force_error") not in (None, "")]
        if final:
            ax.scatter([float(r[key]) for r in final], [float(r["force_error"]) for r in final], s=22, facecolors="none",
                       edgecolors=INK, linewidths=0.8, label="final_eval (frozen setting)")
        if target is not None:
            ax.axhline(target, color=INK, linestyle=":", linewidth=0.9, label=f"pre-registered max {target:g}")
        ax.set_xlabel(label)
        ax.set_yscale("log")
        ax.grid(True, color=GRID, linewidth=0.5)
    axes[0].set_ylabel("|F_frag - F_parent| per core atom (eV/A)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6, frameon=False)
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    return fig


# ---------------------------------------------------------------------------------------------- 4 learning curves


def fig_learning_curves(curves: Mapping[str, Sequence[Sequence[Mapping[str, Any]]]] | None, *, metric: str = "force_rmse",
                        x_key: str = "cumulative_cpu_core_s", variants: Sequence[str] | None = None, title: str | None = None) -> Any:
    """Metric vs cumulative measured DFT cost: thin line per repetition, thick line through the per-round mean.

    The mean is drawn only for rounds that every repetition reached, at the mean measured cost.
    """

    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    names = [n for n in (variants or list(curves or {})) if curves and n in curves]
    plotted = 0
    scale = 3600.0 if x_key == "cumulative_cpu_core_s" else 1.0
    for name in names:
        reps = [[p for p in rep if p.get(metric) is not None and p.get(x_key) is not None] for rep in curves[name]]
        reps = [rep for rep in reps if rep]
        if not reps:
            continue
        colour = colour_for(name, names)
        for rep in reps:
            ax.plot([p[x_key] / scale for p in rep], [p[metric] for p in rep], color=colour, linewidth=0.8, alpha=0.45, marker="o", markersize=3)
        depth = min(len(rep) for rep in reps)
        mean_x = [np.mean([rep[k][x_key] for rep in reps]) / scale for k in range(depth)]
        mean_y = [np.mean([rep[k][metric] for rep in reps]) for k in range(depth)]
        ax.plot(mean_x, mean_y, color=colour, linewidth=2.0, marker="o", markersize=5, label=f"{ARM_LABELS.get(name, name)} (n={len(reps)})")
        plotted += 1
    if not plotted:
        mark_missing(ax, "no executed arm runs")
        return fig
    ax.set_xlabel("cumulative DFT cost (CPU core-hours, measured)" if x_key == "cumulative_cpu_core_s" else x_key)
    ax.set_ylabel(f"{metric} on final_eval (eV/A)")
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- 5 ablations


def fig_ablations(final: Mapping[str, Sequence[float | None]] | None, *, metric: str = "force_rmse") -> Any:
    """Final metric per variant: bar = mean over repetitions, dots = individual repetitions."""

    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    entries = [(name, [v for v in values if v is not None]) for name, values in (final or {}).items()]
    entries = [(name, values) for name, values in entries if values]
    if not entries:
        mark_missing(ax, "no executed ablation runs")
        return fig
    names = [name for name, _ in entries]
    for index, (name, values) in enumerate(entries):
        colour = colour_for(name, names)
        ax.bar(index, np.mean(values), width=0.6, color=colour, alpha=0.35, edgecolor=colour)
        ax.scatter([index] * len(values), values, color=colour, s=18, zorder=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([ARM_LABELS.get(n, n) for n in names], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(f"final {metric} (eV/A)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------- 6 timeline


def fig_timeline(report: Mapping[str, Any] | None) -> Any:
    """Closed-loop timeline from the controller's report JSON (``workflow.controller.summarize`` output,
    optionally with ``event_steps``): segments coloured by model, event triggers, promotions/rollbacks."""

    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    segments = list((report or {}).get("segments") or [])
    if not segments:
        mark_missing(ax, "no closed-loop controller report")
        return fig
    models = []
    for seg in segments:
        if seg.get("model_id") not in models:
            models.append(seg.get("model_id"))
    start = 0
    boundaries = []
    for seg in segments:
        end = seg.get("end_step")
        if end is None:
            continue
        index = models.index(seg.get("model_id"))
        ax.barh(0, end - start, left=start, height=0.4, color=CATEGORICAL[index % len(CATEGORICAL)], edgecolor="white", linewidth=2)
        if boundaries and boundaries[-1][1] != seg.get("model_id"):
            boundaries.append((start, seg.get("model_id")))
        elif not boundaries:
            boundaries.append((start, seg.get("model_id")))
        start = end
    steps = list((report or {}).get("event_steps") or [])
    if steps:
        ax.scatter(steps, [0.45] * len(steps), marker="v", color="#e34948", s=30, label="monitor event", zorder=3)
    promotions = list((report or {}).get("promotions") or [])
    actions = {p.get("model_id"): p.get("action") for p in promotions}
    for step, model in boundaries[1:]:
        action = actions.get(model, "model change")
        ax.axvline(step, color=INK, linewidth=0.9, linestyle="--")
        ax.text(step, 0.62, f"{action}: {str(model)[:10]}", fontsize=7, color=INK, rotation=0, ha="left")
    for index, model in enumerate(models):
        ax.barh([], [], color=CATEGORICAL[index % len(CATEGORICAL)], label=f"model {str(model)[:12]}")
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.9)
    ax.set_xlabel("MD step")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    if not steps:
        ax.text(0.01, 0.9, "event steps not in report", transform=ax.transAxes, fontsize=7, color=MUTED)
    fig.tight_layout()
    return fig


__all__ = ["CATEGORICAL", "colour_for", "fig_ablations", "fig_fidelity", "fig_fidelity_distance", "fig_hotspots", "fig_receptive_field", "missing_reasons", "fig_learning_curves", "fig_risk_vs_error",
           "fig_timeline", "mark_missing", "save"]
