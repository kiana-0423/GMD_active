"""Figures of the repair benchmark (PDF/SVG/PNG). Every panel without executed data says so.

Arms keep fixed colours and marker shapes (colour never carries identity alone: legend + marker +
the report's tables). One y-axis per panel; families are small multiples on a shared scale.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from tools.experiments.alkane_hotspots.figures import CATEGORICAL, GRID, INK, MUTED, _plt, mark_missing, save

MARKERS = ("o", "s", "D", "^", "v", "P", "X", "*")


def arm_style(arms: Sequence[str]) -> dict[str, dict[str, str]]:
    """Fixed order: the configuration's arm order -> categorical slot and marker (never by rank)."""

    return {arm: {"color": CATEGORICAL[i % len(CATEGORICAL)], "marker": MARKERS[i % len(MARKERS)]} for i, arm in enumerate(arms)}


def _axes(ax: Any) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7)


def _family_axes(plt: Any, families: Sequence[str]) -> tuple[Any, Any]:
    fig, axes = plt.subplots(1, len(families), figsize=(2.6 * len(families) + 0.6, 3.2), sharey=True, squeeze=False)
    return fig, axes[0]


def fig_region_errors(evaluation: Mapping[str, Any] | None, *, arms: Sequence[str], families: Sequence[str], threshold: float) -> Any:
    """Challenged-region force RMSE after repair, per run (dots), against the frozen initial model (line)."""

    plt = _plt()
    fig, axes = _family_axes(plt, families)
    if not evaluation or not evaluation.get("runs"):
        mark_missing(axes[0], "sealed evaluation not executed")
        return fig
    style = arm_style(arms)
    initial = evaluation["initial_model"]["challenge_regions"]
    for ax, family in zip(axes, families):
        _axes(ax)
        base = (initial.get(family) or {}).get("force_rmse")
        if base is not None:
            ax.axhline(base, color=MUTED, linewidth=1.2, linestyle="--", label="A frozen initial")
        ax.axhline(threshold, color=INK, linewidth=0.8, linestyle=":", label="success threshold")
        for i, arm in enumerate(a for a in arms if a in {r["arm"] for r in evaluation["runs"]}):
            values = [r["outcome"]["region_rmse_after"] for r in evaluation["runs"] if r["arm"] == arm and r["family"] == family
                      and r["outcome"]["region_rmse_after"] is not None]
            x = i + np.linspace(-0.15, 0.15, max(len(values), 1))[: len(values)]
            ax.scatter(x, values, s=28, color=style[arm]["color"], marker=style[arm]["marker"], edgecolors="white", linewidths=0.6, label=arm, zorder=3)
        ax.set_title(family, fontsize=8, color=INK)
        ax.set_xticks([])
    axes[0].set_ylabel("challenged-region force RMSE (eV/A)", fontsize=7, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        h, lab = ax.get_legend_handles_labels()
        for item, name in zip(h, lab):
            if name not in labels:
                handles.append(item)
                labels.append(name)
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    return fig


def fig_retention(evaluation: Mapping[str, Any] | None, *, arms: Sequence[str], tolerance: float) -> Any:
    """Relative change of the sealed in-domain test force RMSE per run (0 = unchanged; above the line = regression)."""

    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    if not evaluation or not evaluation.get("runs"):
        mark_missing(ax, "sealed evaluation not executed")
        return fig
    _axes(ax)
    style = arm_style(arms)
    present = [a for a in arms if a in {r["arm"] for r in evaluation["runs"]}]
    for i, arm in enumerate(present):
        values = [r["outcome"]["relative_retention_change"] for r in evaluation["runs"] if r["arm"] == arm and r["outcome"]["relative_retention_change"] is not None]
        x = i + np.linspace(-0.2, 0.2, max(len(values), 1))[: len(values)]
        ax.scatter(x, values, s=24, color=style[arm]["color"], marker=style[arm]["marker"], edgecolors="white", linewidths=0.6, zorder=3)
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
    ax.axhline(tolerance, color=INK, linewidth=0.8, linestyle=":", label=f"retention tolerance (+{tolerance:.0%})")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, fontsize=7)
    ax.set_ylabel("in-domain test RMSE change vs initial", fontsize=7, color=INK)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def fig_stability(evaluation: Mapping[str, Any] | None, *, arms: Sequence[str], families: Sequence[str]) -> Any:
    """Fraction of stable trajectories (every trajectory in the denominator), initial model vs each arm."""

    plt = _plt()
    fig, axes = _family_axes(plt, families)
    if not evaluation or not evaluation.get("runs"):
        mark_missing(axes[0], "sealed evaluation not executed")
        return fig
    style = arm_style(arms)
    for ax, family in zip(axes, families):
        _axes(ax)
        rows = [("A_initial", evaluation["initial_model"]["stability"].get(family) or {}, MUTED)]
        for arm in arms:
            runs = [r for r in evaluation["runs"] if r["arm"] == arm and r["family"] == family and r.get("stability")]
            if runs:
                n = sum(r["stability"]["n_trajectories"] for r in runs)
                k = sum(r["stability"]["n_stable"] for r in runs)
                rows.append((arm, {"n_trajectories": n, "n_stable": k}, style[arm]["color"]))
        for i, (name, item, colour) in enumerate(rows):
            n, k = item.get("n_trajectories") or 0, item.get("n_stable") or 0
            ax.bar(i, k / n if n else 0.0, width=0.6, color=colour, edgecolor="white", linewidth=2)
            ax.text(i, (k / n if n else 0.0) + 0.02, f"{k}/{n}", ha="center", fontsize=6, color=INK)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[0] for r in rows], rotation=35, ha="right", fontsize=6)
        ax.set_ylim(0, 1.12)
        ax.set_title(family, fontsize=8, color=INK)
    axes[0].set_ylabel("stable trajectories (fraction)", fontsize=7, color=INK)
    fig.tight_layout()
    return fig


def fig_failures(rows: Sequence[Mapping[str, Any]] | None, *, families: Sequence[str]) -> Any:
    """Initial model: challenged-region error vs the DFT force in the region; failures above the threshold."""

    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    data = [r for r in rows or [] if r.get("status") == "evaluated" and "region" in r]
    if not data:
        mark_missing(ax, "no evaluated challenge structures")
        return fig
    _axes(ax)
    style = arm_style(list(families))
    for family in families:
        mine = [r for r in data if r["family"] == family]
        if mine:
            ax.scatter([r["region"]["reference_force_rms"] for r in mine], [r["region"]["force_rmse"] for r in mine], s=30, color=style[family]["color"],
                       marker=style[family]["marker"], edgecolors="white", linewidths=0.6, label=family, zorder=3)
    threshold = min(r["failure_threshold"] for r in data)
    ax.axhline(threshold, color=INK, linewidth=0.8, linestyle=":", label="failure threshold")
    ax.set_xlabel("DFT force RMS in the challenged region (eV/A)", fontsize=7, color=INK)
    ax.set_ylabel("model error RMSE in the region (eV/A)", fontsize=7, color=INK)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def fig_qualification(reports: Sequence[Mapping[str, Any]] | None, *, target: float) -> Any:
    """Per qualification stage: id_val per-structure force RMSE against the acceptance target."""

    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    if not reports:
        mark_missing(ax, "no qualification stage executed")
        return fig
    _axes(ax)
    for i, report in enumerate(reports):
        values = [row["force_rmse"] for row in report["id_val_per_structure"]]
        x = i + np.linspace(-0.15, 0.15, max(len(values), 1))[: len(values)]
        ax.scatter(x, values, s=24, color=CATEGORICAL[0], edgecolors="white", linewidths=0.6, zorder=3)
        ax.scatter([i], [report["id_val_accuracy"]["force_rmse"]], s=60, marker="_", color=INK, zorder=4)
        ax.text(i, max(values) * 1.05, "accepted" if report["accepted"] else "not accepted", ha="center", fontsize=6, color=INK)
    ax.axhline(target, color=INK, linewidth=0.8, linestyle=":", label=f"acceptance target {target} eV/A")
    ax.set_xticks(range(len(reports)))
    ax.set_xticklabels([f"stage {r['qualification_stage']} ({r['n_train']} train)" for r in reports], fontsize=7)
    ax.set_ylabel("id_val force RMSE per structure (eV/A)", fontsize=7, color=INK)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def fig_qualification_md(reports: Sequence[Mapping[str, Any]] | None, *, steps: int) -> Any:
    """Per qualification MD trajectory: the step of the first broken covalent bond (hollow marker at the top: intact)."""

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    rows = [(r["qualification_stage"], t) for r in (reports or []) for t in r["md_trajectories"]]
    if not rows:
        mark_missing(ax, "no qualification MD executed")
        return fig
    _axes(ax)
    starts = list(dict.fromkeys((stage, t["partition"], t["start"]) for stage, t in rows))
    seeds = sorted({t["seed"] for _, t in rows})
    for k, seed in enumerate(seeds):
        style = {"color": CATEGORICAL[k % len(CATEGORICAL)], "marker": MARKERS[k % len(MARKERS)]}
        for stage, t in rows:
            if t["seed"] != seed:
                continue
            x = starts.index((stage, t["partition"], t["start"])) + (k - (len(seeds) - 1) / 2) * 0.18
            broken = t.get("first_broken_step")
            if broken is None:
                ax.scatter([x], [steps], s=40, marker=style["marker"], facecolors="none", edgecolors=style["color"], linewidths=1.2, zorder=3)
            else:
                ax.scatter([x], [broken], s=40, marker=style["marker"], color=style["color"], edgecolors="white", linewidths=0.6, zorder=3)
        ax.scatter([], [], marker=style["marker"], color=style["color"], label=f"seed {seed}")
    ax.scatter([], [], marker="o", facecolors="none", edgecolors=MUTED, label=f"intact through {steps} steps")
    ax.axhline(steps, color=INK, linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(starts)))
    ax.set_xticklabels([f"{start.replace('C16-', '')}\n{part}" + (f"\nstage {stage}" if len({s for s, _, _ in starts}) > 1 else "")
                        for stage, part, start in starts], fontsize=6)
    ax.set_ylim(0, steps * 1.12)
    ax.set_ylabel("MD step of the first broken bond", fontsize=7, color=INK)
    ax.legend(fontsize=7, frameon=False, ncol=len(seeds) + 1, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    return fig


def fig_training_curves(curves: Mapping[str, Mapping[str, Any]] | None, *, target: float) -> Any:
    """Training (solid) and validation (dashed) force RMSE per ensemble member; a dot marks the selected epoch."""

    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    members = {f"{stage} {name}": c for stage, by_member in (curves or {}).items() for name, c in by_member.items()}
    if not members:
        mark_missing(ax, "no training log")
        return fig
    _axes(ax)
    for i, (name, c) in enumerate(members.items()):
        color = CATEGORICAL[i % len(CATEGORICAL)]
        ax.plot(c["epoch"], c["train_force"], color=color, linewidth=1.4, label=f"{name.replace('_', ' ')} train")
        ax.plot(c["epoch"], c["val_force"], color=color, linewidth=1.4, linestyle="--", label=f"{name.replace('_', ' ')} validation")
        ax.scatter([c["best_epoch"]], [c["best_val_force"]], s=36, color=color, edgecolors="white", linewidths=0.8, zorder=4)
    ax.axhline(target, color=INK, linewidth=0.8, linestyle=":", label=f"acceptance target {target} eV/A")
    ax.set_yscale("log")
    ax.set_xlabel("epoch", fontsize=7, color=INK)
    ax.set_ylabel("force RMSE (eV/A)", fontsize=7, color=INK)
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    return fig


def fig_compute(ledger: Mapping[str, Any] | None, projections: Mapping[str, Mapping[str, Any]] | None = None) -> Any:
    """Per stage: allowance (track), measured consumption (bar), and the projected upper bound of a blocked stage (tick)."""

    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    if not ledger or "allowances" not in ledger:
        mark_missing(ax, "no ledger")
        return fig
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7)
    allow = ledger["allowances"]["stages_core_h"]
    used = ledger["consumed_core_h"]["stages"]
    names = list(allow)[::-1]
    y = np.arange(len(names))
    ax.barh(y, [allow[n] for n in names], height=0.6, color=GRID, label="allowance")
    ax.barh(y, [used.get(n, 0.0) for n in names], height=0.6, color=CATEGORICAL[0], label="measured consumption")
    for i, n in enumerate(names):
        if used.get(n):
            ax.text(used[n] + 1.0, i, f"{used[n]:.1f} of {allow[n]:.0f}", va="center", fontsize=6, color=INK)
    shown = False
    for proj in (projections or {}).values():
        for sc in proj.get("scopes", []):
            if not sc.get("fits") and sc["scope"] in names:
                i = names.index(sc["scope"])
                at = allow[sc["scope"]] - sc["remaining_core_h"] + sc["upper90_core_h"]
                ax.plot([at, at], [i - 0.4, i + 0.4], color=CATEGORICAL[1], linewidth=2.0, label=None if shown else "consumed + projected next stage (upper)")
                ax.text(at + 1.5, i, (f"q{proj['qualification_stage']} " if proj.get("qualification_stage") is not None else "") + f"upper {at:.1f}",
                        fontsize=6, color=INK, va="center")
                shown = True
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=7)
    ax.set_xlabel("CPU core-hours", fontsize=7, color=INK)
    ax.legend(fontsize=6, frameon=False, loc="center right")
    fig.tight_layout()
    return fig


__all__ = ["arm_style", "fig_compute", "fig_failures", "fig_qualification", "fig_qualification_md", "fig_training_curves", "fig_region_errors", "fig_retention", "fig_stability", "save"]
