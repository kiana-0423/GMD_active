"""Render PHAL manuscript figures and tables from bundled, measured pilot data."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "phal-paper-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

LOG = logging.getLogger(__name__)
BLUE, ORANGE, GREEN, PURPLE, GRAY = "#0072B2", "#D55E00", "#009E73", "#7B5EA7", "#606770"
PALETTE = [BLUE, GREEN, ORANGE, PURPLE, GRAY]
ARMS = ["random_local", "committee_local", "phal_local", "phal_full_system", "no_lj"]
NAMES = ["Random local", "Committee local", "PHAL local", "PHAL full", "PHAL without LJ"]
UNIT = r"eV $\mathrm{\AA}^{-1}$"


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "savefig.dpi": 300, "axes.axisbelow": True})


def save(fig: Any, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(folder / f"{name}.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    LOG.info("Rendered %s", name)


def architecture(folder: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set(xlim=(0, 10), ylim=(0, 5.7))
    ax.axis("off")
    nodes = [
        (0.15, 3.65, "GMD", "MD trajectory segments\nForces, species and atom IDs", BLUE),
        (3.55, 3.65, "PHAL monitoring", "Staged scores and events\nHotspot localization", ORANGE),
        (6.95, 3.65, "Local extraction", "Core, buffer and H caps\nGeometry and force fidelity", ORANGE),
        (6.95, 1.75, "CP2K", "Reference forces\nValidated job collection", GREEN),
        (3.55, 1.75, "GMD_SGT", "Weighted force training\nCandidate model export", PURPLE),
        (0.15, 1.75, "Model promotion", "Validation and export checks\nAccept or retain; restart MD", BLUE),
    ]
    for x, y, title, body, color in nodes:
        ax.add_patch(FancyBboxPatch((x, y), 2.9, 1.2, boxstyle="round,pad=0.04,rounding_size=0.09",
                                   facecolor=color + "0E", edgecolor=color, linewidth=1.2))
        ax.text(x + 1.45, y + .89, title, ha="center", va="center", weight="bold", fontsize=10, color=color)
        ax.text(x + 1.45, y + .40, body, ha="center", va="center", fontsize=7.8, linespacing=1.55)
    for start, end in [((3.1, 4.25), (3.5, 4.25)), ((6.5, 4.25), (6.9, 4.25)),
                       ((8.4, 3.55), (8.4, 3.02)), ((6.9, 2.35), (6.5, 2.35)),
                       ((3.5, 2.35), (3.1, 2.35)), ((1.6, 3.02), (1.6, 3.55))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, color=GRAY, linewidth=1.1))
    ax.add_patch(FancyBboxPatch((.15, .30), 9.7, .70, boxstyle="round,pad=0.04,rounding_size=0.07",
                               edgecolor="#B5BDC5", facecolor="#F5F7F9"))
    ax.text(5, .65, "Persistent state  •  Provenance  •  Job reconciliation  •  Recovery and rollback",
            ha="center", va="center", fontsize=9)
    ax.text(5, 5.42, "A local reference-labeling loop in the GMD ecosystem", ha="center", weight="bold", fontsize=11)
    save(fig, folder, "fig01_workflow")


def molecules(data: dict[str, Any], folder: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.5))
    titles = {"normal": "Ordinary conformation", "torsion": "Local torsion", "bend": "Local bending", "contact": "Nonbonded contact"}
    records = {r["family"]: r for r in data["geometries"]}
    for letter, ax, family in zip("abcd", axes.flat, titles):
        row = records[family]
        xyz = np.asarray(row["positions"])
        centered = xyz - xyz.mean(axis=0)
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        xy = centered @ basis[:2].T
        if xy[15, 0] < xy[0, 0]:
            xy[:, 0] *= -1
        for i, j in row["bonds"]:
            ax.plot(xy[[i, j], 0], xy[[i, j], 1], color="#9AA2A9", lw=.8, zorder=1)
        ax.scatter(xy[16:, 0], xy[16:, 1], s=13, c="#D9E0E6", edgecolors="#9AA2A9", linewidths=.35, zorder=2)
        ax.scatter(xy[:16, 0], xy[:16, 1], s=43, c=BLUE, edgecolors="white", linewidths=.4, zorder=3)
        hot = row["perturbation"]["atoms"]
        if hot:
            ax.scatter(xy[hot, 0], xy[hot, 1], s=[76 if i < 16 else 32 for i in hot],
                       facecolors="none", edgecolors=ORANGE, linewidths=1.15, zorder=4)
        ax.set(xlim=(-11.4, 11.4), ylim=(-6.0, 6.0), aspect="equal")
        ax.axis("off")
        ax.set_title(f"({letter}) {titles[family]}", loc="left", pad=1)
    fig.legend([Line2D([], [], marker="o", color="none", markerfacecolor=BLUE, markersize=6),
                Line2D([], [], marker="o", color="none", markerfacecolor="#D9E0E6", markeredgecolor="#9AA2A9", markersize=5),
                Line2D([], [], marker="o", color="none", markerfacecolor="none", markeredgecolor=ORANGE, markersize=7)],
               ["Carbon", "Hydrogen", "Imposed perturbation site"], loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(bottom=.10, hspace=.15, wspace=.10)
    save(fig, folder, "fig02_configurations")


def fidelity(data: dict[str, Any], folder: Path) -> None:
    order = ["graph-h2", "graph-h3", "graph-h4", "cluster-c1.5-b3", "cluster-c1.5-b4", "cluster-c1.5-b5"]
    labels = ["G2", "G3", "G4", "C3*", "C4", "C5"]
    by_id = {r["setting_id"]: r for r in data["calibration"]}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    specs = [("pooled_core_force_rmse", data["targets"]["core_force_rmse"], "Component RMSE", .035),
             ("max_core_force_error", data["targets"]["max_core_force_error"], "Maximum vector error", .24)]
    for col, (field, target, title, upper) in enumerate(specs):
        ax = axes[0, col]
        vals = [by_id[key][field] for key in order]
        ax.bar(np.arange(6), vals, color=[ORANGE if key == "cluster-c1.5-b3" else BLUE for key in order], width=.65)
        ax.axhline(target, color=GRAY, linestyle="--", lw=1, label="Acceptance threshold")
        ax.set(xticks=np.arange(6), xticklabels=labels, ylabel=f"{title} ({UNIT})", ylim=(0, upper))
        ax.set_title(f"({'ab'[col]}) C16 calibration", loc="left")
        ax.grid(axis="y", alpha=.15)
        ax = axes[1, col]
        vals = [data[key][field] for key in ("c16", "c32")]
        ax.bar([0, 1], vals, color=[BLUE, GREEN], width=.5)
        ax.axhline(target, color=GRAY, linestyle="--", lw=1)
        ax.set(xticks=[0, 1], xticklabels=["C16 held-out", "C32 transfer"],
               ylabel=f"{title} ({UNIT})", ylim=(0, target * 1.28))
        ax.set_title(f"({'cd'[col]}) Frozen C3 setting", loc="left")
        for x, value in enumerate(vals):
            ax.text(x, value + target * .035, f"{value:.4f}", ha="center", fontsize=8)
        ax.grid(axis="y", alpha=.15)
    fig.legend([Line2D([], [], color=GRAY, linestyle="--")], ["Acceptance threshold"], loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, .045, 1, 1))
    save(fig, folder, "fig03_fragment_fidelity")


def boundary_cost(data: dict[str, Any], folder: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    ax = axes[0]
    rows = [r for r in data["boundary"] if r["system"] in ("C16 held-out", "C32 transfer")]
    for system, marker, offset in [("C16 held-out", "o", -.08), ("C32 transfer", "^", .08)]:
        for element, color in [("C", ORANGE), ("H", BLUE)]:
            selected = [r for r in rows if r["system"] == system and r["element"] == element]
            ax.scatter([r["distance"] + offset for r in selected], [max(r["error"], 1e-8) for r in selected],
                       s=19, marker=marker, c=color, alpha=.75, linewidths=.25, edgecolors="white",
                       label=f"{system[:3]} {element}")
    ax.set(yscale="log", xlabel="Bond distance to fragment cut", ylabel=f"Atom vector error ({UNIT})")
    ax.set_title("(a) Core error and cut distance", loc="left")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.grid(alpha=.15)
    ax = axes[1]
    costs = data["c32_costs"]
    parents = ["C32-b00-contact0", "C32-b00-torsion0"]
    full = [data["c16"]["mean_parent_cost_core_s"] / 3600] + [
        next(r["core_s"] for r in costs if r["parent_id"] == p and r["kind"] == "full") / 3600 for p in parents]
    local = [data["c16"]["mean_fragment_cost_core_s"] / 3600] + [
        np.mean([r["core_s"] for r in costs if r["parent_id"] == p and r["kind"] == "fragment"]) / 3600 for p in parents]
    x = np.arange(3)
    ax.bar(x - .18, full, width=.34, color=BLUE, label="Full parent")
    ax.bar(x + .18, local, width=.34, color=ORANGE, label="Local fragment")
    ax.set(yscale="log", ylim=(.06, 45), xticks=x, xticklabels=["C16\nheld-out", "C32\ncontact", "C32\ntorsion"],
           ylabel="Allocated CPU core-hours / label")
    for i, value in enumerate(full):
        ax.text(i - .18, value * 1.15, f"{value:.2f}", ha="center", fontsize=7.5)
    for i, value in enumerate(local):
        ax.text(i + .18, value * 1.15, f"{value:.2f}", ha="center", fontsize=7.5)
    ax.set_title("(b) Measured reference cost", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=.15)
    fig.tight_layout()
    save(fig, folder, "fig04_boundary_and_cost")


def acquisition(data: dict[str, Any], folder: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for arm, name, color in zip(ARMS, NAMES, PALETTE):
        for rep, record in enumerate(data["arms"][arm]):
            rounds = record["rounds"]
            y = [r["force_rmse"] for r in rounds]
            for ax, field, divisor in [(axes[0], "cumulative_cpu_core_s", 3600), (axes[1], "n_labels", 1)]:
                ax.plot([r[field] / divisor for r in rounds], y, color=color,
                        linestyle="-" if rep == 0 else "--", marker="o" if rep == 0 else "s",
                        markersize=3.8, linewidth=1.2, alpha=.85)
    axes[0].set(xlabel="Attributed DFT cost (CPU core-hours)", xlim=(-.05, 2.95))
    axes[1].set(xlabel="Acquired labels", xticks=[0, 2, 4, 6, 8], xlim=(-.2, 8.2))
    for letter, ax in zip("ab", axes):
        ax.set(ylabel=f"Held-out force RMSE ({UNIT})", ylim=(0, .82))
        ax.set_title(f"({letter}) Exploratory static-pool study", loc="left")
        ax.axhline(data["accuracy_targets"][0], linestyle=":", color="#A0A0A0", lw=1)
        ax.grid(alpha=.15)
    handles = [Line2D([], [], color=color, label=name, linewidth=2) for name, color in zip(NAMES, PALETTE)]
    handles += [Line2D([], [], color="black", marker="o", label="Seed 11", linewidth=1),
                Line2D([], [], color="black", marker="s", linestyle="--", label="Seed 23", linewidth=1)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, .19, 1, 1))
    save(fig, folder, "fig05_acquisition")


def closed_loop(data: dict[str, Any], folder: Path) -> None:
    d = data["closed_loop"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    ax = axes[0]
    time = np.asarray(d["event_steps"]) * .0005
    ax.scatter(time, np.full(len(time), .65), marker="|", s=160, color=ORANGE, label="Monitor event")
    ax.plot([0, 2], [.65, .65], color="#A7AFB5", lw=.7, zorder=0)
    for t, label in [(1, "+2 labels"), (2, "+1 label")]:
        ax.axvline(t, ymin=.2, ymax=.8, color=BLUE, linestyle="--", lw=1)
        ax.text(t - .03, .20, label, ha="right", fontsize=7.6)
    ax.text(1.04, .07, "Model updates", ha="center", fontsize=7.6)
    ax.text(.5, 1.04, "Cycle 0", ha="center", fontsize=8)
    ax.text(1.5, 1.04, "Cycle 1", ha="center", fontsize=8)
    ax.set(xlim=(-.05, 2.08), ylim=(0, 1.25), yticks=[], xlabel="Simulation time (ps)")
    ax.spines["left"].set_visible(False)
    ax.set_title("(a) Real closed loop", loc="left")
    ax = axes[1]
    vals = d["validation_force_rmse"]
    label_counts = d["cumulative_new_labels"]
    ax.plot(label_counts, vals, "-o", color=BLUE, markersize=5)
    for i, value in zip(label_counts, vals):
        ax.annotate(f"{value:.3f}", (i, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    ax.set(xticks=label_counts, xlabel="Cumulative new labels",
           ylabel=f"Validation RMSE ({UNIT})", ylim=(0, .60))
    ax.set_title("(b) Validation error", loc="left")
    ax.grid(axis="y", alpha=.15)
    ax = axes[2]
    evaluation = d["frozen_evaluation"]
    for index, color in enumerate([BLUE, GREEN, ORANGE]):
        a = evaluation["initial"]["per_seed"][index]
        b = evaluation["final"]["per_seed"][index]
        values = [a["n_intact"] / a["n_frames"], b["n_intact"] / b["n_frames"]]
        offset = (index - 1) * .04
        ax.plot(np.asarray([0, 1]) + offset, values, "-o", color=color, markersize=4, label=f"Seed {a['seed']}")
    ax.set(xticks=[0, 1], xticklabels=["Initial", "Final"], ylim=(-.04, 1.13), xlim=(-.2, 1.2),
           ylabel="Fraction of frames with parent topology")
    ax.set_title("(c) Frozen-model MD", loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    ax.grid(axis="y", alpha=.15)
    fig.tight_layout(w_pad=1.4)
    save(fig, folder, "fig06_closed_loop")


def tables(data: dict[str, Any], out: Path) -> None:
    folder = out / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Setting & RMSE & Max.\ error & Mean atoms & Core-s/fragment \\", r"\midrule"]
    order = ["graph-h2", "graph-h3", "graph-h4", "cluster-c1.5-b3", "cluster-c1.5-b4", "cluster-c1.5-b5"]
    names = ["Graph, 2 hops", "Graph, 3 hops", "Graph, 4 hops", r"Radial, 3 \AA\ buffer (selected)", r"Radial, 4 \AA\ buffer", r"Radial, 5 \AA\ buffer"]
    by_id = {r["setting_id"]: r for r in data["calibration"]}
    for key, name in zip(order, names):
        r = by_id[key]
        lines.append(f"{name} & {r['pooled_core_force_rmse']:.4f} & {r['max_core_force_error']:.4f} & {r['mean_n_atoms']:.1f} & {r['mean_fragment_cost_core_s']:.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (folder / "fidelity_calibration.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Arm & \multicolumn{2}{c}{RMSE at four labels} & Mean RMSE & Mean core-s \\",
             r"& Seed 11 & Seed 23 & & \\", r"\midrule"]
    for arm, name in zip(ARMS[:4], NAMES[:4]):
        rows = [r["rounds"][1] for r in data["arms"][arm]]
        lines.append(f"{name} & {rows[0]['force_rmse']:.4f} & {rows[1]['force_rmse']:.4f} & {np.mean([r['force_rmse'] for r in rows]):.4f} & {np.mean([r['cumulative_cpu_core_s'] for r in rows]):.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (folder / "acquisition_four_labels.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    try:
        data = json.loads((args.root / "data/plot_data.json").read_text())
        style()
        architecture(args.root / "figures")
        molecules(data, args.root / "figures")
        fidelity(data, args.root / "figures")
        boundary_cost(data, args.root / "figures")
        acquisition(data, args.root / "figures")
        closed_loop(data, args.root / "figures")
        tables(data, args.root)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Figure generation failed: {exc}") from exc


if __name__ == "__main__":
    main()
