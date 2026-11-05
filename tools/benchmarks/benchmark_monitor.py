# Ported from PHAL (tests/scripts/benchmark_monitor.py) at 96263b6; MIT License, see NOTICE.
"""Benchmark the GMD_active monitor kernels (direct O(N^2) references vs shared neighbour-list paths).

For each atom count, a deterministic periodic hydrogen box (jittered cubic lattice at a fixed number
density, default 0.064 atoms/A^3 = 2.5 A spacing, fixed seed) is built and these kernels are timed:

=================  ============================================  =================================================
metric             direct (reference) path                       fast (production) path
=================  ============================================  =================================================
``neighbor_list``  --                                            ``utils.neighbors.build_neighbor_pairs`` (one per
                                                                 frame, shared by every metric below)
``rmin``           ``monitors.geometry.                          ``monitors.geometry.minimum_neighbor_distances``
                   minimum_neighbor_distances_exact``            on the shared pairs
``coordination``   ``monitors.coordination.                      ``monitors.coordination.smooth_coordination_numbers``
                   smooth_coordination_numbers_reference``       on the shared pairs
``lj_residual``    --  (only one implementation exists)          ``monitors.lj_local.LocalLJEstimator`` (linear,
                                                                 every atom) on the shared pairs
=================  ============================================  =================================================

``--include-nonlinear`` adds ``lj_residual_nonlinear`` (the bounded scipy diagnostic estimator) as
its own single-path row. PHAL reported it as the "direct" column of ``lj_residual``, which compared
two different algorithms; here it is never presented as a speedup.

PHAL used uniformly random positions in a fixed 40 A cube: at 100 atoms almost no pair lies within
the 6 A cutoff, so the neighbour-list paths timed near-empty loops. A constant density keeps the
neighbour count per atom (~60 within 6 A) realistic at every size; ``box_A`` records the cube edge.

Forces are a physical-ish 12-6 LJ field (epsilon 0.01 eV, sigma 2.0 A, plus 1 meV/A noise) summed
over the shared pairs, instead of PHAL's all-ones array, so the LJ fit does representative work.

Timing and memory are separate passes: every kernel is called once to warm up, timed over
``--repeats`` calls with ``time.perf_counter`` (no tracer active), then called once more under
``tracemalloc`` for the peak Python allocation. PHAL ran the timed loop under ``tracemalloc``,
which inflated the times.

Usage (from the repository root, with the package installed or ``PYTHONPATH=src``)::

    python tools/benchmarks/benchmark_monitor.py --atoms 100,500,1000 --repeats 3 --output bench.csv --json-output bench.json
    python tools/benchmarks/benchmark_monitor.py --quick --baseline bench.json --threshold 0.2 [--warn-only]

Exit status 1 if any ``fast_s`` regressed by more than ``--threshold`` (fraction) against the
baseline JSON, unless ``--warn-only``; 2 if the package is not importable.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import sys
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

LOGGER = logging.getLogger("gmd_active_learning.tools.benchmark_monitor")

CSV_COLUMNS = (
    "metric", "n_atoms", "direct_s", "fast_s", "speedup", "direct_peak_mb", "fast_peak_mb", "direct_impl", "fast_impl", "box_A",
)
DEFAULT_ATOMS = "100,500,1000"
QUICK_ATOMS = "32,64"
DEFAULT_THRESHOLD = 0.2
DEFAULT_SEED = 7
DEFAULT_DENSITY = 0.064  # atoms / A^3 (2.5 A lattice spacing)
CUTOFF = 6.0
LJ_EPSILON = 0.01
LJ_SIGMA = 2.0
FORCE_NOISE = 1.0e-3


@dataclass(frozen=True, slots=True)
class Kernel:
    metric: str
    direct: Callable[[], object] | None
    fast: Callable[[], object]
    direct_impl: str
    fast_impl: str


@dataclass(frozen=True, slots=True)
class System:
    positions: np.ndarray
    numbers: np.ndarray
    cell: np.ndarray
    pbc: tuple[bool, bool, bool]
    forces: np.ndarray


def build_system(n_atoms: int, *, seed: int = DEFAULT_SEED, density: float = DEFAULT_DENSITY) -> System:
    """Deterministic periodic H box: jittered cubic lattice (no overlaps) with LJ forces."""

    from gmd_active_learning.utils.neighbors import build_neighbor_pairs

    if n_atoms < 2:
        raise ValueError("benchmark systems need at least 2 atoms")
    if density <= 0.0:
        raise ValueError("density must be positive")
    rng = np.random.default_rng(seed)
    per_side = int(np.ceil(n_atoms ** (1.0 / 3.0)))
    spacing = density ** (-1.0 / 3.0)
    box_length = per_side * spacing
    grid = np.stack(np.meshgrid(*(np.arange(per_side),) * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    sites = grid[rng.permutation(len(grid))[:n_atoms]]
    positions = (sites + 0.5) * spacing + rng.uniform(-0.15, 0.15, size=(n_atoms, 3)) * spacing
    cell = np.eye(3) * box_length
    pbc = (True, True, True)
    pairs = build_neighbor_pairs(positions, cell, pbc, CUTOFF)
    radius = pairs.distances
    prefactor = 24.0 * LJ_EPSILON * (-(2.0 * LJ_SIGMA**12 / radius**14) + (LJ_SIGMA**6 / radius**8))
    forces = np.zeros((n_atoms, 3))
    np.add.at(forces, pairs.first, prefactor[:, None] * pairs.vectors)
    forces += FORCE_NOISE * rng.standard_normal(forces.shape)
    return System(positions=positions, numbers=np.ones(n_atoms, dtype=np.int64), cell=cell, pbc=pbc, forces=forces)


def kernels(system: System, *, include_nonlinear: bool = False) -> list[Kernel]:
    from gmd_active_learning.monitors.coordination import (
        smooth_coordination_numbers,
        smooth_coordination_numbers_reference,
    )
    from gmd_active_learning.monitors.geometry import minimum_neighbor_distances, minimum_neighbor_distances_exact
    from gmd_active_learning.monitors.lj_local import LocalLJEstimator
    from gmd_active_learning.utils.neighbors import build_neighbor_pairs

    pairs = build_neighbor_pairs(system.positions, system.cell, system.pbc, CUTOFF)
    linear = LocalLJEstimator(estimator="linear", cutoff=CUTOFF)
    out = [
        Kernel(
            "neighbor_list",
            None,
            lambda: build_neighbor_pairs(system.positions, system.cell, system.pbc, CUTOFF),
            "",
            "utils.neighbors.build_neighbor_pairs",
        ),
        Kernel(
            "rmin",
            lambda: minimum_neighbor_distances_exact(system.positions, system.cell, system.pbc),
            lambda: minimum_neighbor_distances(pairs),
            "monitors.geometry.minimum_neighbor_distances_exact",
            "monitors.geometry.minimum_neighbor_distances",
        ),
        Kernel(
            "coordination",
            lambda: smooth_coordination_numbers_reference(system.positions, system.numbers, system.cell, system.pbc, cutoff=CUTOFF),
            lambda: smooth_coordination_numbers(pairs, system.numbers, cutoff=CUTOFF),
            "monitors.coordination.smooth_coordination_numbers_reference",
            "monitors.coordination.smooth_coordination_numbers",
        ),
        Kernel("lj_residual", None, lambda: linear(pairs, system.forces), "", "monitors.lj_local.LocalLJEstimator[linear]"),
    ]
    if include_nonlinear:
        nonlinear = LocalLJEstimator(estimator="nonlinear", cutoff=CUTOFF)
        out.append(
            Kernel("lj_residual_nonlinear", None, lambda: nonlinear(pairs, system.forces), "", "monitors.lj_local.LocalLJEstimator[nonlinear]")
        )
    return out


def time_call(fn: Callable[[], object], repeats: int) -> float:
    """Mean wall time per call (s) after one untimed warm-up call; no tracer is active."""

    fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def peak_memory_mb(fn: Callable[[], object]) -> float:
    """Peak Python allocation of one call (MiB), measured in its own pass."""

    tracemalloc.start()
    try:
        fn()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / (1024 * 1024)


def benchmark(
    atom_counts: Sequence[int], *, repeats: int, seed: int, density: float = DEFAULT_DENSITY, include_nonlinear: bool = False
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_atoms in atom_counts:
        system = build_system(n_atoms, seed=seed, density=density)
        for kernel in kernels(system, include_nonlinear=include_nonlinear):
            fast_s = time_call(kernel.fast, repeats)
            direct_s = None if kernel.direct is None else time_call(kernel.direct, repeats)
            fast_peak = peak_memory_mb(kernel.fast)
            direct_peak = None if kernel.direct is None else peak_memory_mb(kernel.direct)
            row = {
                "metric": kernel.metric,
                "n_atoms": int(n_atoms),
                "direct_s": None if direct_s is None else round(direct_s, 6),
                "fast_s": round(fast_s, 6),
                "speedup": None if direct_s is None else (round(direct_s / fast_s, 2) if fast_s > 0 else float("inf")),
                "direct_peak_mb": None if direct_peak is None else round(direct_peak, 3),
                "fast_peak_mb": round(fast_peak, 3),
                "direct_impl": kernel.direct_impl,
                "fast_impl": kernel.fast_impl,
                "box_A": round(float(system.cell[0, 0]), 4),
            }
            LOGGER.info("%s n=%d fast=%.6fs direct=%s", kernel.metric, n_atoms, fast_s, direct_s)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------
# baseline regression gate


def baseline_key(row: dict[str, Any]) -> str:
    return f"{row['metric']}:{row['n_atoms']}"


def load_baseline(path: Path) -> dict[str, float]:
    """``metric:n_atoms -> fast_s`` from a JSON written by ``--json-output`` (or a bare list of rows)."""

    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("results", [])
    baseline: dict[str, float] = {}
    for row in rows:
        value = row.get("fast_s")
        if value is None or value == "":
            continue
        baseline[baseline_key(row)] = float(value)
    return baseline


def regressions(rows: Sequence[dict[str, Any]], baseline: dict[str, float], *, threshold: float) -> list[str]:
    failures: list[str] = []
    for row in rows:
        key = baseline_key(row)
        reference = baseline.get(key)
        if reference is None or reference <= 0.0:
            continue
        current = float(row["fast_s"])
        if current > reference * (1.0 + threshold):
            failures.append(f"{key} fast_s {current:.6f}s > baseline {reference:.6f}s by more than {threshold:.0%}")
    return failures


# ---------------------------------------------------------------------------------------------
# CLI


def _parse_atoms(raw: str) -> list[int]:
    counts = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not counts or any(count < 2 for count in counts):
        raise argparse.ArgumentTypeError(f"--atoms needs comma-separated counts >= 2, got {raw!r}")
    return counts


def _write_csv(handle: TextIO, rows: Sequence[dict[str, Any]]) -> None:
    writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if row[key] is None else row[key] for key in CSV_COLUMNS})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--atoms", default=None, help=f"comma-separated atom counts (default {DEFAULT_ATOMS}; --quick: {QUICK_ATOMS})")
    parser.add_argument("--repeats", type=int, default=None, help="timed calls per kernel (default 1; --quick: 1)")
    parser.add_argument("--quick", action="store_true", help="small systems for a smoke run in a few seconds")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed for positions and force noise")
    parser.add_argument("--density", type=float, default=DEFAULT_DENSITY, help="number density in atoms/A^3 (default 0.064)")
    parser.add_argument("--include-nonlinear", action="store_true", help="also time the scipy nonlinear LJ estimator")
    parser.add_argument("--output", type=Path, default=None, help="CSV output path (default: stdout)")
    parser.add_argument("--json-output", type=Path, default=None, help="JSON output path (usable as a later --baseline)")
    parser.add_argument("--baseline", type=Path, default=None, help="JSON baseline for the fast_s regression gate")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="allowed fast_s regression fraction (default 0.2)")
    parser.add_argument("--warn-only", action="store_true", help="report regressions without failing")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    try:
        import gmd_active_learning  # noqa: F401
    except ImportError:
        print("gmd_active_learning is not importable: install the package (pip install -e .) or set PYTHONPATH=src", file=sys.stderr)
        return 2
    if args.threshold < 0.0:
        parser.error("--threshold must be >= 0")
    try:
        atom_counts = _parse_atoms(args.atoms if args.atoms is not None else (QUICK_ATOMS if args.quick else DEFAULT_ATOMS))
    except (argparse.ArgumentTypeError, ValueError) as exc:
        parser.error(str(exc))
    repeats = max(1, args.repeats if args.repeats is not None else 1)

    if args.density <= 0.0:
        parser.error("--density must be positive")
    rows = benchmark(atom_counts, repeats=repeats, seed=args.seed, density=args.density, include_nonlinear=args.include_nonlinear)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            _write_csv(handle, rows)
    else:
        _write_csv(sys.stdout, rows)

    failures: list[str] = []
    if args.baseline is not None:
        failures = regressions(rows, load_baseline(args.baseline), threshold=args.threshold)
        for failure in failures:
            print(f"PERF REGRESSION: {failure}", file=sys.stderr)
    if args.json_output is not None:
        import gmd_active_learning

        payload = {
            "threshold": args.threshold,
            "seed": args.seed,
            "repeats": repeats,
            "quick": bool(args.quick),
            "density_per_A3": args.density,
            "cutoff_A": CUTOFF,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "gmd_active_learning": getattr(gmd_active_learning, "__version__", "unknown"),
            },
            "results": rows,
            "regressions": failures,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if failures and not args.warn_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
