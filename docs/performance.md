# Monitor performance benchmark

`tools/benchmarks/benchmark_monitor.py` times the per-frame kernels of the staged monitor
(`gmd_active_learning.monitors`). It compares the direct O(N²) reference helpers with the
shared-neighbour-list paths that `StagedOODMonitor` actually uses, and it can gate on regressions
against a stored baseline. It is a port of PHAL's `tests/scripts/benchmark_monitor.py` (ledger
UTIL-008). The deliberate differences are listed at the end.

The benchmark imports the package normally. Run it from the repository root with the package
installed (`pip install -e .`) or with `PYTHONPATH=src`:

```bash
# smoke run, a few seconds
PYTHONPATH=src python tools/benchmarks/benchmark_monitor.py --quick

# regression data for later comparison
PYTHONPATH=src python tools/benchmarks/benchmark_monitor.py --atoms 100,500,1000,5000 --repeats 5 \
    --output benchmark_monitor.csv --json-output benchmark_monitor.json

# gate a change against that baseline (exit 1 if any fast_s is more than 20 % slower)
PYTHONPATH=src python tools/benchmarks/benchmark_monitor.py --atoms 100,500,1000,5000 --repeats 5 \
    --baseline benchmark_monitor.json [--threshold 0.2] [--warn-only]
```

## What is timed

Each atom count gets a deterministic periodic hydrogen box: a jittered cubic lattice at
`--density` (default 0.064 atoms/Å³, which is 2.5 Å spacing, about 60 neighbours within 6 Å) and
`--seed` (default 7). Forces are a 12-6 Lennard-Jones field (ε = 0.01 eV, σ = 2.0 Å) summed over the
neighbour pairs, plus 1 meV/Å of noise. The neighbour cutoff is 6 Å, the default `lj_cutoff`.

| `metric` | direct path | fast path |
|---|---|---|
| `neighbor_list` | none | `utils.neighbors.build_neighbor_pairs`: the single per-frame neighbour list shared by every metric below |
| `rmin` | `monitors.geometry.minimum_neighbor_distances_exact` | `monitors.geometry.minimum_neighbor_distances` on the shared pairs |
| `coordination` | `monitors.coordination.smooth_coordination_numbers_reference` | `monitors.coordination.smooth_coordination_numbers` on the shared pairs |
| `lj_residual` | none: only one implementation exists | `monitors.lj_local.LocalLJEstimator` (linear) over every atom, on the shared pairs |
| `lj_residual_nonlinear` (with `--include-nonlinear`) | none | the bounded scipy estimator (`estimator: nonlinear`, diagnostics only) |

The fast paths of `rmin`, `coordination` and `lj_residual` take an already built neighbour list,
because the monitor builds it once per frame and shares it. Add the `neighbor_list` row to get the
cost of one metric computed on its own. The lazy LJ evaluation in the monitor usually fits far
fewer atoms than the `lj_residual` row, which fits all of them.

## Output columns

CSV goes to `--output`, or to stdout. The first seven columns are PHAL's.

| Column | Meaning |
|---|---|
| `metric` | the kernel (see above) |
| `n_atoms` | atoms in the synthetic periodic cell |
| `direct_s` / `fast_s` | mean wall time per call in seconds, after one untimed warm-up call, over `--repeats` calls. Empty when the metric has no direct path. |
| `speedup` | `direct_s / fast_s`. Empty for single-path metrics. |
| `direct_peak_mb` / `fast_peak_mb` | peak Python allocation of one call in MiB, from `tracemalloc` in a separate pass |
| `direct_impl` / `fast_impl` | the function that was timed |
| `box_A` | edge of the cubic cell in Å |

`--json-output` writes:

- `{"threshold", "seed", "repeats", "quick", "density_per_A3", "cutoff_A", "environment", "results", "regressions"}`
- `results` holds the same rows, with `null` for the empty cells.

That file can be passed back as `--baseline`. A PHAL-style bare list of rows is accepted too.

## Regression gate

With `--baseline`, every row whose `metric:n_atoms` key appears in the baseline is compared on
`fast_s`. A row regresses when `fast_s > baseline_fast_s × (1 + threshold)`. The threshold defaults
to 0.2, so 20 % slower fails. Each regression is printed to stderr as `PERF REGRESSION: ...` and
stored in the JSON `regressions` list.

The exit status is 1 if anything regressed, unless `--warn-only` is set. Rows with no baseline entry,
or a baseline `fast_s` of 0, are not compared. The exit status is 2 when `gmd_active_learning` cannot
be imported.

## Differences from PHAL

| Item | PHAL | GMD_active |
|---|---|---|
| Timing and memory | Timed loop ran under `tracemalloc`, inflating the times. | Timing with no tracer, after a warm-up call. Peak memory is measured in a separate call. |
| `lj_residual` | "direct" was the nonlinear scipy fit and "fast" the linear projection. The speedup compared two algorithms. | Single-path row (linear, the production path). The nonlinear estimator is an optional row of its own. |
| Forces | All ones (non-physical). | LJ forces plus small noise, so the fit does representative work. |
| System | Uniform random positions in a fixed 40 Å cube. At 100 atoms almost no pair is within 6 Å. | Jittered lattice at constant density, so the neighbour count per atom is realistic at every size and no atoms overlap. |
| Neighbour-list cost | Hidden: the list was prebuilt outside the timed calls. | Reported as its own `neighbor_list` row. |
| Imports | `sys.path` injection of `src/`. | Imports the installed (or `PYTHONPATH`) package. |
| Extras | none | `--quick`, `--seed`, `--density`, `--include-nonlinear`, `direct_impl`/`fast_impl`/`box_A` columns, JSON environment block |

## Limitations

- The numbers depend on the machine, the BLAS library and the load. Treat them as regression data for
  one runtime, not as portable scientific benchmarks. Compare baselines only on the same machine.
- With small `--repeats`, timings of a few microseconds are noisy. Use `--repeats 5` or more before
  gating. `--quick` is a smoke test, not a baseline.
- The direct and fast paths do not compute identical quantities when the cell is smaller than twice the
  cutoff (the `--quick` boxes). The O(N²) references use one minimum image per pair, while the
  neighbour list counts every periodic image within the cutoff. The benchmark measures cost only.
- `tracemalloc` sees Python-level allocations, including NumPy buffers. Memory that native ASE or
  SciPy code allocates outside the Python allocator is not counted.
