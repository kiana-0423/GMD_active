# CP2K capability contract (`gmd-cp2k-capabilities-v1`)

`gmd_active_learning.adapters.dft.scf.capabilities.detect_cp2k_capabilities()` reports exactly
what one CP2K binary advertises. It never infers a feature it cannot see.

1. **Resolve the executable.** Resolution order:
   1. the argument;
   2. `scf.ml_scf.executable`;
   3. `scf.capabilities.executable`;
   4. `$CP2K_BIN`;
   5. `cp2k.psmp`, `.popt`, `.ssmp`, `.sopt` or `cp2k` on `PATH`.
2. **Probe the banner.** It runs `<cp2k> --version` with `os.environ` updated by `scf.ml_scf.environment`.
   - This is a deliberate fix of PHAL `cp2k_capabilities.py:246`, which ignored the configured environment. A deployment that set the toolchain `LD_LIBRARY_PATH` only in config got an empty banner, and every run silently degraded (cp2k-ml-scf RECONSTRUCTION.md §4f, finding F-22).
   - Exit 127, or "error while loading shared libraries", is reported as `loader_failure: true`. That is a dynamic-linker failure, not a missing executable.
3. **Parse the banner.** The parser reads the version, the git commit (the `git:` prefix is optional) and `cp2kflags`. Flags are matched as **exact tokens**; PHAL matched substrings.
4. **Report every feature.** Each feature in the table below gets `advertised` (banner evidence), `validated` (a real run on this machine) and `evidence`.

| Feature | Advertised when |
| --- | --- |
| `native_scf` | a version banner was parsed |
| `ao_matrix_printing` | a version banner was parsed (upstream CP2K feature) |
| `ml_density_guess` (mode A, `<input>.ml_density`, `phal-ml-scf-v2`) | `phal_ml_scf` in `cp2kflags` |
| `ml_inprocess_guess` (mode B, TorchScript) | never: no build token proves it, and `libtorch` alone does not (cp2k-ml-scf hook 0009 is a placeholder) |
| `ml_mixing` | `phal_ml_mixing` in `cp2kflags` |
| `scf_controller`, `pulay_recorder` | never: they existed only in the lost fork |

`validated` is set only by a real run. `run_native_validation()`, or the CLI flag
`scf capabilities --validate-native DIR`, runs the cp2k-ml-scf H2 control case with
`&AO_MATRICES`, pinned to `OMP_NUM_THREADS=1`:

- `native_scf` is validated from a converged SCF history;
- `ao_matrix_printing` is validated from a parsed density and overlap.

ML features cannot be validated without the modified binary.

## Recorded: native CP2K 2026.2 on the migration machine

`python -m gmd_active_learning.adapters.dft.scf.cli scf capabilities --executable /opt/homebrew/bin/cp2k.psmp --validate-native <dir> --json`
→ [capabilities_native_cp2k_2026.2.json](capabilities_native_cp2k_2026.2.json):

| Field | Value |
| --- | --- |
| version / commit | `2026.2` / `c92cc08` (upstream `REVISION` file is `git:c92cc08`, VERSION_LOCK.md) |
| cp2kflags | `omp libint fftw3 libxc parallel scalapack mpi_f08 no_statm_access` |
| `native_scf` | advertised, **validated**: converged in 10 steps, matching the cp2k-ml-scf H2 reference of 10 steps at −1.165933374533096 Ha |
| `ao_matrix_printing` | advertised, **validated**: Tr(P S) = 2.000000000 |
| `ml_density_guess`, `ml_inprocess_guess`, `ml_mixing`, `scf_controller`, `pulay_recorder` | **not supported** |

As expected, the native binary has no ML-SCF features. The ML wrapper therefore
degrades every run on it to `native_fallback`. That behaviour is exercised for real in
`tests/scf/test_scf_cp2k_native.py::test_the_ml_wrapper_degrades_to_a_native_run_on_a_native_binary`.

## Declared, not validated here: cp2k-ml-scf

The source is the reconstruction's own `README.md`, `VERSION_LOCK.md` and `RECONSTRUCTION.md`.
`capabilities.CP2K_ML_SCF_DECLARED` holds the same record in machine-readable form. Its
binaries are Linux/aarch64; they could not run on the migration machine, and Docker was not
running. **None of this was validated by GMD_active.**

| Item | Declared |
| --- | --- |
| Upstream pin | CP2K v2026.2 @ `67b5da876dd6a76b8b021d5a04d1c81ba79a4c50` |
| ML build `cp2kflags` | `omp libint fftw3 libxc elpa parallel scalapack mpi_f08 phal_ml_scf libtorch` → contract: `ml_density_guess` advertised; `ml_mixing` not; `ml_inprocess_guess` not |
| Native build `cp2kflags` | `omp libint fftw3 libxc elpa parallel scalapack mpi_f08` |
| Implemented | `SCF_GUESS ML` from `<input>.ml_density` (protocol `phal-ml-scf-v2`), deterministic native fallback, `ML-SCF\|` log lines |
| Reported results | H2 replay of its own converged density: 1 step at −1.165933374533587 Ha. 14 rejection paths each reproduce the atomic result. O2 triplet replay: first residual 1.45e-6 |
| Not implemented | mode B, ML mixing, controller, Pulay recorder |
| Environment read | `PHAL_ML_SCF_ENABLED`, `_DENSITY`, `_PROTOCOL`, `_FALLBACK`, `_GUESS_ALPHA`, `_STRUCTURE_DIGEST`, `_BASIS` |
| Run-time requirement | the toolchain `LD_LIBRARY_PATH` in the environment of both the probe and the run; the build RUNPATH is absolute (`/workspace/cp2k-ml-scf/...`) |

These declarations are pinned in tests from the recorded banners (`tests/scf/test_scf_ml_backend.py`,
`tests/scf/test_scf_capabilities_and_benchmark.py`). The tests show how GMD_active *would*
classify those binaries. They are not evidence that the binaries behave as declared.

No change to cp2k-ml-scf was needed. Its generator `tests/generate_exchange_vectors.py`
imports PHAL's writer (`hotspot_al.cp2k_ml_scf.density.exchange`). The ported writer
`gmd_active_learning.adapters.dft.scf.density.exchange` is byte-compatible. Re-pointing that import
is a one-line change on the cp2k-ml-scf side, to make once `ref/` is removed.
