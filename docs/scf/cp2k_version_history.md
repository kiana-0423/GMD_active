# CP2K version history (archive)

> **Archive page, ported verbatim from PHAL** (`docs/cp2k_migration_v2026_2.md` at 96263b6,
> MIT License, see NOTICE).
>
> **The authoritative pin is `cp2k-ml-scf/VERSION_LOCK.md`:** CP2K `v2026.2` @
> `67b5da876dd6a76b8b021d5a04d1c81ba79a4c50` (tag object `09496e055e132aa3dba53a7751ebdf432b4ebb78`,
> `REVISION` file `git:c92cc08`). GMD_active records the same strings in
> `capabilities.CP2K_ML_SCF_DECLARED`. `tests/scf/test_scf_version_pin.py` asserts that they agree
> with VERSION_LOCK.md whenever the sibling repository is present.
>
> Parts of the narrative below are **superseded**:
>
> - The fork it describes was lost and was never under version control. The current
>   `cp2k-ml-scf` is a clean-room reconstruction.
> - The reconstruction drops `-D__PHAL_ML_MIXING` and does not advertise
>   `phal_ml_mixing`. §10 below still shows both flags.
> - §8 and §8b are chronological: the ML build is "not run" and then "integrated".
> - §9 item 2, "density_gnn has not been trained or measured", predates the 0.85x measurement.
> - The run-time toolchain requirement (`LD_LIBRARY_PATH`) is documented only in the fork,
>   at VERSION_LOCK.md "Required runtime environment". GMD_active supplies it through
>   `scf.ml_scf.environment`, to both the probe and the run.
>
> Native CP2K 2026.2 as installed on the migration machine reports `git:c92cc08`, matching the
> pinned `REVISION` ([capabilities.md](capabilities.md)).

---

# CP2K migration: v2024.1 → v2026.2

Date: 2026-08-01

Scope: CP2K version migration, environment cleanup, and container update. No
ML-SCF scientific logic was modified and the CP2K integration was not rewritten.
What *did* change beyond a version string is called out in
[Consequences of the CMake-only build](#consequences-of-the-cmake-only-build)
and [Hook coordinate changes](#hook-coordinate-changes); both were forced by
upstream, not chosen.

---

## 1. Removed versions

### CP2K source trees

| Path | Version | Commit | Size | Action |
| --- | --- | --- | --- | --- |
| `../cp2k-ml-scf/cp2k/` | v2024.1 | `077bf32593c66d7a8606997006b97a99204d7b92` | 426 MB | deleted, re-cloned at v2026.2 |
| `../cp2k-ml-scf/cp2k_src/` | v2024.1 (duplicate of the above) | `077bf325…` | 472 MB | deleted, not replaced |
| `../cp2k-ml-scf/build/native-cmake/` | stale CMake tree (`/usr/bin/f95`, cmake 3.28.3) | — | 565 MB | deleted |
| `../cp2k-ml-scf/validation/build/*.o`, `*.mod` | Stage-15 stub objects | — | ~350 KB | deleted |
| `../cp2k-ml-scf/patches/__pycache__/` | — | — | — | deleted |

`cp2k_src/` was an unreferenced second checkout of the same tag carrying in-tree
`exe/`, `lib/`, `obj/` and `regtesting/` directories. Its only unique file,
`arch/phal-minimal.ssmp`, was verified byte-identical to
`validation/arch/phal-minimal.ssmp` before deletion.

**Total reclaimed: ~1.46 GB.**

### Binaries

None existed. `exe/phal-minimal/` and `lib/phal-minimal/ssmp/` were empty and no
`cp2k.psmp`, `cp2k.popt` or `cp2k.ssmp` was present on disk or on `PATH`.
Nothing was removed under this heading.

### Docker images

| Image | ID | Size | Action |
| --- | --- | --- | --- |
| `phal/cp2k-ml:cp2k-ml-development` | `aa0937228b66` | 2.06 GB | removed |

The image contained no CP2K, but it was built from the pre-migration Dockerfile
(no Libint/LibXC/ELPA, `libdbcsr-dev` from apt) and its tag has been superseded.
`versions.yaml` records the retired tags under `images.superseded_tags`.

### Environment variables

`CP2K_HOME`, `CP2K_ROOT` and `CP2K_DATA_DIR` were **not set** in the
environment. `CP2K_ROOT` exists only as a default-valued shell variable inside
the fork's `build.sh` and still points at the live checkout, so it was kept.

### Runtime artifacts

`runtime/cp2k/` and `runtime/cp2k_ml/` held 13 `.gitkeep` placeholders and one
`scf_dataset/metadata.json` describing an empty dataset (`n_samples: 0` for all
splits). No outputs, checkpoints, density artifacts or logs from the old CP2K
existed. **Nothing was deleted**; the directory structure required by Phase 8 was
already in place and is unchanged. No user dataset was touched.

---

## 2. New version

| Field | Value |
| --- | --- |
| `cp2k_version` | `v2026.2` |
| `cp2k_commit` | `67b5da876dd6a76b8b021d5a04d1c81ba79a4c50` |
| tag object | `09496e055e132aa3dba53a7751ebdf432b4ebb78` |
| support branch | `support/v2026.2` (head is the same commit) |
| `REVISION` | `git:c92cc08` |
| build system | CMake (upstream removed the Makefile) |
| binary | `cp2k.psmp` |

Recorded in `../cp2k-ml-scf/VERSION_LOCK.md` (authoritative) and mirrored into
`containers/versions.yaml` under `future_stacks.cp2k_ml`.

The external `cp2k-ml-scf/` directory is **not itself a git repository** — only
the nested `cp2k/` checkout is — so there is no fork commit to pin. This is
recorded as `fork.vcs: untracked` rather than left as a null, and is the one
part of the reproducibility story that is still weak.

---

## 3. Consequences of the CMake-only build

This is the substantive part of the migration; the version number was the easy
half.

**v2026.2 deleted the `Makefile` and the entire `arch/` directory.** The fork's
build path depended on both:

| Removed upstream | Was used for | Replacement |
| --- | --- | --- |
| `Makefile` | `make -C cp2k ARCH=local VERSION=psmp` | `cmake -S … -B … && cmake --build` |
| `arch/*.psmp` | appending `DFLAGS += -D__PHAL_ML_SCF` | `-DCMAKE_Fortran_FLAGS`, `-DCMAKE_CXX_FLAGS` |
| implicit `src/` compilation | picking up installed ML sources automatically | explicit `CP2K_SRCS_F` / `CP2K_SRCS_CPP` lists |
| `exe/<arch>/cp2k.psmp` | output location | `build/<target>/bin/cp2k.psmp` |

Three consequences:

1. **`../cp2k-ml-scf/build.sh` was rewritten** for CMake, with named targets
   `cp2k-native` and `cp2k-ml`, a `TOOLCHAIN_ENV` hook, a pinned-commit drift
   warning, and an explicit assertion that `cp2k.psmp` was produced. The
   two-binary control/ML design and the native-regression gate are unchanged.

2. **Two build-registration hooks were added** (`0003-cmake-register-fortran-sources`,
   `0003-cmake-register-cxx-source`). Without them the four ML sources installed
   into `src/` are copied and never compiled, because the CMake build uses
   explicit source lists rather than globbing. These hooks touch only
   `src/CMakeLists.txt`; no ML-SCF logic is involved. The C++ one appends via
   `list(APPEND …)` rather than editing upstream's `set(…)` line, so the
   upstream line stays byte-identical.

3. **`configs/cp2k.yaml` executable path updated** from
   `/workspace/cp2k-ml-scf/cp2k/exe/local/cp2k.psmp` (Makefile layout, no longer
   produced by anything) to `/workspace/cp2k-ml-scf/build/cp2k-ml/bin/cp2k.psmp`.

Also realigned to the new layout:

- `containers/scripts/validate_cp2k_ml.py` — build directory
  `build/native-cmake` → `build/cp2k-native` (matching `build.sh`'s target
  name), and `DEFAULT_IMAGE` to the new tag.
- `../cp2k-ml-scf/README.md`, `docs/ml_scf_user_guide.md` — clone commands,
  binary paths, and the arch-file `DFLAGS` snippet replaced with the CMake
  equivalents.
- `../cp2k-ml-scf/.gitignore` — ignores the two new build trees.

`src/hotspot_al/config/default.yaml` still defaults the *standard* backend to
`cp2k.popt`. That remains valid: v2026.2's CMake install creates a
`popt → psmp` symlink (`CMakeLists.txt:1401-1402`). Left unchanged deliberately.

### Pre-existing test breakage, fixed in passing

`../cp2k-ml-scf/tests/test_apply_hooks.py` was failing 4 of 47 tests **before**
this migration. Its synthetic CP2K fixtures still encoded the hooks-v1 shape —
`SELECT CASE (density_guess)` and `CASE (broyden_mix)` — which `hooks.json`'s own
v2 note describes as structures that "do not exist" in real CP2K. The fixtures
were rewritten to the real IF-cascade / `base_uses.f90` shape verified against
v2026.2, the marker IDs updated from the v1 numbering (`0002`) to v2 (`0003`,
`0004`), and coverage added for the two new CMake registration hooks. 47/47 now
pass. No applier logic and no ML-SCF source changed.

---

## 4. Hook coordinate changes

`patches/apply_hooks.py --cp2k-root cp2k --check` against v2026.2:

| Hook | v2024.1 | v2026.2 | Status |
| --- | --- | --- | --- |
| `0001-scf-guess-ml-use` | `qs_initial_guess.F` | `qs_initial_guess.F:102` | would-apply |
| `0001-scf-guess-ml-branch` | `qs_initial_guess.F` | `qs_initial_guess.F:652` | would-apply |
| `0002-mixing-method-ml-use` | `qs_scf_loop_utils.F` | `qs_scf_loop_utils.F:68` | would-apply |
| `0003-cmake-register-fortran-sources` | n/a | `src/CMakeLists.txt:618` | would-apply (new) |
| `0003-cmake-register-cxx-source` | n/a | `src/CMakeLists.txt:1505` | would-apply (new) |
| `0001-scf-guess-ml-constant` | `input_constants.F:~201` | `input_constants.F:208` | manual (was manual) |
| `0001-scf-guess-ml-enum` | `input_cp2k_dft.F:~5136` | **`input_cp2k_scf.F:207`** | manual (was manual) |
| `0002-mixing-method-ml-constant` | `qs_density_mixing_types.F:~46`, `~500` | `:50-55`, `:543` | manual (was manual) |
| `0002-mixing-method-ml-branch` | `qs_scf_loop_utils.F:~466`, `~588` | `:514`, `:641` | manual (was manual) |

**All four automatic anchors survived the two-year jump.** No hook broke. The
four manual hooks were manual before the migration too — that is not a
regression the version bump caused.

Three coordinate facts in `hooks.json` were stale and would have misled whoever
performs the manual edits. They were corrected as *metadata*; no insertion
payload and no ML-SCF logic was changed:

- The `SCF_GUESS` keyword **moved file**, from `input_cp2k_dft.F` to
  `input_cp2k_scf.F`. The old note asserted that `input_cp2k_scf.F` "does not
  exist" — now inverted.
- **`ml_mixing_nr = 7` would now collide**: `new_pulay_mixing_nr = 7` was added
  upstream. The note now specifies `8`. Following the old note verbatim would
  have produced a build in which `MIXING_METHOD ML` silently selected the new
  Pulay variant — a wrong result, not a build failure.
- All line numbers re-measured against the pinned commit.

`verified_against` is now `cp2k v2026.2`.

---

## 5. Docker changes

`containers/Dockerfile.cp2k-ml`, rewritten as three stages:

| Stage | Purpose | Ships? |
| --- | --- | --- |
| `toolchain` | build Libint, LibXC, ELPA, FFTW, OpenBLAS, ScaLAPACK | no, discarded |
| `development` | the image PHAL uses | yes |
| `probe` | toolchain self-check | yes |

Added, per the Phase 4 requirement:

| Requirement | Provision |
| --- | --- |
| Base Ubuntu 24.04 | pinned by digest `sha256:4fbb8e6a…` |
| gcc, gfortran | apt `4:13.2.0-7ubuntu1` |
| cmake, ninja, make | apt |
| OpenMPI | apt `4.1.6-7ubuntu2` |
| BLAS, LAPACK, FFTW | apt headers + OpenBLAS/FFTW from the toolchain |
| **Libint, LibXC, ELPA** | **new** — CP2K toolchain script |
| Python, PyTorch, LibTorch | PyTorch 2.5.1 wheel, `/opt/libtorch` symlink |
| git, wget, curl, vim | apt |

Removed: `libdbcsr-dev`. DBCSR ships inside the CP2K tree at v2026.2, so the apt
package is both redundant and a version-skew risk. No CP2K package is installed
from apt.

### The no-clone rule, and its one exception

Docker must not clone CP2K, and it does not. Libint, LibXC and ELPA have no
usable Ubuntu 24.04 packages at the versions v2026.2 expects, so they are built
by CP2K's own toolchain script — which lives inside the CP2K tree. The
`toolchain` stage therefore extracts **only** `tools/toolchain/` from the pinned
commit tarball (`tar --strip-components=3`), asserts no `src/` landed, builds the
dependency prefix, and is then discarded. The `development` stage copies only
the resulting `install/` directory.

Two build-time assertions enforce this, and fail the image rather than warn:

```dockerfile
test ! -d /opt/cp2k-toolchain-src/src
! find / -xdev -name "qs_scf.F" -print -quit | grep -q .
```

**This is a deliberate deviation from the letter of "Docker must not clone
CP2K", taken because the chosen dependency strategy requires it.** It preserves
the intent — no CP2K source in any shipped layer, source stays external, the
image compiles nothing — but it is a deviation and should be read as one. The
alternative (apt Libint/LibXC, no ELPA) avoids the fetch entirely at the cost of
older library versions.

### Why `--install-all` is not used

The first working toolchain invocation used `--install-all` plus
`--with-libtorch=no`. That silently does not work, and the failure mode is worth
recording because it is invisible in the flag list:

`install_cp2k_toolchain.sh` lines 1215-1221 force `with_libtorch=__INSTALL__`
whenever DeePMD or GauXC is enabled, and `--install-all` enables both. The
explicit `--with-libtorch=no` is overridden with no warning. The result is a
second LibTorch inside the toolchain prefix, copied into the final image
alongside the PyTorch wheel's — precisely the Python/C++ runtime divergence
`VERSION_LOCK.md` states cannot occur. `--install-all` also failed to link
DeePMD on arm64, which is how the problem surfaced at all.

Packages are therefore selected explicitly, and a post-condition asserts the
prefix contains no `libtorch*` directory:

```dockerfile
&& ! ls -d /opt/cp2k-toolchain-src/install/libtorch* >/dev/null 2>&1
```

`--with-openmpi=system` is load-bearing beyond reuse: it sets `MPI_MODE=openmpi`
(line 813-817), which is what makes ScaLAPACK and ELPA link against the pinned
OpenMPI. Without it the script defaults to MPICH.

### Build metadata

`CP2K_VERSION` and `CP2K_COMMIT` are now build args resolved from
`versions.yaml`, surfaced as `org.phal.cp2k-version` / `org.phal.cp2k-commit`
labels and as `CP2K_VERSION_EXPECTED` / `CP2K_COMMIT_EXPECTED` in the image
environment. The commit appears nowhere as a literal in the Dockerfile, so
`versions.yaml` stays the single source of truth — asserted by a test.

---

## 6. Dependency changes

| Dependency | Before | After |
| --- | --- | --- |
| MPI | OpenMPI 4.1.6 (apt) | unchanged, version pinned |
| BLAS / LAPACK | `libblas-dev`, `liblapack-dev` | OpenBLAS from toolchain |
| ScaLAPACK | apt | toolchain |
| FFTW | apt | toolchain |
| **Libint** | **absent** | **toolchain** |
| **LibXC** | **absent** | **toolchain** |
| **ELPA** | **absent** | **toolchain** |
| DBCSR | `libdbcsr-dev` (apt) | bundled in the CP2K tree |
| COSMA | absent | still absent (explicitly `--with-cosma=no`) |
| CUDA | absent | still absent (`CP2K_USE_ACCEL=NONE`) |
| LibTorch | PyTorch 2.5.1 wheel | unchanged |

Libint, LibXC and ELPA are **additions**, not replacements: the pre-migration
environment could not build hybrid functionals or use ELPA at all.

---

## 7. Version tracking

`containers/versions.yaml` → `future_stacks.cp2k_ml` previously had
`upstream_version: null` and `fork_commit: null` — it never recorded v2024.1 at
all, which is how an unpinned build survived this long. Now:

- `upstream.{version,commit,tag_object,support_branch,revision_string,build_system}`
- `superseded_upstream.{version,commit,build_system,removed_on}` — so a stale
  artifact can still be identified by what it was built from
- `compiler.*`, `mpi.*`, `build_tools.*` as exact apt package versions
- `math_libraries.*` naming the provider per library
- `ml_backend.{pytorch,libtorch}: enabled`
- `build_targets.{native,ml,binary}`
- `images.superseded_tags`
- `base_image_digest`

**No null remains anywhere in the stack** — enforced by a recursive test rather
than by review.

---

## 8. Validation results

| Check | Result |
| --- | --- |
| `hooks.json` valid JSON | pass |
| `apply_hooks.py --check` vs v2026.2 | pass — 4 would-apply, 4 manual, 0 missing/ambiguous |
| `build.sh` (fork) syntax + guards | pass |
| `containers/build.sh plan cp2k-ml` | pass |
| `containers/build.sh plan cp2k-ml-probe` | pass |
| `containers/build.sh plan` for base/md/train/train-probe/md-probe | pass — no regression |
| `build.sh` rejects unknown target, floating tag, missing registry, missing OCI URI | pass |
| `pytest tests/ --ignore=tests/integration` (PHAL) | 587 passed, 3 skipped (torch absent on host) |
| `pytest tests/` (fork) | 47 passed |
| `docker build cp2k-ml` | **pass** — `phal/cp2k-ml:cp2k-ml-v2026-2-development`, 686 MB content |
| no CP2K source in image | **pass** (build-time assertion) |
| no second LibTorch in toolchain prefix | **pass** (build-time assertion) |
| toolchain `setup` sources cleanly | **pass** (build-time assertion) |
| CP2K v2026.2 compiles | **pass** — 4153/4153, `build/cp2k-native/bin/cp2k.psmp` |
| `cp2k --version` shows v2026.2 | **pass** — see banner below |
| Native SCF: H2 | **pass** — converged in 10 steps, E = -1.16593337453310 Ha |
| Native SCF: H2O | **pass** — converged in 13 steps, E = -17.22096054599224 Ha |
| ML-SCF: TorchScript load | **pass** — `guess.ts.pt`, `mixer.ts.pt` load under torch 2.5.1 |
| ML-SCF: interface init | **pass** — see below |
| `cp2k-ml` (ML-enabled) build | **not run** — blocked on the manual hooks |

### The binary

```
 CP2K version 2026.2
 Source code revision 67b5da8
 cp2kflags: omp libint fftw3 libxc elpa parallel scalapack
 compiler: GCC version 13.3.0
```

The revision matches the pinned commit, and `libint`, `libxc` and `elpa` are
present in `cp2kflags` — the three dependencies that did not exist before this
migration are live in the binary, not merely installed.

CMake's configure summary places LibXC, FFTW3, Libint2 and ELPA under
*enabled* with real toolchain paths, and LibTorch under "Dependencies not
included in this build", which is correct for the native control.

### ML interface initialisation

`parse_version_banner` was run against the real binary's banner:

```json
{"version": "2026.2", "git_commit": "67b5da8", "ml_scf_supported": false,
 "ml_mixing_supported": false, "torch_supported": false}
```

All three `false` values are the correct answer for the native control build,
which is compiled without `-D__PHAL_ML_SCF`. PHAL correctly declines to treat it
as ML-capable.

**One migration-caused regression was found and fixed here.** v2024.1 printed
`Source code revision git:b4a17a5`; v2026.2 prints `Source code revision
67b5da8` — upstream dropped the `git:` prefix. `_GIT_COMMIT` in
`cp2k_ml_scf/cp2k_capabilities.py` required that prefix, so after the migration
every recorded provenance would have carried `git_commit: null` while looking
otherwise healthy. The prefix is now optional; both banner formats parse, and
`test_revision_is_read_with_and_without_the_git_prefix` pins it. This is a
one-regex version-compatibility fix, not a change to any ML-SCF behaviour.

### Still not run

The **ML-enabled** build (`./build.sh --ml-only`) has not been performed. It
cannot succeed yet: the four manual hooks are unapplied, so `scf_guess_ml` and
`ml_mixing_nr` are undeclared and the patched sources would not compile. That is
pre-existing, not caused by the migration — but it means the two ML keywords are
still unproven against v2026.2. Everything the ML path depends on *underneath*
(the toolchain, LibTorch, TorchScript loading, the capability contract, and a
converging native SCF) is now verified.

`versions.yaml` therefore keeps `validation.status: not_validated`:
`external_source_build` and `toolchain_probe` are now passing, but
`ml_interface` and `scientific_comparison` are not, and the status reflects the
weakest link deliberately.

---

## 8b. ML-SCF hook integration (2026-08-01)

`SCF_GUESS ML` is integrated into CP2K v2026.2 and validated by real execution.
`MIXING_METHOD ML` is **not** integrated; see §9.

### What was applied

All four previously-manual hooks are now automatic, so the integration is
reproducible from `hooks.json` rather than from a written procedure. Two
applier capabilities were added to make that possible:

- `position: "replace"` — the constant lists and the three `SCF_GUESS`
  enumeration lists are continued Fortran statements whose last element carries
  no `&`. They cannot be extended by insertion, only by replacing a line. The
  replacement must match the whole line, or the rest of it would be silently
  discarded.
- token-wise marker matching — `marker in text` matched
  `PHAL-ML-SCF-HOOK-0003` against a *different* hook's marker
  `PHAL-ML-SCF-HOOK-0003B`, so the USE-statement hook reported
  `already-applied` and never inserted anything. The build then failed on an
  unknown symbol. A skipped hook that reports success is precisely the failure
  this applier exists to prevent; matching is now token-wise and two tests pin
  it.

| Hook | File | What |
| --- | --- | --- |
| 0001 | `input_constants.F` | `scf_guess_ml = 12` |
| 0002, 0002B, 0002C | `input_cp2k_scf.F` | `ML` in the `SCF_GUESS` c-vals / desc / i-vals |
| 0002D, 0003B | `input_cp2k_scf.F`, `qs_initial_guess.F` | import the new constant |
| 0003, 0003C | `qs_initial_guess.F` | ML `USE` statements, local declarations |
| 0004 | `qs_initial_guess.F` | try the prediction, then fall into the atomic branch |
| 0011 | `qs_initial_guess.F` | install the accepted density, enforce `Tr(P S) = N` |
| 0012 | `cp2k_info.F` | advertise `phal_ml_scf` / `phal_ml_mixing` |
| 0008, 0009 | `src/CMakeLists.txt` | register the ML sources |

### Design of the guess hook

The ML branch does **not** set `did_guess` and does not replace the atomic
branch. It requests a prediction and then always falls into the untouched
atomic branch, which is what reserves the DBCSR blocks and, for OT runs, builds
the initial MO set. Hook 0011 then overwrites the block values in place. On
rejection nothing is overwritten, so the fallback path is upstream's own code
rather than a copy of it — which is why every rejection below reproduces the
atomic energy exactly.

### Two things the migration broke that only execution revealed

1. **`-D__PHAL_ML_SCF` advertises nothing on its own.** `cp2k_info.F` builds
   the `cp2kflags` string from an explicit `#if defined()` list, so an unknown
   macro is invisible there. `phal cp2k-scf capabilities` reads exactly that
   line, so a fully working ML build reported itself ML-incapable until hook
   0012 was added. The fork README's claim that the `DFLAGS` alone are
   sufficient was true for the Makefile era and is not true now.

2. **The fork's own native-regression gate could not parse v2026.2 output.**
   v2026.2 prints `ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]  -32.09…`;
   earlier releases printed `… energy (a.u.):  -32.09…`. The harness regex
   required the colon, so the gate raised “no total energy found” on every run
   rather than comparing anything. Fixed to accept both label forms.

### Electron count: `Tr(P)` vs `Tr(P S)` — resolved in protocol v2

**Superseded.** The mismatch described below was fixed on 2026-08-01 by moving
the exchange contract to `phal-ml-scf-v2`, which carries the overlap matrix and
evaluates the electron count as `Tr(P S)`. Full treatment in
[`density_representation.md`](density_representation.md). The paragraph below is
kept because it records how the problem presented before it was understood.



The validation gate `ml_validate_density` tests the **trace** of the proposed
matrix against the electron count the caller passes. In a non-orthogonal AO
basis the electron count is `Tr(P S)`, not `Tr(P)`. Measured on the converged
densities used for the tests:

| System | `Tr(P)` | `Tr(P S)` | ratio |
| --- | --- | --- | --- |
| H2 | 0.9705 | 2.0000 | 0.485 |
| H2O | 6.3990 | 8.0000 | 0.800 |

A converged CP2K density therefore failed that gate outright unless the producer
normalised it to `Tr(P) = N` first — which produced a density carrying the wrong
electron count, accepted only because the test measured the wrong quantity.

Protocol v2 now carries `S` and evaluates `Tr(P S)`, so the raw converged
densities above are accepted unmodified. Hook 0011 still re-imposes the count
inside CP2K with upstream's own idiom (`dbcsr_dot(P, S)` then `dbcsr_scale`), as
a second line of defence against a prediction that is merely mis-scaled.

### Validation — real CP2K execution

Binary: `build/cp2k-ml/bin/cp2k.psmp`, CP2K 2026.2, revision `67b5da8`,
`cp2kflags: omp libint fftw3 libxc elpa parallel scalapack phal_ml_scf
phal_ml_mixing libtorch`.

ML symbols are in the build: 14 in `libcp2k.so.2026.2`, including
`__ml_scf_interface_MOD_*`, `phal_ml_initial_guess` and `phal_ml_mix`. (They are
not in `cp2k.psmp` itself, which is a thin driver — neither are
`qs_initial_guess`'s.)

The test feeds each system its own converged density back as the initial guess,
using PHAL's AO-matrix parser and the fork's own exchange writer:

| Run | Steps | Total energy (Ha) |
| --- | --- | --- |
| H2, `SCF_GUESS ATOMIC` | 10 | −1.16593337453310 |
| H2, `SCF_GUESS ML` | **1** | −1.16593337453359 |
| H2O, `SCF_GUESS ATOMIC` | 13 | −17.22096054599225 |
| H2O, `SCF_GUESS ML` | **1** | −17.22096054599407 |

Energies agree to 5×10⁻¹³ and 1.8×10⁻¹² Ha — within SCF convergence noise.

Every rejection path was exercised, and each one reproduces the atomic result:

| Condition | Logged reason | Steps | Energy |
| --- | --- | --- | --- |
| `PHAL_ML_SCF_ENABLED` unset | not enabled for this run | 10 | −1.16593337453310 |
| density file absent | no ML density file was found | 10 | −1.16593337453310 |
| wrong `DIM` | wrong shape for this system | 10 | −1.16593337453310 |
| NaN injected | contains non-finite values | 10 | −1.16593337453309 |
| broken symmetry | matrix is not Hermitian | 10 | −1.16593337453310 |

`tests/test_native_regression.py`, run with both binaries:
**`PASS: native CP2K behaviour is unchanged by the ML-SCF patch`** — identical
energies, forces and SCF step counts for an input using no ML keyword.

Suites: fork 49 passed, PHAL 587 passed / 3 skipped.

## 9. Remaining work

1. **`MIXING_METHOD ML` is not integrated.** `SCF_GUESS ML` is done; the mixing
   keyword is not. What it needs, and the one decision that is not a mechanical
   edit:

   - `ml_mixing_nr = 8` (not 7 — upstream took 7 for `new_pulay_mixing_nr`, and
     reusing it would silently select the new Pulay variant rather than error),
     plus `ml_mix` in `input_constants.F` and the `METHOD` enumeration in
     `qs_density_mixing_types.F`.
   - `qs_mixing_utils.F` must allocate Pulay's buffers for the new method:
     `mixing_method >= gspace_mixing_nr` and `>= pulay_mixing_nr` already admit
     8, but two `== pulay_mixing_nr .OR. == new_pulay_mixing_nr` conditions do
     not. Those two lines are **byte-identical**, so the applier refuses them as
     ambiguous; it needs an explicit `occurrence` selector before they can be
     hooked reproducibly.
   - `qs_scf_loop_utils.F`: add the constant to both `SELECT CASE` lists.
   - `qs_gspace_mixing.F`: a dispatch branch calling `ml_mixer_propose`, falling
     through to `pulay_mixing` when it declines. `mix_rho` is
     `mixing_method >= gspace_mixing_nr`, so 8 already routes here — which is
     what makes Pulay the literal fallback the design asks for.
   - Mixer state must persist across SCF iterations. `ml_mixer_type` is not held
     anywhere in `qs_scf_env_type`; a module-level `SAVE` in the CP2K glue is
     the least invasive home.

   **The open decision:** `ml_mixer_propose` gates on
   `SUM(proposal) ≈ expected_electrons`. In CP2K's g-space density that sum is
   not the electron count (the G=0 coefficient is), and in the AO density matrix
   it is not either (`Tr(P S)` is) — the same mismatch documented in §8b. The
   gate is caller-parameterised, so the honest reading is "preserve the current
   iterate's aggregate", but that is a semantic choice about what the mixer
   guarantees, and changing `ml_mixer_propose` is out of scope by instruction.
   This should be settled before the branch is written, not after.

   Note also that with no trained mixer, `phal_ml_mix` returns non-zero and the
   only reachable path in a real run is decline-then-Pulay. Integration can be
   proven; acceleration cannot, until a mixer model exists.

2. **Benchmark a real model.** Every ML number above comes from replaying a
   converged density, which proves the mechanism, not the science. The
   `density_gnn` predictor has not been trained or measured against the atomic
   guess.
3. **`../cp2k-ml-scf/` is not under version control.** `VERSION_LOCK.md` pins
   CP2K but nothing pins the fork's own sources. `git init` there would close
   the last reproducibility gap.
4. **Test fixtures still print `CP2K version 2024.1`** in
   `tests/unit/test_scf_capabilities_and_benchmark.py`,
   `test_scf_ml_regression.py`, `test_scf_production.py` and
   `tests/e2e/test_ml_scf_pipeline.py`. These are parser inputs — they assert
   that the version parser extracts whatever CP2K printed, not which version is
   installed — so they are correct as written and were left alone under the
   "do not modify ML-SCF logic" constraint. Refreshing the strings is cosmetic.
5. **Reconsider the toolchain fetch** if the Phase 4 no-clone rule is meant
   strictly; see §5.

---

## 10. Resulting architecture

```text
PHAL
 │
 ├── Standard CP2K backend      src/hotspot_al/backends/cp2k.py
 │      executable: cp2k.popt   (v2026.2 symlinks popt → psmp)
 │
 └── CP2K-ML backend            src/hotspot_al/backends/cp2k_ml_scf.py
        executable: /workspace/cp2k-ml-scf/build/cp2k-ml/bin/cp2k.psmp
        │
        └── CP2K v2026.2 @ 67b5da87…  +  ML-SCF hooks
                 SCF_GUESS ML       integrated, validated
                 MIXING_METHOD ML   not integrated (§9)
```

Both binaries exist and are the control/treatment pair the fork is built around:

```text
build/cp2k-native/bin/cp2k.psmp   unpatched control
build/cp2k-ml/bin/cp2k.psmp       -D__PHAL_ML_SCF -D__PHAL_ML_MIXING
                                  proven identical on ML-free input
```

No legacy CP2K installation remains: no v2024.1 source tree, no v2024.1 build
artifact, no image built from the pre-migration Dockerfile, and no environment
variable referencing any of them.
