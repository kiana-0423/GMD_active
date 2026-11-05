> **Ported from PHAL** (`docs/density_representation.md` at 96263b6, MIT License, see NOTICE). The body below is
> kept **verbatim**, including every measured number. Those measurements were taken by
> PHAL on the lost `cp2k-ml-scf` fork, with models and datasets that no longer exist;
> they cannot be re-run here as written. The rules it states are enforced by the ported validators; see [contracts.md](contracts.md).
> Code paths named below (`hotspot_al.*`, `scripts/*`) map to GMD_active as listed in
> [README.md](README.md#module-map). Status of each feature: [README.md](README.md).

# Density representation and the electron count

Every density matrix in this project lives in CP2K's **atomic-orbital basis**,
and that basis is **not orthonormal**. One consequence governs the whole ML-SCF
contract:

> The number of electrons in an AO density matrix is `Tr(P S)`, not `Tr(P)`.

`P` is the density matrix, `S` the AO overlap. This document exists because the
validation gate originally used `Tr(P)`, and that was wrong in a way that looked
like a model-quality problem rather than a units problem.

---

## 1. Why `Tr(P)` is not the electron count

Basis functions centred on different atoms overlap. Writing `|i⟩` for AO `i`:

```
S_ij = ⟨i|j⟩        S_ii = 1 (normalised),  S_ij ≠ 0 for i ≠ j
Ne   = Tr(P S) = Σ_ij P_ij S_ij
Tr(P)          = Σ_i  P_ii
```

`Tr(P)` counts only the diagonal, discarding every bond-population term
`P_ij S_ij` with `i ≠ j`. Those terms are not a correction — in a molecule they
are a large fraction of the total, because that is what a chemical bond *is*.

`Tr(P) = Ne` holds only when `S = I`, i.e. an orthonormal basis. No CP2K
Gaussian basis set is orthonormal.

## 2. Measured, not asserted

From converged CP2K v2026.2 runs (DZVP-MOLOPT-GTH, PBE), recorded by PHAL's own
AO-matrix parser:

| System | `n_ao` | `Ne` | `Tr(P)` | `Tr(P S)` | `Tr(P)/Ne` |
| --- | --- | --- | --- | --- | --- |
| H2 | 10 | 2 | 0.970451 | 2.000000 | 0.485 |
| H2O | 23 | 8 | 6.399031 | 8.000000 | 0.800 |

`Tr(P S)` reproduces the electron count exactly. `Tr(P)` is low by 51.5% and
20.0%. The error is not small, not systematic in a correctable way, and grows
with bonding density — H2, which is all bond, is the worse case.

These two systems are pinned as fixtures in
`../cp2k-ml-scf/tests/fixtures/{h2,h2o}_converged.npz` and asserted in
`tests/test_ml_scf_exchange.py`.

## 3. What went wrong

`ml_validate_density` compared `Tr(P)` against the electron count the caller
passed, with a 0.1% tolerance. The consequences were both directions of wrong:

- **A correct density was rejected.** A converged CP2K density fails a
  `Tr(P) ≈ Ne` test by 20-50%, so it was refused as "does not integrate to the
  requested electron count" and the run fell back to the atomic guess. The
  fallback is silent by design, so this reads as "the model isn't helping".
- **An incorrect density was accepted.** To get through the gate, a producer had
  to scale `P` by `Ne/Tr(P)` — multiplying H2's density by 2.06. That density
  carries the wrong number of electrons; it passed only because the test was
  measuring the wrong quantity.

The same assumption appeared independently in the TorchScript export wrapper,
which normalised against `torch.eye(n_ao)` — an identity overlap. It produced
models normalised to `Tr(P) = Ne`, which is the same error at the source.

## 4. The contract now

**Exchange protocol `phal-ml-scf-v2`.** A density-matrix file carries the
overlap:

```
PHAL-ML-SCF phal-ml-scf-v2
FORMAT         phal-density-matrix-v1
REPRESENTATION ao-nonorthogonal
STRUCTURE      sha256:...
MODEL          sha256:...
ELECTRONS      8.0
NSPIN          1
DIM            23
BASIS          DZVP-MOLOPT-GTH
OVERLAP        present
VALUES
<dim*dim*nspin reals, column-major>
OVERLAP_VALUES
<dim*dim reals, column-major>
```

**Validation.** `ml_validate_density` (Fortran) and `validate_density` (the
Python mirror) both:

1. check shape and spin channels,
2. check every element is finite — before any arithmetic, since comparisons
   against NaN are all false,
3. check `P = P†` to 1e-8,
4. compute `Ne = Σ_ij P_ij S_ij` and compare to the expected count at 0.1%.

A density matrix that carries **no** overlap is rejected with `no_overlap`. It is
not checked with `Tr(P)` instead: an unverifiable count is refused, not
approximated. Atomic-population files are unaffected — there the count is the
plain sum, and no overlap is involved.

**Torch models.** `export_density_guess` scripts a wrapper whose signature takes
the overlap as its sixth argument, normalises with `Tr(P S)`, and symmetrises
the output. Export runs a probe with a deliberately **non-orthonormal** `S`
(unit diagonal, 0.25 off-diagonal) and refuses to write a model that is not
Hermitian, not finite, or normalised to anything other than `Tr(P S) = Ne`. An
identity probe would let the old bug through, which is why it is not used.

**Recorded artifacts.** `density_metadata.json` states the representation
explicitly, so no consumer has to infer it:

| Field | Meaning |
| --- | --- |
| `representation` | `ao-nonorthogonal` |
| `has_overlap` | whether `overlap_matrix.npy` was written |
| `electron_count_method` | `trace_ps`, or `trace_p` when CP2K printed no overlap |
| `electron_count` | the value that method produced |
| `basis_set` | the basis the matrix is expressed in |
| `spin_channels` | 1 restricted, 2 spin-polarised |
| `n_ao` | matrix dimension |
| `matrix_format` | `ao-density-matrix-spin-major-v1` |

`electron_count_method` is the field that matters: a dataset entry reading
`trace_p` is carrying an approximation that is 20-50% low, and must not be
trained against or validated with as though it were exact.

## 5. Enabling the overlap in CP2K

CP2K prints `S` only when asked. PHAL's input writer emits this whenever density
recording is on:

```
&PRINT
  &AO_MATRICES
    DENSITY .TRUE.
    OVERLAP .TRUE.
    NDIGITS 12
  &END AO_MATRICES
&END PRINT
```

Without `OVERLAP .TRUE.` the recorder falls back to `trace_p` and says so. That
is a degraded artifact, not a usable one, for anything that needs the count.

## 6. Inside CP2K

CP2K itself has always been right about this; only the ML layer was not. The
`SCF_GUESS ML` install hook enforces the count using CP2K's own idiom, the same
one the sparse guess uses:

```fortran
CALL dbcsr_dot(p_rmpv(ispin)%matrix, s_sparse(1)%matrix, trps1)
rscale = REAL(nelectron, dp)/trps1
CALL dbcsr_scale(p_rmpv(ispin)%matrix, rscale)
```

So even a prediction that slips through the gate is renormalised to the exact
electron count before the SCF sees it. The gate's job is to catch a density that
is wrong in ways rescaling cannot fix — wrong shape, wrong symmetry, not finite.

## 7. SCF trajectories

A neural mixer learns the map from (density history, residual history) to the
next density, so it needs the *path* the SCF took, not just its endpoint.
`record_trajectory` keeps every iteration:

```
runtime/cp2k/scf/<task>/trajectory/
├── overlap.npy      the AO overlap S
├── metadata.json    n_frames, n_ao, spin_channels, n_electrons, frames[]
├── iter_0000/       density.npy + metadata.json (iteration, residual, energy)
├── iter_0001/
├── ...
└── final/           the converged density
```

Three switches gate it — `enabled`, `record_density`, `record_trajectory` —
because it is the most expensive artifact the pipeline produces.

**The overlap is stored once.** `S` depends only on geometry and basis set, both
fixed for the whole SCF, so the frames share one file instead of `N` identical
copies of an `nao × nao` matrix. Each frame's metadata names it, so a frame
stays self-describing.

**`Tr(P S) = Ne` holds at every iteration**, not only at convergence — measured
to ~1e-12 across all frames of both fixtures. That is what makes the electron
count usable as a per-frame gate: a frame that fails it is corrupt, not merely
unconverged. `Tr(P)` meanwhile *drifts* along the trajectory (H2: 1.463 → 0.970,
H2O: 7.454 → 6.399), so it is not even wrong by a fixed factor that a consumer
could correct for.

| System | frames | `Tr(P)` first → last | `Tr(P S)` every frame |
| --- | --- | --- | --- |
| H2 | 11 (10 iterations + final) | 1.463 → 0.970 | 2.000000000 |
| H2O | 14 (13 iterations + final) | 7.454 → 6.399 | 8.000000000 |

Ordering is taken from the recorded frame index, never from directory listing
order — `final` sorts *before* `iter_0000` alphabetically, so a loader that
trusted the filesystem would put the converged density first.

## 7b. `Ne` is not `sum(Z)` either

`Tr(P S)` gives you *an* electron count. Which one it has to equal is the second
half of the contract, and it was wrong for as long as the first half was right.

Every calculation here uses GTH pseudopotentials. The density CP2K works with
holds **valence** electrons only:

| system | `sum(Z)` | `Ne` | error if you use `sum(Z)` |
| --- | --- | --- | --- |
| H₂O | 10 | 8 | +25% |
| CO₂ | 22 | 16 | +38% |
| Si (2 atoms) | 28 | 8 | +250% |
| Cu₂O | 66 | 28 | +136% |

CP2K accepts a 10⁻³ relative deviation, so a model normalised to the nuclear
charge is rejected on every system. That is the correct behaviour and an
almost undebuggable symptom: the run converges normally, from the atomic guess,
and reports the ML guess as failed.

### Where the counts come from

They are **fitted from the data**, not transcribed. Every recorded density
states the count it integrates to, so `ValenceTable.fit` solves for the
per-element valences and verifies that they reproduce every sample. On the
27-system dataset the fit returns H=1, C=4, N=5, O=6, F=7, Al=3, Si=4, Li=3,
Ti=12, Ni=18, Cu=11, Zn=12 — every one matching the `-qN` suffix of the
potential CP2K shipped, which is what
`test_the_fitted_table_matches_cp2ks_own_potential_file` asserts.

An underdetermined dataset is refused rather than guessed. H₂O alone admits
both (H=1, O=6) and (H=3, O=2); both integrate to 8, and only one is right for
anything else.

### Where it is enforced

* **Export** refuses a checkpoint with no valence table, and refuses one whose
  counts all equal `Z`. The probe composition is chosen to maximise the
  `Z - valence` gap, because a probe of hydrogen alone would certify the bug —
  H has one valence electron and `Z = 1`.
* **`verify_deployed`** re-checks a deployed model against water and reports
  "it normalises to sum(Z) = 10.000" by name when it does.
* **CP2K** rescales an in-process prediction to the expected count, but only
  within ±2%: enough for float noise, not enough to absorb a nuclear-charge
  normalisation. Beyond that it rejects and logs both numbers.

The middle option — silently rescaling whatever arrives — is the one to avoid.
It turns a model trained against the wrong quantity into a density that merely
integrates correctly.

## 8. Rule of thumb

Any time you write `trace`, `sum`, or `np.trace` on a density matrix and mean
"how many electrons is this", you need `S`. If you do not have `S`, you do not
have the electron count, and the honest answer is to say so rather than report
the diagonal sum.

And any time you write `sum(atomic_numbers)` and mean "how many electrons
should this be", you need the pseudopotential's valence counts. `Z` is the
nuclear charge; under GTH it is not the number of electrons in the density.

## 9. ML Pulay mixing

`MIXING_METHOD ML_PULAY` predicts the **DIIS coefficient vector**, not a
density. That choice follows directly from this document.

CP2K's Pulay mixing solves `min c^T B c` subject to `sum(c) = 1`, where
`B_ij = <r_i | r_j>`, and forms `rho_in(next) = sum_i c_i (rho_in(i) + beta r_i)`.
Only `c` is replaced; the buffers, metric, Kerker damping and density update
are upstream's code.

Predicting coefficients makes the two properties this document is about
**structural rather than checked**:

| Property | Why it holds automatically |
| --- | --- |
| `Tr(P S) = Ne` | `Tr(·S)` is linear, so an affine combination of densities that each carry `Ne` carries `Ne` |
| `P = P†` | an affine combination of Hermitian matrices is Hermitian |

A model that predicts a raw density has to be checked for both and rescaled
when it fails. A model that predicts coefficients cannot fail either one, for
any output at all, as long as `sum(c) = 1` — which the exported module imposes
by construction and the Fortran gate re-checks.

That leaves exactly one run-time question: **is the proposal an improvement?**
In the DIIS quadratic form the residual of a candidate is `sqrt(c^T B c)`, an
`nb x nb` contraction with no Kohn-Sham build, so the accept test is exact
within the DIIS model rather than a heuristic. A proposal worse than native
Pulay's is rejected before it touches the density, and the fallback is the
native solve that already ran.

`B` is `nb x nb` where `nb` is the history depth (at most `NBUFFER`), so the
model is independent of `nao` — one model serves any system.
