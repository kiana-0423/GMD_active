# Draft review and validation record

## Scope and authorization

The user requested an English arXiv-oriented LaTeX draft in a new directory, accompanying figures, and omission of unfinished results. The user explicitly chose placeholders for author information. The academic-research-suite workflow was used for source tracing, drafting, citation checking, and figure review; the request supplied the writing authorization. This is an adapted artifact-production workflow, not a claimed complete or blinded peer-review process.

Only completed `alkane_hotspots` evidence was frozen. Ongoing `alkane_repair` data, partial q0 labels, projected repairs, the larger campaign, and unexecuted C64/C128 outcomes are excluded. No simulation, training run, compute-cap change, publication, Git commit, or push was performed for this manuscript task.

## Review outcomes

- **Originality:** framed as integration and pilot validation. Prior local active-learning work is cited; no first-in-field claim is made.
- **Method description:** source inspection checked the LJ sign, committee normalization, actual enabled online metrics, weighted force-loss denominator, caps and energy masks, and optimizer scheduling.
- **Evidence sufficiency:** supports the demonstrated software loop and local-reference fidelity. Acquisition superiority, physical-feature benefit, qualified extrapolation repair, long-time stability, and equilibrium accuracy remain unsupported and are explicitly limited.
- **Argument coherence:** local DFT fidelity, model-context sensitivity, acquisition performance, and short-trajectory behavior are kept distinct. The complete online loop is not inferred solely from offline acquisition plots.
- **Writing and figures:** actual repeats and failed settings remain visible; costs, denominators, exploratory protocol changes, and diagnostic definitions are stated. Figure previews were visually inspected and node-text/annotation overlaps corrected.

Separate agents checked implementation evidence and reference metadata. The main drafting agent reconciled those audits. The reviews were not blinded and should not be described as external peer review.

## Corrections made during audit

1. Training warm-up counts changed from epochs to optimizer steps.
2. Recovery wording corrected to retrospective strict rejection of the same promoted model.
3. The seven fidelity sites identified as three perturbation sites plus four controls.
4. Eight static trigger observations identified as four structures evaluated in each of two rounds.
5. Fingerprint description corrected to pair-distance density.
6. Topology retention defined using exact bond-set equality and the fixed ASE-radius multiplier.
7. Energy loss specified as energy-per-atom MAE; no force gradient from zero-weight targets stated with its batch normalization.
8. The local/full comparison explicitly discloses incomplete acquisition pairing in repetition 1.
9. Figure 06 uses actual cumulative label counts on its numeric x-axis.

## Checks

`scripts/validate_bundle.py` implements 19 offline checks: snapshot hashes; exclusion of ongoing inputs; per-atom calibration and transfer pooling; gate failures; four-label arithmetic; acquisition pairing; cost accumulation; YAML target agreement; online label accounting; topology denominators; molecular geometry identities; figure/input existence; citation/BibTeX matching; cross-references; LaTeX brace/environment balance; author placeholders; JSON helpers/nonfinite-value rejection; and table reproduction.

The test output is saved in `validation.txt`. Figure generation has been exercised on the bundled data. Source checks are not a LaTeX compiler and cannot certify page layout. **LaTeX was not compiled**, as requested. Author information, contributions, competing interests, and funding remain explicit placeholders. No manuscript PDF or submission-ready certification is claimed.

The acronym checker was applied to a text extraction of the draft. Software names, chemical/basis labels, file formats, and standard units are distinguished from ordinary prose acronyms; see `acronyms.md` and `acronyms.json` for its raw diagnostic output and `acronym_allowlist.txt` for allowed terms.

The raw checker reports partial coverage because its initials parser does not recognize the hyphenated expansions of DFT, LJ, RMSE, and SCF. Those four first-use definitions were checked manually in the LaTeX source. Partial coverage is retained in the report rather than presented as a clean automated pass.
