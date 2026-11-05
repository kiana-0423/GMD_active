# SCF data contracts: rule → validator → test

The specifications are [density_representation.md](density_representation.md) and
[pulay_dataset.md](pulay_dataset.md), ported verbatim from PHAL. The cp2k-ml-scf
reconstruction was derived from them (its RECONSTRUCTION.md §4 cites them as "Exact"), so
they are a cross-repository contract. This page maps every rule they state to the ported code
that enforces it and to the test that proves it. Rules where GMD_active deliberately departs
from PHAL's code (not from its documents) are marked **fixed**.

| Rule | Enforced by | Test |
| --- | --- | --- |
| `Ne = Tr(P S)`, never `Tr(P)`, in the AO basis | `ao_matrices.electron_count`, `recording.trajectory.validate_frame` | `test_scf_trajectory.py::test_trace_p_alone_would_not_conserve`, `::test_validate_frame_uses_the_overlap_not_the_trace` |
| An unverifiable count is refused, not approximated (no overlap → refuse). **Fixed:** PHAL's dataset validator and training loader fell back to `Tr(P)` | `validation.validate_density_artifacts`, `tools/scf/training/dataset_loader.density_electron_count` | `test_scf_fixes.py::test_a_density_without_an_overlap_is_refused_not_approximated`, `test_scf_training.py::test_density_matrix_target_reads_an_npz_artifact` |
| A density matrix is exported only with its overlap | `density.exchange.write_density_exchange`, `density.predictor.validate_prediction` | `test_scf_density_exchange.py::test_density_matrix_without_an_overlap_is_refused` |
| Exchange format `phal-ml-scf-v2`: header keys, `-` for empty fields, column-major values, exact value counts | `density.exchange` | `test_scf_density_exchange.py::test_exchange_header_matches_the_documented_layout`, `::test_values_are_written_in_fortran_order`, `::test_reading_a_truncated_file_fails_loudly`, `test_scf_phal_parity.py::test_exchange_files_are_byte_identical` |
| Validation order: shape → finite → Hermitian 1e-8 → `Tr(P S)` 0.1% | `density.predictor.validate_prediction`, `recording.trajectory.validate_frame` | `test_scf_trajectory.py::test_validate_frame_rejects_*`, `test_scf_density_predictor.py` |
| A density holds **valence** electrons under GTH (water 8, not 10). **Fixed:** PHAL's predictor validation and the dummy, MLP and GNN predictors used `sum(Z) − charge` | `density.predictor.expected_density_electrons`, used by `validate_prediction` and every predictor | `test_scf_fixes.py::test_the_valence_count_is_the_reference_for_a_density`, `test_scf_valence.py::test_water_under_gth_has_eight_electrons_not_ten` |
| A deployed guess normalises to valence, never to the nuclear charge; a valence table equal to Z is refused at export | `deploy.export_density_guess`, `deploy.verify_deployed`, `density.gnn_model.ValenceTable` | `test_scf_production.py::test_a_deployed_guess_normalises_to_valence_not_nuclear_charge`, `::test_a_nuclear_charge_table_is_refused_at_export` |
| `&AO_MATRICES DENSITY/OVERLAP` is required for density recording; `&EACH QS_SCF 1` for trajectories | `recording.input.ao_matrices_print_block` | `test_scf_density_recording.py::test_ao_matrices_are_printed_only_when_recording_density`; real run: `test_scf_cp2k_native.py::test_scf_history_density_and_trajectory_are_recorded_from_a_real_run` |
| Trajectory frame order is the recorded index (P0 … Pfinal), never the directory order | `recording.trajectory.load_trajectory` | `test_scf_trajectory.py::test_loaded_trajectory_is_ordered_p0_to_pfinal` |
| Every frame conserves `Ne`, is finite and Hermitian | `recording.trajectory.validate_frame` | `test_scf_trajectory.py::test_every_iteration_conserves_electrons`, real H2 run above (11 frames at 2.000000) |
| Pulay sample acceptance: converged; matrices present; representation `cp2k_pulay_metric`, metric `gspace_kerker`; overlap present and finite; frames finite and Hermitian; matrices reproduce CP2K's coefficients (1e-6); ≥ 4 frames | `tools/scf/pulay_dataset.validate_sample`, `pulay_matrix.verify_record` | `test_pulay_dataset.py::test_*_is_rejected`, `test_pulay_matrix_record.py` |
| B-matrix conditioning is reported and never applied | `pulay_matrix.b_matrix_diagnostics` | `test_pulay_b_diagnostics.py` |
| Structure identity (`structure_hash`) is shared with exchange files and prediction logs | `schema.structure_hash` | `test_scf_phal_parity.py::test_structure_hash_matches` |
| Recorded density artifacts round-trip losslessly. **Fixed:** PHAL dropped 5 of 14 fields | `schema.SCFDensityArtifacts` | `test_scf_fixes.py::test_density_artifacts_survive_a_save_and_reload` |

Known residual inconsistency, recorded but not changed: the dataset validator's Hermiticity tolerance
(1e-6, `validation.validate_density_artifacts`) is looser than the exchange gate and the trajectory
gate (both 1e-8). The recording-side gate is configurable (`scf.recording.hermitian_tolerance`).
