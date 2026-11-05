# GMD probe

Ported in spirit from PHAL's `containers/probes/md-probe` (LAMMPS +
pair_allegro), re-targeted to GMD + GMD_SGT.

This probe validates only that the `gmd` binary built with
`-DGMD_ENABLE_TORCH=ON` loads a GMD_SGT TorchScript export and integrates with
it. It is not a formal MD simulation.

- `h2.xyz`: two H atoms in a 12 A periodic box (GMD xyz format).
- `run0.run`: single-point evaluation. GMD refuses `run 0`, so it runs one step
  and the validator gates on the step-0 row of `output.log` (the energy of the
  initial configuration).
- `run10.run`: 10 velocity-Verlet steps.
- `sgt-probe.pt` (generated at image build time by
  `containers/scripts/build_gmd_probe_model.py`, not committed): a seeded,
  randomly initialised GMD_SGT `AllegroStyleBackbone`, marked
  - `purpose = integration_probe_only`
  - `scientifically_valid = false`
  - `training_steps = 0`

`model_path` is relative, so it resolves against the run file's directory; the
validator copies these inputs and the model into a writable work directory
(the image itself is read-only under Apptainer).

GMD's TorchScript adapter evaluates on the CPU (`torch::kCPU`), so on a GH200
node these runs use the Grace CPU even when the job requests a GPU.
