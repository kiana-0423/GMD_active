# PHAL Milestone-3 release baseline (historical provenance)

The three files in this directory are **verbatim copies** of PHAL
(`lammps_monitor`, <https://github.com/kiana-0423/lammps_monitor>)
`docs/releases/` at commit `96263b6`:

| File | PHAL source |
| --- | --- |
| `milestone3-baseline.md` | `docs/releases/milestone3-baseline.md` |
| `baseline.yaml` | `docs/releases/baseline.yaml` |
| `manifest.json` | `docs/releases/manifest.json` |

They are kept as historical evidence only (ledger entry DOC-006). Do not edit
them; `tests/tools/test_containers.py` checks that they stay byte-identical to
the PHAL originals whenever the reference tree `ref/lammps_monitor` is present.

What they describe, and what they do not:

- They freeze PHAL's **LAMMPS + pair_allegro / NequIP + Allegro** runtime
  (train-probe and md-probe images, PBS jobs `2360507.opbs` and
  `2362776.opbs` on Miyabi-G GH200). None of that stack is used by GMD_active,
  which targets GMD, GMD_SGT and CP2K (see `docs/containers/`).
- The baseline was frozen at PHAL commit `8ea02eb` (tag
  `milestone3-md-probe`, 2026-07-11), which **predates** the migration
  reference `96263b6`. Later PHAL changes are not covered by it; in particular
  the default-config move (PHAL commit `6811f35`) broke `Dockerfile.train-probe`
  at `96263b6` (finding F-32), so the validated train-probe image cannot be
  rebuilt from `96263b6`.
- The evidence the files cite (`build/evidence/release/milestone3/`,
  `runtime/logs/...`) was **git-ignored in PHAL and is not in either
  repository**. The digests and hashes below can therefore not be re-verified
  from repository contents; they are PHAL's own records.

GMD_active's own images have no release baseline yet: nothing has been built or
validated (see `containers/versions.yaml`, all stacks `not_built` /
`not_validated`). When one exists it should use the same structure
(`<name>-baseline.md`, `baseline.yaml`, `manifest.json`) with evidence committed
or archived somewhere reachable.
