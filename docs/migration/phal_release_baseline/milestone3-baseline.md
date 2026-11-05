# PHAL Milestone 3 Release Baseline

This document freezes the first formally validated PHAL runtime baseline. It
records completed validation; it is not a development roadmap or a claim of
scientific model accuracy.

## Git baseline

- Branch: `main`
- Commit: `8ea02ebdb460153a403e046aa4434ef2b8bb7dba`
- Annotated tag: `milestone3-md-probe`
- Release scope: architecture, `train-probe`, and `md-probe`

## Validated architecture

The Core Domain, Application, Backend Interface, and Infrastructure dependency
boundaries passed the offline architecture suite. Milestone 3 did not change
the Core Domain or active-learning workflow.

## Train probe

- Image: `ghcr.io/kiana-0423/train-probe@sha256:513aeaf7e997caa3b743ecc4b7c4d0defdfabd01a13de3f83841f2dc21810e86`
- SIF: `build/containers/phal-train-probe-m2-train-probe-a-arm64.sif`
- SIF SHA256: `fd0e73861d22fce07b7d920ea406c3185db916bed93aa3fba5ca14d9f022bc47`
- PBS job: `2360507.opbs` on `mg0006`, exit status 0
- GPU: NVIDIA GH200 120GB, capability 9.0
- Torch: `2.9.0a0+50eac811a6.nv25.09`; CUDA runtime: `13.0`
- NequIP: `0.18.0`; Allegro: `0.8.3`
- Imports, CUDA tensor test, CPU inference, GPU inference, finite energy,
  finite forces, and force gradients passed.

## MD probe

- Image: `ghcr.io/kiana-0423/md-probe@sha256:d684ffa628091e6277c3f4b98c4de6cd5ef53aa59b210387f253ac0b5e957885`
- SIF: `build/containers/md-probe-m3-md-probe-a-arm64.sif`
- SIF SHA256: `0b6307d80fff7a94aa69b807194854dc05d30fba6fe463f41d8906aace923627`
- Probe model SHA256: `0c800c6d94f697f461033c8ec19c0b52cc587bac873d07dd7435955a06aa2f54`
- LAMMPS: `10 Sep 2025`; official `pair_allegro` backend: `v0.7.0`
- PBS job: `2362776.opbs` on `mg0004`, exit status 0
- Kokkos CUDA `allegro/kk` used the GH200 device.
- `lmp -h`, pair registration, `run 0`, `run 10`, neighbor construction,
  and finite Temp/PE/KE/TotEng passed without NaN, Inf, CUDA error, illegal
  memory access, or segmentation fault.

The included model is an integration probe with zero training steps and is not
scientifically valid. The frozen evidence snapshot is under
`build/evidence/release/milestone3/`; canonical machine-readable metadata is in
`baseline.yaml` and `manifest.json` in this directory.
