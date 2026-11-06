# ADR 005: GMD is controlled in segments through files, cooperatively

## Status

Accepted (GMD_active 0.4).

## Context

PHAL tailed a LAMMPS dump with one frame of lag, could not tell "timeout" from "LAMMPS exited",
and controlled the process with signals (SIGSTOP/SIGCONT, SIGTERM then SIGKILL) (F-19, F-29). A
restart from a GMD checkpoint lost atomic numbers, so a restarted ML run evaluated every atom as
padding (F-02). The active-learning loop needs: frames as soon as they exist, a way to stop or
checkpoint on a monitor decision, model changes between segments, and a restart that is exactly
the continuous run.

## Decision

- GMD runs in **segments** of `workflow.segment_steps`. Each segment is a separate GMD process
  started from the previous segment's checkpoint (GMD checkpoint v3, which stores Z and the model
  identity). A promoted model takes effect only at a segment boundary (`model_change` directive),
  recorded in the store.
- Frames are **flushed files** (`frame_file`, `gmd-frame-v1` extxyz with stable atom IDs, species,
  forces, energy, model hash) read incrementally by `FrameTail`; a status file (`gmd-status-v1`)
  reports progress and termination, so end of stream is distinct from a poll timeout.
- Control is **cooperative**: the adapter creates the stop file or the checkpoint-request file and
  GMD acts at its next `control_poll_interval` boundary. No signals are used for control; SIGTERM
  is only for abandoning a hung process.
- The GMD TorchScript force path is serial; the adapter never adds `--np`, and the launcher is
  validated to request one rank.

## Consequences

- Stop latency is up to `control_poll_interval` steps plus the reader's poll delay after the
  triggering frame. This is documented and configurable, not hidden.
- Segment boundaries are natural persistence points: the monitor state, open events and the
  segment record are saved there, so a crashed controller resumes at the last completed segment
  (`MDEngine.completed_segment` reconciles a segment whose outcome is unknown).
- Continuous and segmented runs agree to 1e-6 A in positions (bitwise with JIT optimisation off),
  tested with the real GMD build (`tests/integration/test_gmd_real.py`).
- GMD needed upstream extensions (frame, stop, checkpoint-request and status files, checkpoint
  v3, model change); see `docs/migration/integration_gaps.md`.
