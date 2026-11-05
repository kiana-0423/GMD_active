"""A fake GMD executable for adapter tests: same CLI, directives, frame and status formats.

``fake_gmd.py input.xyz run.in`` writes ``gmd-frame-v1`` frames every ``frame_interval`` steps
(forces = -0.1 * displacement from the start, deterministic), honours the stop file, writes a
checkpoint and ``gmd-status-v1`` status. Behaviour switches (environment):

* ``FAKE_GMD_EXIT``: exit code at the end (non-zero also writes state "failed").
* ``FAKE_GMD_CHUNK``: write each frame in chunks of this many bytes, flushing between chunks.
* ``FAKE_GMD_DELAY``: seconds to sleep per step (lets readers observe partial files).
* ``FAKE_GMD_TRUNCATE_AFTER``: after this many frames, truncate the frame file and rewrite it.
* ``FAKE_GMD_MARKER``: path of a file touched when the program starts (proves it ran).
* ``FAKE_GMD_BURST``: ``step:atom:fx`` adds a force burst (eV/A along x) on one atom at one step,
  only while the model file's sha256 starts with ``FAKE_GMD_BURST_MODEL`` when that is set.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path


def main(xyz: str, run_in: str) -> int:
    if os.environ.get("FAKE_GMD_MARKER"):
        Path(os.environ["FAKE_GMD_MARKER"]).write_text("ran\n")
    directives: dict[str, str] = {}
    for line in Path(run_in).read_text().splitlines():
        if line.strip():
            key, _, value = line.strip().partition(" ")
            directives[key] = value.strip()
    lines = Path(xyz).read_text().splitlines()
    n = int(lines[0].split()[0])
    box = [float(x) for x in lines[1].split()[:3]]
    symbols = [line.split()[0] for line in lines[2 : 2 + n]]
    start = [[float(x) for x in line.split()[1:4]] for line in lines[2 : 2 + n]]
    n_steps = int(directives["run"])
    every = int(directives.get("frame_interval", "1"))
    first = 0
    restart = directives.get("restart_from")
    if restart:
        first = int(json.loads(Path(restart).read_text())["step"])
    model = directives.get("model_path", "")
    sha = hashlib.sha256(Path(model).read_bytes()).hexdigest() if model and Path(model).is_file() else "none"
    frame_path = Path(directives.get("frame_file", "frames.extxyz"))
    stop_file = Path(directives.get("stop_file", "STOP"))
    chunk = int(os.environ.get("FAKE_GMD_CHUNK", "0"))
    delay = float(os.environ.get("FAKE_GMD_DELAY", "0"))
    truncate_after = int(os.environ.get("FAKE_GMD_TRUNCATE_AFTER", "0"))
    written: list[str] = []
    burst = os.environ.get("FAKE_GMD_BURST", "")
    burst_step, burst_atom, burst_fx = (int(burst.split(":")[0]), int(burst.split(":")[1]), float(burst.split(":")[2])) if burst else (-1, -1, 0.0)
    if os.environ.get("FAKE_GMD_BURST_MODEL") and not sha.startswith(os.environ["FAKE_GMD_BURST_MODEL"]):
        burst_step = -1  # an improved model no longer produces the burst

    def frame_text(step: int) -> str:
        shift = 0.001 * step  # continuous across restarts
        out = [str(n), f'Lattice="{box[0]} 0 0 0 {box[1]} 0 0 0 {box[2]}" Properties=id:I:1:species:S:1:pos:R:3:forces:R:3 pbc="T T T" '
               f"step={step} time_fs={0.5 * step} energy={-1.0 - 0.01 * step} format=gmd-frame-v1 run_id={directives.get('run_id', 'none')} "
               f"force_field=ml model_sha256={sha}"]
        for i, (symbol, pos) in enumerate(zip(symbols, start)):
            p = [pos[0] + shift * (i + 1), pos[1], pos[2]]
            f = [-0.1 * shift * (i + 1) + (burst_fx if (step == burst_step and i == burst_atom) else 0.0), 0.0, 0.0]
            out.append(f"{i} {symbol} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f} {f[0]:.10f} {f[1]:.10f} {f[2]:.10f}")
        return "\n".join(out) + "\n"

    def emit(text: str) -> None:
        with frame_path.open("a") as handle:
            if chunk <= 0:
                handle.write(text)
                handle.flush()
                return
            for k in range(0, len(text), chunk):
                handle.write(text[k : k + chunk])
                handle.flush()
                time.sleep(delay / 4 if delay else 0.0)

    state, step = "completed", first
    for step in range(first, first + n_steps + 1):
        if step != first and stop_file.exists():
            state = "stopped"
            break
        if (step - first) % every == 0:
            text = frame_text(step)
            emit(text)
            written.append(text)
            if truncate_after and len(written) == truncate_after:
                frame_path.write_text("")  # truncated by an external process ...
                time.sleep(max(3 * delay, 0.05))  # ... long enough for a reader to notice ...
                for previous in written:  # ... and rewritten from the start
                    emit(previous)
        time.sleep(delay)
    exit_code = int(os.environ.get("FAKE_GMD_EXIT", "0"))
    checkpoint = Path(directives.get("checkpoint_file", "segment.gmdchk"))
    checkpoint.write_text(json.dumps({"step": step}))
    status = {"format_version": "gmd-status-v1", "state": "failed" if exit_code else state, "step": step, "start_step": first,
              "checkpoint": checkpoint.name, "model_sha256": sha, "model_changed_from": None}
    Path(directives.get("status_file", "status.json")).write_text(json.dumps(status))
    if exit_code:
        print("fake GMD failure: forces exploded", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
