"""Regenerate tests/fixtures/gmd/: frames and status written by the real patched GMD (Torch build).

Runs GMD through the GMD adapter on ``tests/fixtures/gmd/water2.xyz`` (two water molecules, the
second listed H-O-H so species order is exercised) with the tiny GMD_SGT fixture export
``tests/fixtures/sgt/member_11/model.pt`` for 6 steps, writing a frame every 2 steps, and stores
GMD's own ``frames.extxyz``, ``status.json`` and ``run.in`` plus a manifest. Absolute paths that GMD
records (model path, checkpoint) are replaced by fixture-relative names; frame lines are untouched.

    PYTHONPATH=src python tools/fixtures/make_gmd_frame_fixture.py [--gmd PATH]

``--gmd`` defaults to $GMD_EXECUTABLE or ../GMD/build-torch/gmd next to this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from gmd_active_learning.adapters.gmd.engine import GMDConfig, GMDEngine
from gmd_active_learning.contracts import UNITS, FrameAction, ModelArtifact
from gmd_active_learning.contracts.ids import file_sha256
from gmd_active_learning.workflow.protocols import SegmentRequest

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "gmd"
EXPORT = REPO / "tests" / "fixtures" / "sgt" / "member_11" / "model.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gmd", default=os.environ.get("GMD_EXECUTABLE", str(REPO.parent / "GMD" / "build-torch" / "gmd")))
    args = parser.parse_args()
    engine = GMDEngine(GMDConfig(executable=args.gmd, initial_structure=str(FIXTURE / "water2.xyz"), frame_interval=2,
                                 directives={"time_step": 0.25, "velocity": 50.0, "velocity_init": "random", "velocity_seed": 7}))
    model = ModelArtifact("fixture-member-11", str(EXPORT), file_sha256(EXPORT), str(EXPORT), file_sha256(EXPORT), None, None, (1, 8), 4.0,
                          dict(UNITS), {}, "fixture", "gmd_sgt", "fixture")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "segment"
        result = engine.run_segment(SegmentRequest("fixture-run", 0, work, model, 6, None, None), lambda frame: FrameAction.CONTINUE)
        if result.status != "completed":
            raise SystemExit(f"GMD run failed: {result.detail}")
        shutil.copy2(work / "frames.extxyz", FIXTURE / "frames.extxyz")
        status = json.loads((work / "status.json").read_text())
        for key, value in list(status.items()):
            if isinstance(value, str) and value.startswith(str(work)):
                status[key] = Path(value).name
        (FIXTURE / "status.json").write_text(json.dumps(status, indent=1, sort_keys=True) + "\n")
        run_in = (work / "run.in").read_text().replace(str(EXPORT), "model.pt")
        (FIXTURE / "run.in").write_text(run_in)
    # GMD has no --version flag; record the commit of the checkout the binary was built from.
    commit = subprocess.run(["git", "-C", str(engine.executable.parent), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "-C", str(engine.executable.parent), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=False)
    manifest = {
        "description": "Frames and status written by the real patched GMD (TorchScript build) through the GMD adapter.",
        "command": "PYTHONPATH=src python tools/fixtures/make_gmd_frame_fixture.py",
        "gmd_commit": (commit.stdout.strip() or "unknown") + ("+local-changes" if dirty.stdout.strip() else ""),
        "gmd_binary": "GMD built with -DGMD_ENABLE_TORCH=ON (frame_file/status_file directives, checkpoint v3)",
        "model_export": "tests/fixtures/sgt/member_11/model.pt",
        "model_export_sha256": file_sha256(EXPORT),
        "files": {name: hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() for name in ("water2.xyz", "frames.extxyz", "status.json", "run.in")},
    }
    (FIXTURE / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
