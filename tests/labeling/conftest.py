"""A scripted fake ``cp2k`` executable for labeling-service tests (real CP2K tests live elsewhere).

The fake reads ``-i <input> -o <output>``, parses PROJECT/RUN_TYPE/COORD/FIXED_ATOMS/MAX_SCF from the
input and writes CP2K-2026.2-style output. Behaviour per invocation comes from ``plan.json`` (a list
of actions consumed in order; the last one repeats) or from rules:

- ``ok``: converged SCF, energy, forces (and a GEO_OPT trajectory for RUN_TYPE GEO_OPT);
- ``scf_fail``: ABORT "SCF run NOT converged", exit 1;   ``abort``: other ABORT, exit 1;
- ``crash``: exit 137 without output;   ``truncated``: output cut before the forces, exit 0;
- ``bad_count``: one force row too few;   ``sleep``: sleep 30 s first;
- rule ``min_max_scf``: if the input's MAX_SCF is below it, the action becomes ``scf_fail``;
- an input containing ``ML_SCF_FAIL`` always fails the SCF (used for the ML-SCF fallback test).
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

FAKE_CP2K = r'''
import json, os, re, sys, time
from pathlib import Path

args = sys.argv[1:]
inp = Path(args[args.index("-i") + 1]); out = Path(args[args.index("-o") + 1])
state = Path(os.environ["FAKE_CP2K_DIR"])
text = inp.read_text()
config = json.loads((state / "plan.json").read_text()) if (state / "plan.json").exists() else {}
plan = config.get("plan", ["ok"])
counter = state / "counter"
n = int(counter.read_text()) if counter.exists() else 0
counter.write_text(str(n + 1))
action = plan[min(n, len(plan) - 1)]
with (state / "calls.log").open("a") as log:
    log.write(json.dumps({"input": str(inp), "action": action}) + "\n")
max_scf = int(re.search(r"MAX_SCF\s+(\d+)", text).group(1))
if config.get("min_max_scf") and max_scf < config["min_max_scf"] and action == "ok":
    action = "scf_fail"
if "ML_SCF_FAIL" in text:
    action = "scf_fail"
if action == "sleep":
    time.sleep(30); action = "ok"
if action == "crash":
    sys.exit(137)
project = re.search(r"PROJECT\s+(\S+)", text).group(1)
run_type = re.search(r"RUN_TYPE\s+(\S+)", text).group(1)
coord = re.search(r"&COORD\n(.*?)\n\s*&END COORD", text, re.S).group(1).splitlines()
atoms = [(line.split()[0], [float(v) for v in line.split()[1:4]]) for line in coord]
fixed = re.search(r"LIST ([\d ]+)", text)
fixed = {int(v) - 1 for v in fixed.group(1).split()} if fixed else set()
lines = [" CP2K| version string:                                       CP2K version 2026.2",
         " CP2K| source code revision number:                                  git:c92cc08",
         "  Step     Update method      Time    Convergence         Total energy    Change",
         "  ------------------------------------------------------------------------------",
         "     1 OT DIIS     0.15E+00    0.0     0.00000900        -1.1000000000 -1.10E+00"]
if action in {"scf_fail", "abort"}:
    message = "SCF run NOT converged. To continue the calculation regardless," if action == "scf_fail" else "Unknown basis set FAKE"
    lines += [" *******************************************************************************",
              " *   ___                                                                       *",
              " *  /   \\                                                                     *",
              " * [ABORT]                                                                     *",
              f" *  \\___/     {message:<62}*",
              " *    |                                                                        *",
              " *******************************************************************************"]
    out.write_text("\n".join(lines) + "\n")
    sys.exit(1)
lines += ["", "  *** SCF run converged in     1 steps ***", "",
          " ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]             -1.100000000000000"]
if action == "truncated":
    out.write_text("\n".join(lines) + "\n")
    sys.exit(0)
rows = atoms[:-1] if action == "bad_count" else atoms
lines += [" FORCES| Atomic forces [hartree/bohr]",
          " FORCES|   Atom     x               y               z               |f|"]
for i, _ in enumerate(rows, start=1):
    fx, fy, fz = 0.01 * i, -0.02 * i, 0.005
    lines.append(f" FORCES| {i:6d} {fx: .8E} {fy: .8E} {fz: .8E} {(fx*fx+fy*fy+fz*fz)**0.5: .8E}")
lines += [" FORCES| Sum     0.00000000E+00  0.00000000E+00  0.00000000E+00", ""]
if run_type == "GEO_OPT":
    moved = [(s, [p[0] + (0.0 if i in fixed else 0.01), p[1], p[2]]) for i, (s, p) in enumerate(atoms)]
    frames = []
    for frame in (atoms, moved):
        frames.append(f"{len(frame)}\n i = 0, E = -1.1\n" + "\n".join(f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}" for s, p in frame))
    (inp.parent / f"{project}-pos-1.xyz").write_text("\n".join(frames) + "\n")
    lines += [" *******************************************************************************",
              " ***                    GEOMETRY OPTIMIZATION COMPLETED                      ***",
              " *******************************************************************************"]
lines += ["  **** **** ******  **  PROGRAM ENDED AT                 2026-11-06 00:00:00.000"]
out.write_text("\n".join(lines) + "\n")
'''


class FakeCP2K:
    def __init__(self, root: Path) -> None:
        self.state = root
        self.state.mkdir(parents=True, exist_ok=True)
        script = root / "fake_cp2k.py"
        script.write_text(FAKE_CP2K, encoding="utf-8")
        self.executable = root / "cp2k.fake"
        self.executable.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        self.executable.chmod(self.executable.stat().st_mode | stat.S_IXUSR)

    def configure(self, plan: list[str] | None = None, **rules: Any) -> None:
        (self.state / "plan.json").write_text(json.dumps({"plan": plan or ["ok"], **rules}), encoding="utf-8")

    def calls(self) -> list[dict[str, Any]]:
        log = self.state / "calls.log"
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


@pytest.fixture()
def fake_cp2k(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeCP2K:
    fake = FakeCP2K(tmp_path / "fake_cp2k")
    monkeypatch.setenv("FAKE_CP2K_DIR", str(fake.state))
    return fake
