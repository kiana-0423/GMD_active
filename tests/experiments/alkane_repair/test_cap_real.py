"""The launcher's wall-clock cap ends real CP2K attempts; the failure and its consumption are recorded (real CP2K)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from .conftest import CONFIGS, copy_profile

pytestmark = [pytest.mark.cp2k]


def test_cpu_cap_terminates_cp2k_and_is_accounted(tmp_path: Path) -> None:
    if shutil.which("cp2k.psmp") is None:
        pytest.skip("cp2k.psmp not on PATH")
    from tools.experiments.alkane_repair.budget import ComputeLedger, load_authorization
    from tools.experiments.alkane_repair.config import load_benchmark
    from tools.experiments.alkane_repair.generate import generate, write_structures
    from tools.experiments.alkane_repair.references import freeze_protocol, label_stage

    bench = copy_profile(tmp_path)
    alpha = yaml.safe_load((tmp_path / "repair_smoke_alpha.yaml").read_text())
    alpha["cp2k"]["launcher"] = ["/usr/bin/timeout", "-k", "2", "4"]
    (tmp_path / "repair_smoke_alpha.yaml").write_text(yaml.safe_dump(alpha))
    auth = yaml.safe_load((CONFIGS / "repair_smoke_authorization.yaml").read_text())
    auth["authorization"]["enforcement"].update(attempt_wall_cap_s=4.0, kill_grace_s=2.0)
    (tmp_path / "repair_smoke_authorization.yaml").write_text(yaml.safe_dump(auth))
    loaded = load_benchmark(bench)
    write_structures(loaded, generate(loaded))
    freeze_protocol(loaded)
    outcome = label_stage(loaded, "initial_model", qualification_stage=0, echo=lambda s: None)
    assert outcome["failed"] and not outcome["completed"]
    ledger = ComputeLedger(loaded, load_authorization(loaded))
    attempts = ledger.attempts()
    assert attempts and not any(a.running for a in attempts)
    # every attempt stopped within its (4 + 2) s wall cap: at most 6 s x 2 cores (plus process start-up)
    assert all(a.core_s < 20.0 for a in attempts)
    listing = subprocess.run(["ps", "-axo", "command="], capture_output=True, text=True).stdout
    assert str(tmp_path) not in listing
