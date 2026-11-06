"""MD/MLIP backend contracts and the full built-in backend selection (ledger BE-001 MD/MLIP part,
BE-003): the workflow protocols are structural, the registry builds typed adapters for
{md: gmd, mlip: gmd_sgt, dft: cp2k, scheduler: pbs} without any module-global factory."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from gmd_active_learning.adapters.dft.cp2k.backend import CP2KBackend
from gmd_active_learning.adapters.gmd.engine import GMDConfig, GMDEngine
from gmd_active_learning.adapters.registry import BackendRegistry, BackendRole, create_backend
from gmd_active_learning.adapters.schedulers.pbs import PBSScheduler
from gmd_active_learning.core.exceptions import CapabilityUnavailableError
from gmd_active_learning.workflow.protocols import FrameEnricher, MDEngine, Trainer

REPO = Path(__file__).resolve().parents[2]


class Incomplete:
    def run_segment(self, request, on_frame):  # noqa: ANN001, ANN201
        return None


def _gmd(tmp_path: Path) -> tuple[Path, Path]:
    executable = tmp_path / "gmd"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    structure = tmp_path / "w.xyz"
    structure.write_text("1\n5 5 5\nAr 0 0 0\n")
    return executable, structure


def test_md_protocol_is_satisfied_only_by_complete_implementations(tmp_path: Path) -> None:
    executable, structure = _gmd(tmp_path)
    assert isinstance(GMDEngine(GMDConfig(executable=str(executable), initial_structure=str(structure))), MDEngine)
    assert not isinstance(Incomplete(), MDEngine)
    assert not isinstance(Incomplete(), Trainer) and not isinstance(Incomplete(), FrameEnricher)


def test_full_builtin_selection_returns_typed_adapters(tmp_path: Path) -> None:
    executable, structure = _gmd(tmp_path)
    sgt = REPO.parent / "GMD_SGT"
    raw = {
        "backends": {"md": {"engine": "gmd"}, "mlip": {"engine": "gmd_sgt"}, "dft": {"engine": "cp2k"}, "scheduler": {"engine": "pbs"},
                     "discover_entry_points": False},
        "md": {"executable": str(executable), "initial_structure": str(structure)},
        "mlip": {"source_path": str(sgt)},
        "cp2k": {"executable": "cp2k.psmp"},
    }
    registry = BackendRegistry()
    assert type(create_backend(raw, BackendRole.MD, registry=registry)) is GMDEngine
    assert type(create_backend(raw, BackendRole.DFT, registry=registry)) is CP2KBackend
    assert type(create_backend(raw, BackendRole.SCHEDULER, registry=registry)) is PBSScheduler
    try:
        mlip = create_backend(raw, BackendRole.MLIP, registry=registry)
    except CapabilityUnavailableError as exc:
        pytest.skip(f"GMD_SGT unavailable: {exc}")
    from gmd_active_learning.adapters.sgt.backend import GMDSGTBackend

    assert type(mlip) is GMDSGTBackend and isinstance(mlip, Trainer) and isinstance(mlip, FrameEnricher)
    assert "gmd_active_learning.adapters.sgt.backend" in sys.modules  # imported on demand by the registry
