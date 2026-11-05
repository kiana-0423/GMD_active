"""Real GMD_SGT inference through the adapter (ledger TST-006; replaces PHAL's Allegro smoke scripts).

Loads a real exported TorchScript model *through GMDSGTBackend* (not a side script) and checks
finite H2O energies/forces for isolated and periodic water. The model is ``$GMD_SGT_MODEL_PATH``
when set, otherwise the tiny seeded fixture export (tests/fixtures/sgt). Skipped, with the reason,
only when GMD_SGT is unavailable or the configured model file does not exist.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from gmd_active_learning.adapters.sgt.backend import GMDSGTBackend, SGTConfig
from gmd_active_learning.core.exceptions import CapabilityUnavailableError

pytestmark = pytest.mark.integration
REPO = Path(__file__).resolve().parents[2]
SGT_PATH = os.environ.get("GMD_SGT_PATH", str(REPO.parent / "GMD_SGT"))
FIXTURES = REPO / "tests" / "fixtures" / "sgt"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())


@pytest.fixture(scope="module")
def setup() -> tuple[GMDSGTBackend, Path, Path | None]:
    try:
        backend = GMDSGTBackend(SGTConfig(source_path=SGT_PATH))
    except CapabilityUnavailableError as exc:
        pytest.skip(f"GMD_SGT unavailable: {exc}")
    configured = os.environ.get("GMD_SGT_MODEL_PATH")
    if configured:
        export = Path(configured)
        if not export.is_file():
            pytest.skip(f"GMD_SGT_MODEL_PATH={configured} does not exist")
        return backend, export, None
    member = MANIFEST["members"][0]
    return backend, FIXTURES / member["export"], FIXTURES / member["checkpoint"]


def test_exported_model_gives_finite_h2o_forces_through_the_adapter(setup) -> None:  # noqa: ANN001
    backend, export, _checkpoint = setup
    info = backend.model_info(export)
    assert info.kind == "export" and {1, 8} <= set(info.species) and info.cutoff > 0
    for water in read(REPO / "tests" / "fixtures" / "datasets" / "tiny_h2o.extxyz", index=":"):
        prediction = backend.predict(export, water.positions, water.numbers)
        assert prediction.forces.shape == (3, 3)
        assert np.isfinite(prediction.energy) and np.isfinite(prediction.forces).all()
        assert abs(prediction.forces.sum(axis=0)).max() < 1e-4  # conservative model: no net force


def test_periodic_water_box_export_matches_checkpoint(setup) -> None:  # noqa: ANN001
    backend, export, checkpoint = setup
    if checkpoint is None:
        pytest.skip("checkpoint comparison needs the fixture model (GMD_SGT_MODEL_PATH gives only an export)")
    water = read(REPO / "tests" / "fixtures" / "datasets" / "tiny_h2o.extxyz", index=0)
    rng = np.random.default_rng(4)
    molecules = [water.positions - water.positions[0] + offset for offset in ([0.5, 0.5, 0.5], [3.4, 3.2, 3.6], [0.8, 3.5, 2.0])]
    box = Atoms("OH2" * 3, positions=np.vstack(molecules) + rng.normal(0, 0.02, (9, 3)), cell=np.eye(3) * 6.0, pbc=True)
    exported = backend.predict(export, box.positions, box.numbers, box.cell.array, box.pbc)
    eager = backend.predict(checkpoint, box.positions, box.numbers, box.cell.array, box.pbc)
    assert np.isfinite(exported.forces).all() and exported.forces.shape == (9, 3)
    np.testing.assert_allclose(exported.forces, eager.forces, atol=1e-4, rtol=0)
    assert exported.energy == pytest.approx(eager.energy, abs=1e-4)
