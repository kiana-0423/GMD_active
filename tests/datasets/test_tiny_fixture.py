"""The tiny seeded synthetic H2O dataset fixture (FIX-002, dataset part) is reproducible and GMD_SGT-ready."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from gmd_active_learning.datasets import read_samples_extxyz

from datasets.test_gmd_sgt_compat import sgt  # noqa: F401  (fixture)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "datasets"


def _generator():
    spec = importlib.util.spec_from_file_location("make_tiny_h2o", FIXTURE_DIR / "make_tiny_h2o.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_tiny_h2o"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_fixture_is_reproducible_from_its_seed() -> None:
    stored = read_samples_extxyz(FIXTURE_DIR / "tiny_h2o.extxyz")
    rebuilt = _generator().build()
    assert [s.sample_id for s in stored] == [s.sample_id for s in rebuilt]
    for a, b in zip(stored, rebuilt):
        np.testing.assert_allclose(a.forces, b.forces, atol=1e-9)
        assert a.energy == pytest.approx(b.energy, abs=1e-12) and a.energy_weight == 1.0
        assert a.split_group == b.split_group


def test_fixture_forces_are_the_negative_energy_gradient() -> None:
    generator = _generator()
    for sample in read_samples_extxyz(FIXTURE_DIR / "tiny_h2o.extxyz"):
        positions = sample.region.positions
        np.testing.assert_allclose(generator.toy_forces(positions), sample.forces, atol=1e-6)


def test_gmd_sgt_reads_the_fixture(sgt) -> None:  # noqa: F811
    items = sgt.read_extxyz(str(FIXTURE_DIR / "tiny_h2o.extxyz"))
    assert len(items) == 8 and {item["split_group"] for item in items} == {f"event:evt-{k}" for k in range(4)}
    assert all(float(item["energy_weight"]) == 1.0 and item["force_weight"].numel() == 3 for item in items)
