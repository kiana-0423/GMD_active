"""SCF-029: exporting a fixture density checkpoint matches PHAL's export (output rtol 1e-10, manifest).

``data/deploy_parity/density.pt`` is a tiny density GNN checkpoint, trained for one epoch with GMD's
``tools.scf.training.density_gnn`` on the water/H2 test examples (torch seed 0).
``phal_water_probe.npz`` and ``phal_manifest.json`` were produced by exporting that checkpoint with
PHAL 96263b6 (``hotspot_al.cp2k_ml_scf.deploy.export_density_guess``) and calling the scripted module
on the water probe below. The comparison therefore needs no ref/. While ref/ exists, PHAL's export is
re-run and must reproduce the frozen files. The manifests may differ only in ``path``,
``source_checkpoint`` and ``model_digest``: the TorchScript archive embeds the module's qualified
name, so the bytes differ. PHAL's manifest has no timestamp.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

DATA = Path(__file__).resolve().parent / "data" / "deploy_parity"
REF = Path(__file__).resolve().parents[2] / "ref" / "lammps_monitor"
UNSTABLE_KEYS = {"path", "source_checkpoint", "model_digest"}


def water_probe(scripted) -> np.ndarray:  # noqa: ANN001
    """The scripted guess on water (O, H, H; 23 AOs) with a fixed non-orthonormal overlap."""

    numbers = torch.as_tensor([[8, 1, 1]], dtype=torch.long)
    positions = torch.as_tensor([[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]], dtype=torch.float32)
    cell = torch.eye(3, dtype=torch.float32).unsqueeze(0) * 10.0
    charge = torch.zeros(1, dtype=torch.float32)
    multiplicity = torch.ones(1, dtype=torch.float32)
    n_ao = 23
    overlap = torch.full((n_ao, n_ao), 0.1, dtype=torch.float32)
    overlap.fill_diagonal_(1.0)
    with torch.no_grad():
        return scripted(numbers, positions, cell, charge, multiplicity, overlap.unsqueeze(0)).double().numpy()


def _export(export, tmp_path: Path, name: str) -> tuple[np.ndarray, dict]:  # noqa: ANN001
    checkpoint = tmp_path / "density.pt"
    if not checkpoint.exists():
        shutil.copy(DATA / "density.pt", checkpoint)
    deployed = export(checkpoint, tmp_path / f"{name}.ts.pt")
    manifest = json.loads(Path(f"{deployed.path}.deploy.json").read_text(encoding="utf-8"))
    return water_probe(torch.jit.load(str(deployed.path))), {key: value for key, value in manifest.items() if key not in UNSTABLE_KEYS}


def phal_capture(tmp_path: Path) -> tuple[np.ndarray, dict]:
    from hotspot_al.cp2k_ml_scf.deploy import export_density_guess

    return _export(export_density_guess, tmp_path, "phal")


def test_deployed_density_guess_matches_phal(tmp_path: Path) -> None:
    from gmd_active_learning.adapters.dft.scf.deploy import export_density_guess, verify_deployed

    output, manifest = _export(export_density_guess, tmp_path, "gmd")
    with np.load(DATA / "phal_water_probe.npz") as data:
        expected = data["density"]
    assert output.shape == expected.shape == (23, 23, 1)
    np.testing.assert_allclose(output, expected, rtol=1e-10, atol=0.0)
    assert manifest == json.loads((DATA / "phal_manifest.json").read_text(encoding="utf-8"))
    assert verify_deployed(tmp_path / "gmd.ts.pt", "density_guess")["output_shape"] == [23, 23, 1]


@pytest.mark.skipif(not (REF / "src" / "hotspot_al").is_dir(), reason="ref/lammps_monitor not present")
def test_frozen_phal_export_is_current(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(REF / "src"))
    output, manifest = phal_capture(tmp_path)
    with np.load(DATA / "phal_water_probe.npz") as data:
        np.testing.assert_array_equal(output, data["density"])
    assert manifest == json.loads((DATA / "phal_manifest.json").read_text(encoding="utf-8"))


if __name__ == "__main__":  # re-capture: python tests/scf/test_scf_deploy_phal_parity.py <tmp_dir>
    sys.path.insert(0, str(REF / "src"))
    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)
    density, phal_manifest = phal_capture(target)
    np.savez(DATA / "phal_water_probe.npz", density=density)
    (DATA / "phal_manifest.json").write_text(json.dumps(phal_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
