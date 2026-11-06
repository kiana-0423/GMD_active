"""Run the SCF-026 short density-training fixture with PHAL's or GMD's code; prints a JSON summary.

Used by tests/scf/test_scf_density_training_parity.py. PHAL's run happens in a subprocess with
cwd=ref/lammps_monitor (its own ``tests`` package provides the identical fixtures).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

EPOCHS = 3


def run(flavour: str, work: Path) -> dict:
    if flavour == "phal":
        from hotspot_al.cp2k_ml_scf.training.train_density import DensityTrainingConfig, train_density_matrix_model
        from tests.unit.test_scf_density_model import identifiable_examples
    else:
        from tests.scf.test_scf_density_model import identifiable_examples
        from tools.scf.training.density_gnn import DensityTrainingConfig, train_density_matrix_model
    examples = identifiable_examples(work / "data", count=3)
    config = DensityTrainingConfig(epochs=EPOCHS, batch_size=2, hidden_size=16, n_interactions=2, seed=0)
    result = train_density_matrix_model(examples, None, work / "density.pt", config=config)
    return {
        "basis": result.basis.to_dict(),
        "valence": None if result.valence is None else result.valence.to_dict(),
        "history": [{"epoch": m.epoch, "split": m.split, "loss": m.loss, "density_mae": m.density_mae,
                     "density_rmse": m.density_rmse, "electron_mae": m.electron_mae} for m in result.history],
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        print(json.dumps(run(sys.argv[1], Path(directory)), sort_keys=True))
