# Ported from PHAL (hotspot_al.cp2k_ml_scf.deploy.export_mixer) at 96263b6; MIT License, see NOTICE.
"""TorchScript export of the experimental neural (learned-DIIS) mixer.

**Negative result, experimental only.** No CP2K build consumes this artifact:
``MIXING_METHOD ML`` was never integrated, and the cp2k-ml-scf reconstruction
declares ML mixing out of scope. It is kept so the measurement can be repeated.
"""

from __future__ import annotations

from pathlib import Path

from gmd_active_learning.adapters.dft.scf._torch import require_torch
from gmd_active_learning.adapters.dft.scf.deploy import (
    MIXER_MODEL_ENV,
    DeployedModel,
    DeployError,
    _finish,
)


def export_mixer(
    checkpoint_path: str | Path, output_path: str | Path, *, experimental: bool = False
) -> DeployedModel:
    """Script a neural mixer and verify CP2K's call signature works.

    **Experimental. This is a documented negative result**, and exporting one
    requires saying so: neural mixing learns the DIIS coefficient vector, and
    native DIIS already computes the exact minimiser of the form it is scored
    on (``docs/ml_pulay_objective.md``). A perfect model ties; an imperfect one
    loses. Measured: 11 -> 15 -> 37 SCF steps as more proposals were accepted.

    The opt-in exists because the failure is invisible from the outside -- a
    run with a neural mixer converges, just slower -- so nothing else would
    stop this being wired into a production configuration by habit.
    """

    if not experimental:
        raise DeployError(
            "export_mixer is experimental and not a production backend: neural mixing "
            "cannot beat native DIIS, which already solves min c^T B c exactly (see "
            "docs/ml_pulay_objective.md). Pass experimental=True to export one anyway. "
            "For acceleration use export_density_guess() and the adaptive controller."
        )

    torch = require_torch()

    from tools.scf.experimental.neural_mixer.model import load_mixer_checkpoint

    source = Path(checkpoint_path)
    model, config = load_mixer_checkpoint(source)

    try:
        scripted = torch.jit.script(model)
    except Exception as exc:
        raise DeployError(f"the mixer could not be scripted: {exc}") from exc

    # Exactly what phal_ml_mix passes: (nhist, nspin, ndim) float64 tensors.
    probe_history = torch.randn(3, 1, 16, dtype=torch.float64)
    probe_residual = torch.randn(3, 1, 16, dtype=torch.float64)
    scripted = scripted.double()
    try:
        with torch.no_grad():
            output = scripted(probe_history, probe_residual)
    except Exception as exc:
        raise DeployError(f"the scripted mixer rejected CP2K's call signature: {exc}") from exc

    if tuple(output.shape) != (1, 16):
        raise DeployError(
            f"the mixer returned shape {tuple(output.shape)}; CP2K expects (nspin, ndim) = (1, 16)"
        )
    if not bool(torch.isfinite(output).all()):
        raise DeployError("the mixer produced non-finite values on a probe input")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(target))

    return _finish(
        kind="mixer",
        target=target,
        source=source,
        environment_variable=MIXER_MODEL_ENV,
        metadata={"max_history": config.max_history, "architecture": config.architecture},
    )


__all__ = ["MIXER_MODEL_ENV", "export_mixer"]
