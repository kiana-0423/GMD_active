# Ported from PHAL (hotspot_al.cp2k_ml_scf.deploy) at 96263b6; MIT License, see NOTICE.
"""Export trained models into the form CP2K's fork loads (TorchScript).

CP2K calls the models in-process through ``phal_ml_torch.cpp``, which uses
``torch::jit::load``.  A Python checkpoint is therefore not deployable as it
stands: it has to be scripted first, and the scripted module has to accept the
exact tensor signature the C++ shim passes.

Export verifies that signature by calling the scripted module the way CP2K
will.  A model that scripts cleanly but cannot be called with CP2K's arguments
would fail at the worst possible moment -- inside a running calculation, where
the only symptom is a silent fallback -- so the check happens here instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gmd_active_learning.adapters.dft.scf._torch import model_digest, require_torch

DEPLOY_VERSION = "phal-scf-deploy-v1"

#: Environment variables the CP2K fork reads to find deployed models.
GUESS_MODEL_ENV = "PHAL_ML_SCF_GUESS_MODEL"
ML_ENABLED_ENV = "PHAL_ML_SCF_ENABLED"
MIXER_MODEL_ENV = "PHAL_ML_SCF_MIXER_MODEL"


class DeployError(RuntimeError):
    """Raised when a model cannot be exported in a form CP2K can load."""


@dataclass(frozen=True, slots=True)
class DeployedModel:
    """A model exported for CP2K, with the provenance to identify it."""

    kind: str
    path: Path
    digest: str
    source_checkpoint: Path
    environment_variable: str
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "deploy_version": DEPLOY_VERSION,
            "kind": self.kind,
            "path": str(self.path),
            "model_digest": self.digest,
            "source_checkpoint": str(self.source_checkpoint),
            "environment_variable": self.environment_variable,
            "metadata": dict(self.metadata),
        }


def export_density_guess(checkpoint_path: str | Path, output_path: str | Path) -> DeployedModel:
    """Script a density model for CP2K's in-process ``SCF_GUESS ML``.

    The scripted wrapper takes CP2K's arguments -- atomic numbers, positions,
    cell, charge, multiplicity -- and returns ``P`` shaped ``(nao, nao, nspin)``
    in Fortran order, which is what ``phal_ml_initial_guess`` copies out.
    """

    torch = require_torch()

    from gmd_active_learning.adapters.dft.scf.density.gnn_model import (
        AOBasis,
        DensityGNNConfig,
        ValenceTable,
        build_density_model,
    )
    from gmd_active_learning.adapters.dft.scf.density.gnn_model import load_density_checkpoint

    source = Path(checkpoint_path)
    checkpoint = load_density_checkpoint(source)
    basis = AOBasis.from_dict(checkpoint["basis"])

    # A checkpoint with no valence table cannot be deployed. The only thing an
    # export could fall back on is the nuclear charge, which is wrong for every
    # pseudopotential calculation -- and wrong quietly, because the resulting
    # density is rejected by CP2K and the run silently uses the atomic guess.
    payload = checkpoint.get("valence")
    if not payload:
        raise DeployError(
            f"{source}: the checkpoint carries no valence electron table, so a deployed "
            "model would have to normalise to sum(Z). That is the nuclear charge, not the "
            "electron count CP2K works with (H2O: 10 against 8 under GTH), and CP2K rejects "
            "such a density. Re-train with a current version, which fits the table from the "
            "dataset."
        )
    valence = ValenceTable.from_dict(payload)
    if valence.is_nuclear_charge():
        raise DeployError(
            f"{source}: every valence count equals the nuclear charge. That is correct only "
            "for an all-electron calculation; under pseudopotentials it means the table was "
            "filled with Z and the model will normalise to the wrong electron count."
        )

    model = build_density_model(DensityGNNConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    wrapper = _build_guess_wrapper(torch, model, basis, valence)
    try:
        scripted = torch.jit.script(wrapper)
    except Exception as exc:
        raise DeployError(f"the density model could not be scripted: {exc}") from exc

    _probe_guess(torch, scripted, basis, valence)

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(target))

    return _finish(
        kind="density_guess",
        target=target,
        source=source,
        environment_variable=GUESS_MODEL_ENV,
        metadata={
            "basis_set": basis.basis_set,
            "ao_per_element": {str(key): value for key, value in sorted(basis.ao_per_element.items())},
            "representation": "ao-nonorthogonal",
            "electron_count_method": "trace_ps",
            # Recorded so a deployed model can be checked against the run it is
            # used in without re-opening the checkpoint.
            "normalization": "valence_electrons",
            "valence_per_element": {
                str(key): value for key, value in sorted(valence.valence_per_element.items())
            },
            "potential_family": valence.potential_family,
            "requires_overlap": True,
            "note": (
                "CP2K must be built with __PHAL_ML_SCF and linked against libtorch for the "
                "in-process guess; otherwise the pre-computed density file path is used. "
                "The wrapper takes the AO overlap S as its sixth argument and normalises to "
                "Tr(P S) = Ne; it cannot be called without one."
            ),
        },
    )


def _probe_guess(torch: Any, scripted: Any, basis: Any, valence: Any) -> None:
    """Call the scripted guess once and check what CP2K will check.

    The properties are the ones ``ml_validate_density`` enforces on the Fortran
    side: Hermitian, finite, and an electron count of ``Tr(P S)``. Failing here
    costs an export; failing in CP2K costs a silent fallback in the middle of a
    production run.

    The probe composition is chosen so that the valence count and the nuclear
    charge *differ* wherever the table allows it. A probe of hydrogen alone
    would pass either way -- H has one valence electron and Z = 1 -- and would
    therefore certify exactly the bug this check exists to catch.
    """

    numbers = _probe_composition(basis, valence)
    n_atoms = len(numbers)
    n_ao = sum(int(basis.ao_per_element[number]) for number in numbers)

    atomic_numbers = torch.as_tensor([numbers], dtype=torch.long)
    positions = torch.zeros((1, n_atoms, 3), dtype=torch.float32)
    for index in range(1, n_atoms):
        positions[0, index, 0] = 1.4 * index
    cell = torch.eye(3, dtype=torch.float32).unsqueeze(0) * 10.0
    charge = torch.zeros(1, dtype=torch.float32)
    multiplicity = torch.ones(1, dtype=torch.float32)

    # A deliberately non-orthonormal probe overlap: unit diagonal with real
    # off-diagonal weight. An identity here would let a model that normalises
    # to Tr(P) pass, which is the other half of the bug this probe exists to
    # catch.
    off = 0.25
    s = torch.full((n_ao, n_ao), off, dtype=torch.float32)
    s.fill_diagonal_(1.0)
    overlap = s.unsqueeze(0)

    try:
        output = scripted(atomic_numbers, positions, cell, charge, multiplicity, overlap)
    except Exception as exc:  # noqa: BLE001 - reported as a deploy failure
        raise DeployError(f"the density model failed on a probe input: {exc}") from exc

    if not bool(torch.isfinite(output).all()):
        raise DeployError("the density model produced non-finite values on a probe input")

    # CP2K reads (nao, nao, nspin).
    if output.shape[0] != n_ao or output.shape[1] != n_ao:
        raise DeployError(
            f"the density model returned {tuple(output.shape)} for a {n_ao}-orbital probe"
        )

    matrix = output.to(torch.float64)
    asymmetry = float((matrix - matrix.transpose(0, 1)).abs().max())
    if asymmetry > 1e-8:
        raise DeployError(f"the density model is not Hermitian on a probe input ({asymmetry:.2e})")

    expected = valence.electrons(numbers)
    nuclear = float(sum(numbers))
    trace_ps = float((matrix * overlap.to(torch.float64).squeeze(0).unsqueeze(-1)).sum())

    if abs(trace_ps - expected) > 1.0e-3 * max(1.0, abs(expected)):
        detail = ""
        if abs(trace_ps - nuclear) <= 1.0e-3 * max(1.0, abs(nuclear)) and nuclear != expected:
            detail = (
                f" It normalises to sum(Z) = {nuclear:.6f}, the nuclear charge. Under "
                "pseudopotentials the density holds valence electrons only, so this model "
                "would be rejected by CP2K on every system."
            )
        raise DeployError(
            f"the density model normalises to Tr(P S) = {trace_ps:.6f}, not {expected:.6f} "
            f"for the probe composition {numbers}.{detail}"
        )


def _probe_composition(basis: Any, valence: Any) -> list[int]:
    """A probe whose valence count differs from its nuclear charge, if possible.

    Prefers the element with the largest ``Z - valence`` gap, because that is
    where a nuclear-charge normalisation is most visible. Falls back to any
    element the model knows when no such element exists (an all-electron
    table), in which case the two counts genuinely coincide.
    """

    shared = sorted(set(basis.ao_per_element) & set(valence.valence_per_element))
    if not shared:
        raise DeployError(
            "the model's AO table and valence table have no element in common; "
            "the checkpoint is inconsistent"
        )

    def gap(number: int) -> int:
        return int(number) - int(valence.valence_per_element[number])

    best = max(shared, key=gap)
    if gap(best) > 0:
        # Two atoms: exercises the pair blocks as well as the diagonal ones.
        return [best, best]
    return [shared[0], shared[0]]


def _build_guess_wrapper(torch: Any, model: Any, basis: Any, valence: Any) -> Any:
    """Wrap the density GNN in the signature CP2K's C++ shim calls."""

    nn = torch.nn
    from torch import Tensor

    # Scripting cannot look elements up in a Python dict at call time, so both
    # tables are baked into tensors indexed by atomic number.
    max_z = max(max(basis.ao_per_element), max(valence.valence_per_element)) + 1
    ao_table = torch.zeros(max_z, dtype=torch.long)
    for number, count in basis.ao_per_element.items():
        ao_table[number] = int(count)

    # Valence electrons per element, NOT the nuclear charge. Under GTH
    # pseudopotentials water carries 8 electrons and not 10; a model normalised
    # to sum(Z) is 25% wrong there and 136% wrong for Cu2O, and CP2K rejects it
    # against a 1e-3 tolerance -- which reads as "the model failed" rather than
    # "the model was normalised against the wrong quantity".
    valence_table = torch.zeros(max_z, dtype=torch.long)
    for number, count in valence.valence_per_element.items():
        valence_table[number] = int(count)

    class GuessWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.register_buffer("ao_table", ao_table)
            self.register_buffer("valence_table", valence_table)

        def forward(
            self,
            atomic_numbers: Tensor,
            positions: Tensor,
            cell: Tensor,
            charge: Tensor,
            multiplicity: Tensor,
            overlap: Tensor,
        ) -> Tensor:
            n_atoms = atomic_numbers.shape[1]
            counts = self.ao_table[atomic_numbers[0]]
            n_ao = int(counts.sum().item())

            atom_index = torch.repeat_interleave(
                torch.arange(n_atoms, device=counts.device), counts
            ).unsqueeze(0)
            slots = torch.cat([torch.arange(int(counts[i].item())) for i in range(n_atoms)]).unsqueeze(0)

            spin_channels = 2 if int(multiplicity[0].item()) > 1 else 1
            # Ne = valence electrons - charge. This is the count CP2K holds the
            # prediction to, and it is not sum(Z).
            electrons = float(self.valence_table[atomic_numbers[0]].sum().item()) - float(
                charge[0].item()
            )
            unpaired = float(multiplicity[0].item()) - 1.0
            if spin_channels == 1:
                channel_electrons = torch.tensor([[electrons]], dtype=torch.float32)
            else:
                channel_electrons = torch.tensor(
                    [[(electrons + unpaired) / 2.0, (electrons - unpaired) / 2.0]], dtype=torch.float32
                )

            globals_ = torch.tensor(
                [
                    [
                        float(charge[0].item()),
                        unpaired,
                        electrons / float(n_atoms),
                        float(spin_channels),
                    ]
                ],
                dtype=torch.float32,
            )

            raw = self.model(
                atomic_numbers,
                positions.to(torch.float32),
                torch.ones((1, n_atoms), dtype=torch.float32),
                globals_,
                atom_index,
                slots,
                torch.ones((1, n_ao), dtype=torch.float32),
            )
            # P = P† is imposed, not learned.
            matrix = 0.5 * (raw[:, :spin_channels] + raw[:, :spin_channels].transpose(-1, -2))

            # Ne = Tr(P S). The overlap comes from CP2K; it used to be assumed
            # to be the identity here, which silently normalised to Tr(P) = Ne
            # and produced a density carrying the wrong number of electrons in
            # every real (non-orthonormal) basis. S is symmetric, so the
            # elementwise product is the full contraction.
            s = overlap.to(matrix.dtype)
            traces = (matrix * s.unsqueeze(1)).sum(dim=(-1, -2))
            scale = channel_electrons / torch.where(
                traces.abs() > 1e-8, traces, torch.ones_like(traces)
            )
            matrix = matrix * scale[:, :, None, None]

            # CP2K reads (nao, nao, nspin) in Fortran order; the C++ shim copies
            # the buffer verbatim, so the layout is produced here.
            return matrix[0].permute(1, 2, 0).to(torch.float64).contiguous()

    return GuessWrapper()


def _finish(
    *,
    kind: str,
    target: Path,
    source: Path,
    environment_variable: str,
    metadata: Mapping[str, Any],
) -> DeployedModel:
    digest = model_digest(target)
    deployed = DeployedModel(
        kind=kind,
        path=target,
        digest=digest,
        source_checkpoint=source,
        environment_variable=environment_variable,
        metadata=dict(metadata),
    )
    manifest = target.with_suffix(target.suffix + ".deploy.json")
    manifest.write_text(json.dumps(deployed.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return deployed


def _deployment_manifest(source: Path) -> dict[str, Any]:
    """The manifest written beside a deployed model."""

    manifest_path = source.with_suffix(source.suffix + ".deploy.json")
    if not manifest_path.is_file():
        raise DeployError(
            f"{manifest_path}: missing deployment manifest, so the model cannot be verified; "
            "re-export it with export_density_guess()"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["_path"] = str(manifest_path)
    return payload


def _probe_n_ao(manifest: Mapping[str, Any], atomic_numbers: list[int]) -> int:
    """AO count for a probe composition, from the deployment manifest."""

    table = (manifest.get("metadata") or {}).get("ao_per_element") or {}
    if not table:
        raise DeployError(f"{manifest['_path']}: manifest carries no ao_per_element table")
    try:
        return int(sum(int(table[str(number)]) for number in atomic_numbers))
    except KeyError as exc:
        raise DeployError(
            f"{manifest['_path']}: the model's basis has no entry for element {exc}; "
            "verification needs one for the probe composition"
        ) from exc


def _probe_electrons(manifest: Mapping[str, Any], atomic_numbers: list[int]) -> float:
    """``Ne`` for a probe composition, from the manifest's valence table.

    A manifest without one is refused rather than defaulted: the only available
    default is the nuclear charge, and accepting it here would let exactly the
    models this check exists to reject pass verification.
    """

    metadata = manifest.get("metadata") or {}
    normalization = metadata.get("normalization")
    table = metadata.get("valence_per_element") or {}
    if not table:
        raise DeployError(
            f"{manifest['_path']}: manifest carries no valence_per_element table, so the "
            "model was exported before electron counts were pinned and normalises to sum(Z). "
            "Re-export it."
        )
    if normalization != "valence_electrons":
        raise DeployError(
            f"{manifest['_path']}: normalization is {normalization!r}, expected "
            "'valence_electrons'"
        )
    try:
        return float(sum(int(table[str(number)]) for number in atomic_numbers))
    except KeyError as exc:
        raise DeployError(
            f"{manifest['_path']}: the model's valence table has no entry for element {exc}"
        ) from exc


def deployment_environment(
    *,
    guess_model: str | Path | None = None,
    mixer_model: str | Path | None = None,
) -> dict[str, str]:
    """The environment CP2K needs to find deployed models."""

    environment: dict[str, str] = {}
    if guess_model is not None:
        environment[GUESS_MODEL_ENV] = str(guess_model)
    if mixer_model is not None:
        environment[MIXER_MODEL_ENV] = str(mixer_model)
    if environment:
        # The master switch. Without it CP2K honours SCF_GUESS ML by falling
        # back to ATOMIC and reporting it -- a working model, a correct run,
        # and no acceleration, which is the least debuggable combination there
        # is. Anything that ships a model path ships this too.
        environment[ML_ENABLED_ENV] = "1"
    return environment


def verify_deployed(path: str | Path, kind: str) -> dict[str, Any]:
    """Load a deployed model back and confirm it still answers CP2K's call."""

    torch = require_torch()
    source = Path(path)
    if not source.is_file():
        raise DeployError(f"deployed model not found: {source}")

    try:
        module = torch.jit.load(str(source))
    except Exception as exc:
        raise DeployError(f"{source}: TorchScript load failed ({exc})") from exc

    if kind == "mixer":
        with torch.no_grad():
            output = module(
                torch.randn(3, 1, 12, dtype=torch.float64), torch.randn(3, 1, 12, dtype=torch.float64)
            )
        shape = tuple(output.shape)
        if shape != (1, 12):
            raise DeployError(f"{source}: mixer returned {shape}, expected (1, 12)")
    elif kind == "density_guess":
        # The guess takes CP2K's overlap as its sixth argument (protocol v2),
        # so verification has to supply one of the right size. The AO count
        # comes from the manifest written beside the model: the wrapper derives
        # nao from its own baked table, and there is no way to ask it.
        # Water: sum(Z) = 10, valence electrons under GTH = 8. A model that
        # passes this probe cannot be normalising to the nuclear charge.
        probe_numbers = [8, 1, 1]
        manifest = _deployment_manifest(source)
        n_ao = _probe_n_ao(manifest, probe_numbers)
        expected_electrons = _probe_electrons(manifest, probe_numbers)
        overlap = torch.full((n_ao, n_ao), 0.05, dtype=torch.float64)
        overlap.fill_diagonal_(1.0)
        with torch.no_grad():
            output = module(
                torch.as_tensor([probe_numbers], dtype=torch.long),
                torch.as_tensor(
                    np.asarray([[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]]),
                    dtype=torch.float64,
                ),
                torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0,
                torch.zeros(1, dtype=torch.float64),
                torch.ones(1, dtype=torch.float64),
                overlap.unsqueeze(0),
            )
        shape = tuple(output.shape)
        if len(shape) != 3 or shape[0] != shape[1]:
            raise DeployError(f"{source}: guess returned {shape}, expected (nao, nao, nspin)")

        matrix = output.to(torch.float64)
        trace_ps = float((matrix * overlap.unsqueeze(-1)).sum())
        nuclear = float(sum(probe_numbers))
        if abs(trace_ps - expected_electrons) > 1.0e-3 * max(1.0, expected_electrons):
            detail = ""
            if abs(trace_ps - nuclear) <= 1.0e-3 * nuclear and nuclear != expected_electrons:
                detail = (
                    f" It normalises to sum(Z) = {nuclear:.3f}: the nuclear charge, not the "
                    "electron count. CP2K would reject this density on every system."
                )
            raise DeployError(
                f"{source}: guess normalises to Tr(P S) = {trace_ps:.6f}, expected "
                f"{expected_electrons:.6f} for H2O.{detail}"
            )
    else:
        raise DeployError(f"unknown deployed model kind {kind!r}")

    return {"path": str(source), "kind": kind, "digest": model_digest(source), "output_shape": list(shape)}


__all__ = [
    "DEPLOY_VERSION",
    "GUESS_MODEL_ENV",
    "MIXER_MODEL_ENV",
    "DeployError",
    "DeployedModel",
    "deployment_environment",
    "export_density_guess",
    "verify_deployed",
]
