"""CP2K output parsing and validation on PHAL's toy fixture and real CP2K 2026.2 outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from ase import units

from gmd_active_learning.adapters.dft.cp2k.config import FailureClass, ValidationSettings
from gmd_active_learning.adapters.dft.cp2k.parse import (
    LEGACY_PHAL_ENERGY,
    parse_cp2k_forces,
    parse_cp2k_output,
    parse_cp2k_output_file,
    read_last_xyz_frame,
)
from gmd_active_learning.adapters.dft.cp2k.validation import (
    classify_failure,
    output_is_complete,
    validation_reasons,
)
from gmd_active_learning.core.exceptions import LabelingError

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "cp2k"
REAL = FIXTURES / "real_2026.2"
CONVERSION = units.Hartree / units.Bohr
DEFAULT = ValidationSettings()


def _raw_force_rows(path: Path) -> np.ndarray:
    rows = [line.split()[2:5] for line in path.read_text().splitlines() if re.match(r"^\s*FORCES\|\s+\d+\s", line)]
    return np.asarray(rows, dtype=float)


# -- PHAL toy fixture (FIX-003) ----------------------------------------------------------------------
def test_toy_fixture_forces_match_phal_conversion() -> None:
    forces = parse_cp2k_forces(FIXTURES / "toy_forces.out")
    expected = np.array([[0.1, 0.0, -0.1], [0.05, 0.025, 0.0125]]) * CONVERSION
    np.testing.assert_allclose(forces, expected, rtol=1e-12)
    parsed = parse_cp2k_output_file(FIXTURES / "toy_forces.out")
    assert parsed.energy is None and not parsed.scf_converged  # PHAL's energy parser also returns None here
    assert validation_reasons(parsed, n_atoms=2, settings=DEFAULT)  # no convergence marker: never a valid label


def test_truncated_toy_fixture_is_rejected(tmp_path: Path) -> None:
    lines = (FIXTURES / "toy_forces.out").read_text().splitlines()
    truncated = tmp_path / "truncated.out"
    truncated.write_text("\n".join(lines[:4]) + "\n")  # second atom row missing
    parsed = parse_cp2k_output_file(truncated)
    assert parsed.n_force_rows == 1
    reasons = validation_reasons(parsed, n_atoms=2, settings=DEFAULT)
    assert any("1 rows for a region of 2 atoms" in reason for reason in reasons)
    (tmp_path / "empty.out").write_text("CP2K output without an ATOMIC FORCES section\n")
    with pytest.raises(LabelingError, match="No CP2K force block"):
        parse_cp2k_forces(tmp_path / "empty.out")


def _force_block(first_x: str) -> list[str]:
    return [
        " FORCES| Atomic forces [hartree/bohr]",
        " FORCES|   Atom     x               y               z               |f|",
        f" FORCES|      1 {first_x}  2.00000000E-01 -3.00000000E-01   3.74165739E-01",
        " FORCES|      2  4.00000000E-01 -5.00000000E-01  6.00000000E-01   8.77496439E-01",
        " FORCES| Sum     3.00000000E-01 -3.00000000E-01  3.00000000E-01",
    ]


def test_force_eval_print_and_last_block_wins(tmp_path: Path) -> None:
    path = tmp_path / "two_blocks.out"
    path.write_text("\n".join([*_force_block("-9.00000000E-01"), "", *_force_block("-1.00000000E-01")]) + "\n")
    forces = parse_cp2k_forces(path)
    np.testing.assert_allclose(forces, np.array([[-0.1, 0.2, -0.3], [0.4, -0.5, 0.6]]) * CONVERSION, rtol=1e-12)


def test_phal_style_synthetic_output_fields() -> None:
    text = "\n".join(
        [
            " CP2K| version string: CP2K version 2025.2",
            " *** SCF run converged in 7 steps ***",
            " ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:  -1.250000",
            " STRESS| Analytical stress tensor [GPa]",
            " STRESS| X  1.0 0.1 0.2",
            " STRESS| Y  0.1 2.0 0.3",
            " STRESS| Z  0.2 0.3 3.0",
            " FORCES| Atomic forces [hartree/bohr]",
            " FORCES| 1  0.1 0.2 0.3 0.4",
            " FORCES| Sum 0.1 0.2 0.3",
            " ML-SCF| accelerator enabled",
            " ML-SCF| fallback used",
        ]
    )
    parsed = parse_cp2k_output(text)
    assert parsed.energy == pytest.approx(-1.25 * units.Hartree)
    assert parsed.scf_steps == 7 and parsed.version == "2025.2"
    np.testing.assert_allclose(parsed.stress, -np.array([[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]]) * units.GPa)
    assert parsed.ml_scf_accelerator and parsed.ml_scf_fallback
    assert json.loads(json.dumps(parsed.provenance()))["energy_ev"] == parsed.energy


# -- real CP2K 2026.2 outputs (FIX-003 addition) -------------------------------------------------------
def test_f12_phal_energy_regex_misses_real_2026_output() -> None:
    text = (REAL / "h2_sp" / "sp.out").read_text()
    assert "ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]" in text
    assert LEGACY_PHAL_ENERGY.search(text) is None  # PHAL lost every energy on 2026.2 (F-12)
    raw = float(re.search(r"ENERGY\| Total FORCE_EVAL \( QS \) energy \[hartree\]\s+(\S+)", text).group(1))
    assert parse_cp2k_output(text).energy == pytest.approx(raw * units.Hartree, rel=1e-15)


@pytest.mark.parametrize("case, n_atoms", [("h2_sp", 2), ("h2_periodic_sp", 2), ("h2_slab_sp", 2), ("ch3_capped_sp", 5)])
def test_real_converged_outputs_are_valid(case: str, n_atoms: int) -> None:
    path = next((REAL / case).glob("*.out"))
    parsed = parse_cp2k_output_file(path)
    assert parsed.scf_converged and parsed.normal_termination and not parsed.aborted
    assert parsed.version == "2026.2" and parsed.git_revision == "c92cc08"
    assert parsed.scf_steps is not None and parsed.scf_steps > 0
    assert parsed.scf_residual is not None and parsed.scf_residual < 1e-5
    assert np.isfinite(parsed.energy) and parsed.energy < 0
    np.testing.assert_allclose(parsed.forces, _raw_force_rows(path) * CONVERSION, rtol=1e-12)
    assert validation_reasons(parsed, n_atoms=n_atoms, settings=DEFAULT) == []
    assert output_is_complete(path.read_text())
    assert validation_reasons(parsed, n_atoms=n_atoms + 1, settings=DEFAULT)  # atom-count check


def test_real_periodic_stress_is_converted_from_bar_with_ase_sign() -> None:
    parsed = parse_cp2k_output_file(REAL / "h2_periodic_sp" / "sp.out")
    assert parsed.stress_unit == "bar"
    raw = np.asarray(parsed.stress_raw)
    np.testing.assert_allclose(parsed.stress, -raw * units.bar, rtol=1e-12)
    np.testing.assert_allclose(parsed.stress, parsed.stress.T, atol=1e-15)
    assert parse_cp2k_output_file(REAL / "h2_sp" / "sp.out").stress is None


def test_real_unconverged_output_is_invalid_with_reason() -> None:
    path = REAL / "h2_unconverged" / "sp.out"
    assert (REAL / "h2_unconverged" / "exit_code").read_text().strip() == "1"
    parsed = parse_cp2k_output_file(path)
    assert not parsed.scf_converged and parsed.scf_not_converged_marker and parsed.aborted
    assert parsed.abort_message.startswith("SCF run NOT converged")
    reasons = validation_reasons(parsed, n_atoms=2, settings=DEFAULT)
    assert any("SCF did not converge" in r for r in reasons)
    assert any("CP2K aborted" in r for r in reasons)
    assert not output_is_complete(path.read_text())
    assert classify_failure(parsed, stage="sp", exit_code=1) is FailureClass.SCF_NOT_CONVERGED


def test_real_geo_opt_markers_and_trajectory() -> None:
    done = parse_cp2k_output_file(REAL / "ch3_capped_hopt" / "hopt.out")
    assert done.geo_opt_completed and not done.geo_opt_max_iter_reached and done.scf_converged
    limited = parse_cp2k_output_file(REAL / "ch3_capped_hopt_maxiter" / "hopt.out")
    assert limited.geo_opt_max_iter_reached and not limited.geo_opt_completed
    assert classify_failure(limited, stage="hopt", exit_code=0) is FailureClass.GEO_OPT_NOT_CONVERGED
    symbols, positions = read_last_xyz_frame(REAL / "ch3_capped_hopt" / "hopt-pos-1.xyz")
    manifest = json.loads((REAL / "manifest.json").read_text())
    offset = np.asarray(manifest["cases"]["ch3_capped_hopt"]["offset"])
    region = json.loads((FIXTURES.parent / "structures" / "regions" / "ch3_capped.json").read_text())
    original = np.asarray(region["positions"])
    relaxed = positions - offset
    assert symbols == ["C", "H", "H", "H", "H"]
    np.testing.assert_allclose(relaxed[:4], original[:4], atol=1e-6)  # FIXED_ATOMS respected by CP2K
    assert 0.0 < np.linalg.norm(relaxed[4] - original[4]) < 0.2  # only the cap moved


def test_truncated_trajectory_frame_is_ignored(tmp_path: Path) -> None:
    lines = (REAL / "ch3_capped_hopt" / "hopt-pos-1.xyz").read_text().splitlines()
    frame = 2 + 5
    n_frames = len(lines) // frame
    assert n_frames >= 2
    (tmp_path / "complete.xyz").write_text("\n".join(lines[: (n_frames - 1) * frame]) + "\n")
    (tmp_path / "truncated.xyz").write_text("\n".join(lines[: n_frames * frame - 2]) + "\n")
    _, expected = read_last_xyz_frame(tmp_path / "complete.xyz")
    _, positions = read_last_xyz_frame(tmp_path / "truncated.xyz")
    np.testing.assert_array_equal(positions, expected)


def test_real_manifest_hashes_match_files() -> None:
    from gmd_active_learning.contracts.ids import file_sha256

    manifest = json.loads((REAL / "manifest.json").read_text())
    assert any("2026.2" in line for line in manifest["cp2k_version"])
    for case, entry in manifest["cases"].items():
        for name, digest in entry["files"].items():
            assert file_sha256(REAL / case / name) == digest, f"{case}/{name} changed"


def test_validation_switches() -> None:
    parsed = parse_cp2k_output_file(FIXTURES / "toy_forces.out")
    relaxed = ValidationSettings(require_scf_convergence=False, require_energy=False, require_normal_termination=False)
    assert validation_reasons(parsed, n_atoms=2, settings=relaxed) == []
    assert validation_reasons(None, n_atoms=2, settings=DEFAULT) == ["CP2K output file was not produced"]


def test_non_finite_values_are_invalid() -> None:
    text = (REAL / "h2_sp" / "sp.out").read_text()
    bad = re.sub(r"(FORCES\|\s+1\s+)\S+", r"\1NaN", text)
    parsed = parse_cp2k_output(bad.replace("NaN", "nan"))
    # A NaN does not even match the numeric row pattern, so the atom count check fails closed.
    assert validation_reasons(parsed, n_atoms=2, settings=DEFAULT)


def test_failure_classification_table() -> None:
    converged = parse_cp2k_output_file(REAL / "h2_sp" / "sp.out")
    assert classify_failure(None, stage="sp", exit_code=None) is FailureClass.EXECUTION_FAILED
    assert classify_failure(converged, stage="sp", exit_code=0, cancelled=True) is FailureClass.CANCELLED
    assert classify_failure(converged, stage="sp", exit_code=0, walltime_exceeded=True) is FailureClass.WALLTIME
    assert classify_failure(converged, stage="sp", exit_code=137) is FailureClass.EXECUTION_FAILED
    assert classify_failure(converged, stage="sp", exit_code=0) is FailureClass.INVALID_RESULT
    aborted = parse_cp2k_output(" * [ABORT]\n *  \\___/  Unknown basis set\n PROGRAM ENDED AT\n")
    assert classify_failure(aborted, stage="sp", exit_code=1) is FailureClass.INVALID_RESULT


REF = Path(__file__).resolve().parents[3] / "ref" / "lammps_monitor"


@pytest.mark.skipif(not (REF / "src" / "hotspot_al").is_dir(), reason="ref/lammps_monitor not present")
def test_force_parsing_and_completion_match_phal_on_every_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(REF / "src"))
    from hotspot_al.backends.cp2k import CP2KBackend as PhalBackend
    from hotspot_al.cp2k.cp2k_force_parser import parse_cp2k_forces as phal_forces
    from hotspot_al.cp2k.cp2k_result import parse_cp2k_result as phal_result

    outputs = [FIXTURES / "toy_forces.out", *sorted(REAL.glob("*/*.out"))]
    for path in outputs:
        text = path.read_text()
        assert output_is_complete(text) == PhalBackend().output_is_complete(text), path
        ours = parse_cp2k_output_file(path)
        if ours.forces is None:
            with pytest.raises(Exception):
                phal_forces(path)
            continue
        np.testing.assert_allclose(ours.forces, phal_forces(path), rtol=1e-12)
        legacy = phal_result(path)
        assert legacy.scf_converged == ours.scf_converged and legacy.scf_iterations == ours.scf_steps
        if path.parent.name.startswith(("h2", "ch3")):
            assert legacy.energy is None and ours.energy is not None  # F-12, fixed here
