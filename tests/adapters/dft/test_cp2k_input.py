"""CP2K input generation from ExtractedRegion (no CP2K needed)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gmd_active_learning.adapters.dft.cp2k.backend import CP2KBackend
from gmd_active_learning.adapters.dft.cp2k.command import build_cp2k_command, cp2k_environment
from gmd_active_learning.adapters.dft.cp2k.config import CP2KConfig, RetryStep
from gmd_active_learning.adapters.dft.cp2k.input import (
    build_cp2k_input,
    periodicity,
    poisson_solver,
    prepare_geometry,
    settings_with_step,
    write_cp2k_inputs,
)
from gmd_active_learning.core.exceptions import (
    ConfigurationError,
    LabelingError,
    UnsupportedCombinationError,
)

REF = Path(__file__).resolve().parents[3] / "ref" / "lammps_monitor"


def test_single_point_has_required_sections(load_region) -> None:
    text = build_cp2k_input(load_region("h2_cluster"), CP2KConfig()).text
    for needle in (
        "RUN_TYPE ENERGY_FORCE",
        "METHOD Quickstep",
        "BASIS_SET_FILE_NAME BASIS_MOLOPT",
        "POTENTIAL_FILE_NAME GTH_POTENTIALS",
        "BASIS_SET DZVP-MOLOPT-SR-GTH",
        "POTENTIAL GTH-PBE",
        "&XC_FUNCTIONAL PBE",
        "CHARGE 0",
        "MULTIPLICITY 1",
        "  &PRINT\n    &FORCES MEDIUM",
    ):
        assert needle in text
    assert "&MOTION" not in text and "UKS" not in text and "STRESS_TENSOR" not in text


@pytest.mark.parametrize(
    "pbc, label", [((False,) * 3, "NONE"), ((True, True, False), "XY"), ((True, False, True), "XZ"), ((True,) * 3, "XYZ")]
)
def test_periodicity_label(pbc, label: str) -> None:
    assert periodicity(pbc) == label


@pytest.mark.parametrize(
    "name, periodic, solver",
    [("h2_cluster", "NONE", "WAVELET"), ("h2_slab", "XY", "MT"), ("h2_periodic", "XYZ", "PERIODIC")],
)
def test_boundary_conditions_follow_the_region_exactly(load_region, name: str, periodic: str, solver: str) -> None:
    built = build_cp2k_input(load_region(name), CP2KConfig())
    assert built.text.count(f"PERIODIC {periodic}\n") == 2  # &POISSON and &CELL agree
    assert f"POISSON_SOLVER {solver}" in built.text
    assert built.stress is (periodic == "XYZ")
    if periodic == "XYZ":
        assert "  STRESS_TENSOR ANALYTICAL" in built.text and "    &STRESS_TENSOR MEDIUM" in built.text


def test_unsupported_solver_for_periodicity_raises_instead_of_changing_bcs(load_region) -> None:
    config = CP2KConfig(poisson={"partially_periodic_solver": "WAVELET"})
    with pytest.raises(UnsupportedCombinationError, match="does not support PERIODIC XY"):
        build_cp2k_input(load_region("h2_slab"), config)
    assert poisson_solver("XY", CP2KConfig(poisson={"partially_periodic_solver": "ANALYTIC"})) == "ANALYTIC"
    assert poisson_solver("NONE", CP2KConfig(poisson={"non_periodic_solver": "MULTIPOLE"})) == "MULTIPOLE"
    with pytest.raises(UnsupportedCombinationError):
        poisson_solver("XY", CP2KConfig(poisson={"partially_periodic_solver": "WAVELET"}))


def test_stress_always_is_rejected_for_isolated_regions(load_region) -> None:
    with pytest.raises(UnsupportedCombinationError, match="meaningless"):
        build_cp2k_input(load_region("h2_cluster"), CP2KConfig(stress="always"))
    assert "STRESS_TENSOR" in build_cp2k_input(load_region("h2_slab"), CP2KConfig(stress="always")).text
    assert "STRESS_TENSOR" not in build_cp2k_input(load_region("h2_periodic"), CP2KConfig(stress="never")).text


def test_isolated_region_without_cell_gets_a_cubic_padded_box(load_region) -> None:
    region = load_region("ch3_capped")
    geometry = prepare_geometry(region, CP2KConfig(cluster_padding=5.0))
    assert geometry.rebuilt_cell
    lengths = np.diag(geometry.cell)
    assert np.allclose(lengths, lengths[0]) and np.allclose(geometry.cell, np.diag(lengths))
    assert np.allclose(geometry.to_region_frame(geometry.positions), region.positions)
    centre = (geometry.positions.min(axis=0) + geometry.positions.max(axis=0)) / 2
    assert np.allclose(centre, lengths / 2)


def test_non_cubic_cell_is_replaced_for_wavelet_but_kept_for_mt(load_region) -> None:
    region = replace(load_region("h2_cluster"), cell=np.diag([12.0, 12.0, 14.0]), positions=np.array([[6.0, 6, 7], [6.74, 6, 7]]))
    assert prepare_geometry(region, CP2KConfig()).rebuilt_cell  # WAVELET aborts on non-cubic cells
    kept = prepare_geometry(region, CP2KConfig(poisson={"non_periodic_solver": "MT"}))
    assert not kept.rebuilt_cell and np.allclose(kept.cell, region.cell)
    cubic = replace(region, cell=np.diag([12.0, 12.0, 12.0]), positions=np.array([[6.0, 6, 6], [6.74, 6, 6]]))
    assert not prepare_geometry(cubic, CP2KConfig()).rebuilt_cell


def test_slab_without_vacuum_is_refused(load_region) -> None:
    region = load_region("h2_slab")
    with pytest.raises(LabelingError, match="non-periodic Z direction"):
        prepare_geometry(region, CP2KConfig(min_vacuum=6.0))


def test_periodic_region_needs_a_cell(load_region) -> None:
    region = load_region("h2_periodic")
    with pytest.raises(LabelingError, match="non-degenerate cell"):
        prepare_geometry(replace(region, cell=np.zeros((3, 3))), CP2KConfig())


def test_odd_electron_region_is_open_shell(load_region) -> None:
    built = build_cp2k_input(load_region("ch3_radical"), CP2KConfig())
    assert "MULTIPLICITY 2" in built.text and "    UKS" in built.text
    assert built.electronic_state.n_electrons == 9


def test_charge_and_multiplicity_come_from_the_region(load_region) -> None:
    region = replace(load_region("ch3_radical"), charge=1, multiplicity=3)
    built = build_cp2k_input(region, CP2KConfig())
    assert "CHARGE 1" in built.text and "MULTIPLICITY 3" in built.text  # 8 electrons: triplet is consistent
    impossible = replace(load_region("ch3_radical"), multiplicity=1)
    assert "MULTIPLICITY 2" in build_cp2k_input(impossible, CP2KConfig(spin_policy="auto")).text
    with pytest.raises(ConfigurationError, match="impossible"):
        build_cp2k_input(impossible, CP2KConfig(spin_policy="strict"))


def test_uncapped_isolated_fragment_gate(load_region, caplog) -> None:
    region = load_region("ch3_radical")
    region.metadata.update({"h_capping_skipped": True, "h_capping_skipped_reason": "metal"})
    with pytest.raises(UnsupportedCombinationError, match="uncapped broken bonds"):
        build_cp2k_input(region, CP2KConfig())
    assert "CHARGE" in build_cp2k_input(region, CP2KConfig(uncapped_fragment_policy="warn")).text
    assert "uncapped broken bonds" in caplog.text
    build_cp2k_input(region, CP2KConfig(uncapped_fragment_policy="allow"))
    periodic = load_region("h2_periodic")
    periodic.metadata["h_capping_skipped"] = True
    build_cp2k_input(periodic, CP2KConfig())  # a periodic region passes the gate


def test_point_charge_embedding_is_refused(load_region) -> None:
    with pytest.raises(UnsupportedCombinationError, match="point-charge embedding"):
        build_cp2k_input(load_region("h2_cluster"), CP2KConfig(), embedding_enabled=True)
    region = load_region("h2_cluster")
    region.metadata["embedding"] = {"status": "ok", "point_charges": [{"position": [0, 0, 5], "charge": 0.1, "element": "O"}]}
    with pytest.raises(UnsupportedCombinationError):
        build_cp2k_input(region, CP2KConfig())
    region.metadata["embedding"] = {"enabled": False, "status": "disabled"}
    build_cp2k_input(region, CP2KConfig())


def test_backend_reads_embedding_switch_from_config(load_region) -> None:
    backend = CP2KBackend.from_config({"cp2k": {}, "embedding": {"enabled": True}})
    with pytest.raises(UnsupportedCombinationError):
        backend.build_input(load_region("h2_cluster"), stage="sp", project_name="sp")


def test_h_only_geo_opt_fixes_every_non_cap_atom(tmp_path: Path, load_region) -> None:
    written = write_cp2k_inputs(load_region("ch3_capped"), tmp_path, CP2KConfig(), job_name="job")
    assert set(written) == {"hopt_input", "single_point_input"}
    hopt = written["hopt_input"].read_text()
    assert "RUN_TYPE GEO_OPT" in hopt and "OPTIMIZER BFGS" in hopt and "MAX_ITER 50" in hopt
    assert "      LIST 1 2 3 4\n" in hopt and "COMPONENTS_TO_FIX XYZ" in hopt
    assert written["single_point_input"].name == "job_sp.inp"
    uncapped = write_cp2k_inputs(load_region("h2_cluster"), tmp_path / "b", CP2KConfig(), job_name="job")
    assert set(uncapped) == {"single_point_input"}
    disabled = write_cp2k_inputs(load_region("ch3_capped"), tmp_path / "c", CP2KConfig(h_only_opt={"enabled": False}))
    assert set(disabled) == {"single_point_input"}


def test_kinds_extra_sections_dispersion_and_file_names(load_region) -> None:
    config = CP2KConfig(
        kinds={"C": {"basis": "TZVP-MOLOPT-GTH", "potential": "GTH-PBE-q4"}},
        extra_sections={"global_lines": ["EXTENDED_FFT_LENGTHS"], "dft_lines": ["&PRINT", "&END PRINT"]},
        dispersion=True,
        basis_set_file="BASIS_MOLOPT_UZH",
    )
    text = build_cp2k_input(load_region("ch3_capped"), config).text
    assert "    &KIND C\n      BASIS_SET TZVP-MOLOPT-GTH\n      POTENTIAL GTH-PBE-q4" in text
    assert "    &KIND H\n      BASIS_SET DZVP-MOLOPT-SR-GTH" in text
    assert "  EXTENDED_FFT_LENGTHS\n&END GLOBAL" in text
    assert "TYPE DFTD3(BJ)" in text and "BASIS_SET_FILE_NAME BASIS_MOLOPT_UZH" in text
    with pytest.raises(ValueError):
        CP2KConfig(kinds={"Xx": {}})


def test_template_rendering(tmp_path: Path, load_region) -> None:
    template = tmp_path / "t.inp"
    template.write_text("# project $project ($run_type)\n$global_section\n$force_eval_section\n$motion_section\n")
    backend = CP2KBackend(CP2KConfig(template_path=str(template)))
    text = backend.build_input(load_region("h2_cluster"), stage="sp", project_name="sp").text
    assert text.startswith("# project sp (ENERGY_FORCE)\n&GLOBAL")
    plain_hash = CP2KBackend(CP2KConfig()).settings_hash()
    assert backend.settings_hash() != plain_hash  # template text is part of the identity
    template.write_text("$unknown_placeholder")
    with pytest.raises(ConfigurationError, match="placeholder"):
        CP2KBackend(CP2KConfig(template_path=str(template))).build_input(load_region("h2_cluster"), stage="sp", project_name="sp")


def test_settings_hash_is_deterministic_and_physics_only() -> None:
    base = CP2KConfig().settings_hash()
    assert base == CP2KConfig().settings_hash()
    assert CP2KConfig(cutoff=500).settings_hash() != base
    assert CP2KConfig(functional="BLYP").settings_hash() != base
    for unchanged in (
        {"executable": "/elsewhere/cp2k.psmp"},
        {"data_dir": "/data"},
        {"retry": {"max_retries": 7}},
        {"resources": {"ntasks": 64}},
        {"task_dir": "x"},
    ):
        assert CP2KConfig.model_validate(unchanged).settings_hash() == base


def test_retry_steps_apply_to_base_settings_without_compounding() -> None:
    base = CP2KConfig(max_scf=100, retry={"max_scf_cap": 250})
    tripled = settings_with_step(base, RetryStep(max_scf_multiplier=3.0))
    assert tripled.max_scf == 250  # 300 capped
    assert settings_with_step(base, RetryStep(max_scf_multiplier=2.0)).max_scf == 200
    nbuffer = settings_with_step(base, RetryStep(use_ot=False, mixing_nbuffer=4))
    assert nbuffer.max_scf == 100 and nbuffer.use_ot is False and nbuffer.mixing.nbuffer == 4
    assert base.max_scf == 100 and base.mixing.nbuffer is None  # base untouched
    longer = settings_with_step(base, RetryStep(h_opt_max_iter_multiplier=2.0))
    assert longer.h_only_opt.max_iter == 100
    with pytest.raises(ValueError):
        RetryStep(max_scf=10, max_scf_multiplier=2.0)


def test_diagonalization_emits_mixing_block(load_region) -> None:
    text = build_cp2k_input(load_region("h2_cluster"), CP2KConfig(use_ot=False, mixing={"nbuffer": 4})).text
    assert "&DIAGONALIZATION" in text and "METHOD BROYDEN_MIXING" in text and "NBUFFER 4" in text and "&OT" not in text


def test_command_launcher_and_environment() -> None:
    config = CP2KConfig(executable="cp2k.psmp", launcher=["mpiexec", "-n", "{ntasks}"], data_dir="/opt/cp2k/data", environment={"OMP_NUM_THREADS": "2"})
    from gmd_active_learning.adapters.schedulers.config import SchedulerResources

    assert build_cp2k_command(config, "sp.inp", "sp.out", resources=SchedulerResources(ntasks=8)) == [
        "mpiexec", "-n", "8", "cp2k.psmp", "-i", "sp.inp", "-o", "sp.out",
    ]
    assert cp2k_environment(config) == {"OMP_NUM_THREADS": "2", "CP2K_DATA_DIR": "/opt/cp2k/data"}
    with pytest.raises(ConfigurationError, match="placeholder"):
        build_cp2k_command(CP2KConfig(launcher=["srun", "{gpus}"]), "a", "b")


def test_execution_request_paths_are_work_dir_relative(tmp_path: Path) -> None:
    backend = CP2KBackend(CP2KConfig(executable="/opt/cp2k/bin/cp2k.psmp"))
    request = backend.execution_request(tmp_path / "tasks" / "job.inp", output_file=tmp_path / "tasks" / "job.out")
    assert request.command == ("/opt/cp2k/bin/cp2k.psmp", "-i", "job.inp", "-o", "job.out")
    assert request.work_dir == tmp_path / "tasks"


def test_disabled_backend_refuses_construction() -> None:
    with pytest.raises(ConfigurationError, match="switched off"):
        CP2KBackend(CP2KConfig(enabled=False))


# -- optional parity with PHAL 96263b6 (skipped once ref/ is removed) ------------------------------------
@pytest.mark.skipif(not (REF / "src" / "hotspot_al").is_dir(), reason="ref/lammps_monitor not present")
@pytest.mark.parametrize("name, periodic", [("ch3_capped", "NONE"), ("ch3_radical", "NONE"), ("h2_periodic", "XYZ")])
def test_inputs_are_byte_identical_to_phal_for_equivalent_settings(monkeypatch, load_region, name: str, periodic: str) -> None:
    monkeypatch.syspath_prepend(str(REF / "src"))
    from ase import Atoms
    from hotspot_al.config import load_config
    from hotspot_al.cp2k.cp2k_input import build_cp2k_input as phal_build
    from hotspot_al.models import ExtractedRegion as PhalRegion

    region = load_region(name)
    if periodic == "NONE":  # give PHAL and GMD the same cubic, padded cell
        geometry = prepare_geometry(region, CP2KConfig())
        region = replace(region, positions=geometry.positions, cell=geometry.cell)
    caps = region.indices_with(*[r for r in region.roles if r.value == "h_cap"]).tolist()
    atoms = Atoms(numbers=region.numbers, positions=region.positions, cell=region.cell, pbc=region.pbc)
    phal_region = PhalRegion(
        atoms=atoms,
        original_indices=list(range(region.n_atoms)),
        core_indices=[0],
        inner_buffer_indices=[],
        outer_buffer_indices=[],
        boundary_indices=[],
        h_cap_indices=caps,
        hotspot_indices=[0],
        metadata={},
    )
    phal_config = load_config()
    phal_config["cp2k"]["periodic"] = periodic
    phal_config["cp2k"]["poisson_solver"] = "WAVELET" if periodic == "NONE" else "PERIODIC"
    ours = CP2KConfig(stress="never")
    for run_type, fixed in (("ENERGY_FORCE", None), ("GEO_OPT", [i for i in range(region.n_atoms) if i not in caps])):
        if run_type == "GEO_OPT" and not caps:
            continue
        expected = phal_build(phal_region, config=phal_config, run_type=run_type, fixed_atom_indices=fixed, project_name="job")
        got = build_cp2k_input(region, ours, run_type=run_type, project_name="job", fixed_atom_indices=fixed).text
        assert got == expected


def test_run_cp2k_with_a_fake_executable_and_history_hook(tmp_path: Path) -> None:
    import stat
    import sys

    fake = tmp_path / "cp2k.fake"
    fake.write_text(
        f'#!{sys.executable}\nimport sys\nargs = sys.argv[1:]\nopen(args[args.index("-o") + 1], "w").write("PROGRAM ENDED AT\\n")\n'
        'sys.exit(int(open(args[args.index("-i") + 1]).read().strip() or 0))\n'
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    from gmd_active_learning.adapters.dft.cp2k.command import run_cp2k

    (tmp_path / "job.inp").write_text("0")
    seen: list[Path] = []

    def hook(output: Path, _input: Path) -> None:
        seen.append(output)
        raise RuntimeError("history recorder broke")

    process, output = run_cp2k(tmp_path / "job.inp", CP2KConfig(executable=str(fake)), history_hook=hook)
    assert process.returncode == 0 and output.read_text() == "PROGRAM ENDED AT\n" and seen == [output]
    (tmp_path / "bad.inp").write_text("3")
    with pytest.raises(LabelingError, match="exited with status 3"):
        run_cp2k(tmp_path / "bad.inp", CP2KConfig(executable=str(fake)))
    with pytest.raises(LabelingError, match="not found"):
        run_cp2k(tmp_path / "job.inp", CP2KConfig(executable=str(tmp_path / "missing")))
