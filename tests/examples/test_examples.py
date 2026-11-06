"""Run every example's ``main()`` on the bundled data (EXT-011, LAB-011 and the PHAL examples 01-06).

No external program is needed except for the live GMD mode of example 06, which runs only when a
GMD Torch build is found (``$GMD_EXECUTABLE`` or the sibling ``../GMD/build-torch/gmd``).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
DATA = EXAMPLES / "data"
FIXTURES = ROOT / "tests" / "fixtures"
PHAL_SRC = ROOT / "ref" / "lammps_monitor" / "src"
GMD_BIN = Path(os.environ.get("GMD_EXECUTABLE", ROOT.parent / "GMD" / "build-torch" / "gmd"))


def load_example(name: str) -> ModuleType:
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    spec = importlib.util.spec_from_file_location(f"example_{name.replace('.py', '')}", EXAMPLES / name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(name: str, argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = load_example(name).main(argv)
    return code, capsys.readouterr().out


# ----------------------------------------------------------------------------------- bundled data
@pytest.mark.parametrize(
    ("bundled", "fixture"),
    [
        ("toy_ethane.lammpstrj", "lammps/toy_hotspot.lammpstrj"),
        ("gmd_water_frames.extxyz", "gmd/frames.extxyz"),
        ("water2.xyz", "gmd/water2.xyz"),
        ("ch3_capped_region.json", "structures/regions/ch3_capped.json"),
        ("ch3_capped_sp.out", "cp2k/real_2026.2/ch3_capped_sp/sp.out"),
    ],
)
def test_bundled_data_are_copies_of_the_test_fixtures(bundled: str, fixture: str) -> None:
    assert (DATA / bundled).read_bytes() == (FIXTURES / fixture).read_bytes()


def test_legacy_dump_generator_is_deterministic() -> None:
    generator = load_example("data/make_legacy_ethane_dump.py")
    assert generator.build() == (DATA / "legacy_ethane_kick.lammpstrj").read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------ 01 monitor
def test_01_monitors_a_legacy_lammps_trajectory(capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run("01_monitor_trajectory.py", [], capsys)
    assert code == 0
    rows = {line.split()[0]: line.split() for line in out.splitlines() if line[:1].isdigit()}
    assert len(rows) == 24 and "processed_frames=24 stop_reason=end_of_stream errors=0" in out
    kicked = rows["200"]  # the 33 eV/A kick on atom id 1
    assert kicked[3] == "True" and kicked[4] in {"save", "checkpoint", "stop"} and "force_large" in kicked[5]
    assert rows["0"][3] == "False"


def test_01_monitors_gmd_frames(capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run("01_monitor_trajectory.py", [str(DATA / "gmd_water_frames.extxyz")], capsys)
    assert code == 0 and "processed_frames=4" in out


def test_01_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        load_example("01_monitor_trajectory.py").main([str(tmp_path / "missing.lammpstrj")])
    assert error.value.code == 2


# ------------------------------------------------------------------------------------ 02 extract
def test_02_extracts_at_least_one_region_from_gmd_frames(capsys: pytest.CaptureFixture[str]) -> None:
    """EXT-011 acceptance: runs on a GMD frame fixture and prints at least one region."""

    code, out = run("02_extract_hotspots.py", [str(DATA / "gmd_water_frames.extxyz"), "--atom-id", "0"], capsys)
    assert code == 0
    n_regions = int(next(line for line in out.splitlines() if line.startswith("n_regions:")).split()[1])
    assert n_regions >= 1 and "mode=cluster" in out and "hotspots=[0]" in out


@pytest.mark.parametrize("mode", ["cluster", "graph"])
def test_02_extracts_from_the_first_triggered_frame(mode: str, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run("02_extract_hotspots.py", ["--mode", mode], capsys)
    assert code == 0 and f"mode={mode}" in out and "n_regions: 1" in out


def test_02_exits_1_without_regions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "off.yaml"
    config.write_text("extraction: {enabled: false}\n", encoding="utf-8")
    code, out = run("02_extract_hotspots.py", ["--config", str(config)], capsys)
    assert code == 1 and "n_regions: 0" in out  # PHAL indexed regions[0] and crashed


# ------------------------------------------------------------------------------------ 03 CP2K inputs
def test_03_writes_capped_cluster_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run("03_generate_cp2k_inputs.py", ["--output-dir", str(tmp_path)], capsys)
    assert code == 0 and "single_point_input" in out
    inputs = sorted(tmp_path.glob("*/hotspot_region_sp.inp"))
    assert len(inputs) == 1
    text = inputs[0].read_text(encoding="utf-8")
    assert "RUN_TYPE ENERGY_FORCE" in text and "PERIODIC NONE" in text and "POISSON_SOLVER WAVELET" in text
    cell = [line.split()[1:] for line in text.splitlines() if line.strip().startswith(("A ", "B ", "C ")) and len(line.split()) == 4]
    lengths = [float(row[index]) for index, row in enumerate(cell[:3])]
    assert lengths[0] == lengths[1] == lengths[2]  # WAVELET needs a cubic box (CP2K 2026.2 aborts otherwise)
    coords = text.split("&COORD")[1].split("&END COORD")[0].strip().splitlines()
    assert len(coords) == 8


def test_03_refuses_an_uncapped_isolated_fragment(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """With capping disabled, a cluster that cuts the C-C bond is refused (PHAL wrote it silently)."""

    config = tmp_path / "nocap.yaml"
    config.write_text("capping: {enabled: false}\nextraction: {mode: graph, graph: {hops: 0, radial_shell: false}}\n", encoding="utf-8")
    code, out = run("03_generate_cp2k_inputs.py", ["--config", str(config), "--output-dir", str(tmp_path / "jobs")], capsys)
    assert code == 1 and "refused" in out
    assert not list((tmp_path / "jobs").glob("*/*.inp"))


# ------------------------------------------------------------------------------------ 04 CP2K forces
def test_04_prints_force_statistics_and_a_valid_verdict(capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run("04_parse_cp2k_forces.py", [], capsys)
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == "(5, 3)" and "label: VALID" in out and "cp2k_version: 2026.2" in out


def test_04_statistics_match_phal(capsys: pytest.CaptureFixture[str]) -> None:
    """LAB-011 acceptance: the same statistics as PHAL's example for the fixture CP2K output."""

    if not PHAL_SRC.is_dir():
        pytest.skip("ref/lammps_monitor is absent; PHAL comparison not possible")
    sys.path.insert(0, str(PHAL_SRC))
    try:
        from hotspot_al.cp2k.cp2k_force_parser import parse_cp2k_forces as phal_parse
    finally:
        sys.path.remove(str(PHAL_SRC))
    module = load_example("04_parse_cp2k_forces.py")
    for output in [DATA / "ch3_capped_sp.out", *sorted((FIXTURES / "cp2k" / "real_2026.2").glob("*_sp/sp.out"))]:
        ours = module.force_statistics(module.parse_cp2k_forces(output))
        phal = module.force_statistics(phal_parse(output))
        assert ours["shape"] == phal["shape"]
        for key in ("min", "max", "mean", "norm_min", "norm_max"):
            np.testing.assert_allclose(ours[key], phal[key], rtol=1e-12, atol=0.0)
    capsys.readouterr()


def test_04_rejects_an_unconverged_output(capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run("04_parse_cp2k_forces.py", [str(FIXTURES / "cp2k/real_2026.2/h2_unconverged/sp.out")], capsys)
    assert code == 1 and "no forces" in out and "label: INVALID" in out


# ------------------------------------------------------------------------------------ 05 dataset
def test_05_publishes_a_gmd_sgt_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from gmd_active_learning.datasets.extxyz_io import read_samples_extxyz

    code, out = run("05_write_sgt_dataset.py", ["--output-dir", str(tmp_path / "store")], capsys)
    assert code == 0 and "force_weight=[1.0, 1.0, 1.0, 1.0, 0.0]" in out
    version = next(line.split()[2] for line in out.splitlines() if line.startswith("dataset version"))
    files = {line.split(": ")[0]: Path(line.split(": ")[1]) for line in out.splitlines() if line.startswith(("train_file", "val_file", "sgt_npz"))}
    assert all(path.is_file() for path in files.values())
    samples = read_samples_extxyz(files["train_file"]) + read_samples_extxyz(files["val_file"])
    assert len(samples) == 1 and samples[0].region.force_weights.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]
    # Deterministic: re-running publishes the same version.
    code, again = run("05_write_sgt_dataset.py", ["--output-dir", str(tmp_path / "store")], capsys)
    assert code == 0 and f"dataset version {version}" in again


def test_05_refuses_an_invalid_label(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    unconverged = FIXTURES / "cp2k/real_2026.2/h2_unconverged/sp.out"
    region = FIXTURES / "structures/regions/h2_cluster.json"
    code, out = run("05_write_sgt_dataset.py", [str(region), str(unconverged), "--output-dir", str(tmp_path / "store")], capsys)
    assert code == 1 and "label INVALID" in out and not (tmp_path / "store").exists()


# ------------------------------------------------------------------------------------ 06 online
def test_06_replays_gmd_frames_and_writes_events(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run("06_online_monitor.py", ["--event-dir", str(tmp_path / "events")], capsys)
    assert code == 0 and "processed_frames=4" in out and "steps=[0, 2, 4, 6]" in out


def test_06_replays_a_triggering_trajectory_into_the_event_store(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from gmd_active_learning.events import read_event

    code, out = run("06_online_monitor.py", [str(DATA / "legacy_ethane_kick.lammpstrj"), "--event-dir", str(tmp_path / "events")], capsys)
    assert code == 0 and "processed_frames=24" in out and "200" in out.split("triggered_steps=")[1]
    events = sorted((tmp_path / "events").glob("*/event.json"))
    assert events and int(out.split("events=")[1]) == len(events)
    assert any(200 in read_event(path.parent).trigger_steps for path in events)


def test_06_live_mode_needs_its_inputs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        load_example("06_online_monitor.py").main(["--live"])
    assert error.value.code == 2


@pytest.mark.integration
def test_06_live_gmd_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    if not GMD_BIN.is_file():
        pytest.skip(f"needs a GMD Torch build ($GMD_EXECUTABLE or {GMD_BIN})")
    model = FIXTURES / "sgt" / "member_11" / "model.pt"
    code, out = run("06_online_monitor.py", ["--live", "--gmd-executable", str(GMD_BIN), "--model", str(model),
                                             "--steps", "10", "--work-dir", str(tmp_path / "run")], capsys)
    assert code == 0 and "processed_frames=11" in out and "errors=0" in out


# ------------------------------------------------------------------------------------ legacy example
def test_minimal_lj_monitor_example_runs(capsys: pytest.CaptureFixture[str]) -> None:
    load_example("minimal_lj_monitor_example.py").main()
    assert "step" in capsys.readouterr().out
