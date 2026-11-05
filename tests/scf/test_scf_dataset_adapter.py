# Ported from PHAL (tests.scf.test_scf_dataset_adapter) at 96263b6; MIT License, see NOTICE.
"""Tests for reading a collected dataset as density-model training data.

The adapter's job is to refuse quietly-wrong samples. Two of them would produce
a model that is wrong everywhere rather than merely inaccurate: a density with
no overlap (its electron count is unverifiable) and one from an unconverged run
(it is not a converged density).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("ase", reason="the adapter reads structures through ASE")

from tools.scf.training.dataset_adapter import (  # noqa: E402
    AdapterError,
    sample_from_directory,
    samples_from_dataset,
    split_samples,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "scf" / "pulay_sample_periodic"
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    """A real sample, with its structure source resolvable from the repo."""

    root = tmp_path / "dataset"
    root.mkdir()
    target = root / "si-diamond"
    shutil.copytree(FIXTURE, target)

    # The recorded source path is repo-relative; the adapter resolves it
    # against the dataset root, so mirror the file there.
    payload = json.loads((target / "sample.json").read_text(encoding="utf-8"))
    source = REPO / payload["structure"]["source"]
    mirrored = root / payload["structure"]["source"]
    mirrored.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, mirrored)
    return target


def test_a_real_sample_becomes_a_training_sample(sample: Path):
    result = sample_from_directory(sample, root=sample.parent)

    assert result is not None
    assert result.sample_id == "si-diamond"
    assert result.structure.atomic_numbers == (14, 14)
    assert result.artifacts.n_ao == 26
    assert result.artifacts.electron_count_method == "trace_ps"
    assert result.trajectory.converged is True


def test_the_two_electron_counts_are_kept_apart(sample: Path):
    """All-electron on the state, valence on the artifacts.

    Confusing them makes every physically correct prediction look like it lost
    electrons: Si is Z=14 but its GTH pseudopotential carries 4.
    """

    result = sample_from_directory(sample, root=sample.parent)

    assert result is not None
    assert result.electronic_state.n_electrons == 28  # sum(Z)
    assert result.artifacts.density_electron_count == pytest.approx(8.0, abs=1e-6)  # valence


def test_a_sample_without_an_overlap_is_skipped(sample: Path):
    """Without S the electron count is Tr(P), which is not the electron count."""

    (sample / "trajectory" / "overlap.npy").unlink()

    assert sample_from_directory(sample, root=sample.parent) is None


def test_an_unconverged_sample_is_skipped(sample: Path):
    payload = json.loads((sample / "sample.json").read_text(encoding="utf-8"))
    payload["scf_converged"] = False
    (sample / "sample.json").write_text(json.dumps(payload), encoding="utf-8")

    assert sample_from_directory(sample, root=sample.parent) is None


def test_a_sample_whose_structure_cannot_be_found_is_skipped(sample: Path):
    payload = json.loads((sample / "sample.json").read_text(encoding="utf-8"))
    payload["structure"]["source"] = "nowhere/absent.extxyz"
    (sample / "sample.json").write_text(json.dumps(payload), encoding="utf-8")

    assert sample_from_directory(sample, root=sample.parent) is None


def test_reading_a_dataset_reports_what_it_dropped(sample: Path):
    (sample.parent / "empty").mkdir()

    samples, tally = samples_from_dataset(sample.parent)

    assert tally["n_samples"] == 1
    # The empty directory, plus the mirrored structure tree this fixture puts
    # under the root; anything that is not a sample is counted, not ignored.
    assert tally["n_skipped"] >= 1
    assert samples[0].sample_id == "si-diamond"


def test_a_missing_dataset_is_an_error(tmp_path: Path):
    with pytest.raises(AdapterError, match="not a dataset directory"):
        samples_from_dataset(tmp_path / "absent")


def test_the_split_is_deterministic(sample: Path):
    samples, _ = samples_from_dataset(sample.parent)

    first = split_samples(samples * 8)
    second = split_samples(samples * 8)

    assert [item.sample_id for item in first[0]] == [item.sample_id for item in second[0]]
    assert len(first[1]) > 0


def test_the_real_dataset_yields_usable_examples():
    root = Path("runtime/cp2k/pulay_dataset")
    if not root.is_dir():
        pytest.skip("the generated dataset is not present")

    from tools.scf.training.dataset_loader import build_examples

    samples, tally = samples_from_dataset(root)
    examples = build_examples(samples, target="density_matrix")

    assert tally["n_samples"] > 20
    assert len(examples) > 20
    # nao varies by more than an order of magnitude across the dataset; a model
    # that treats it as a fixed dimension cannot be trained on this.
    sizes = [example.density_target.shape[-1] for example in examples]
    assert max(sizes) > 10 * min(sizes)
