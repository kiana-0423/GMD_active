"""``scheduler`` section and typed resources."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from gmd_active_learning.adapters.schedulers.config import SchedulerConfig, SchedulerResources

SECTIONS = Path(__file__).resolve().parents[3] / "configs" / "sections"


def test_packaged_scheduler_section_equals_model_defaults() -> None:
    data = yaml.safe_load((SECTIONS / "scheduler.yaml").read_text())["scheduler"]
    assert SchedulerConfig.model_validate(data) == SchedulerConfig()


def test_resources_are_typed_and_validated() -> None:
    resources = SchedulerResources(nodes=2, walltime="1-00:00:00", memory="16G")
    assert resources.walltime_seconds == 86400 and resources.memory_parts() == (16, "G")
    for bad in ({"nodes": 0}, {"walltime": "soon"}, {"memory": "lots"}, {"directives": "#SBATCH -N1"}):
        with pytest.raises(ValidationError):
            SchedulerResources.model_validate(bad)


def test_resource_merge_override_wins_and_directives_concatenate() -> None:
    base = SchedulerResources(nodes=1, walltime=600, extra_directives=["--qos=a"])
    merged = base.merged(SchedulerResources(walltime=1200, extra_directives=["--exclusive"]))
    assert merged.nodes == 1 and merged.walltime == 1200 and merged.extra_directives == ["--qos=a", "--exclusive"]
    assert base.merged(None) is base
