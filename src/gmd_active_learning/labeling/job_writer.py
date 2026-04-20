from __future__ import annotations

from pathlib import Path


def write_job_script(job_dir: str | Path, scheduler: str, command: str) -> Path:
    job_path = Path(job_dir) / ("submit.slurm" if scheduler == "slurm" else "submit.pbs")
    if scheduler == "slurm":
        content = f"#!/bin/bash\n#SBATCH -J gmd_al\n#SBATCH -N 1\n{command}\n"
    else:
        content = f"#!/bin/bash\n#PBS -N gmd_al\n{command}\n"
    job_path.write_text(content, encoding="utf-8")
    return job_path
