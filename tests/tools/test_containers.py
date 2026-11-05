# Ported from PHAL (tests/unit/test_container_milestone2.py, test_container_milestone3.py,
# test_container_cp2k_ml.py) at 96263b6; MIT License, see NOTICE.
"""Static and host-level checks for containers/, tools/hpc/pbs/ and the container docs.

Nothing here needs docker, apptainer or PBS. Images cannot be built in this
environment, so these tests establish only what can be shown without them:
the Dockerfiles are well formed and self-consistent, every pin is exact, the
build plan resolves, floating references are refused, the PBS templates hash
the SIF at job time, and the validators behave correctly on this host (with
real GMD / GMD_SGT / CP2K executables when the sibling checkouts provide them,
skipped otherwise).
"""

from __future__ import annotations

import fnmatch
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONTAINERS = ROOT / "containers"
SCRIPTS = CONTAINERS / "scripts"
BUILD_SCRIPT = CONTAINERS / "build.sh"
VERSIONS = CONTAINERS / "versions.yaml"
PBS_DIR = ROOT / "tools" / "hpc" / "pbs"
DOCS = ROOT / "docs" / "containers"
BASELINE_DIR = ROOT / "docs" / "migration" / "phal_release_baseline"
PHAL_REF = ROOT / "ref" / "lammps_monitor"
WORKSPACE = ROOT.parent
GMD_SGT_CHECKOUT = WORKSPACE / "GMD_SGT"

DOCKERFILES = {
    "gmd": CONTAINERS / "Dockerfile.gmd",
    "sgt-train": CONTAINERS / "Dockerfile.sgt-train",
    "cp2k": CONTAINERS / "Dockerfile.cp2k",
    "cp2k-ml": CONTAINERS / "Dockerfile.cp2k-ml",
}
TARGETS = {
    # target: (dockerfile key, stack key, stage)
    "gmd": ("gmd", "gmd", "runtime"),
    "sgt-train": ("sgt-train", "sgt_train", "runtime"),
    "cp2k": ("cp2k", "cp2k", "runtime"),
    "cp2k-probe": ("cp2k", "cp2k", "probe"),
    "cp2k-ml": ("cp2k-ml", "cp2k_ml", "development"),
    "cp2k-ml-probe": ("cp2k-ml", "cp2k_ml", "probe"),
}
PBS_SCRIPTS = sorted(PBS_DIR.glob("*.pbs"))
FLOATING = {"latest", "stable", "main", "master", "head"}
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
DOCKER_INSTRUCTIONS = {
    "FROM", "RUN", "CMD", "LABEL", "EXPOSE", "ENV", "ADD", "COPY", "ENTRYPOINT",
    "VOLUME", "USER", "WORKDIR", "ARG", "ONBUILD", "STOPSIGNAL", "HEALTHCHECK", "SHELL",
}
BASE_ENVIRONMENT = {"PATH", "HOME", "LD_LIBRARY_PATH", "USER", "HOSTNAME"}

BASH = shutil.which("bash") or "bash"
requires_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")


# --------------------------------------------------------------------------- helpers


def manifest() -> dict[str, Any]:
    return yaml.safe_load(VERSIONS.read_text(encoding="utf-8"))


def strings(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for nested in value.values():
            yield from strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from strings(nested)
    elif isinstance(value, str):
        yield value


def leaves(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield from leaves(nested, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from leaves(nested, f"{path}[{index}]")
    else:
        yield path, value


def is_floating(value: str) -> bool:
    lowered = value.lower()
    return lowered in FLOATING or any(lowered.endswith(f":{word}") or lowered.endswith(f"#{word}")
                                      for word in FLOATING)


def clean_env(**overrides: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GMD_AL_")}
    env.update(overrides)
    return env


def run_build(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run([BASH, str(BUILD_SCRIPT), *args], cwd=ROOT, env=env or clean_env(),
                          check=False, text=True, capture_output=True, timeout=60)


def plan(target: str, **env: str) -> dict[str, list[str]]:
    result = run_build("plan", target, env=clean_env(**env))
    assert result.returncode == 0, result.stderr
    parsed: dict[str, list[str]] = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        parsed.setdefault(key, []).append(value)
    return parsed


class Instruction:
    def __init__(self, keyword: str, args: str, line: int) -> None:
        self.keyword = keyword
        self.args = args
        self.line = line


def parse_dockerfile(path: Path) -> list[Instruction]:
    """Minimal Dockerfile parser: comments, continuation lines, instruction keywords."""
    instructions: list[Instruction] = []
    buffer: list[str] = []
    start = 0
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if stripped.startswith("#") or (not stripped and not buffer):
            continue
        if not buffer:
            start = number
        if stripped.endswith("\\"):
            buffer.append(stripped[:-1])
            continue
        buffer.append(stripped)
        text = " ".join(part.strip() for part in buffer if part.strip())
        buffer = []
        keyword, _, args = text.partition(" ")
        instructions.append(Instruction(keyword, args.strip(), start))
    assert not buffer, f"{path.name}: dangling line continuation"
    return instructions


class Stage:
    def __init__(self, base: str, alias: str | None) -> None:
        self.base = base
        self.alias = alias
        self.instructions: list[Instruction] = []


def stages(path: Path) -> tuple[list[Instruction], list[Stage]]:
    global_args: list[Instruction] = []
    result: list[Stage] = []
    for instruction in parse_dockerfile(path):
        if instruction.keyword.upper() == "FROM":
            tokens = [t for t in instruction.args.split() if not t.startswith("--")]
            alias = tokens[2] if len(tokens) >= 3 and tokens[1].upper() == "AS" else None
            result.append(Stage(tokens[0], alias))
        elif not result:
            global_args.append(instruction)
        else:
            result[-1].instructions.append(instruction)
    return global_args, result


def arg_decls(instructions: list[Instruction]) -> dict[str, str | None]:
    decls: dict[str, str | None] = {}
    for instruction in instructions:
        if instruction.keyword.upper() == "ARG":
            name, sep, default = instruction.args.partition("=")
            decls[name.strip()] = default if sep else None
    return decls


def env_keys(instructions: list[Instruction]) -> set[str]:
    keys: set[str] = set()
    for instruction in instructions:
        if instruction.keyword.upper() == "ENV":
            keys.update(re.findall(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_]*)=", instruction.args))
    return keys


def stage_chain(all_stages: list[Stage], stage_name: str) -> list[Stage]:
    by_alias = {stage.alias: stage for stage in all_stages if stage.alias}
    chain = [by_alias[stage_name]]
    while chain[-1].base in by_alias:
        chain.append(by_alias[chain[-1].base])
    return chain


def copy_instructions(path: Path) -> Iterator[tuple[Instruction, list[str], str | None]]:
    for instruction in parse_dockerfile(path):
        if instruction.keyword.upper() not in {"COPY", "ADD"}:
            continue
        tokens = instruction.args.split()
        flags = [t for t in tokens if t.startswith("--")]
        positional = [t for t in tokens if not t.startswith("--")]
        source_from = next((f.split("=", 1)[1] for f in flags if f.startswith("--from=")), None)
        yield instruction, positional[:-1], source_from


def dockerignore_patterns(path: Path) -> list[tuple[bool, str]]:
    patterns = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        patterns.append((negated, line[1:].strip("/") if negated else line.strip("/")))
    return patterns


def _pattern_matches(path: str, pattern: str) -> bool:
    parts = path.split("/")
    candidates = [pattern] + ([pattern[3:]] if pattern.startswith("**/") else [])
    for index in range(1, len(parts) + 1):
        prefix = "/".join(parts[:index])
        if any(fnmatch.fnmatchcase(prefix, candidate) for candidate in candidates):
            return True
    return False


def is_ignored(path: str, patterns: list[tuple[bool, str]]) -> bool:
    """Docker semantics: patterns in order, the last match wins, '!' re-includes."""
    ignored = False
    for negated, pattern in patterns:
        if _pattern_matches(path, pattern):
            ignored = not negated
    return ignored


def load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), SCRIPTS / name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gmd_sgt_env() -> dict[str, str] | None:
    """Environment in which `import gmd_sgt` works, or None."""
    env = clean_env(PYTHONDONTWRITEBYTECODE="1")
    if GMD_SGT_CHECKOUT.is_dir():
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(GMD_SGT_CHECKOUT), env.get("PYTHONPATH")]))
    probe = subprocess.run([sys.executable, "-c", "import torch, gmd_sgt"], env=env,
                           capture_output=True, text=True, check=False)
    return env if probe.returncode == 0 else None


def local_gmd() -> Path | None:
    for candidate in (os.environ.get("GMD_EXECUTABLE"), str(WORKSPACE / "GMD" / "build-torch" / "gmd")):
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return Path(candidate)
    return None


def run_json(command: list[str], env: dict[str, str], timeout: float = 300) -> tuple[int, dict[str, Any]]:
    completed = subprocess.run(command, env=env, capture_output=True, text=True, check=False,
                               timeout=timeout, cwd=ROOT)
    assert completed.stdout.strip(), completed.stderr
    return completed.returncode, json.loads(completed.stdout)


# --------------------------------------------------------------------------- files and layout


def test_expected_files_exist() -> None:
    expected = [
        *DOCKERFILES.values(),
        *(Path(f"{path}.dockerignore") for path in DOCKERFILES.values()),
        BUILD_SCRIPT,
        VERSIONS,
        CONTAINERS / "README.md",
        SCRIPTS / "validate_platform.py",
        SCRIPTS / "validate_sgt_stack.py",
        SCRIPTS / "validate_gmd_stack.py",
        SCRIPTS / "build_gmd_probe_model.py",
        SCRIPTS / "validate_cp2k_stack.py",
        SCRIPTS / "validate_cp2k_ml_stack.py",
        CONTAINERS / "probes" / "gmd" / "README.md",
        CONTAINERS / "probes" / "gmd" / "h2.xyz",
        CONTAINERS / "probes" / "gmd" / "run0.run",
        CONTAINERS / "probes" / "gmd" / "run10.run",
        CONTAINERS / "probes" / "cp2k" / "h2.inp",
        PBS_DIR / "validate_gmd_probe.pbs",
        PBS_DIR / "validate_sgt_train_probe.pbs",
        PBS_DIR / "validate_cp2k_probe.pbs",
        PBS_DIR / "diagnose_hpcx.pbs",
        PBS_DIR / "submit.sh",
        DOCS / "README.md",
        DOCS / "architecture.md",
        DOCS / "runtime.md",
        DOCS / "apptainer.md",
        DOCS / "docker.md",
        DOCS / "probe-procedure.md",
    ]
    missing = [str(path.relative_to(ROOT)) for path in expected if not path.is_file()]
    assert not missing


def test_build_and_validator_scripts_are_executable() -> None:
    for path in (BUILD_SCRIPT, PBS_DIR / "submit.sh", *SCRIPTS.glob("*.py")):
        assert path.stat().st_mode & stat.S_IXUSR, path


def test_dead_containers_dockerignore_is_not_reintroduced() -> None:
    # PHAL's containers/.dockerignore was never consulted: the build context is
    # the repository root. BuildKit reads <Dockerfile>.dockerignore instead.
    assert not (CONTAINERS / ".dockerignore").exists()
    for target in TARGETS:
        assert plan(target)["dockerignore"] == [f"{DOCKERFILES[TARGETS[target][0]]}.dockerignore"]


def test_dropped_milestone1_scaffolds_are_not_referenced() -> None:
    """CTR-002: PHAL's empty base/md/train Dockerfiles are dropped; nothing points at them."""
    pattern = re.compile(r"Dockerfile\.(base|md|train|md-probe|train-probe)(?![\w.-])")
    roots = [CONTAINERS, PBS_DIR, DOCS, ROOT / "src", ROOT / "configs"]
    offenders = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix == ".pyc":
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                # Provenance lines that name the PHAL original are not dependencies.
                if pattern.search(line) and "PHAL" not in line:
                    offenders.append(f"{path.relative_to(ROOT)}:{number}")
    for name in ("Dockerfile.base", "Dockerfile.md", "Dockerfile.train", "Dockerfile.md-probe",
                 "Dockerfile.train-probe"):
        assert not (CONTAINERS / name).exists()
    assert not offenders


# --------------------------------------------------------------------------- Dockerfiles


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_dockerfile_parses(name: str) -> None:
    path = DOCKERFILES[name]
    instructions = parse_dockerfile(path)
    assert instructions
    for instruction in instructions:
        assert instruction.keyword.upper() in DOCKER_INSTRUCTIONS, (path.name, instruction.line)
        assert instruction.keyword == instruction.keyword.upper(), (path.name, instruction.line)
    global_args, all_stages = stages(path)
    assert all(i.keyword == "ARG" for i in global_args), "only ARG may precede the first FROM"
    assert all_stages, "no FROM instruction"
    aliases = [stage.alias for stage in all_stages if stage.alias]
    assert len(aliases) == len(set(aliases)), "duplicate stage names"
    for stage in all_stages:
        # Every base is either the pinned BASE_IMAGE arg or an earlier stage.
        assert stage.base == "${BASE_IMAGE}" or stage.base in aliases, stage.base


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_build_target_stage_exists(target: str) -> None:
    dockerfile, _, stage = TARGETS[target]
    _, all_stages = stages(DOCKERFILES[dockerfile])
    assert stage in {s.alias for s in all_stages}
    assert plan(target)["docker_stage"] == [stage]


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_copy_sources_exist_in_build_context(name: str) -> None:
    """F-32 regression: every COPY/ADD source must exist (PHAL copied a vanished config/)."""
    path = DOCKERFILES[name]
    _, all_stages = stages(path)
    aliases = {stage.alias for stage in all_stages if stage.alias}
    named_contexts = {entry.split("=", 1)[0]
                      for target, (key, _, _) in TARGETS.items() if key == name
                      for entry in plan(target).get("named_context", [])}
    copies = list(copy_instructions(path))
    assert copies
    for instruction, sources, source_from in copies:
        assert sources, f"{path.name}:{instruction.line} has no source"
        if source_from is not None:
            assert source_from in aliases or source_from in named_contexts, (
                f"{path.name}:{instruction.line}: --from={source_from} is neither a stage nor a "
                "named context supplied by build.sh")
            continue
        for source in sources:
            assert not source.startswith(("/", "..")), source
            assert (ROOT / source).exists(), f"{path.name}:{instruction.line}: missing {source}"


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_dockerfile_specific_ignore_admits_copy_sources_only(name: str) -> None:
    path = DOCKERFILES[name]
    patterns = dockerignore_patterns(Path(f"{path}.dockerignore"))
    assert patterns[0] == (False, "*"), "the ignore file must be an allowlist"
    for _, sources, source_from in copy_instructions(path):
        if source_from is not None:
            continue
        for source in sources:
            files = [ROOT / source] if (ROOT / source).is_file() else [
                p for p in (ROOT / source).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
            for file in files:
                relative = file.relative_to(ROOT).as_posix()
                assert not is_ignored(relative, patterns), f"{relative} is excluded from the context"
    for excluded in (".git/HEAD", "src/gmd_active_learning/__init__.py", "tests/conftest.py",
                     "ref/lammps_monitor/README.md", "runtime/logs/x.log", "build/containers/x.sif",
                     "data/x.extxyz", "containers/scripts/__pycache__/x.cpython-311.pyc"):
        assert is_ignored(excluded, patterns), excluded


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_dockerfile_variables_are_declared(name: str) -> None:
    """Every ${VAR} is an ARG of its stage, an ENV of the stage chain, or base environment.

    Global ARGs (before the first FROM) are visible only in FROM lines; using one
    inside a stage without redeclaring it silently expands to an empty string.
    """
    path = DOCKERFILES[name]
    global_args, all_stages = stages(path)
    global_decls = arg_decls(global_args)
    by_alias = {stage.alias: stage for stage in all_stages if stage.alias}
    reference = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)|\$([A-Za-z_][A-Za-z0-9_]*)")
    for stage in all_stages:
        for match in reference.finditer(stage.base):
            assert (match.group(1) or match.group(2)) in global_decls
        ancestry = [stage]
        while ancestry[-1].base in by_alias:
            ancestry.append(by_alias[ancestry[-1].base])
        envs = set().union(*(env_keys(s.instructions) for s in ancestry))
        allowed = set(arg_decls(stage.instructions)) | envs | BASE_ENVIRONMENT
        for instruction in stage.instructions:
            for match in reference.finditer(instruction.args):
                variable = match.group(1) or match.group(2)
                assert variable in allowed, f"{path.name}:{instruction.line}: ${variable} is not declared"


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_build_args_match_dockerfile_args(target: str) -> None:
    dockerfile, _, stage = TARGETS[target]
    global_args, all_stages = stages(DOCKERFILES[dockerfile])
    chain = stage_chain(all_stages, stage)
    # Stages used via COPY --from are built too.
    by_alias = {s.alias: s for s in all_stages if s.alias}
    for s in list(chain):
        for instruction in s.instructions:
            match = re.search(r"--from=(\S+)", instruction.args)
            if instruction.keyword == "COPY" and match and match.group(1) in by_alias:
                chain.extend(stage_chain(all_stages, match.group(1)))
    declared: dict[str, str | None] = dict(arg_decls(global_args))
    for s in chain:
        declared.update(arg_decls(s.instructions))
    passed = {line.split("=", 1)[0] for line in plan(target)["build_arg"]}
    required = {name for name, default in declared.items() if default is None}
    assert required <= passed, f"ARGs without default not supplied by build.sh: {required - passed}"
    assert passed <= set(declared), f"build args not consumed by the Dockerfile: {passed - set(declared)}"


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_dockerfiles_have_no_floating_or_pinned_literals(name: str) -> None:
    text = DOCKERFILES[name].read_text(encoding="utf-8")
    for word in ("latest", "stable", "master"):
        assert not re.search(rf"\b{word}\b", text, re.I), word
    assert not re.search(r"(?<![_.-])\bmain\b(?![_.-])", text)
    # Pins come from versions.yaml through build args, never as literals.
    for value in strings(manifest()["stacks"]):
        if re.fullmatch(r"[0-9a-f]{40}", value) or value.startswith("sha256:"):
            assert value not in text, f"{name} hard-codes {value}"


def test_gmd_dockerfile_builds_gmd_with_torch_from_the_base_image() -> None:
    text = DOCKERFILES["gmd"].read_text(encoding="utf-8")
    lowered = text.lower()
    for required in (
        "FROM ${BASE_IMAGE} AS builder",
        "FROM ${BASE_IMAGE} AS runtime",
        "-D GMD_ENABLE_TORCH=ON",
        "-D BUILD_TESTING=OFF",
        "torch.utils.cmake_prefix_path",
        "--target gmd",
        "COPY --from=builder /opt/gmd /opt/gmd",
        "COPY --from=gmd_src . /opt/src/gmd",
        "COPY --from=gmd_sgt_src . /opt/src/gmd_sgt",
        ".gmd_al_source_commit",
        'project(GMD VERSION ${GMD_CMAKE_VERSION}',
        "pip install --no-deps",
        "build_gmd_probe_model.py",
        "validate_gmd_stack.py --mode static",
    ):
        assert required in text, required
    assert "pip install torch" not in lowered
    assert "nvidia/cuda" not in lowered
    assert "apt-get install cuda" not in lowered
    assert "validate_gmd_stack.py --mode runtime" not in text  # never run gmd on a builder


def test_sgt_train_dockerfile_is_pinned_and_inherits_cuda_torch() -> None:
    text = DOCKERFILES["sgt-train"].read_text(encoding="utf-8")
    lowered = text.lower()
    assert "arg base_image" in lowered
    assert "from ${base_image}" in lowered
    assert "pip install torch" not in lowered
    assert "apt-get install cuda" not in lowered
    assert "nvidia/cuda" not in lowered
    assert "copy config" not in lowered  # F-32
    assert "COPY --from=gmd_sgt_src . /opt/src/gmd_sgt" in text
    assert "python -m pip install --no-deps /opt/src/gmd_sgt" in text
    assert "gmd_sgt.__version__ != '${GMD_SGT_VERSION}'" in text
    assert "validate_sgt_stack.py --devices cpu" in text
    for path in ("/runtime/models", "/runtime/datasets", "/runtime/checkpoints", "/runtime/logs"):
        assert path in text


def test_cp2k_dockerfile_is_a_plain_production_build_of_the_pinned_commit() -> None:
    text = DOCKERFILES["cp2k"].read_text(encoding="utf-8")
    passed_args = {line.strip().rstrip("\\").strip() for line in text.splitlines()
                   if not line.lstrip().startswith("#")}
    for required in (
        "FROM ${BASE_IMAGE} AS toolchain",
        "FROM toolchain AS build",
        "FROM ${BASE_IMAGE} AS runtime",
        "FROM runtime AS probe",
        'https://github.com/cp2k/cp2k/archive/${CP2K_COMMIT}.tar.gz',
        "-DCP2K_USE_MPI=ON",
        "-DCP2K_USE_LIBINT2=ON",
        "-DCP2K_USE_LIBXC=ON",
        "-DCP2K_USE_ELPA=ON",
        "-DCP2K_USE_FFTW3=ON",
        "-DCP2K_USE_LIBTORCH=OFF",
        "--with-libtorch=no",
        "--with-openmpi=system",
        "test -x /opt/cp2k/bin/cp2k.psmp",
        "CP2K_DATA_DIR=/opt/cp2k/share/cp2k/data",
        "source \"${TOOLCHAIN_ENV:-/opt/cp2k-toolchain-src/install/setup}\"",
        'grep -q "CP2K version ${CP2K_VERSION#v}"',
        "validate_cp2k_stack.py",
    ):
        assert required in text, required
    assert "--install-all" not in passed_args
    assert "ML_SCF" not in text and "__PHAL_ML_SCF" not in text
    commit = manifest()["stacks"]["cp2k"]["upstream"]["commit"]
    assert commit not in text and len(commit) == 40


def test_cp2k_ml_dockerfile_is_toolchain_only() -> None:
    text = DOCKERFILES["cp2k-ml"].read_text(encoding="utf-8")
    lowered = text.lower()
    assert "ubuntu:24.04" in text
    for dependency in ("gfortran", "cmake", "ninja-build", "make", "libopenmpi-dev", "libblas-dev",
                       "liblapack-dev", "libfftw3-dev", "libscalapack-openmpi-dev", "git", "wget",
                       "curl", "vim", '"torch==${PYTORCH_VERSION}"'):
        assert dependency in text
    assert "libdbcsr-dev" not in text
    for option in ("--with-libint=install", "--with-libxc=install", "--with-elpa=install"):
        assert option in text
    passed_args = {line.strip().rstrip("\\").strip() for line in text.splitlines()
                   if not line.lstrip().startswith("#")}
    assert "--install-all" not in passed_args
    for option in ("--with-libtorch=no", "--with-deepmd=no", "--with-gauxc=no"):
        assert option in text
    assert "install/libtorch*" in text
    assert "--with-openmpi=system" in text
    for apt_package in ("libint2-dev", "libxc-dev", "libelpa-dev", "cp2k-data", "cp2k-bin"):
        assert apt_package not in text
    assert "git clone" not in lowered
    assert "git checkout" not in lowered
    assert "cmake --build" not in lowered
    assert "copy cp2k" not in lowered
    assert "cp2k-ml-scf" in text
    for stage in (" as toolchain", " as development", " as probe"):
        assert stage in lowered
    assert 'tar -xz --strip-components=3 "cp2k-${CP2K_COMMIT}/tools/toolchain"' in text
    assert "test ! -d /opt/cp2k-toolchain-src/src" in text
    assert '-name "qs_scf.F"' in text
    assert "ARG CP2K_COMMIT" in text


def test_cp2k_ml_installs_ase_at_build_time() -> None:
    text = DOCKERFILES["cp2k-ml"].read_text(encoding="utf-8")
    assert manifest()["stacks"]["cp2k_ml"]["ml_runtime"]["ase"]
    assert "ARG ASE_VERSION" in text
    assert '"ase==${ASE_VERSION}"' in text
    assert 'test -n "${ASE_VERSION}"' in text
    assert "import ase, ase.io, ase.build" in text
    build = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert "stacks.cp2k_ml.ml_runtime.ase" in build


# --------------------------------------------------------------------------- versions.yaml


def test_versions_manifest_platform_and_honest_status() -> None:
    data = manifest()
    assert data["schema_version"] == 1
    platform_section = data["platform"]
    assert platform_section == {
        "os": "linux", "architecture": "arm64", "target_system": "miyabi-g", "cpu": "nvidia-grace",
        "gpu": "gh200", "scheduler": "pbs-professional", "apptainer": "1.3.5",
        "default_oci_platform": "linux/arm64",
    }
    assert data["status"] == "not_built"
    for name, stack in data["stacks"].items():
        assert stack["status"] in {"not_built", "development"}, name
        assert stack["validation"]["status"] == "not_validated", name
        # Nothing has been built: no output digest or SIF hash may be recorded.
        for key in ("oci_digest", "oci_image", "sif_sha256"):
            if key in stack:
                assert stack[key] is None, f"{name}.{key}"
        for path, value in leaves(stack["validation"]):
            if path != "phal_record":
                assert value in {"not_run", "not_validated"}, f"{name}.validation.{path}"
    assert all(value is None for key, value in data["validated"].items() if key.startswith("selected_"))


def test_every_version_pin_is_exact() -> None:
    data = manifest()
    assert not [value for value in strings(data) if is_floating(value)]
    range_tokens = re.compile(r"[<>~^*]|==|!=|\.x\b|,")
    for path, value in leaves(data["stacks"]):
        key = path.rsplit(".", 1)[-1]
        if key in {"commit", "base_tag_commit", "tag_object"}:
            assert isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value), path
        elif key.endswith("digest"):
            assert value is None or re.fullmatch(r"sha256:[0-9a-f]{64}", value), path
        elif key in {"version", "e3nn", "ase", "pyyaml", "pytorch", "numpy", "scipy", "torch",
                     "python", "cuda", "apptainer", "cmake_project_version"}:
            assert isinstance(value, str) and value and not range_tokens.search(value), path
        elif key in {"image_tag", "probe_tag", "development_tag"}:
            assert re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", value), path
    stacks = data["stacks"]
    assert stacks["cp2k"]["upstream"]["version"] == "v2026.2"
    assert stacks["cp2k"]["upstream"]["commit"] == "67b5da876dd6a76b8b021d5a04d1c81ba79a4c50"
    assert stacks["cp2k_ml"]["upstream"] == {**stacks["cp2k_ml"]["upstream"],
                                             **{k: stacks["cp2k"]["upstream"][k]
                                                for k in ("version", "commit", "tag_object")}}


def test_cp2k_ml_manifest_has_no_null_version_pins() -> None:
    stack = manifest()["stacks"]["cp2k_ml"]
    assert stack["repository"] == "external"
    assert stack["source"] == "git-managed-fork"
    assert stack["upstream"]["build_system"] == "cmake"
    assert stack["build_targets"]["binary"] == "cp2k.psmp"
    assert stack["math_libraries"]["provider"] == "cp2k-toolchain"
    assert stack["status"] == "development"
    for section in ("upstream", "fork", "compiler", "mpi", "build_tools", "math_libraries",
                    "ml_runtime", "images"):
        assert all(value is not None for _, value in leaves(stack[section])), section


def test_gmd_and_sgt_train_share_one_torch() -> None:
    """A model exported by the training image must load in GMD's LibTorch."""
    stacks = manifest()["stacks"]
    gmd, sgt = stacks["gmd"], stacks["sgt_train"]
    for key in ("base_image", "base_image_digest"):
        assert gmd[key] == sgt[key]
    assert gmd["inherited"]["torch"] == sgt["inherited"]["torch"]
    assert gmd["gmd_sgt"]["commit"] == sgt["gmd_sgt"]["commit"]
    assert gmd["gmd_sgt"]["version"] == sgt["gmd_sgt"]["version"]
    assert gmd["python_packages"] == sgt["python_packages"]
    assert gmd["gmd"]["cmake"]["enable_torch"] is True
    assert gmd["probe_model"]["scientifically_valid"] is False
    assert gmd["probe_model"]["training_steps"] == 0


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_pins_exist_in_sibling_checkouts() -> None:
    stacks = manifest()["stacks"]
    checked = 0
    gmd_repo = WORKSPACE / "GMD"
    if (gmd_repo / ".git").exists():
        commit = stacks["gmd"]["gmd"]["commit"]
        cmake = _git(gmd_repo, "show", f"{commit}:CMakeLists.txt")
        assert cmake.returncode == 0, cmake.stderr
        assert f"project(GMD VERSION {stacks['gmd']['gmd']['cmake_project_version']}" in cmake.stdout
        assert "option(GMD_ENABLE_TORCH" in cmake.stdout
        tag = _git(gmd_repo, "rev-parse", f"{stacks['gmd']['gmd']['base_tag']}^{{commit}}")
        if tag.returncode == 0:
            assert tag.stdout.strip() == stacks["gmd"]["gmd"]["base_tag_commit"]
        checked += 1
    if (GMD_SGT_CHECKOUT / ".git").exists():
        commit = stacks["sgt_train"]["gmd_sgt"]["commit"]
        init = _git(GMD_SGT_CHECKOUT, "show", f"{commit}:gmd_sgt/__init__.py")
        assert init.returncode == 0, init.stderr
        assert f'__version__ = "{stacks["sgt_train"]["gmd_sgt"]["version"]}"' in init.stdout
        checked += 1
    lock = WORKSPACE / "cp2k-ml-scf" / "VERSION_LOCK.md"
    if lock.is_file():
        text = lock.read_text(encoding="utf-8")
        assert stacks["cp2k"]["upstream"]["commit"] in text
        assert stacks["cp2k_ml"]["ml_runtime"]["pytorch"] in text
        checked += 1
    if not checked:
        pytest.skip("no sibling checkouts (../GMD, ../GMD_SGT, ../cp2k-ml-scf) to compare against")


# --------------------------------------------------------------------------- build.sh


@requires_bash
@pytest.mark.parametrize("target", sorted(TARGETS))
def test_plan_lists_base_image_tag_and_build_args(target: str) -> None:
    dockerfile, stack_key, stage = TARGETS[target]
    stack = manifest()["stacks"][stack_key]
    resolved = plan(target)
    assert resolved["target"] == [target]
    assert resolved["platform"] == ["linux/arm64"]
    assert resolved["build_context"] == [str(ROOT)]
    assert resolved["dockerfile"] == [str(DOCKERFILES[dockerfile])]
    base = f"{stack['base_image']}@{stack['base_image_digest']}"
    assert resolved["base_image"] == [base]
    assert resolved["base_image_digest"] == [stack["base_image_digest"]]
    tag = resolved["image_tag"][0]
    expected_tag = {"cp2k-probe": stack.get("probe_tag"),
                    "cp2k-ml": stack.get("images", {}).get("development_tag"),
                    "cp2k-ml-probe": stack.get("images", {}).get("probe_tag")}.get(target, stack.get("image_tag"))
    assert tag == expected_tag
    assert resolved["image"] == [f"gmd-al/{target}:{tag}"]
    args = dict(line.split("=", 1) for line in resolved["build_arg"])
    assert args["BASE_IMAGE"] == base
    assert args["IMAGE_TAG"] == tag
    if target == "gmd":
        assert args["GMD_COMMIT"] == stack["gmd"]["commit"]
        assert args["GMD_SGT_COMMIT"] == stack["gmd_sgt"]["commit"]
        contexts = dict(line.split("=", 1) for line in resolved["named_context"])
        assert set(contexts) == {"gmd_src", "gmd_sgt_src"}
        assert contexts["gmd_src"].endswith(stack["gmd"]["commit"])
    elif target == "sgt-train":
        assert args["GMD_SGT_VERSION"] == stack["gmd_sgt"]["version"]
        assert args["GMD_SGT_COMMIT"] == stack["gmd_sgt"]["commit"]
    else:
        assert args["CP2K_VERSION"] == "v2026.2"
        assert args["CP2K_COMMIT"] == stack["upstream"]["commit"]
    if target.startswith("cp2k-ml"):
        assert resolved["cp2k_source"] == ["external"]
        assert resolved["source_mount"][0].endswith("cp2k-ml-scf:/workspace/cp2k-ml-scf")
        assert args["PYTORCH_VERSION"] == stack["ml_runtime"]["pytorch"]
        assert args["ASE_VERSION"] == stack["ml_runtime"]["ase"]


@requires_bash
def test_dry_run_prints_docker_commands_without_docker(tmp_path: Path) -> None:
    # A PATH without docker, git or apptainer proves --dry-run needs none of them.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("awk", "tr", "dirname", "basename", "cat", "sed"):
        located = shutil.which(tool)
        assert located, tool
        (bin_dir / tool).symlink_to(located)
    env = clean_env(PATH=str(bin_dir), GMD_AL_REGISTRY="registry.example")
    for command, expected in (
        (["--dry-run", "build", "gmd"], ["docker buildx build", "--build-context gmd_src=",
                                         "--target runtime", "--load", "git -C"]),
        (["push", "sgt-train", "--dry-run"], ["--push", "--metadata-file",
                                              "registry.example/gmd-al/sgt-train:"]),
        (["--dry-run", "oci", "cp2k"], ["type=oci", "oci_archive="]),
        (["--dry-run", "build-all"], ["target=gmd", "target=sgt-train", "target=cp2k"]),
    ):
        result = subprocess.run([BASH, str(BUILD_SCRIPT), *command], cwd=ROOT, env=env, text=True,
                                capture_output=True, check=False)
        assert result.returncode == 0, (command, result.stderr)
        for fragment in expected:
            assert fragment in result.stdout, (command, fragment)
        assert "dry_run=1" in result.stdout


@requires_bash
@pytest.mark.parametrize(
    ("args", "env", "message"),
    [
        (("plan", "gmd"), {"GMD_AL_IMAGE_TAG": "latest"}, "fixed tag or digest"),
        (("plan", "cp2k"), {"GMD_AL_BASE_IMAGE": "ubuntu:latest"}, "fixed tag or digest"),
        (("plan", "sgt-train"), {"GMD_AL_BASE_IMAGE": "nvcr.io/nvidia/pytorch:stable"}, "fixed tag or digest"),
        (("plan", "gmd"), {"GMD_AL_GMD_SOURCE": "https://github.com/kiana-0423/GMD.git#main"},
         "fixed tag or digest"),
        (("sif", "gmd"), {"GMD_AL_OCI_URI": "docker://registry.example/gmd-al/gmd:main"}, "fixed tag or digest"),
        (("tag", "cp2k", "latest"), {}, "fixed tag or digest"),
    ],
)
def test_floating_refs_are_rejected(args: tuple[str, ...], env: dict[str, str], message: str) -> None:
    result = run_build(*args, env=clean_env(**env))
    assert result.returncode != 0
    assert message in result.stderr


def _manifest_variant(tmp_path: Path, old: str, new: str) -> Path:
    text = VERSIONS.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "versions.yaml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


@requires_bash
@pytest.mark.parametrize(
    ("old", "new", "target", "message"),
    [
        ('commit: "54ac558e30f9fb85a3474f7749016e24aa109a46"', 'commit: "main"', "gmd", "fixed tag or digest"),
        ('commit: "54ac558e30f9fb85a3474f7749016e24aa109a46"', 'commit: "54ac558"', "gmd", "40-character"),
        ('version: "v2026.2"', 'version: "latest"', "cp2k", "fixed tag or digest"),
        ('e3nn: "0.6.0"', 'e3nn: ">=0.5"', "gmd", "exact version"),
        ('image_tag: "cp2k-v2026.2-a1"', 'image_tag: "stable"', "cp2k", "fixed tag or digest"),
        ('version: "0.4.0"', 'version: "<GMD_SGT_VERSION>"', "gmd", "placeholder"),
    ],
)
def test_manifest_with_floating_or_inexact_pin_is_refused(tmp_path: Path, old: str, new: str, target: str,
                                                          message: str) -> None:
    variant = _manifest_variant(tmp_path, old, new)
    result = run_build("plan", target, env=clean_env(GMD_AL_VERSIONS_FILE=str(variant)))
    assert result.returncode == 2
    assert message in result.stderr


@requires_bash
def test_unpinned_base_digest_is_reported_and_push_refused(tmp_path: Path) -> None:
    digest = manifest()["stacks"]["cp2k"]["base_image_digest"]
    variant = _manifest_variant(tmp_path, f'base_image_digest: "{digest}"', "base_image_digest: null")
    env = clean_env(GMD_AL_VERSIONS_FILE=str(variant), GMD_AL_REGISTRY="registry.example")
    result = run_build("plan", "cp2k", env=env)
    assert result.returncode == 0, result.stderr
    assert "base_image=ubuntu:24.04\n" in result.stdout
    assert "base_image_digest=unpinned" in result.stdout
    pushed = run_build("--dry-run", "push", "cp2k", env=env)
    assert pushed.returncode == 2
    assert "refusing to push" in pushed.stderr
    bad = _manifest_variant(tmp_path, f'base_image_digest: "{digest}"', 'base_image_digest: "sha256:abc"')
    assert run_build("plan", "cp2k", env=clean_env(GMD_AL_VERSIONS_FILE=str(bad))).returncode == 2


@requires_bash
def test_sif_requires_registry_uri_before_apptainer() -> None:
    result = run_build("sif", "sgt-train")
    assert result.returncode == 2
    assert "GMD_AL_OCI_URI is required" in result.stderr
    assert "Docker daemon" in result.stderr


@requires_bash
def test_unknown_target_is_rejected() -> None:
    result = run_build("plan", "md-probe")
    assert result.returncode == 2
    assert "unknown target" in result.stderr


def test_build_script_contract() -> None:
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert '"${PROJECT_ROOT}"' in text
    for variable in ("GMD_AL_PLATFORM", "GMD_AL_BASE_IMAGE", "GMD_AL_IMAGE_TAG", "GMD_AL_REGISTRY",
                     "GMD_AL_OCI_URI", "GMD_AL_GMD_SOURCE", "GMD_AL_GMD_SGT_SOURCE", "GMD_AL_CP2K_ML_SOURCE"):
        assert variable in text
    assert "--metadata-file" in text
    assert "PHAL_" not in text
    assert "Dockerfile.gmd" not in text  # target resolution stays generic
    assert 'ROLE_TARGETS="gmd sgt-train cp2k"' in text
    assert "git archive" in text  # images are built from the pinned commit, not a work tree
    assert "sha256" in text and "sif_sha256" in text


# --------------------------------------------------------------------------- validators


PLATFORM_KEYS = {
    "machine", "python", "platform", "torch", "torch_cuda", "cuda_available", "cuda_device_count",
    "gpu_name", "cuda_device_capability", "cuda_tensor_test", "gmd_sgt", "e3nn", "gmd_sgt_import",
    "required_modules", "require_cuda", "errors", "failures", "package_versions", "status",
}


def test_validate_platform_emits_documented_keys_on_this_host() -> None:
    env = gmd_sgt_env() or clean_env(PYTHONDONTWRITEBYTECODE="1")
    code, payload = run_json([sys.executable, str(SCRIPTS / "validate_platform.py")], env)
    assert set(payload) == PLATFORM_KEYS
    assert payload["require_cuda"] is False
    assert payload["status"] in {"passed", "failed"}
    assert code == (0 if payload["status"] == "passed" else 1)
    assert payload["cuda_tensor_test"] is (payload["cuda_available"] and payload["cuda_tensor_test"])


def test_validate_platform_require_cuda_fails_without_cuda() -> None:
    env = gmd_sgt_env() or clean_env(PYTHONDONTWRITEBYTECODE="1")
    probe = subprocess.run([sys.executable, "-c", "import torch, sys; sys.exit(torch.cuda.is_available())"],
                           env=env, capture_output=True, check=False)
    if probe.returncode != 0:
        pytest.skip("this host has CUDA (or no torch); the no-CUDA failure path cannot be shown")
    env["GMD_AL_REQUIRE_CUDA"] = "1"
    code, payload = run_json([sys.executable, str(SCRIPTS / "validate_platform.py")], env)
    assert code == 1
    assert payload["status"] == "failed"
    assert payload["require_cuda"] is True
    assert "torch imported but CUDA is unavailable" in payload["errors"]


def test_validate_platform_reports_a_missing_torch_instead_of_crashing(tmp_path: Path) -> None:
    (tmp_path / "torch.py").write_text("raise ImportError('simulated: torch is not installed')\n")
    env = clean_env(PYTHONPATH=str(tmp_path), PYTHONDONTWRITEBYTECODE="1", GMD_AL_REQUIRE_CUDA="1")
    code, payload = run_json([sys.executable, str(SCRIPTS / "validate_platform.py")], env)
    assert code == 1
    assert set(payload) == PLATFORM_KEYS
    assert payload["torch"] is None
    assert payload["status"] == "failed"
    assert any(f["stage"] == "import-torch" and "simulated" in f["exception_message"]
               for f in payload["failures"])
    assert "CUDA is required but torch could not be imported" in payload["errors"]


def test_validator_sources_record_gpu_and_inference_evidence() -> None:
    platform_validator = (SCRIPTS / "validate_platform.py").read_text(encoding="utf-8")
    sgt_validator = (SCRIPTS / "validate_sgt_stack.py").read_text(encoding="utf-8")
    assert "GMD_AL_REQUIRE_CUDA" in platform_validator
    assert 'environ.get("PHAL_REQUIRE_CUDA")' not in platform_validator + sgt_validator
    assert "get_device_capability" in platform_validator
    assert 'device="cuda"' in platform_validator
    assert "cuda_tensor_test" in platform_validator
    assert "traceback.format_exc()" in platform_validator
    for field in ("energy_finite", "forces_finite", "force_gradient_computed", "gmd-sgt-{device}-inference",
                  "script_model", "GMD_AL_REQUIRE_CUDA"):
        assert field in sgt_validator


@pytest.mark.integration
def test_validate_sgt_stack_runs_a_random_tiny_model_on_cpu() -> None:
    env = gmd_sgt_env()
    if env is None:
        pytest.skip("torch + gmd_sgt are not importable (no ../GMD_SGT checkout or install)")
    code, payload = run_json([sys.executable, "-W", "ignore", str(SCRIPTS / "validate_sgt_stack.py")], env)
    assert code == 0, payload["errors"]
    assert payload["status"] == "passed"
    cpu = payload["inference"]["cpu"]
    assert cpu["finite_energy_and_forces"] and cpu["forces_nonzero"] and cpu["force_gradient_computed"]
    assert payload["torchscript"]["status"] == "passed"
    assert payload["probe"]["scientifically_valid"] is False
    assert payload["inference"]["cuda"]["status"] in {"passed", "skipped"}


def test_gmd_log_parser_gates_on_finite_energies(tmp_path: Path) -> None:
    module = load_script("validate_gmd_stack.py")
    header = ("# step  time[fs]  PE[eV]  KE[eV]  E_total[eV]  T[K]  P[bar]  V[A^3]  shake_iter  "
              "shake_error[A]  rattle_iter  rattle_error[A/fs]  P_valid\n# comment line\n")
    good = tmp_path / "good.log"
    good.write_text(header + "0  0.0  -1.7  0.0001  -1.6999  1.0  nan  1728.0  0  0.0  0  0.0  0\n"
                             "1  0.1  -1.7  0.0001  -1.6999  1.0  nan  1728.0  0  0.0  0  0.0  0\n")
    parsed = module.parse_log(good)
    assert parsed["steps"] == [0, 1] and parsed["finite"] is True  # nan P with P_valid 0 is fine
    bad = tmp_path / "bad.log"
    bad.write_text(header + "0  0.0  nan  0.0001  nan  1.0  1.0  1728.0  0  0.0  0  0.0  1\n")
    assert module.parse_log(bad)["finite"] is False
    invalid_pressure = tmp_path / "pressure.log"
    invalid_pressure.write_text(header + "0  0.0  -1.7  0.0001  -1.6999  1.0  inf  1728.0  0  0.0  0  0.0  1\n")
    assert module.parse_log(invalid_pressure)["finite"] is False
    assert module.parse_log(tmp_path / "missing.log")["finite"] is False


def test_gmd_static_check_fails_for_a_missing_binary(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPTS / "validate_gmd_stack.py"), "--mode", "static",
         "--gmd", str(tmp_path / "nope" / "gmd"), "--probe-dir", str(tmp_path)],
        capture_output=True, text=True, check=False, env=clean_env(PYTHONDONTWRITEBYTECODE="1"))
    payload = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert payload["overall_status"] == "FAIL"
    assert payload["static"]["gmd_exists"] is False
    assert payload["gmd_runtime"] == "not_run"


@pytest.mark.integration
def test_validate_gmd_stack_runs_0_and_10_steps_with_a_random_sgt_model(tmp_path: Path) -> None:
    env = gmd_sgt_env()
    gmd = local_gmd()
    if env is None or gmd is None:
        pytest.skip("needs gmd built with GMD_ENABLE_TORCH=ON ($GMD_EXECUTABLE or ../GMD/build-torch/gmd) "
                    "and an importable gmd_sgt")
    command = [sys.executable, "-W", "ignore", str(SCRIPTS / "validate_gmd_stack.py"), "--mode", "runtime",
               "--gmd", str(gmd), "--probe-dir", str(CONTAINERS / "probes" / "gmd"), "--export-model",
               "--workdir", str(tmp_path / "work")]
    code, payload = run_json(command, env)
    assert code == 0, (payload["errors"], payload.get("run0", {}).get("stderr_tail"))
    assert payload["overall_status"] == "PASS"
    assert payload["static_validation"] == "passed"
    assert payload["run0"]["log"]["steps"][0] == 0 and payload["run0_passed"]
    assert payload["run10"]["log"]["steps"] == list(range(11)) and payload["run10_passed"]
    assert payload["run10"]["status_file"]["model_sha256"] == payload["probe_model_sha256"]
    metadata = payload["probe_model_metadata"]
    assert metadata["purpose"] == "integration_probe_only"
    assert metadata["scientifically_valid"] is False and metadata["training_steps"] == 0
    # Static mode against the same binary and model passes without running gmd.
    code, static = run_json([sys.executable, "-W", "ignore", str(SCRIPTS / "validate_gmd_stack.py"), "--mode",
                             "static", "--gmd", str(gmd), "--model", str(tmp_path / "work" / "sgt-probe.pt")], env)
    assert code == 0 and static["overall_status"] == "PASS" and static["run10"] == "not_run"


def _fake_cp2k(tmp_path: Path, converged: bool) -> Path:
    script = tmp_path / ("cp2k-good" if converged else "cp2k-bad")
    line = "  *** SCF run converged in    10 steps ***" if converged else "  *** SCF run NOT converged ***"
    script.write_text(f"""#!{sys.executable}
import sys
if "--version" in sys.argv:
    print(" CP2K version 2026.2\\n Source code revision git:c92cc08\\n cp2kflags: omp libint mpi_f08")
    sys.exit(0)
out = sys.argv[sys.argv.index("-o") + 1]
with open(out, "w") as handle:
    handle.write({line!r} + "\\n")
    handle.write(" ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]   -1.165933374533096\\n")
    handle.write("  **** **** ******  **  PROGRAM ENDED AT  2026-11-06\\n")
""")
    script.chmod(0o755)
    return script


@pytest.mark.parametrize("converged", [True, False])
def test_validate_cp2k_stack_logic_with_a_scripted_executable(tmp_path: Path, converged: bool) -> None:
    fake = _fake_cp2k(tmp_path, converged)
    completed = subprocess.run(
        [sys.executable, str(SCRIPTS / "validate_cp2k_stack.py"), "--cp2k", str(fake),
         "--expected-version", "v2026.2", "--workdir", str(tmp_path / "work")],
        capture_output=True, text=True, check=False, env=clean_env(PYTHONDONTWRITEBYTECODE="1"))
    payload = json.loads(completed.stdout)
    assert payload["version_check"] == "passed"
    assert payload["scf_converged"] is converged
    assert payload["overall_status"] == ("passed" if converged else "failed")
    assert completed.returncode == (0 if converged else 1)
    wrong = subprocess.run(
        [sys.executable, str(SCRIPTS / "validate_cp2k_stack.py"), "--cp2k", str(fake),
         "--expected-version", "v2024.1", "--workdir", str(tmp_path / "wrong")],
        capture_output=True, text=True, check=False, env=clean_env(PYTHONDONTWRITEBYTECODE="1"))
    assert wrong.returncode == 1 and json.loads(wrong.stdout)["version_check"] == "failed"


@pytest.mark.cp2k
def test_validate_cp2k_stack_converges_h2_with_a_real_cp2k(tmp_path: Path) -> None:
    executable = os.environ.get("CP2K_EXE") or shutil.which("cp2k.psmp")
    if executable is None:
        pytest.skip("no cp2k.psmp on PATH (or $CP2K_EXE)")
    expected = manifest()["stacks"]["cp2k"]["upstream"]["version"]
    env = clean_env(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    code, payload = run_json([sys.executable, str(SCRIPTS / "validate_cp2k_stack.py"), "--cp2k", executable,
                              "--workdir", str(tmp_path / "h2")], env, timeout=900)
    if payload.get("cp2k_version") != expected.lstrip("v"):
        pytest.skip(f"host CP2K is {payload.get('cp2k_version')}, not the pinned {expected}")
    assert code == 0, payload["errors"]
    assert payload["scf_converged"] and payload["energy_finite"]
    assert "SCF run converged" in (tmp_path / "h2" / "h2.out").read_text()


def test_cp2k_probe_input_is_native_dft() -> None:
    text = (CONTAINERS / "probes" / "cp2k" / "h2.inp").read_text(encoding="utf-8")
    assert "METHOD QUICKSTEP" in text
    assert "MAX_SCF 50" in text
    assert "RUN_TYPE ENERGY" in text
    assert "ML_SCF" not in text
    assert "/workspace/" not in text  # data files resolve through CP2K's data directory
    validator = (SCRIPTS / "validate_cp2k_stack.py").read_text(encoding="utf-8")
    assert '"SCF run converged"' in validator


def test_gmd_probe_model_is_marked_non_scientific() -> None:
    model_text = (SCRIPTS / "build_gmd_probe_model.py").read_text(encoding="utf-8")
    readme = (CONTAINERS / "probes" / "gmd" / "README.md").read_text(encoding="utf-8")
    for token in ("integration_probe_only", "scientifically_valid", "training_steps"):
        assert token in model_text
    assert "integration_probe_only" in readme
    assert "scientifically_valid = false" in readme
    assert "not a formal MD simulation" in readme
    for run_file in ("run0.run", "run10.run"):
        text = (CONTAINERS / "probes" / "gmd" / run_file).read_text(encoding="utf-8")
        assert "force_field ml" in text and "model_path sgt-probe.pt" in text
        assert "status_file status.json" in text


# --------------------------------------------------------------------------- PBS templates


def test_pbs_templates_present() -> None:
    names = {path.name for path in PBS_SCRIPTS}
    assert {"validate_gmd_probe.pbs", "validate_sgt_train_probe.pbs", "validate_cp2k_probe.pbs",
            "diagnose_hpcx.pbs"} <= names


@requires_bash
@pytest.mark.parametrize("path", [*PBS_SCRIPTS, PBS_DIR / "submit.sh"], ids=lambda p: p.name)
def test_pbs_scripts_pass_bash_syntax_check(path: Path) -> None:
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("path", PBS_SCRIPTS, ids=lambda p: p.name)
def test_pbs_templates_compute_the_sif_digest_and_hard_code_nothing(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert not HEX64.search(text), "hard-coded sha256 (PHAL F-32)"
    assert re.search(r'SIF_SHA256=\$\(sha256sum "\$\{?\w+\}?" \| awk', text)
    assert '"sif_sha256":' in text
    assert "group_list=" not in "\n".join(line for line in text.splitlines() if line.startswith("#PBS"))
    assert "gw34" not in text and "<MIYABI_" not in text
    assert re.search(r"^#PBS -q \S+", text, re.M)  # default queue, overridable with qsub -q
    assert "module purge" in text
    assert 'module load "${GMD_AL_APPTAINER_MODULE}"' in text
    assert 'cd "${PBS_O_WORKDIR' in text
    assert "APPTAINER_TMPDIR" in text
    assert "apptainer exec --cleanenv --nv" in text
    assert "NOT RUN YET" in text
    assert "PHAL_" not in text


@pytest.mark.parametrize("name", ["validate_gmd_probe.pbs", "validate_sgt_train_probe.pbs", "validate_cp2k_probe.pbs"])
def test_validation_templates_bind_runtime_and_apply_the_mpi_workaround(name: str) -> None:
    text = (PBS_DIR / name).read_text(encoding="utf-8")
    assert '--bind "${RUNTIME_HOST_DIR}:/runtime"' in text
    assert "unset OPAL_PREFIX" in text
    if name != "validate_cp2k_probe.pbs":
        assert '/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:${LD_LIBRARY_PATH:-}' in text
        assert "APPTAINERENV_GMD_AL_REQUIRE_CUDA=1" in text
    assert "summary-${PBS_JOBID}.json" in text


def _fake_pbs_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    (bin_dir / "module").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "apptainer").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = version ]; then echo 1.3.5; exit 0; fi\n'
        'echo "{\\"fake_apptainer\\": true}"\n'
        'exit "${FAKE_APPTAINER_RC:-0}"\n')
    (bin_dir / "date").write_text("#!/bin/sh\necho 2026-11-06T00:00:00+09:00\n")
    if shutil.which("sha256sum") is None:
        (bin_dir / "sha256sum").write_text('#!/bin/sh\nexec shasum -a 256 "$@"\n')
    for path in bin_dir.iterdir():
        path.chmod(0o755)
    return bin_dir


@requires_bash
@pytest.mark.parametrize("rc", [0, 1])
@pytest.mark.parametrize(
    ("name", "image_var", "stages_expected"),
    [
        ("validate_gmd_probe.pbs", "GMD_AL_GMD_IMAGE", {"platform", "gmd_static", "gmd_runtime"}),
        ("validate_sgt_train_probe.pbs", "GMD_AL_SGT_TRAIN_IMAGE", {"platform", "sgt_stack"}),
        ("validate_cp2k_probe.pbs", "GMD_AL_CP2K_IMAGE", {"cp2k_h2"}),
    ],
)
def test_pbs_job_summary_hashes_the_image_at_job_time(tmp_path: Path, name: str, image_var: str,
                                                      stages_expected: set[str], rc: int) -> None:
    """Run the job script with stub module/apptainer: the summary carries the real SIF sha256."""
    image = tmp_path / "probe.sif"
    image.write_bytes(os.urandom(4096))
    workdir = tmp_path / "submit"
    workdir.mkdir()
    env = clean_env(
        PATH=f"{_fake_pbs_bin(tmp_path)}{os.pathsep}{os.environ.get('PATH', '')}",
        PBS_O_WORKDIR=str(workdir), PBS_JOBID="12345.opbs", PBS_QUEUE="debug-g",
        TMPDIR=str(tmp_path / "tmp"), GMD_AL_PBS_GROUP="testgroup", FAKE_APPTAINER_RC=str(rc),
        **{image_var: str(image)})
    result = subprocess.run(["bash", str(PBS_DIR / name)], cwd=workdir, env=env, capture_output=True,
                            text=True, check=False, timeout=60)
    assert result.returncode == rc, result.stderr
    summary_files = list((workdir / "runtime" / "logs").rglob("summary-12345.opbs.json"))
    assert len(summary_files) == 1
    summary = json.loads(summary_files[0].read_text(encoding="utf-8"))
    assert summary["sif_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert summary["job_id"] == "12345.opbs"
    assert summary["queue"] == "debug-g" and summary["group"] == "testgroup"
    assert set(summary["stages"]) == stages_expected
    assert summary["status"] == ("passed" if rc == 0 else "failed")
    assert summary["exit_status"] == rc


@requires_bash
def test_pbs_job_refuses_a_missing_image(tmp_path: Path) -> None:
    env = clean_env(PATH=f"{_fake_pbs_bin(tmp_path)}{os.pathsep}{os.environ.get('PATH', '')}",
                    PBS_O_WORKDIR=str(tmp_path), PBS_JOBID="1.opbs",
                    GMD_AL_GMD_IMAGE=str(tmp_path / "absent.sif"))
    result = subprocess.run(["bash", str(PBS_DIR / "validate_gmd_probe.pbs")], cwd=tmp_path, env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "SIF image not found" in result.stderr


@requires_bash
def test_submit_takes_queue_and_group_from_variables() -> None:
    script = PBS_DIR / "submit.sh"
    result = subprocess.run(["bash", str(script), "--dry-run", "gmd", "/images/gmd.sif"], capture_output=True,
                            text=True, check=False, env=clean_env(GMD_AL_PBS_QUEUE="regular-g",
                                                                  GMD_AL_PBS_GROUP="gx99"))
    assert result.returncode == 0, result.stderr
    assert "qsub -q regular-g -W group_list=gx99" in result.stdout
    assert "GMD_AL_GMD_IMAGE=/images/gmd.sif" in result.stdout
    assert "validate_gmd_probe.pbs" in result.stdout
    missing = subprocess.run(["bash", str(script), "--dry-run", "cp2k", "/images/cp2k.sif"],
                             capture_output=True, text=True, check=False, env=clean_env())
    assert missing.returncode == 2 and "GMD_AL_PBS_GROUP is required" in missing.stderr


# --------------------------------------------------------------------------- docs and provenance


def test_container_docs_describe_the_pbs_flow_honestly() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in DOCS.glob("*.md"))
    assert "Slurm launch semantics" not in combined
    assert "PBS" in combined
    for phrase in ("OCI", "SIF", "/runtime", "not built", "versions.yaml", "Dockerfile.gmd.dockerignore"):
        assert phrase in combined, phrase
    readme = (CONTAINERS / "README.md").read_text(encoding="utf-8")
    assert "not built" in readme.lower() and "docs/containers" in readme


BASELINE_FILES = ("milestone3-baseline.md", "baseline.yaml", "manifest.json")


def test_release_baseline_archive_is_documented_as_historical() -> None:
    for name in BASELINE_FILES:
        assert (BASELINE_DIR / name).is_file()
    readme = (BASELINE_DIR / "README.md").read_text(encoding="utf-8")
    for phrase in ("96263b6", "git-ignored", "LAMMPS", "predates", "verbatim"):
        assert phrase in readme, phrase
    yaml.safe_load((BASELINE_DIR / "baseline.yaml").read_text(encoding="utf-8"))
    json.loads((BASELINE_DIR / "manifest.json").read_text(encoding="utf-8"))


@pytest.mark.skipif(not (PHAL_REF / "docs" / "releases").is_dir(), reason="ref/lammps_monitor is not present")
@pytest.mark.parametrize("name", BASELINE_FILES)
def test_release_baseline_copies_are_byte_identical_to_phal(name: str) -> None:
    assert (BASELINE_DIR / name).read_bytes() == (PHAL_REF / "docs" / "releases" / name).read_bytes()
