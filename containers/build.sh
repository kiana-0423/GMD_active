#!/usr/bin/env bash
# Ported from PHAL (containers/build.sh) at 96263b6; MIT License, see NOTICE.
#
# Container build orchestrator for the GMD_active images (GMD, GMD_SGT, CP2K).
# `plan` and `--dry-run` need only bash + awk: no docker, git or apptainer.
# Kept compatible with bash 3.2 (macOS /bin/bash).
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
VERSIONS_FILE=${GMD_AL_VERSIONS_FILE:-"${SCRIPT_DIR}/versions.yaml"}
OUTPUT_DIR=${GMD_AL_CONTAINER_OUTPUT_DIR:-"${PROJECT_ROOT}/build/containers"}
EVIDENCE_DIR=${GMD_AL_EVIDENCE_DIR:-"${PROJECT_ROOT}/build/evidence"}
IMAGE_NAMESPACE=${GMD_AL_IMAGE_NAMESPACE:-"gmd-al"}
REGISTRY=${GMD_AL_REGISTRY:-}
TARGETS="gmd sgt-train cp2k cp2k-probe cp2k-ml cp2k-ml-probe"
# Production role images built by build-all (cp2k-ml is experimental tooling).
ROLE_TARGETS="gmd sgt-train cp2k"
DRY_RUN=0

# ---------------------------------------------------------------- manifest

# Resolve a dotted key path (a.b.c) to its scalar value. Supports the subset of
# YAML used by versions.yaml: 2-space indentation, scalar values, optional
# double quotes, trailing "# comment" on unquoted values.
yaml_path_value() {
    path=$1
    awk -v target="${path}" '
        function clean(value,    end) {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            if (substr(value, 1, 1) == "\"") {
                end = index(substr(value, 2), "\"")
                if (end > 0) return substr(value, 2, end - 1)
                return substr(value, 2)
            }
            sub(/[[:space:]]+#.*$/, "", value)
            return value
        }
        /^[[:space:]]*(#|$)/ { next }
        /^[[:space:]]*-/ { next }
        {
            match($0, /^[ ]*/)
            level = int(RLENGTH / 2) + 1
            content = substr($0, RLENGTH + 1)
            colon = index(content, ":")
            if (colon == 0) next
            key = clean(substr(content, 1, colon - 1))
            value = clean(substr(content, colon + 1))
            stack[level] = key
            for (i = level + 1; i <= 12; i++) delete stack[i]
            current = stack[1]
            for (i = 2; i <= level; i++) current = current "." stack[i]
            if (current == target && value != "") {
                print value
                exit
            }
        }
    ' "${VERSIONS_FILE}"
}

# ---------------------------------------------------------------- guards

die() {
    echo "build.sh: $*" >&2
    exit 2
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_value() {
    name=$1
    value=$2
    if [[ -z "${value}" || "${value}" == "null" || "${value}" == "~" ]]; then
        die "required value is not configured: ${name}"
    fi
    case "${value}" in
        *"<"*">"*|*TODO*|*PLACEHOLDER*|*placeholder*)
            die "${name} is a placeholder, not a pin: ${value}" ;;
    esac
}

lower() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

# Floating references are never accepted: they make a build irreproducible.
reject_floating_ref() {
    name=$1
    value=$(lower "$2")
    case "${value}" in
        latest|stable|main|master|head|*:latest|*:stable|*:main|*:master|*"#main"|*"#master"|*"#head")
            die "${name} must use a fixed tag or digest, got: $2" ;;
    esac
}

# An exact version: no ranges, wildcards or floating names.
require_exact_version() {
    name=$1
    value=$2
    require_value "${name}" "${value}"
    reject_floating_ref "${name}" "${value}"
    case "${value}" in
        *[\<\>=~^\*,\ ]*|*.x|*.x.*)
            die "${name} must be an exact version, got: ${value}" ;;
    esac
}

require_commit() {
    name=$1
    value=$2
    require_value "${name}" "${value}"
    reject_floating_ref "${name}" "${value}"
    if [[ ! "${value}" =~ ^[0-9a-f]{40}$ ]]; then
        die "${name} must be a full 40-character commit hash, got: ${value}"
    fi
}

validate_target() {
    case " ${TARGETS} " in
        *" $1 "*) ;;
        *) die "unknown target: $1 (expected: ${TARGETS})" ;;
    esac
}

# ---------------------------------------------------------------- resolution

stack_for() {
    case "$1" in
        gmd) echo gmd ;;
        sgt-train) echo sgt_train ;;
        cp2k|cp2k-probe) echo cp2k ;;
        cp2k-ml|cp2k-ml-probe) echo cp2k_ml ;;
    esac
}

dockerfile_for() {
    case "$1" in
        cp2k-ml|cp2k-ml-probe) echo "${SCRIPT_DIR}/Dockerfile.cp2k-ml" ;;
        cp2k-probe) echo "${SCRIPT_DIR}/Dockerfile.cp2k" ;;
        *) echo "${SCRIPT_DIR}/Dockerfile.$1" ;;
    esac
}

build_stage_for() {
    case "$1" in
        gmd|sgt-train|cp2k) echo runtime ;;
        cp2k-probe|cp2k-ml-probe) echo probe ;;
        cp2k-ml) echo development ;;
    esac
}

is_remote_source() {
    case "$1" in
        https://*|http://*|git@*|ssh://*|git://*) return 0 ;;
        *) return 1 ;;
    esac
}

# Sets <PREFIX>_CONTEXT (what docker receives) and <PREFIX>_SOURCE_KIND.
resolve_source() {
    label=$1
    source=$2
    commit=$3
    if is_remote_source "${source}"; then
        reject_floating_ref "${label} source" "${source}"
        RESOLVED_CONTEXT="${source}#${commit}"
        RESOLVED_KIND="git-remote"
    else
        RESOLVED_CONTEXT="${OUTPUT_DIR}/sources/${label}-${commit}"
        RESOLVED_KIND="git-archive"
    fi
}

version_context() {
    target=$1
    STACK=$(stack_for "${target}")
    PLATFORM=${GMD_AL_PLATFORM:-$(yaml_path_value platform.default_oci_platform)}
    BASE_IMAGE_DIGEST=$(yaml_path_value "stacks.${STACK}.base_image_digest")
    manifest_base=$(yaml_path_value "stacks.${STACK}.base_image")
    BASE_IMAGE=${GMD_AL_BASE_IMAGE:-${manifest_base}}
    if [[ -z "${GMD_AL_BASE_IMAGE:-}" && -n "${BASE_IMAGE_DIGEST}" && "${BASE_IMAGE_DIGEST}" != "null" ]]; then
        if [[ ! "${BASE_IMAGE_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
            die "stacks.${STACK}.base_image_digest is not a sha256 digest: ${BASE_IMAGE_DIGEST}"
        fi
        BASE_IMAGE="${BASE_IMAGE}@${BASE_IMAGE_DIGEST}"
    fi
    case "${BASE_IMAGE}" in
        *@sha256:*) BASE_PINNED=1 ;;
        *) BASE_PINNED=0 ;;
    esac
    if [[ "${target}" == "cp2k-ml" ]]; then
        IMAGE_TAG=${GMD_AL_IMAGE_TAG:-$(yaml_path_value stacks.cp2k_ml.images.development_tag)}
    elif [[ "${target}" == "cp2k-ml-probe" ]]; then
        IMAGE_TAG=${GMD_AL_IMAGE_TAG:-$(yaml_path_value stacks.cp2k_ml.images.probe_tag)}
    elif [[ "${target}" == "cp2k-probe" ]]; then
        IMAGE_TAG=${GMD_AL_IMAGE_TAG:-$(yaml_path_value stacks.cp2k.probe_tag)}
    else
        IMAGE_TAG=${GMD_AL_IMAGE_TAG:-$(yaml_path_value "stacks.${STACK}.image_tag")}
    fi

    require_value GMD_AL_PLATFORM "${PLATFORM}"
    require_value GMD_AL_BASE_IMAGE "${BASE_IMAGE}"
    require_value GMD_AL_IMAGE_TAG "${IMAGE_TAG}"
    reject_floating_ref GMD_AL_BASE_IMAGE "${BASE_IMAGE}"
    reject_floating_ref GMD_AL_IMAGE_TAG "${IMAGE_TAG}"
    if [[ ! "${IMAGE_TAG}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]; then
        die "GMD_AL_IMAGE_TAG is not a valid image tag: ${IMAGE_TAG}"
    fi

    # Build args (NAME=value), named build contexts (name=location) and notes.
    ARG_LINES=("BASE_IMAGE=${BASE_IMAGE}" "IMAGE_TAG=${IMAGE_TAG}")
    CONTEXT_LINES=()
    LOCAL_SOURCES=()
    case "${target}" in
        gmd)
            GMD_VERSION=$(yaml_path_value stacks.gmd.gmd.version)
            GMD_COMMIT=$(yaml_path_value stacks.gmd.gmd.commit)
            GMD_CMAKE_VERSION=$(yaml_path_value stacks.gmd.gmd.cmake_project_version)
            GMD_SGT_VERSION=$(yaml_path_value stacks.gmd.gmd_sgt.version)
            GMD_SGT_COMMIT=$(yaml_path_value stacks.gmd.gmd_sgt.commit)
            E3NN_VERSION=$(yaml_path_value stacks.gmd.python_packages.e3nn)
            ASE_VERSION=$(yaml_path_value stacks.gmd.python_packages.ase)
            PYYAML_VERSION=$(yaml_path_value stacks.gmd.python_packages.pyyaml)
            require_exact_version gmd_version "${GMD_VERSION}"
            require_commit gmd_commit "${GMD_COMMIT}"
            require_exact_version gmd_cmake_project_version "${GMD_CMAKE_VERSION}"
            require_exact_version gmd_sgt_version "${GMD_SGT_VERSION}"
            require_commit gmd_sgt_commit "${GMD_SGT_COMMIT}"
            require_exact_version e3nn "${E3NN_VERSION}"
            require_exact_version ase "${ASE_VERSION}"
            require_exact_version pyyaml "${PYYAML_VERSION}"
            ARG_LINES+=(
                "GMD_VERSION=${GMD_VERSION}"
                "GMD_COMMIT=${GMD_COMMIT}"
                "GMD_CMAKE_VERSION=${GMD_CMAKE_VERSION}"
                "GMD_SGT_VERSION=${GMD_SGT_VERSION}"
                "GMD_SGT_COMMIT=${GMD_SGT_COMMIT}"
                "E3NN_VERSION=${E3NN_VERSION}"
                "ASE_VERSION=${ASE_VERSION}"
                "PYYAML_VERSION=${PYYAML_VERSION}"
            )
            gmd_source=${GMD_AL_GMD_SOURCE:-"${PROJECT_ROOT}/../GMD"}
            resolve_source gmd "${gmd_source}" "${GMD_COMMIT}"
            CONTEXT_LINES+=("gmd_src=${RESOLVED_CONTEXT}")
            if [[ "${RESOLVED_KIND}" == "git-archive" ]]; then LOCAL_SOURCES+=("gmd|${gmd_source}|${GMD_COMMIT}|${RESOLVED_CONTEXT}"); fi
            sgt_source=${GMD_AL_GMD_SGT_SOURCE:-"${PROJECT_ROOT}/../GMD_SGT"}
            resolve_source gmd_sgt "${sgt_source}" "${GMD_SGT_COMMIT}"
            CONTEXT_LINES+=("gmd_sgt_src=${RESOLVED_CONTEXT}")
            if [[ "${RESOLVED_KIND}" == "git-archive" ]]; then LOCAL_SOURCES+=("gmd_sgt|${sgt_source}|${GMD_SGT_COMMIT}|${RESOLVED_CONTEXT}"); fi
            ;;
        sgt-train)
            GMD_SGT_VERSION=$(yaml_path_value stacks.sgt_train.gmd_sgt.version)
            GMD_SGT_COMMIT=$(yaml_path_value stacks.sgt_train.gmd_sgt.commit)
            E3NN_VERSION=$(yaml_path_value stacks.sgt_train.python_packages.e3nn)
            ASE_VERSION=$(yaml_path_value stacks.sgt_train.python_packages.ase)
            PYYAML_VERSION=$(yaml_path_value stacks.sgt_train.python_packages.pyyaml)
            require_exact_version gmd_sgt_version "${GMD_SGT_VERSION}"
            require_commit gmd_sgt_commit "${GMD_SGT_COMMIT}"
            require_exact_version e3nn "${E3NN_VERSION}"
            require_exact_version ase "${ASE_VERSION}"
            require_exact_version pyyaml "${PYYAML_VERSION}"
            ARG_LINES+=(
                "GMD_SGT_VERSION=${GMD_SGT_VERSION}"
                "GMD_SGT_COMMIT=${GMD_SGT_COMMIT}"
                "E3NN_VERSION=${E3NN_VERSION}"
                "ASE_VERSION=${ASE_VERSION}"
                "PYYAML_VERSION=${PYYAML_VERSION}"
            )
            sgt_source=${GMD_AL_GMD_SGT_SOURCE:-"${PROJECT_ROOT}/../GMD_SGT"}
            resolve_source gmd_sgt "${sgt_source}" "${GMD_SGT_COMMIT}"
            CONTEXT_LINES+=("gmd_sgt_src=${RESOLVED_CONTEXT}")
            if [[ "${RESOLVED_KIND}" == "git-archive" ]]; then LOCAL_SOURCES+=("gmd_sgt|${sgt_source}|${GMD_SGT_COMMIT}|${RESOLVED_CONTEXT}"); fi
            ;;
        cp2k|cp2k-probe)
            CP2K_VERSION=$(yaml_path_value stacks.cp2k.upstream.version)
            CP2K_COMMIT=$(yaml_path_value stacks.cp2k.upstream.commit)
            require_exact_version cp2k_version "${CP2K_VERSION}"
            require_commit cp2k_commit "${CP2K_COMMIT}"
            ARG_LINES+=("CP2K_VERSION=${CP2K_VERSION}" "CP2K_COMMIT=${CP2K_COMMIT}")
            ;;
        cp2k-ml|cp2k-ml-probe)
            CP2K_VERSION=$(yaml_path_value stacks.cp2k_ml.upstream.version)
            CP2K_COMMIT=$(yaml_path_value stacks.cp2k_ml.upstream.commit)
            PYTORCH_VERSION=$(yaml_path_value stacks.cp2k_ml.ml_runtime.pytorch)
            NUMPY_VERSION=$(yaml_path_value stacks.cp2k_ml.ml_runtime.numpy)
            SCIPY_VERSION=$(yaml_path_value stacks.cp2k_ml.ml_runtime.scipy)
            PYYAML_VERSION=$(yaml_path_value stacks.cp2k_ml.ml_runtime.pyyaml)
            ASE_VERSION=$(yaml_path_value stacks.cp2k_ml.ml_runtime.ase)
            require_exact_version cp2k_version "${CP2K_VERSION}"
            require_commit cp2k_commit "${CP2K_COMMIT}"
            require_exact_version pytorch "${PYTORCH_VERSION}"
            require_exact_version numpy "${NUMPY_VERSION}"
            require_exact_version scipy "${SCIPY_VERSION}"
            require_exact_version pyyaml "${PYYAML_VERSION}"
            require_exact_version ase "${ASE_VERSION}"
            ARG_LINES+=(
                "CP2K_VERSION=${CP2K_VERSION}"
                "CP2K_COMMIT=${CP2K_COMMIT}"
                "PYTORCH_VERSION=${PYTORCH_VERSION}"
                "NUMPY_VERSION=${NUMPY_VERSION}"
                "SCIPY_VERSION=${SCIPY_VERSION}"
                "PYYAML_VERSION=${PYYAML_VERSION}"
                "ASE_VERSION=${ASE_VERSION}"
            )
            ;;
    esac
}

image_ref() {
    repository="${IMAGE_NAMESPACE}/$1"
    if [[ -n "${REGISTRY}" ]]; then
        repository="${REGISTRY%/}/${repository}"
    fi
    echo "${repository}:${IMAGE_TAG}"
}

platform_slug() {
    echo "${PLATFORM//\//-}"
}

# ---------------------------------------------------------------- plan

print_plan() {
    target=$1
    validate_target "${target}"
    version_context "${target}"
    echo "target=${target}"
    echo "platform=${PLATFORM}"
    echo "dockerfile=$(dockerfile_for "${target}")"
    echo "dockerignore=$(dockerfile_for "${target}").dockerignore"
    echo "docker_stage=$(build_stage_for "${target}")"
    echo "build_context=${PROJECT_ROOT}"
    echo "base_image=${BASE_IMAGE}"
    if [[ "${BASE_PINNED}" == 1 ]]; then
        echo "base_image_digest=${BASE_IMAGE#*@}"
    else
        echo "base_image_digest=unpinned"
        echo "warning: ${target} base image is not digest-pinned; push is refused until a digest is recorded" >&2
    fi
    echo "image_tag=${IMAGE_TAG}"
    echo "image=$(image_ref "${target}")"
    echo "versions=${VERSIONS_FILE}"
    echo "runtime_mount=${PROJECT_ROOT}/runtime:/runtime"
    echo "evidence_dir=${EVIDENCE_DIR}"
    for line in "${ARG_LINES[@]}"; do
        echo "build_arg=${line}"
    done
    for line in ${CONTEXT_LINES[@]+"${CONTEXT_LINES[@]}"}; do
        echo "named_context=${line}"
    done
    for entry in ${LOCAL_SOURCES[@]+"${LOCAL_SOURCES[@]}"}; do
        IFS='|' read -r label source commit dest <<EOF
${entry}
EOF
        echo "source.${label}=${source} (git archive ${commit})"
    done
    if [[ "${target}" == "cp2k-ml" || "${target}" == "cp2k-ml-probe" ]]; then
        source_dir=${GMD_AL_CP2K_ML_SOURCE:-"${PROJECT_ROOT}/../cp2k-ml-scf"}
        echo "cp2k_source=external"
        echo "source_mount=${source_dir}:/workspace/cp2k-ml-scf"
    fi
    echo "status=$(yaml_path_value "stacks.$(stack_for "${target}").status")"
}

# ---------------------------------------------------------------- docker

docker_args() {
    DOCKER_ARGS=(--platform "${PLATFORM}" --target "$(build_stage_for "${target}")")
    for line in "${ARG_LINES[@]}"; do
        DOCKER_ARGS+=(--build-arg "${line}")
    done
    for line in ${CONTEXT_LINES[@]+"${CONTEXT_LINES[@]}"}; do
        DOCKER_ARGS+=(--build-context "${line}")
    done
    DOCKER_ARGS+=(--file "$(dockerfile_for "${target}")")
}

# Export the pinned commit of each local sibling checkout with `git archive`, so
# the image is built from exactly that commit and never from a dirty work tree
# or its build directories.
prepare_sources() {
    for entry in ${LOCAL_SOURCES[@]+"${LOCAL_SOURCES[@]}"}; do
        IFS='|' read -r label source commit dest <<EOF
${entry}
EOF
        if [[ "${DRY_RUN}" == 1 ]]; then
            echo "+ git -C ${source} archive ${commit} | tar -x -C ${dest}"
            continue
        fi
        require_command git
        [[ -d "${source}" ]] || die "${label} source checkout not found: ${source} (set GMD_AL_$(lower "${label}" | tr '[:lower:]' '[:upper:]')_SOURCE)"
        git -C "${source}" cat-file -e "${commit}^{commit}" 2>/dev/null \
            || die "${label}: pinned commit ${commit} is not present in ${source}"
        rm -rf "${dest}"
        mkdir -p "${dest}"
        git -C "${source}" archive --format=tar "${commit}" | tar -x -C "${dest}"
        printf '%s\n' "${commit}" > "${dest}/.gmd_al_source_commit"
    done
}

run_or_print() {
    if [[ "${DRY_RUN}" == 1 ]]; then
        printf '+'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

dry_run_banner() {
    if [[ "${DRY_RUN}" == 1 ]]; then
        print_plan "$1"
        echo "dry_run=1"
    fi
}

build_image() {
    target=$1
    validate_target "${target}"
    version_context "${target}"
    dry_run_banner "${target}"
    [[ "${DRY_RUN}" == 1 ]] || require_command docker
    prepare_sources
    docker_args
    run_or_print docker buildx build "${DOCKER_ARGS[@]}" \
        --tag "$(image_ref "${target}")" --load "${PROJECT_ROOT}"
}

push_image() {
    target=$1
    validate_target "${target}"
    version_context "${target}"
    dry_run_banner "${target}"
    [[ -n "${REGISTRY}" ]] || die "GMD_AL_REGISTRY is required for push"
    if [[ "${BASE_PINNED}" != 1 && "${GMD_AL_ALLOW_UNPINNED_BASE:-0}" != 1 ]]; then
        die "refusing to push ${target}: base image ${BASE_IMAGE} has no digest (record one in versions.yaml or set GMD_AL_ALLOW_UNPINNED_BASE=1 for a throwaway image)"
    fi
    [[ "${DRY_RUN}" == 1 ]] || { require_command docker; mkdir -p "${EVIDENCE_DIR}"; }
    metadata_file=${GMD_AL_METADATA_FILE:-"${EVIDENCE_DIR}/${target}-build-metadata.json"}
    prepare_sources
    docker_args
    run_or_print docker buildx build "${DOCKER_ARGS[@]}" \
        --tag "$(image_ref "${target}")" \
        --metadata-file "${metadata_file}" \
        --push "${PROJECT_ROOT}"
}

tag_image() {
    target=$1
    destination_tag=$2
    validate_target "${target}"
    version_context "${target}"
    reject_floating_ref destination_tag "${destination_tag}"
    [[ "${DRY_RUN}" == 1 ]] || require_command docker
    repository=$(image_ref "${target}")
    run_or_print docker tag "${repository}" "${repository%:*}:${destination_tag}"
}

export_oci() {
    target=$1
    validate_target "${target}"
    version_context "${target}"
    dry_run_banner "${target}"
    [[ "${DRY_RUN}" == 1 ]] || { require_command docker; mkdir -p "${OUTPUT_DIR}"; }
    archive="${OUTPUT_DIR}/gmd-al-${target}-${IMAGE_TAG}-$(platform_slug).oci.tar"
    prepare_sources
    docker_args
    run_or_print docker buildx build "${DOCKER_ARGS[@]}" \
        --output "type=oci,dest=${archive}" "${PROJECT_ROOT}"
    echo "oci_archive=${archive}"
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

build_sif() {
    target=$1
    validate_target "${target}"
    version_context "${target}"
    source_uri=${GMD_AL_OCI_URI:-}
    if [[ -z "${source_uri}" ]]; then
        die "GMD_AL_OCI_URI is required for sif; Miyabi must pull from a registry (docker://...) or read an OCI archive (oci-archive:...) and cannot use a local Docker daemon"
    fi
    reject_floating_ref GMD_AL_OCI_URI "${source_uri}"
    output="${OUTPUT_DIR}/gmd-al-${target}-${IMAGE_TAG}-$(platform_slug).sif"
    dry_run_banner "${target}"
    if [[ "${DRY_RUN}" == 1 ]]; then
        run_or_print apptainer build "${output}" "${source_uri}"
        return
    fi
    require_command apptainer
    mkdir -p "${OUTPUT_DIR}" "${EVIDENCE_DIR}"
    apptainer build "${output}" "${source_uri}"
    # Computed from the artifact, never copied from a manifest.
    digest=$(sha256_of "${output}")
    printf '%s  %s\n' "${digest}" "$(basename "${output}")" > "${EVIDENCE_DIR}/${target}-sif.sha256"
    echo "sif=${output}"
    echo "sif_sha256=${digest}"
}

usage() {
    cat <<'EOF'
Usage: containers/build.sh [--dry-run] COMMAND [TARGET] [ARG]

Targets: gmd, sgt-train, cp2k (role images); cp2k-probe (cp2k + an H2 single
point run at build time); cp2k-ml, cp2k-ml-probe (experimental development
toolchain for the external cp2k-ml-scf fork).

Commands:
  plan TARGET            Print the resolved build plan (base image, digest, tag,
                         build args, named contexts). Needs no docker.
  build TARGET           Build one image with docker buildx (--load).
  push TARGET            Build with buildx and push to GMD_AL_REGISTRY; refuses
                         a base image without a recorded digest.
  build-all              Build the gmd, sgt-train and cp2k role images.
  tag TARGET TAG         Add an explicit, non-floating tag.
  oci TARGET             Export a platform-specific OCI archive.
  sif TARGET             Convert GMD_AL_OCI_URI to a SIF and record its sha256.
  help                   Show this message.

--dry-run prints the plan and the exact commands instead of running them; no
docker, git or apptainer is needed.

Environment overrides:
  GMD_AL_PLATFORM            Defaults to platform.default_oci_platform (linux/arm64).
  GMD_AL_BASE_IMAGE          Overrides the manifest base image (no digest appended).
  GMD_AL_IMAGE_TAG           Overrides the manifest image tag.
  GMD_AL_REGISTRY            Registry prefix used by image refs and push.
  GMD_AL_IMAGE_NAMESPACE     Defaults to gmd-al.
  GMD_AL_OCI_URI             Required by sif: docker://REGISTRY/gmd-al/TARGET:TAG
                             or oci-archive:PATH.
  GMD_AL_GMD_SOURCE          GMD checkout or git URL (default ../GMD).
  GMD_AL_GMD_SGT_SOURCE      GMD_SGT checkout or git URL (default ../GMD_SGT).
  GMD_AL_CP2K_ML_SOURCE      cp2k-ml-scf checkout mounted at /workspace/cp2k-ml-scf.
  GMD_AL_VERSIONS_FILE       Defaults to containers/versions.yaml.
  GMD_AL_CONTAINER_OUTPUT_DIR  Defaults to build/containers.
  GMD_AL_EVIDENCE_DIR        Defaults to build/evidence.
  GMD_AL_METADATA_FILE       Buildx metadata output for push.
  GMD_AL_ALLOW_UNPINNED_BASE Set to 1 to push from an undigested base (throwaway only).

Floating references (latest, stable, main, master) are rejected everywhere.
The build context is the repository root; BuildKit reads the Dockerfile-specific
ignore file containers/Dockerfile.<target>.dockerignore. No command installs
Docker or Apptainer.
EOF
}

# ---------------------------------------------------------------- main

# --dry-run may appear anywhere on the command line.
ARGS=()
for argument in "$@"; do
    if [[ "${argument}" == "--dry-run" ]]; then
        DRY_RUN=1
    else
        ARGS+=("${argument}")
    fi
done
set -- ${ARGS[@]+"${ARGS[@]}"}
command_name=${1:-help}

case "${command_name}" in
    plan) print_plan "${2:?TARGET is required}" ;;
    build) build_image "${2:?TARGET is required}" ;;
    push) push_image "${2:?TARGET is required}" ;;
    build-all)
        for role in ${ROLE_TARGETS}; do
            build_image "${role}"
        done
        ;;
    tag) tag_image "${2:?TARGET is required}" "${3:?TAG is required}" ;;
    oci) export_oci "${2:?TARGET is required}" ;;
    sif) build_sif "${2:?TARGET is required}" ;;
    help|-h|--help) usage ;;
    *) usage >&2; exit 2 ;;
esac
