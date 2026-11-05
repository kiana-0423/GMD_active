#!/usr/bin/env bash
# Submit a GMD_active SIF validation job to PBS Pro with the queue and group
# taken from variables instead of hard-coded #PBS lines.
#
#   GMD_AL_PBS_GROUP=<group> tools/hpc/pbs/submit.sh gmd /abs/path/gmd.sif
#
# Probes: gmd | sgt-train | cp2k | hpcx (HPC-X diagnosis for NGC-based SIFs).
# Environment:
#   GMD_AL_PBS_QUEUE     queue (default debug-g)
#   GMD_AL_PBS_GROUP     project group for -W group_list (required)
#   GMD_AL_PBS_WALLTIME  optional walltime override (HH:MM:SS)
#   GMD_AL_CP2K_RANKS    MPI ranks for the cp2k probe (default 2)
# --dry-run prints the qsub command instead of running it.
#
# Not exercised on Miyabi yet: no job has been submitted from GMD_active.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DRY_RUN=0
ARGS=()
for argument in "$@"; do
    if [[ "${argument}" == "--dry-run" ]]; then
        DRY_RUN=1
    else
        ARGS+=("${argument}")
    fi
done
set -- ${ARGS[@]+"${ARGS[@]}"}

usage() {
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
}

probe=${1:-}
image=${2:-}
if [[ -z "${probe}" || -z "${image}" ]]; then
    usage >&2
    exit 2
fi

case "${probe}" in
    gmd) script=validate_gmd_probe.pbs; image_var=GMD_AL_GMD_IMAGE ;;
    sgt-train) script=validate_sgt_train_probe.pbs; image_var=GMD_AL_SGT_TRAIN_IMAGE ;;
    cp2k) script=validate_cp2k_probe.pbs; image_var=GMD_AL_CP2K_IMAGE ;;
    hpcx) script=diagnose_hpcx.pbs; image_var=GMD_AL_PROBE_IMAGE ;;
    *) echo "unknown probe: ${probe} (expected gmd, sgt-train, cp2k, hpcx)" >&2; exit 2 ;;
esac

queue=${GMD_AL_PBS_QUEUE:-debug-g}
group=${GMD_AL_PBS_GROUP:-}
if [[ -z "${group}" ]]; then
    echo "GMD_AL_PBS_GROUP is required (the project group for -W group_list)" >&2
    exit 2
fi
case "${image}" in
    /*) ;;
    *) image="$(pwd)/${image}" ;;
esac

variables="${image_var}=${image},GMD_AL_PBS_GROUP=${group}"
if [[ "${probe}" == cp2k && -n "${GMD_AL_CP2K_RANKS:-}" ]]; then
    variables="${variables},GMD_AL_CP2K_RANKS=${GMD_AL_CP2K_RANKS}"
fi
command=(qsub -q "${queue}" -W "group_list=${group}")
if [[ -n "${GMD_AL_PBS_WALLTIME:-}" ]]; then
    command+=(-l "walltime=${GMD_AL_PBS_WALLTIME}")
fi
command+=(-v "${variables}" "${SCRIPT_DIR}/${script}")

if [[ "${DRY_RUN}" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
command -v qsub >/dev/null 2>&1 || { echo "qsub not found: run this on a Miyabi login node" >&2; exit 2; }
"${command[@]}"
