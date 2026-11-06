#!/usr/bin/env bash
# Ported from PHAL (tests/scripts/run_all_offline.sh, run_integration_if_available.sh) at 96263b6; MIT License, see NOTICE.
# Run the complete offline GMD_active suite (no GMD binary, GMD_SGT checkout or CP2K needed; tests that
# need them skip with a reason). From the repository root:
#
#   bash tools/test/run_all_offline.sh                 # offline suite
#   GMD_AL_RUN_INTEGRATION=1 bash tools/test/run_all_offline.sh   # + integration/cp2k tests that can run here
#
# Integration tests read: GMD_EXECUTABLE (GMD built with -DGMD_ENABLE_TORCH=ON; default ../GMD/build-torch/gmd),
# GMD_SGT_PATH (GMD_SGT >= 0.4 checkout; default ../GMD_SGT or an installed gmd_sgt),
# CP2K_EXECUTABLE (labeling, closed loop; default cp2k.psmp on PATH), CP2K_BIN / GMD_SCF_CP2K (SCF tooling).
# Extra arguments are passed to pytest, e.g. `bash tools/test/run_all_offline.sh -x -q tests/monitors`.
set -euo pipefail

PYTHON="${PYTHON:-python}"
cd "$(dirname "$0")/../.."
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" -m compileall -q src tests examples tools >/dev/null
"$PYTHON" -m pytest -p no:cacheprovider -q -m "not integration and not cp2k and not slow" "$@"

if [[ "${GMD_AL_RUN_INTEGRATION:-}" == "1" ]]; then
  echo "== integration / cp2k tests (skipped individually when a program is missing)"
  "$PYTHON" -m pytest -p no:cacheprovider -q -rs -m "integration or cp2k" "$@"
fi
