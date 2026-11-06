#!/usr/bin/env bash
# Clean-installation acceptance check (final migration gate).
#
# Exports the committed tree (which never contains ref/), builds a wheel, installs it into a fresh
# virtual environment, deletes src/ from the export so only the installed package can be imported,
# then runs the CLI and the full test suite. Real-backend tests run when GMD_EXECUTABLE,
# GMD_SGT_PATH and a CP2K executable are available; otherwise they skip with a reason.
#
# Usage: tools/migration/clean_install_check.sh <python-with-deps> <work-dir> [pytest args...]
set -euo pipefail

PYTHON=${1:?python interpreter providing numpy/scipy/torch/ase/pydantic/pytest}
WORK=${2:?scratch directory}
shift 2
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)

rm -rf "$WORK"
mkdir -p "$WORK/export"
git -C "$REPO" archive HEAD | tar -x -C "$WORK/export"
test ! -e "$WORK/export/ref" || { echo "export unexpectedly contains ref/"; exit 1; }

"$PYTHON" -m venv --system-site-packages "$WORK/venv"
VENV_PY="$WORK/venv/bin/python"
if "$VENV_PY" -c "import gmd_active_learning" 2>/dev/null; then
  echo "gmd_active_learning is already importable in the base environment; refusing to test a stale install"; exit 1
fi
"$VENV_PY" -m pip wheel --quiet --no-deps --no-build-isolation -w "$WORK/dist" "$WORK/export"
"$VENV_PY" -m pip install --quiet --no-deps "$WORK"/dist/*.whl
rm -rf "$WORK/export/src"

cd "$WORK/export"
"$VENV_PY" -c "import gmd_active_learning, sys; p = gmd_active_learning.__file__; print('installed package:', p); sys.exit(0 if 'site-packages' in p else 1)"
"$WORK/venv/bin/gmd-al" --help > /dev/null
"$WORK/venv/bin/gmd-al" validate-config --config configs/example_run.yaml > /dev/null
PYTHONDONTWRITEBYTECODE=1 "$VENV_PY" -m pytest -p no:cacheprovider -q -W ignore -rs "$@"
