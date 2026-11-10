#!/usr/bin/env bash
# Install this checkout, editable, into one or more virtual environments.
#
#   scripts/install-into.sh ~/code/project-a/.venv ~/code/project-b/.venv
#   PLOTTERS_EXTRAS=stats,static scripts/install-into.sh ~/code/project-a/.venv
#
# Editable means the venv imports straight from this folder, so a `git pull` here updates every
# environment at once. Extras (PLOTTERS_EXTRAS): "stats" adds scipy, "static" adds matplotlib.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
extras="${PLOTTERS_EXTRAS:-}"
target="$repo${extras:+[$extras]}"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <venv-dir> [<venv-dir> ...]   (env: PLOTTERS_EXTRAS=stats,static)" >&2
  exit 2
fi

for venv in "$@"; do
  python="${venv%/}/bin/python"
  if [ ! -x "$python" ]; then
    echo "skip: no python at $python" >&2
    continue
  fi
  echo "-> $venv ($("$python" --version 2>&1))"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --quiet --python "$python" -e "$target"
  else
    "$python" -m pip install --quiet -e "$target"
  fi
  "$python" -c "import plotters; print('   plotters', plotters.__version__, 'from', plotters.__file__)"
done
