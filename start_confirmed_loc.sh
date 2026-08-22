#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"

exec "$python_bin" \
  "$repo_root/scripts/loc_live/run_best_churro_on_confirmed_untranscribed_loc.py" \
  "$@"
