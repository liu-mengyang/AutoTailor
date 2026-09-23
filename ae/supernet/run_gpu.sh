#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
source "${AE_ENV:-$ROOT/ae/host.env}"
[[ "$AE_ROLE" == gpu ]] || { echo 'Run on the ImageNet GPU server' >&2; exit 2; }
MODE=${1:-smoke}
[[ "$MODE" == smoke || "$MODE" == full ]] || exit 2
: "${RESULT_ROOT:?}" "${ASSET_ROOT:?}"
for model in resnetCompound_comp_swq_speedup_256_best.pth.tar timm_resnet50.onnx; do
  [[ -f "$ASSET_ROOT/compound-supernet/$model" ]] || {
    echo 'A prepared model is missing. Please contact the authors to restore the server environment.' >&2
    exit 1
  }
done
mkdir -p "$RESULT_ROOT/compound-frontier/source"
rsync -a --exclude __pycache__ "$HERE/source/" "$RESULT_ROOT/compound-frontier/source/"
args=(--output "${2:-/results/compound-frontier/$MODE}")
[[ "$MODE" == smoke ]] && args+=(--smoke)
exec bash "$ROOT/ae/container.sh" run python \
  /results/compound-frontier/source/exp_scripts/supernet_frontier.py "${args[@]}"
