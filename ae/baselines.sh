#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# -gt 0 ]] || { echo 'Usage: bash ae/baselines.sh {build|check|run} [command...]'; exit 2; }
action=$1; shift
case "$action" in build|check|run) ;; *) echo 'Expected build, check or run' >&2; exit 2;; esac
source "${AE_ENV:-$ROOT/ae/host.env}"
[[ "$AE_ROLE" == gpu ]] || { echo 'Baseline environment belongs on the GPU server.' >&2; exit 2; }
image=${BASELINE_IMAGE:-autotailor-ae:baselines}
docker info >/dev/null
if [[ "$action" == build ]]; then
 exec docker build -f "$ROOT/docker/Dockerfile.baselines" -t "$image" "$ROOT"
fi
: "${IMAGENET_ROOT:?}" "${ASSET_ROOT:?}" "${RESULT_ROOT:?}"
[[ -d "$IMAGENET_ROOT/ILSVRC/Data/CLS-LOC/val" && -d "$ASSET_ROOT" ]] || { echo 'Missing dataset or asset directory' >&2; exit 2; }
mkdir -p "$RESULT_ROOT"
args=(--rm --user "$(id -u):$(id -g)" --gpus "${GPU_DEVICES:-device=0}" --shm-size 8g
 --mount "type=bind,src=$IMAGENET_ROOT,dst=/datasets/imagenet,readonly"
 --mount "type=bind,src=$ASSET_ROOT,dst=/assets,readonly"
 --mount "type=bind,src=$RESULT_ROOT,dst=/results")
if [[ "$action" == check ]]; then set -- python /opt/baselines/baseline_smoke.py all; fi
if [[ $# -eq 0 ]]; then set -- bash; fi
[[ -t 0 && -t 1 ]] && args+=(-it)
exec docker run "${args[@]}" "$image" "$@"
