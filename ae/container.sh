#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
usage() { echo "Usage: $0 {build|check|run} [command ...]; config: AE_ENV or ae/host.env"; }
[[ $# -gt 0 ]] || { usage; exit 2; }
action=$1; shift
case "$action" in build|check|run) ;; *) usage; exit 2;; esac
ENV_FILE=${AE_ENV:-$ROOT/ae/host.env}
[[ -f "$ENV_FILE" ]] || { echo "Copy ae/phone.env.example or ae/gpu.env.example to $ENV_FILE" >&2; exit 2; }
source "$ENV_FILE"
: "${AE_ROLE:?}" "${AE_IMAGE:?}"
[[ "$AE_ROLE" == phone || "$AE_ROLE" == gpu ]] || exit 2
docker info >/dev/null || { echo "Docker access required for $(id -un); ask the server administrator, then reconnect SSH." >&2; exit 1; }
if [[ "$action" == build ]]; then
  docker build -f "$ROOT/docker/Dockerfile.$AE_ROLE" -t "$AE_IMAGE" "$ROOT"
  exit
fi
: "${ASSET_ROOT:?}" "${RESULT_ROOT:?}"
[[ -d "$ASSET_ROOT" ]] || { echo "Missing ASSET_ROOT=$ASSET_ROOT" >&2; exit 2; }
mkdir -p "$RESULT_ROOT"
args=(--rm --user "$(id -u):$(id -g)" --mount "type=bind,src=$ASSET_ROOT,dst=/assets,readonly" --mount "type=bind,src=$RESULT_ROOT,dst=/results" -e "AE_ROLE=$AE_ROLE" -e USER=ae -e LOGNAME=ae)
if [[ "$AE_ROLE" == phone ]]; then
  : "${ANDROID_SERIAL:?}"; ADB_PORT=${ADB_PORT:-5037}
  [[ "$(adb -H 127.0.0.1 -P "$ADB_PORT" -s "$ANDROID_SERIAL" get-state)" == device ]] || exit 1
  args+=(--network host -e "ADB_SERVER_SOCKET=tcp:127.0.0.1:$ADB_PORT" -e "ADB_PORT=$ADB_PORT" -e "ANDROID_SERIAL=$ANDROID_SERIAL" -e "AE_PHONE_SERIALS=${AE_PHONE_SERIALS:-}")
else
  : "${IMAGENET_ROOT:?}"
  [[ -d "$IMAGENET_ROOT/ILSVRC/Data/CLS-LOC/train" && -d "$IMAGENET_ROOT/ILSVRC/Data/CLS-LOC/val" && -f "$IMAGENET_ROOT/LOC_val_solution.csv" ]] || { echo 'Incomplete ImageNet root' >&2; exit 2; }
  args+=(--gpus "${GPU_DEVICES:-device=0}" --shm-size 8g --mount "type=bind,src=$IMAGENET_ROOT,dst=/datasets/imagenet,readonly")
fi
if [[ "$action" == check ]]; then
  set -- python ae/preflight.py "$AE_ROLE"
elif [[ $# -eq 0 ]]; then
  set -- bash
fi
[[ -t 0 && -t 1 ]] && args+=(-it)
exec docker run "${args[@]}" "$AE_IMAGE" "$@"
