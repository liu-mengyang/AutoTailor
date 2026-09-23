#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
source "${AE_ENV:-$ROOT/ae/host.env}"
[[ "$AE_ROLE" == phone ]] || { echo 'Run on the phone server' >&2; exit 2; }
: "${RESULT_ROOT:?}"
[[ $# == 1 && -f "$1" ]] || { echo "Usage: $0 /path/to/pnnx-20240410-linux.zip" >&2; exit 2; }
TOOL_DIR="$RESULT_ROOT/compound-frontier/tools"
mkdir -p "$TOOL_DIR"
docker build --target ncnn -f "$ROOT/docker/Dockerfile.phone" -t autotailor-ae:ncnn-build "$ROOT"
docker run --rm --mount "type=bind,src=$HERE,dst=/input,readonly" \
  --mount "type=bind,src=$TOOL_DIR,dst=/output" autotailor-ae:ncnn-build bash -c '
  cp /input/benchmark_check.cpp /opt/ncnn/benchmark/benchncnn_custom.cpp
  cmake --build /opt/ncnn/build-android --target benchncnn_custom --parallel 4
  cp /opt/ncnn/build-android/benchmark/benchncnn_custom /output/benchncnn_check_fp32
  chmod 755 /output/benchncnn_check_fp32
  '
unzip -n -q "$1" -d "$TOOL_DIR"
chmod +x "$TOOL_DIR/pnnx-20240410-linux/pnnx"
cp "$HERE/supernet_phone.py" "$HERE/supernet-phone-watch.sh" "$TOOL_DIR/"
echo "Phone tools installed in $TOOL_DIR"
