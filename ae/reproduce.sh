#!/usr/bin/env bash
# Run on the prepared GPU server. Connection settings are supplied privately.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${1:-} == --help ]]; then
  echo 'Usage: bash ae/reproduce.sh [run-id]'
  echo 'Run on the prepared GPU server; reuse a run-id to resume matching results.'
  exit 0
fi
[[ $# -le 1 ]] || { echo 'Expected at most one run-id' >&2; exit 2; }
RUN_ID=${1:-reviewer-$(date -u +%Y%m%dT%H%M%SZ)-$$}
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo 'Invalid run-id' >&2; exit 2; }
HOST_ENV=${AE_ENV:-$ROOT/ae/host.env}
REMOTE_ENV=${AE_REPRODUCE_ENV:-$ROOT/ae/reproduce.local.env}
for config in "$HOST_ENV" "$REMOTE_ENV"; do
  [[ -f "$config" ]] || { echo "Missing prepared configuration: $config. Contact the authors for server setup." >&2; exit 2; }
done
source "$HOST_ENV"
source "$REMOTE_ENV"
[[ "$AE_ROLE" == gpu ]] || { echo 'Run this script on the GPU server.' >&2; exit 2; }
: "${RESULT_ROOT:?}" "${PHONE_HOST:?}" "${PHONE_REPO:?}" "${PHONE_RESULT_ROOT:?}"
# Restrict paths used in SSH/rsync commands to unambiguous shell-safe names.
[[ "$PHONE_HOST" =~ ^[A-Za-z0-9][A-Za-z0-9_.@-]*$ ]] || exit 2
for path in "$PHONE_REPO" "$PHONE_RESULT_ROOT"; do
  [[ "$path" =~ ^/[A-Za-z0-9_./-]+$ ]] || { echo 'Remote paths must be absolute, without spaces or shell characters.' >&2; exit 2; }
done
cd "$ROOT"
RUN_REL=compound-frontier/$RUN_ID
LOCAL_RUN=$RESULT_ROOT/$RUN_REL
REMOTE_RUN=$PHONE_RESULT_ROOT/$RUN_REL
mkdir -p "$LOCAL_RUN"
exec > >(tee -a "$LOCAL_RUN/reproduce.log") 2>&1
printf 'Run ID: %s\n' "$RUN_ID"
echo 'Checking GPU, dataset, phone, and conversion tools...'
bash ae/container.sh check
ssh -o BatchMode=yes "$PHONE_HOST" "bash -s" -- "$PHONE_REPO" "$PHONE_RESULT_ROOT" <<'REMOTE'
set -euo pipefail
cd "$1"
bash ae/container.sh check
test -x "$2/compound-frontier/tools/pnnx-20240410-linux/pnnx"
test -x "$2/compound-frontier/tools/benchncnn_check_fp32"
# Device identifiers are configured privately on the phone server.
source ae/host.env
read -r -a serials <<< "${AE_PHONE_SERIALS:?Ask the authors to configure the two phones}"
[[ ${#serials[@]} -eq 2 && "${serials[0]}" != "${serials[1]}" ]]
for serial in "${serials[@]}"; do
  adb -H 127.0.0.1 -P "${ADB_PORT:-5037}" -s "$serial" get-state
done
REMOTE
echo 'Evaluating and exporting the trained SuperNet candidates...'
bash ae/supernet/run_gpu.sh full "/results/$RUN_REL"
echo 'Transferring exports and reference outputs to the phone server...'
ssh -o BatchMode=yes "$PHONE_HOST" "mkdir -p $REMOTE_RUN"
rsync -a -e 'ssh -o BatchMode=yes' --include='/*.json' --include='/*.pt' \
  --include='/*.reference.npy' --exclude='*' "$LOCAL_RUN/" "$PHONE_HOST:$REMOTE_RUN/"
# Stage the script from this checkout; container images need not be rebuilt.
rsync -a -e 'ssh -o BatchMode=yes' ae/supernet/supernet_phone.py \
  "$PHONE_HOST:$REMOTE_RUN/supernet_phone.py"
echo 'Measuring phone latency...'
ssh -o BatchMode=yes "$PHONE_HOST" "bash -s" -- "$PHONE_REPO" "$RUN_REL" <<'REMOTE'
set -euo pipefail
cd "$1"
bash ae/container.sh run python "/results/$2/supernet_phone.py" "/results/$2"
REMOTE
echo 'Collecting measurements and plotting...'
rsync -a -e 'ssh -o BatchMode=yes' "$PHONE_HOST:$REMOTE_RUN/latency-fp32/" "$LOCAL_RUN/latency-fp32/"
cp ae/supernet/plot_supernet_frontier.py "$LOCAL_RUN/plot_supernet_frontier.py"
bash ae/container.sh run python "/results/$RUN_REL/plot_supernet_frontier.py" \
  "/results/$RUN_REL/accuracy.json" "/results/$RUN_REL/latency-fp32" \
  --output "/results/$RUN_REL/figures"
printf 'Results: %s/figures\n' "$LOCAL_RUN"
