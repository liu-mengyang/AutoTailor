#!/usr/bin/env bash
set -euo pipefail
for attempt in $(seq 1 120); do
  python /results/compound-frontier/tools/supernet_phone.py /results/compound-frontier/full
  count=$(find /results/compound-frontier/full/latency-fp32 -name result.json | wc -l)
  if [[ "$count" -eq 32 ]]; then
    echo ALL_16_CANDIDATES_ON_BOTH_PHONES_COMPLETE
    exit 0
  fi
  sleep 20
done
echo 'Timed out waiting for completed accuracy exports' >&2
exit 1
