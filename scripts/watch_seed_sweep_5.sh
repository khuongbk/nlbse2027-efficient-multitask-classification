#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
seeds=(42 43 44 45 46)
while true; do
    complete=1
    for seed in "${seeds[@]}"; do
        [ -f "logs/seed_sweep/seed${seed}.exit_code" ] || complete=0
    done
    [ "$complete" -eq 1 ] && break
    sleep 60
done
failed=0
for seed in "${seeds[@]}"; do
    code=$(cat "logs/seed_sweep/seed${seed}.exit_code")
    [ "$code" = 0 ] || failed=1
done
if [ "$failed" -ne 0 ]; then
    echo "At least one seed failed; see logs/seed_sweep/seed*.log" \
        | tee logs/seed_sweep/summary_status.txt
    exit 1
fi
${PYTHON:-python} scripts/summarize_seed_sweep.py \
    | tee logs/seed_sweep/summary_status.txt
