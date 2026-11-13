#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
python=/home/ngaltn/miniconda3/envs/nlbse27/bin/python

run_one() {
    local seed="$1"
    local gpu="$2"
    local config="configs/seed_sweep/A2_R11_all_seed${seed}.json"
    local log="logs/seed_sweep/seed${seed}.log"
    "$python" -u -m nlbse27ee.train --config "$config" --gpu "$gpu" > "$log" 2>&1
    local result=$?
    printf '%s\n' "$result" > "logs/seed_sweep/seed${seed}.exit_code"
    echo "seed ${seed} finished with exit code ${result}"
    return "$result"
}

run_one 45 1 & pid45=$!
run_one 46 2 & pid46=$!
failed=0
wait "$pid45" || failed=1
wait "$pid46" || failed=1
exit "$failed"
