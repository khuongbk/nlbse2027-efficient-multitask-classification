#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
python="${PYTHON:-python}"
mkdir -p logs/seed_sweep
"$python" scripts/generate_seed_sweep.py

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

pids=()
run_one 42 1 & pids+=("$!")
run_one 43 2 & pids+=("$!")
run_one 44 3 & pids+=("$!")
failed_all=0
for pid in "${pids[@]}"; do wait "$pid" || failed_all=1; done

pids=()
run_one 45 1 & pids+=("$!")
run_one 46 2 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid" || failed_all=1; done
exit "$failed_all"
