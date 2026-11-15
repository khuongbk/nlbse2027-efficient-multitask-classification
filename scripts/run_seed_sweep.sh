#!/usr/bin/env bash
# Five-seed sweep of the A2_R11_all recipe (see results/seed_sweep_2026-11-12/README.md).
# Needs teacher/unixcoder12_teacher.pt and cache/oof/fold{0..3}.npz from stage A2.
#   GPUS="0 1 2" bash scripts/run_seed_sweep.sh      # one run per GPU at a time
#   PYTHON=/path/to/python                           # interpreter with requirements.txt
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
python="${PYTHON:-python}"
read -r -a gpus <<< "${GPUS:-0 1 2 3}"
seeds=(42 43 44 45 46)
mkdir -p logs/seed_sweep
"$python" scripts/generate_seed_sweep.py || exit 1

worker() {   # runs seeds k, k+n, k+2n, ... on one GPU
    local k=$1 gpu=${gpus[$1]} failed=0
    for ((i = k; i < ${#seeds[@]}; i += ${#gpus[@]})); do
        local name="A2_R11_all_seed${seeds[$i]}"
        if "$python" -u -m nlbse27ee.train --config "configs/seed_sweep/$name.json" --gpu "$gpu" \
                > "logs/seed_sweep/$name.log" 2>&1; then
            echo "done   $name (gpu $gpu)"
        else
            echo "FAILED $name (gpu $gpu), see logs/seed_sweep/$name.log"; failed=1
        fi
    done
    return $failed
}

pids=()
for k in "${!gpus[@]}"; do worker "$k" & pids+=($!); done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" -eq 0 ] || exit 1
# Runs share GPUs and time each other; remeasure runtime on an idle GPU before summarizing:
#   python scripts/remeasure_seed_sweep.py --gpu 0
"$python" scripts/summarize_seed_sweep.py
