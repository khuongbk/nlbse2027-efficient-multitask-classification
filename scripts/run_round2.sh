#!/usr/bin/env bash
# Round 2: 6-layer early-exit students combining round-1 winners (one run per GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
gpu=0
for name in R8_6L_ffn1536_exits R9_6L_ffn2048_exits R10_6L_exits_e14 R11_6L_ffn1536_exits_e14; do
    python -m nlbse27ee.train --config "configs/$name.json" --gpu "$gpu" > "logs/$name.log" 2>&1 \
        || echo "FAILED $name (see logs/$name.log)" &
    gpu=$((gpu + 1))
done
wait
python -m nlbse27ee.summarize runs | tee logs/summary.md
