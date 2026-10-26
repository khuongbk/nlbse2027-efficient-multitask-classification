#!/usr/bin/env bash
# Stage A2 (research, clean selection): improvements on top of the stage-A winner (R11: 6 layers, FFN 1536, early exit).
#   1. four fold teachers (one per GPU), each trained without one quarter of the inner training rows;
#      their logits on the held-out quarter and on the validation rows are out-of-fold soft targets
#   2. four R11 variants: EMA weights; + exit 0 and attention-pooled exits; + out-of-fold distillation; all
# Selection and the one-time public-test measurement work as in stage A.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

pids=()
for fold in 0 1 2 3; do
    if [ ! -f "cache/oof/fold$fold.npz" ]; then
        python -m nlbse27ee.train --config "configs/T0_fold$fold.json" --gpu "$fold" \
            > "logs/T0_fold$fold.log" 2>&1 &
        pids+=($!)
    fi
done
for pid in "${pids[@]}"; do wait "$pid" || { echo "FAILED fold teacher (see logs/T0_fold*.log)"; exit 1; }; done

gpu=0
for name in A2_R11_ema A2_R11_ema_exit0_attn A2_R11_oof_ema A2_R11_all; do
    python -m nlbse27ee.train --config "configs/$name.json" --gpu "$gpu" > "logs/$name.log" 2>&1 \
        || echo "FAILED $name (see logs/$name.log)" &
    gpu=$((gpu + 1))
done
wait
python -m nlbse27ee.summarize runs | tee logs/summary_stage_a2.md
