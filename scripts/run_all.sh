#!/usr/bin/env bash
# Full pipeline on one 4-GPU machine; every step resumes or is skipped when already done.
#   1. teacher: UniXcoder-base-nine (12 layers) fine-tuned on all training rows, DDP on all GPUs
#   2. runtime calibration with the organizers' baseline cell, teacher logits (cached in cache/)
#   3. ablation grid R0-R7, two runs per GPU, then a summary table
# Usage: bash run_grid.sh            (NPROC=<gpus> to change the DDP width)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
NPROC=${NPROC:-4}
# Some containers hang on GPU peer-to-peer transfers; going through host memory is safe everywhere.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}

if [ ! -f teacher/unixcoder12_teacher.pt ]; then
    python -m torch.distributed.run --nproc_per_node "$NPROC" -m nlbse27ee.train \
        --config configs/T0_teacher12.json 2>&1 | tee -a logs/T0_teacher12.log
fi
test -f teacher/unixcoder12_teacher.pt || { echo "teacher training failed"; exit 1; }

python -m nlbse27ee.train --config configs/R2_4L_exits.json --gpu 0 --baseline-only 2>&1 \
    | tee logs/baseline_calibration.log
python - <<'PY' 2>&1 | tee logs/prepare_cache.log
from nlbse27ee.train import TrainConfig, setup, encode_inputs, teacher_targets
for name in ("R2_4L_exits", "R0_4L_plain", "R6_4L_exits_lines"):
    ctx = setup(TrainConfig.from_json(f"configs/{name}.json", gpu=0))
    encode_inputs(ctx)
    teacher_targets(ctx)
PY

lane() {  # gpu, configs...
    local gpu=$1; shift
    for name in "$@"; do
        python -m nlbse27ee.train --config "configs/$name.json" --gpu "$gpu" \
            > "logs/$name.log" 2>&1 || echo "FAILED $name (see logs/$name.log)"
    done
}
lane 0 R2_4L_exits R6_4L_exits_lines &
lane 1 R4_3L_exits R0_4L_plain &
lane 2 R5_4L_ffn1536_exits R1_4L_compact_flag &
lane 3 R7_6L_exits R3_4L_exits_noKD &
wait
python -m nlbse27ee.summarize runs | tee logs/summary.md
