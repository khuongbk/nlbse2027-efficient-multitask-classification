#!/usr/bin/env bash
# Stage A (research, clean selection). The teacher and every student train on 92% of the training
# set; checkpoints, thresholds, exit margins and the winner are chosen on the other 8% (stratified,
# seed 42). Each model is measured on the public test exactly once, at the end.
# Usage: bash scripts/run_stage_a.sh        (NPROC=<gpus> to change the teacher's DDP width)
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

python -m nlbse27ee.train --config configs/R8_6L_ffn1536_exits.json --gpu 0 --baseline-only 2>&1 \
    | tee logs/baseline_calibration.log
python - <<'PY' 2>&1 | tee logs/prepare_cache.log
from nlbse27ee.train import TrainConfig, setup, encode_inputs, teacher_targets
for name in ("R8_6L_ffn1536_exits", "R0_4L_plain", "R6_4L_exits_lines"):
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
lane 0 R10_6L_exits_e14 R0_4L_plain R1_4L_compact_flag &
lane 1 R11_6L_ffn1536_exits_e14 R2_4L_exits R3_4L_exits_noKD &
lane 2 R7_6L_exits R9_6L_ffn2048_exits R4_3L_exits &
lane 3 R8_6L_ffn1536_exits R5_4L_ffn1536_exits R6_4L_exits_lines &
wait
python -m nlbse27ee.summarize runs | tee logs/summary_stage_a.md
