#!/usr/bin/env bash
# Stage B (submission). Retrain the teacher and the stage-A winner on all 214,586 training rows,
# each stopped at its stage-A best epoch (same learning-rate schedule), then calibrate thresholds
# and exit margins on the public test, which the guide designates for tuning.
# The winner is the student with the best stage-A selection-split score unless WINNER is set.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs configs/final
NPROC=${NPROC:-4}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}

WINNER=${WINNER:-$(python - <<'PY'
from nlbse27ee.summarize import collect
rows = [r for r in collect("runs") if r["selection_split"] == "inner" and not r["run"].startswith("T0")]
print(rows[0]["run"])
PY
)}
echo "stage-B winner: $WINNER"
python - "$WINNER" <<'PY'
import json, sys
winner = sys.argv[1]
for source, target, extra in (
        ("T0_teacher12", "T0_teacher12_full", {"export_teacher_to": "teacher/unixcoder12_teacher_full.pt"}),
        (winner, winner + "_full", {"teacher_checkpoint": "teacher/unixcoder12_teacher_full.pt"})):
    cfg = json.load(open(f"configs/{source}.json"))
    best = json.load(open(f"runs/{source}/results.json"))["training"]["best_epoch"]
    cfg.update(name=target, validation="none", stop_epoch=best, **extra)
    open(f"configs/final/{target}.json", "w").write(json.dumps(cfg, indent=2) + "\n")
    print(target, "stop_epoch", best)
PY

if [ ! -f teacher/unixcoder12_teacher_full.pt ]; then
    python -m torch.distributed.run --nproc_per_node "$NPROC" -m nlbse27ee.train \
        --config configs/final/T0_teacher12_full.json 2>&1 | tee -a logs/T0_teacher12_full.log
fi
python -m nlbse27ee.train --config "configs/final/${WINNER}_full.json" --gpu 0 2>&1 | tee "logs/${WINNER}_full.log"
python -m nlbse27ee.summarize runs | tee logs/summary_stage_b.md
