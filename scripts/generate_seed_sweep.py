#!/usr/bin/env python3
"""Write the five A2_R11_all seed configs (training seeds 42-46, fixed 92/8 split of seed 42).

The configs in configs/seed_sweep/ are the ones the reported sweep ran with; regenerating them
reproduces them except for the GPU, which run_seed_sweep.sh passes on the command line.
"""

import json
from pathlib import Path

SEEDS = (42, 43, 44, 45, 46)
# The sweep ran on V100s without a baseline timing run; exit margins were searched with this
# assumed GPU-to-T4 runtime factor (the L40S runs measured 5.32). It only shifts the
# runtime/accuracy trade-off of the margin search; F1 and GFLOPs are measured, not scaled.
RUNTIME_FACTOR = 2.0

ROOT = Path(__file__).resolve().parents[1]
base = json.loads((ROOT / "configs/A2_R11_all.json").read_text())
out = ROOT / "configs/seed_sweep"
out.mkdir(parents=True, exist_ok=True)
for seed in SEEDS:
    config = {**base,
              "name": f"A2_R11_all_seed{seed}",
              "work_root": "runs/seed_sweep",
              "seed": seed,               # init, dropout, batch order
              "split_seed": 42,           # same inner split, folds and OOF targets for every seed
              "importance_seed": seed,    # rows used to rank FFN neurons before pruning
              "calibrate_runtime_with_baseline": False,
              "t4_runtime_factor": RUNTIME_FACTOR}
    (out / f"A2_R11_all_seed{seed}.json").write_text(json.dumps(config, indent=2) + "\n")
print("wrote", len(SEEDS), "configs to", out)
