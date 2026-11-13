#!/usr/bin/env python3
"""Generate five A2_R11_all student-seed configs on a fixed 92/8 split."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = json.loads((ROOT / 'configs/A2_R11_all.json').read_text())
out = ROOT / 'configs/seed_sweep'
out.mkdir(parents=True, exist_ok=True)
gpus = {42: 1, 43: 2, 44: 3, 45: 1, 46: 2}
for seed, gpu in gpus.items():
    config = dict(base)
    config.update({
        'name': f'A2_R11_all_seed{seed}',
        'work_root': 'runs/seed_sweep',
        'cache_dir': 'cache',
        'gpu': gpu,
        'seed': seed,
        'split_seed': 42,
        'importance_seed': seed,
        'calibrate_runtime_with_baseline': False,
        't4_runtime_factor': 2.0,
    })
    (out / f'A2_R11_all_seed{seed}.json').write_text(json.dumps(config, indent=2) + '\n')
print('Wrote five configs to', out)
