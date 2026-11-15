#!/usr/bin/env python3
"""Summarize per-seed public metrics; do not select a seed using public results."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
seeds = (42, 43, 44, 45, 46)
rows = []
for seed in seeds:
    path = ROOT / 'runs/seed_sweep' / f'A2_R11_all_seed{seed}' / 'results.json'
    result = json.loads(path.read_text())
    final = result['calibration']['final']
    metrics = final['metrics']
    vul, mat = metrics['Vulnerability'], metrics['MAT']
    vul_accuracy = (vul['tp'] + vul['tn']) / (vul['tp'] + vul['tn'] + vul['fp'] + vul['fn'])
    mat_accuracy = (mat['tp'] + mat['tn']) / (mat['tp'] + mat['tn'] + mat['fp'] + mat['fn'])
    rows.append({
        'seed': seed,
        'best_epoch': result['training']['best_epoch'],
        'validation_selection_f1': result['calibration']['selection']['mean_f1'],
        'Vul_accuracy': vul_accuracy,
        'Vul_precision': metrics['Vulnerability']['precision'],
        'Vul_recall': metrics['Vulnerability']['recall'],
        'Vul_f1': metrics['Vulnerability']['f1'],
        'MAT_accuracy': mat_accuracy,
        'MAT_precision': metrics['MAT']['precision'],
        'MAT_recall': metrics['MAT']['recall'],
        'MAT_f1': metrics['MAT']['f1'],
        'mean_accuracy': (vul_accuracy + mat_accuracy) / 2,
        'mean_precision': metrics['mean_precision'],
        'mean_recall': metrics['mean_recall'],
        'mean_f1': metrics['mean_f1'],
        'local_runtime_s': final['local_runtime_s'],
        'profiler_gflops': final['gflops'],
        'estimated_t4_score': final['estimated_t4_score'],
        'split_sha256': result['calibration']['split_sha256'],
    })

remeasure_path = ROOT / 'results/seed_sweep_2026-11-12/v100_sequential_remeasurement.json'
if remeasure_path.exists():
    remeasured = json.loads(remeasure_path.read_text())
    by_seed = {row['seed']: row for row in remeasured['measurements']}
    for row in rows:
        measurement = by_seed[row['seed']]
        row['local_runtime_s'] = measurement['runtime_s']
        row['profiler_gflops'] = measurement['gflops']
        row['local_v100_score'] = measurement['local_v100_score']
        row['estimated_t4_score'] = measurement['estimated_t4_score_factor2']

fields = [key for key in rows[0] if key not in ('seed', 'best_epoch', 'split_sha256')]
summary = {'seeds': rows, 'summary_mean': {}, 'summary_std_sample': {},
           'split_sha256_values': sorted(set(row['split_sha256'] for row in rows)),
           'selection_note': 'Each run selected on inner validation. Public metrics are aggregated only; no seed is selected by public test.'}
for key in fields:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    summary['summary_mean'][key] = float(values.mean())
    summary['summary_std_sample'][key] = float(values.std(ddof=1))

out = ROOT / 'results/seed_sweep_2026-11-12'
out.mkdir(parents=True, exist_ok=True)
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
lines = ['# A2_R11_all five-seed summary', '',
         'The five student runs use one fixed teacher, fixed OOF logits and the same 92/8 split. '
         'Each checkpoint and exit policy was selected on inner validation. Public metrics below are aggregated; '
         'the public test did not select a seed.', '',
         '| Seed | Best epoch | Val mean F1 | Vul F1 | MAT F1 | Mean F1 | Runtime (s) | GFLOPs | Est. T4 score |',
         '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for row in rows:
    lines.append(f"| {row['seed']} | {row['best_epoch']} | {row['validation_selection_f1']:.4f} | "
                 f"{row['Vul_f1']:.4f} | {row['MAT_f1']:.4f} | {row['mean_f1']:.4f} | "
                 f"{row['local_runtime_s']:.2f} | {row['profiler_gflops']:.0f} | {row['estimated_t4_score']:.4f} |")
lines.extend(['', '| Metric | Mean | Sample SD |', '|---|---:|---:|'])
for key in ('Vul_accuracy', 'Vul_precision', 'Vul_recall', 'Vul_f1',
            'MAT_accuracy', 'MAT_precision', 'MAT_recall', 'MAT_f1',
            'mean_accuracy', 'mean_precision', 'mean_recall', 'mean_f1',
            'local_runtime_s', 'profiler_gflops', 'estimated_t4_score'):
    lines.append(f"| {key} | {summary['summary_mean'][key]:.6f} | {summary['summary_std_sample'][key]:.6f} |")
lines += ['', 'All seeds must have the same split SHA256: ' + ', '.join(summary['split_sha256_values']) + '.']
if remeasure_path.exists():
    lines += ['', 'Runtime and profiler GFLOPs were remeasured sequentially on one V100 to avoid timing overlap. '
              'The T4 score is an estimate using the fixed 2.0 runtime factor; it is not an official T4/hidden-test score.']
(out / 'summary.md').write_text('\n'.join(lines) + '\n')
print(out / 'summary.md')
