#!/usr/bin/env python3
"""Remeasure all five seed exports sequentially on an idle V100 for score means."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs/seed_sweep'
DEVICE = torch.device('cuda:1')
sys.path.insert(0, str(RUNS / 'A2_R11_all_seed42/export'))
from nlbse27ee.data import Packed  # noqa: E402
from nlbse27ee.evaluate import organizer_protocol  # noqa: E402
from nlbse27ee.export import load_model  # noqa: E402
from nlbse27ee.preprocess import pack  # noqa: E402
from nlbse27ee.exits import official_score  # noqa: E402


def main():
    assert torch.cuda.is_available() and 'V100' in torch.cuda.get_device_name(DEVICE)
    data = load_dataset('NLBSE/nlbse27-code-comment-classification')['test']
    functions = [str(value or '') for value in data['Function']]
    labels = np.asarray(data['Label'], dtype=np.int64)
    rows = []
    for seed in (42, 43, 44, 45, 46):
        print('remeasuring seed', seed, 'on', torch.cuda.get_device_name(DEVICE), flush=True)
        export = RUNS / f'A2_R11_all_seed{seed}/export'
        model, encoder = load_model(export, DEVICE)
        packed = Packed(*pack(encoder(functions)))
        measured = organizer_protocol(model, packed, labels, encoder.tokenizer.pad_token_id,
                                      DEVICE, batch_size=600, repetitions=10, padding='sorted')
        previous = json.loads((RUNS / f'A2_R11_all_seed{seed}/results.json').read_text())
        old_f1 = previous['calibration']['final']['metrics']['mean_f1']
        assert abs(measured['metrics']['mean_f1'] - old_f1) < 1e-8
        t4_score = official_score(measured['metrics']['mean_f1'], measured['avg_runtime_s'] * 2,
                                  measured['avg_gflops'])
        row = {'seed': seed, 'gpu': torch.cuda.get_device_name(DEVICE),
               'metrics': measured['metrics'], 'runtime_s': measured['avg_runtime_s'],
               'gflops': measured['avg_gflops'], 'local_v100_score': measured['score_local_runtime'],
               'estimated_t4_score_factor2': t4_score, 'padding': 'sorted',
               'batch_size': 600, 'repetitions': 10}
        rows.append(row)
        print('seed', seed, 'mean F1', row['metrics']['mean_f1'], 'runtime', row['runtime_s'],
              'GFLOPs', row['gflops'], 'V100 score', row['local_v100_score'],
              'T4 estimate', row['estimated_t4_score_factor2'], flush=True)
        del model, encoder, packed
        torch.cuda.empty_cache()

    fields = ('mean_f1', 'mean_precision', 'mean_recall')
    summary = {'measurements': rows, 'means': {}, 'sample_sd': {},
               'note': 'Sequential V100 remeasurement; score is still not an official T4/hidden-test score.'}
    for key in ('runtime_s', 'gflops', 'local_v100_score', 'estimated_t4_score_factor2'):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary['means'][key] = float(values.mean())
        summary['sample_sd'][key] = float(values.std(ddof=1))
    for key in fields:
        values = np.asarray([row['metrics'][key] for row in rows], dtype=np.float64)
        summary['means'][key] = float(values.mean())
        summary['sample_sd'][key] = float(values.std(ddof=1))
    path = ROOT / 'results/seed_sweep_2026-11-12/v100_sequential_remeasurement.json'
    path.write_text(json.dumps(summary, indent=2) + '\n')
    print('saved', path)


if __name__ == '__main__':
    main()
