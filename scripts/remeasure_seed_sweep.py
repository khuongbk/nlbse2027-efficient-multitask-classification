#!/usr/bin/env python3
"""Remeasure the five seed exports one after another on an idle GPU.

During training the seeds share GPUs, so their runtimes overlap. This script reloads each export,
runs the organizers' measurement loop alone on one GPU, checks that F1 matches the training-time
result and writes runtime, GFLOPs and scores to
results/seed_sweep_2026-11-12/v100_sequential_remeasurement.json (the reported run used a V100).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True   # keep __pycache__ (and local paths) out of the export folders
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402

from nlbse27ee.data import Packed  # noqa: E402
from nlbse27ee.evaluate import organizer_protocol  # noqa: E402
from nlbse27ee.exits import official_score  # noqa: E402
from nlbse27ee.export import load_model  # noqa: E402
from nlbse27ee.preprocess import pack  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)
RUNS = ROOT / "runs/seed_sweep"
OUT = ROOT / "results/seed_sweep_2026-11-12/v100_sequential_remeasurement.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--runtime-factor", type=float, default=2.0,
                        help="assumed GPU-to-T4 runtime factor for the estimated score")
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    gpu = torch.cuda.get_device_name(device)
    data = load_dataset("NLBSE/nlbse27-code-comment-classification")["test"]
    functions = [str(value or "") for value in data["Function"]]
    labels = np.asarray(data["Label"], dtype=np.int64)

    rows = []
    for seed in SEEDS:
        run = RUNS / f"A2_R11_all_seed{seed}"
        model, encoder = load_model(run / "export", device)
        packed = Packed(*pack(encoder(functions)))
        measured = organizer_protocol(model, packed, labels, encoder.tokenizer.pad_token_id, device,
                                      batch_size=600, repetitions=10, padding="sorted")
        trained = json.loads((run / "results.json").read_text())["calibration"]["final"]
        assert abs(measured["metrics"]["mean_f1"] - trained["metrics"]["mean_f1"]) < 1e-8, seed
        rows.append({
            "seed": seed, "gpu": gpu, "metrics": measured["metrics"],
            "runtime_s": measured["avg_runtime_s"], "gflops": measured["avg_gflops"],
            "local_v100_score": measured["score_local_runtime"],   # key name kept for old files
            "estimated_t4_score_factor2": official_score(
                measured["metrics"]["mean_f1"], measured["avg_runtime_s"] * args.runtime_factor,
                measured["avg_gflops"]),
            "padding": "sorted", "batch_size": 600, "repetitions": 10})
        print(f"seed {seed}: F1 {measured['metrics']['mean_f1']:.4f}, "
              f"{measured['avg_runtime_s']:.2f} s, {measured['avg_gflops']:.0f} GFLOPs", flush=True)
        del model, encoder, packed
        torch.cuda.empty_cache()

    summary = {"measurements": rows, "means": {}, "sample_sd": {},
               "note": f"Sequential remeasurement on {gpu}; the score uses an assumed runtime factor "
                       f"of {args.runtime_factor} and is not an official T4 or hidden-test score."}
    values = {key: [row[key] for row in rows]
              for key in ("runtime_s", "gflops", "local_v100_score", "estimated_t4_score_factor2")}
    values.update({key: [row["metrics"][key] for row in rows]
                   for key in ("mean_f1", "mean_precision", "mean_recall")})
    for key, column in values.items():
        summary["means"][key] = float(np.mean(column))
        summary["sample_sd"][key] = float(np.std(column, ddof=1))
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print("saved", OUT)


if __name__ == "__main__":
    main()
