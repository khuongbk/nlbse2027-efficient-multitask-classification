"""Measure one exported model under the three evaluation padding modes.

``python -m nlbse27ee.padding_ablation --model <export dir or HF repo> --out results/padding.json``

Same organizer loop for every mode (batch 600, fp16 autocast, profiler FLOPs,
repetitions); only how batches are formed and padded changes. Predictions are
identical across modes up to fp16 rounding, so F1 is reported as a sanity check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import PADDING_MODES, Packed
from .evaluate import organizer_protocol
from .exits import official_score
from .export import load_model
from .preprocess import pack


def run(model_source, batch_size: int = 600, repetitions: int = 10, t4_factor: float | None = None,
        modes=PADDING_MODES) -> list[dict]:
    from datasets import load_dataset

    device = torch.device("cuda")
    model, encoder = load_model(model_source, device)
    test = load_dataset("NLBSE/nlbse27-code-comment-classification")["test"]
    assert len(test) == 30_655
    packed = Packed(*pack(encoder(test["Function"])))
    labels = np.asarray(test["Label"], dtype=np.int64)
    rows = []
    for mode in modes:
        result = organizer_protocol(model, packed, labels, encoder.tokenizer.pad_token_id, device,
                                    batch_size=batch_size, repetitions=repetitions, padding=mode)
        row = {"padding": mode, "mean_f1": result["metrics"]["mean_f1"],
               "gflops": result["avg_gflops"], "runtime_s": result["avg_runtime_s"],
               "padded_tokens": result["padded_tokens"], "real_tokens": result["real_tokens"],
               "score_with_this_gpu_runtime": result["score_local_runtime"],
               "gpu": result["gpu"], "batch_size": batch_size, "repetitions": repetitions}
        if t4_factor:
            row["estimated_t4_runtime_s"] = result["avg_runtime_s"] * t4_factor
            row["estimated_t4_score"] = official_score(row["mean_f1"], row["estimated_t4_runtime_s"],
                                                       row["gflops"])
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def markdown(rows: list[dict]) -> str:
    lines = ["| Padding | Padded tokens | Mean F1 | GFLOPs | Runtime (s) | Score (this GPU's runtime) |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['padding']} | {r['padded_tokens']:,} | {r['mean_f1']:.4f} | {r['gflops']:,.0f} | "
                     f"{r['runtime_s']:.2f} | {r['score_with_this_gpu_runtime']:.4f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=600)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--t4-factor", type=float)
    parser.add_argument("--out")
    args = parser.parse_args()
    rows = run(args.model, args.batch_size, args.repetitions, args.t4_factor)
    print(markdown(rows))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
