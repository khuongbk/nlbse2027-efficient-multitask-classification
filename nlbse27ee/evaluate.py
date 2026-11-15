"""The organizers' measurement protocol (CodeBERT_baseline.ipynb, cell 3) for our model.

Identical loop: batch 600, fp16 autocast, torch.profiler(with_flops=True) around
10 repetitions per batch, wall-clock ``time.time()``; tokenization and the transfer
to the GPU are outside the timed region. Reported runtime and GFLOPs are the sums
over all batches divided by the number of repetitions, i.e. one pass over the set.

Padding is a measured choice, not an assumption (see ``data.inference_batches``):
the reference notebook pads the whole set to one length (``global``). Sorting alone
changes nothing under global padding; the saving comes from padding each batch to
its own longest row (``dynamic``), and sorting by length (``sorted``) makes batches
homogeneous. Predictions are always written back in the original row order.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .data import Packed, inference_batches, pad_rows, sorted_batches
from .exits import LABELS, label_metrics, official_score

T4_BASELINE_RUNTIME_S = 192.0984256744385   # organizer notebook output on Colab T4
T4_BASELINE_GFLOPS = 2666509.7131617307


def organizer_protocol(model, packed: Packed, labels: np.ndarray, pad_id: int, device,
                       batch_size: int = 600, repetitions: int = 10,
                       padding: str = "sorted") -> dict:
    model.eval()
    predictions = np.full((len(packed), 2), -1, dtype=np.int64)
    exit_layers = np.zeros(len(packed), dtype=np.int64)
    total_time = total_flops = 0.0
    padded_tokens = 0
    for rows, width in inference_batches(packed.lengths, batch_size, padding):
        input_ids, attention_mask = pad_rows(packed, rows, pad_id, width=width)
        padded_tokens += input_ids.numel()
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        begin = time.time()
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                with torch.profiler.profile(with_flops=True) as profiler:
                    for _ in range(repetitions):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        total_time += time.time() - begin
        total_flops += sum(event.flops for event in profiler.key_averages()) / 1e9
        predictions[rows] = (torch.sigmoid(outputs.logits) > 0.5).int().cpu().numpy()
        if hasattr(outputs, "exit_layer"):
            exit_layers[rows] = outputs.exit_layer.cpu().numpy()
        del input_ids, attention_mask, outputs
        torch.cuda.empty_cache()
    assert (predictions >= 0).all(), "some rows were not predicted"
    n_exits = getattr(getattr(model, "cfg", None), "exits", 0)   # keep empty trailing exits
    metrics = label_metrics(labels, predictions)
    runtime, gflops = total_time / repetitions, total_flops / repetitions
    return {"metrics": metrics, "avg_runtime_s": runtime, "avg_gflops": gflops,
            "score_local_runtime": official_score(metrics["mean_f1"], runtime, gflops),
            "exit_histogram": np.bincount(exit_layers, minlength=n_exits).tolist(),
            "gpu": torch.cuda.get_device_name(device), "batch_size": batch_size,
            "repetitions": repetitions, "padding": padding,
            "padded_tokens": int(padded_tokens), "real_tokens": int(packed.lengths.sum()),
            "predictions": predictions}


def baseline_protocol(device, texts: list[str], labels: np.ndarray,
                      model_id: str = "NLBSE/nlbse27-code-comment-classification-baseline",
                      repetitions: int = 10) -> dict:
    """Run the organizers' CodeBERT baseline cell verbatim to calibrate V100 -> T4 runtime."""
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import RobertaForSequenceClassification, RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained(model_id)
    model = RobertaForSequenceClassification.from_pretrained(model_id).to(device).eval()
    tokenized = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True,
                          max_length=512)
    loader = DataLoader(TensorDataset(tokenized["input_ids"], tokenized["attention_mask"]),
                        batch_size=600)
    predictions, total_time, total_flops = [], 0.0, 0.0
    for input_ids, attention_mask in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        begin = time.time()
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                with torch.profiler.profile(with_flops=True) as profiler:
                    for _ in range(repetitions):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        total_time += time.time() - begin
        total_flops += sum(event.flops for event in profiler.key_averages()) / 1e9
        predictions.extend((torch.sigmoid(outputs.logits) > 0.5).int().cpu().numpy().tolist())
        del input_ids, attention_mask, outputs
        torch.cuda.empty_cache()
    metrics = label_metrics(labels, np.array(predictions))
    runtime, gflops = total_time / repetitions, total_flops / repetitions
    return {"metrics": metrics, "avg_runtime_s": runtime, "avg_gflops": gflops,
            "t4_runtime_factor": T4_BASELINE_RUNTIME_S / runtime,
            "gpu": torch.cuda.get_device_name(device)}


@torch.no_grad()
def predict_labels(model, encoder, texts, device="cuda", batch_size: int = 600) -> np.ndarray:
    """[n, 2] labels in [Vulnerable, MAT] order, in the input order (for the hidden test)."""
    from .preprocess import pack

    packed = Packed(*pack(encoder(texts)))
    labels = np.zeros((len(packed), 2), dtype=np.int64)
    model.eval()
    for rows in sorted_batches(packed.lengths, batch_size):
        input_ids, attention_mask = pad_rows(packed, rows, encoder.tokenizer.pad_token_id)
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        labels[rows] = (outputs.logits > 0).int().cpu().numpy()
    return labels


@torch.no_grad()
def collect_exit_logits(model, packed: Packed, pad_id: int, device, batch_size: int = 600) -> np.ndarray:
    """[layers, rows, 2] raw logits of every exit, same batching/precision as inference."""
    model.eval()
    out = np.zeros((model.cfg.exits, len(packed), 2), dtype=np.float32)
    for rows in sorted_batches(packed.lengths, batch_size):
        input_ids, attention_mask = pad_rows(packed, rows, pad_id)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model.all_exits(input_ids.to(device), attention_mask.to(device))
        out[:, rows] = logits.float().cpu().numpy()
    return out


def report_table(result: dict) -> str:
    metrics = result["metrics"]
    lines = ["| Label | Precision | Recall | F1 |", "|---|---:|---:|---:|"]
    for name in LABELS:
        m = metrics[name]
        lines.append(f"| {name} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")
    lines.append(f"| **Mean** | {metrics['mean_precision']:.4f} | {metrics['mean_recall']:.4f} | "
                 f"{metrics['mean_f1']:.4f} |")
    return "\n".join(lines)
