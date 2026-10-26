"""Metrics, the official score, exit thresholds, and score-aware exit-margin search.

The FLOPs simulator mirrors what ``torch.profiler(with_flops=True)`` counts for
``EarlyExitUniXcoder.forward``: matrix multiplies (addmm) plus the elementwise
mean-pooling multiply; fused attention (SDPA) is not counted by the profiler.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve

LABELS = ("Vulnerability", "MAT")
MAX_AVG_RUNTIME = 1000.0
MAX_AVG_GFLOPS = 3_000_000.0


def official_score(avg_f1: float, avg_runtime: float, avg_gflops: float) -> float:
    return (0.6 * avg_f1
            + 0.2 * max((MAX_AVG_RUNTIME - avg_runtime) / MAX_AVG_RUNTIME, 0.0)
            + 0.2 * max((MAX_AVG_GFLOPS - avg_gflops) / MAX_AVG_GFLOPS, 0.0))


def label_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    """Per-label TP/FP/TN/FN, precision, recall, F1 and their means (organizer formulas)."""
    out = {}
    for j, name in enumerate(LABELS):
        y, p = labels[:, j] == 1, predictions[:, j] == 1
        tp, fp = int(np.sum(p & y)), int(np.sum(p & ~y))
        fn, tn = int(np.sum(~p & y)), int(np.sum(~p & ~y))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[name] = {"precision": precision, "recall": recall, "f1": f1,
                     "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    for metric in ("precision", "recall", "f1"):
        out["mean_" + metric] = float(np.mean([out[name][metric] for name in LABELS]))
    return out


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """F1-optimal threshold placed midway below the chosen score, so ``score > t`` is exact."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    chosen = thresholds[int(np.argmax(f1))]
    below = scores[scores < chosen]
    return float((chosen + below.max()) / 2) if below.size else float(chosen - 1e-3)


def exit_thresholds(all_logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """[layers, 2] logit thresholds, each tuned for F1 on that exit alone."""
    return np.array([[best_threshold(labels[:, j], all_logits[k, :, j]) for j in range(2)]
                     for k in range(all_logits.shape[0])], dtype=np.float64)


def exit_layers(all_logits: np.ndarray, thresholds: np.ndarray, margins: np.ndarray) -> np.ndarray:
    distance = np.abs(all_logits - thresholds[:, None, :])
    confident = (distance >= margins[:, None, :]).all(-1)
    confident[-1] = True
    return confident.argmax(0)


def decisions(all_logits, thresholds, layer) -> np.ndarray:
    rows = np.arange(all_logits.shape[1])
    return (all_logits[layer, rows] - thresholds[layer] > 0).astype(np.int64)


class FlopsModel:
    """GFLOPs the profiler reports for one pass over length-sorted batches.

    Exit ``e`` is evaluated on the rows still active (exit index >= e), at the padded
    width of those rows; transformer layer ``j`` runs on the rows whose exit index is at
    least ``j + exit0``. Pooling costs ``hidden`` FLOPs per token (mean) or ``3 * hidden``
    (attention: score projection + weighted sum); the classifier ``4 * hidden`` per row.
    """

    def __init__(self, lengths: np.ndarray, ffn_sizes: list[int], hidden: int = 768,
                 batch_size: int = 600, scale: float = 1.0, exit0: bool = False, head: str = "mean"):
        self.scale = scale  # multiply results, e.g. to express a subset at full-test-set size
        self.batches = [np.sort(lengths[b]) for b in _sorted_index_batches(lengths, batch_size)]
        self.layer_cost = np.array([2 * hidden * (4 * hidden + 2 * f) for f in ffn_sizes], dtype=np.float64)
        self.pool_cost = hidden * (3 if head == "attn" else 1)
        self.head = 2 * hidden * 2
        self.offset = int(exit0)
        self.exits = len(ffn_sizes) + self.offset
        self.order = np.argsort(lengths, kind="stable")
        self.batch_size = batch_size

    def gflops(self, layer_of_row: np.ndarray) -> float:
        total = 0.0
        layer_sorted = layer_of_row[self.order]
        for b, lengths in enumerate(self.batches):
            exits = layer_sorted[b * self.batch_size:b * self.batch_size + len(lengths)]
            for e in range(self.exits):
                active = exits >= e
                n = int(active.sum())
                if n == 0:
                    break
                width = int(lengths[active].max())
                cost = self.pool_cost + (self.layer_cost[e - self.offset] if e >= self.offset else 0.0)
                total += n * width * cost + n * self.head
        return total * self.scale / 1e9


def _sorted_index_batches(lengths, batch_size):
    order = np.argsort(lengths, kind="stable")
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


DEFAULT_GRID = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, np.inf])


def search_margins(all_logits: np.ndarray, labels: np.ndarray, thresholds: np.ndarray,
                   flops: FlopsModel, runtime_per_gflop: float, grid=DEFAULT_GRID,
                   passes: int = 3) -> dict:
    """Coordinate ascent over exits on per-label margins maximizing the official score.

    A row exits only when both labels are confident, so each exit's two margins are
    searched jointly (a one-label move from (inf, inf) can never change anything).
    Runtime is modelled as proportional to profiler GFLOPs (calibrated on a measured
    no-exit run), so the search trades F1 against both efficiency terms.
    """
    layers = all_logits.shape[0]
    margins = np.full((layers, 2), np.inf)

    def evaluate(m):
        layer = exit_layers(all_logits, thresholds, m)
        metrics = label_metrics(labels, decisions(all_logits, thresholds, layer))
        gflops = flops.gflops(layer)
        runtime = gflops * runtime_per_gflop
        return official_score(metrics["mean_f1"], runtime, gflops), metrics, gflops, runtime, layer

    best = evaluate(margins)
    for _ in range(passes):
        improved = False
        for k in range(layers - 1):
            for vulnerability in grid:
                for mat in grid:
                    trial = margins.copy()
                    trial[k] = (vulnerability, mat)
                    result = evaluate(trial)
                    if result[0] > best[0] + 1e-7:
                        best, margins, improved = result, trial, True
        if not improved:
            break
    score, metrics, gflops, runtime, layer = best
    return {"margins": margins, "score": score, "metrics": metrics, "gflops": gflops,
            "runtime_s": runtime,
            "exit_histogram": np.bincount(layer, minlength=layers).tolist()}


def holdout_check(all_logits, labels, flops_factory, runtime_per_gflop, seed: int = 0) -> dict:
    """Tune thresholds+margins on one random half, report the other half (optimism check)."""
    rng = np.random.default_rng(seed)
    half = rng.permutation(labels.shape[0]) < labels.shape[0] // 2
    out = {}
    for name, fit, test in (("A->B", half, ~half), ("B->A", ~half, half)):
        thresholds = exit_thresholds(all_logits[:, fit], labels[fit])
        tuned = search_margins(all_logits[:, fit], labels[fit], thresholds,
                               flops_factory(fit), runtime_per_gflop)
        layer = exit_layers(all_logits[:, test], thresholds, tuned["margins"])
        held = label_metrics(labels[test], decisions(all_logits[:, test], thresholds, layer))
        out[name] = {"fit_mean_f1": tuned["metrics"]["mean_f1"], "heldout_mean_f1": held["mean_f1"]}
    return out
