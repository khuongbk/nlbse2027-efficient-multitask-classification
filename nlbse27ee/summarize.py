"""One table over all finished runs: ``python -m nlbse27ee.summarize runs``.

Runs are ranked by the score estimated on their selection split, which is how a
winner is chosen without looking at the public test. Public-test columns are the
one-time measurement (held out when the run used ``validation: inner``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def collect(root) -> list[dict]:
    rows = []
    for path in sorted(Path(root).glob("*/results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if "calibration" not in result:      # fold teachers only write out-of-fold logits
            continue
        cfg, calibration = result["config"], result["calibration"]
        final, metrics = calibration["final"], calibration["final"]["metrics"]
        selection = calibration.get("selection", {})
        rows.append({
            "run": cfg["name"], "selection_split": cfg.get("validation", "public"),
            "layers": len(cfg["teacher_layers"]), "ffn": cfg["ffn_size"],
            "best_epoch": result["training"]["best_epoch"],
            "selection_F1": selection.get("mean_f1", metrics["mean_f1"]),
            "selection_score": selection.get("estimated_score", final["estimated_t4_score"]),
            "public_held_out": final.get("held_out", False),
            "vuln_P": metrics["Vulnerability"]["precision"], "vuln_R": metrics["Vulnerability"]["recall"],
            "vuln_F1": metrics["Vulnerability"]["f1"], "MAT_P": metrics["MAT"]["precision"],
            "MAT_R": metrics["MAT"]["recall"], "MAT_F1": metrics["MAT"]["f1"],
            "public_F1": metrics["mean_f1"], "GFLOPs": final["gflops"],
            "est_T4_runtime_s": final["estimated_t4_runtime_s"],
            "public_score_est": final["estimated_t4_score"], "exit_histogram": final["exit_histogram"],
        })
    return sorted(rows, key=lambda row: -row["selection_score"])


def markdown(rows: list[dict]) -> str:
    columns = ["run", "selection_split", "best_epoch", "selection_score", "public_held_out", "public_F1",
               "vuln_F1", "MAT_F1", "GFLOPs", "est_T4_runtime_s", "public_score_est", "exit_histogram"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = []
        for c in columns:
            value = row[c]
            if isinstance(value, float):
                cells.append(f"{value:,.1f}" if c in ("GFLOPs", "est_T4_runtime_s") else f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(markdown(collect(sys.argv[1] if len(sys.argv) > 1 else "runs")))
