"""One table over all finished runs: ``python -m nlbse27ee.summarize runs``."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def collect(root) -> list[dict]:
    rows = []
    for path in sorted(Path(root).glob("*/results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        cfg, final = result["config"], result["calibration"]["final"]
        metrics = final["metrics"]
        rows.append({
            "run": cfg["name"], "layers": len(cfg["teacher_layers"]), "ffn": cfg["ffn_size"],
            "compact": cfg["compact_mode"], "flags": cfg["use_flags"], "exits": cfg["use_exits"],
            "kd_alpha": cfg["kd_alpha"], "best_epoch": result["training"]["best_epoch"],
            "vuln_P": metrics["Vulnerability"]["precision"], "vuln_R": metrics["Vulnerability"]["recall"],
            "vuln_F1": metrics["Vulnerability"]["f1"], "MAT_P": metrics["MAT"]["precision"],
            "MAT_R": metrics["MAT"]["recall"], "MAT_F1": metrics["MAT"]["f1"],
            "mean_F1": metrics["mean_f1"], "GFLOPs": final["gflops"],
            "local_runtime_s": final["local_runtime_s"], "est_T4_runtime_s": final["estimated_t4_runtime_s"],
            "est_T4_score": final["estimated_t4_score"], "exit_histogram": final["exit_histogram"],
        })
    return sorted(rows, key=lambda row: -row["est_T4_score"])


def markdown(rows: list[dict]) -> str:
    columns = ["run", "mean_F1", "vuln_F1", "MAT_F1", "GFLOPs", "est_T4_runtime_s", "est_T4_score",
               "exit_histogram"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = [f"{row[c]:.4f}" if isinstance(row[c], float) and c not in ("GFLOPs", "est_T4_runtime_s")
                 else f"{row[c]:,.1f}" if isinstance(row[c], float) else str(row[c]) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(markdown(collect(sys.argv[1] if len(sys.argv) > 1 else "runs")))
