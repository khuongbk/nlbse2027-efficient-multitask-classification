# A2_R11_all five-seed summary

The five student runs use one fixed teacher, fixed OOF logits and the same 92/8 split. Each checkpoint and exit policy was selected on inner validation. Public metrics below are aggregated; the public test did not select a seed.

| Seed | Best epoch | Val mean F1 | Vul F1 | MAT F1 | Mean F1 | Runtime (s) | GFLOPs | Est. T4 score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 6 | 0.9431 | 0.9414 | 0.9265 | 0.9339 | 5.04 | 59936 | 0.9544 |
| 43 | 11 | 0.9442 | 0.9415 | 0.9277 | 0.9346 | 5.41 | 57070 | 0.9548 |
| 44 | 10 | 0.9449 | 0.9430 | 0.9308 | 0.9369 | 5.47 | 58913 | 0.9560 |
| 45 | 9 | 0.9452 | 0.9427 | 0.9226 | 0.9327 | 4.98 | 58963 | 0.9537 |
| 46 | 10 | 0.9444 | 0.9427 | 0.9310 | 0.9369 | 5.27 | 56085 | 0.9563 |

| Metric | Mean | Sample SD |
|---|---:|---:|
| Vul_accuracy | 0.962518 | 0.000547 |
| Vul_precision | 0.951764 | 0.002723 |
| Vul_recall | 0.932988 | 0.001724 |
| Vul_f1 | 0.942279 | 0.000753 |
| MAT_accuracy | 0.997749 | 0.000108 |
| MAT_precision | 0.984882 | 0.004242 |
| MAT_recall | 0.876832 | 0.003542 |
| MAT_f1 | 0.927719 | 0.003462 |
| mean_accuracy | 0.980134 | 0.000291 |
| mean_precision | 0.968323 | 0.002335 |
| mean_recall | 0.904910 | 0.001446 |
| mean_f1 | 0.934999 | 0.001855 |
| local_runtime_s | 5.234204 | 0.218238 |
| profiler_gflops | 58193.497618 | 1569.620409 |
| estimated_t4_score | 0.955026 | 0.001101 |

All seeds must have the same split SHA256: bae8e73a0dffcdfc31d1fc6d5c6c58caed367ce35abe6c1a8fd0ca86d4136cc8.

Runtime and profiler GFLOPs were remeasured sequentially on one V100 to avoid timing overlap. The T4 score is an estimate using the fixed 2.0 runtime factor; it is not an official T4/hidden-test score.
