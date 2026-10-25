# Efficient multi-task code classification for NLBSE'27

Submission code for the [NLBSE'27 tool competition on code classification](https://nlbse2027.github.io/tools/):
one model that reads a C/C++ `Function` and predicts two binary labels, `[Vulnerable, MAT]`.

The competition ranks models by

```
score = 0.60 * mean F1
      + 0.20 * max((1000 - runtime_s) / 1000, 0)
      + 0.20 * max((3,000,000 - GFLOPs) / 3,000,000, 0)
```

measured on a Colab T4 with the organizers' notebook. The CodeBERT baseline scores **0.7299**
(mean F1 0.910, 192 s, 2,666,510 GFLOPs). Because the GFLOPs cap sits just above the baseline,
compute is worth as much as accuracy, so the method is built around spending less compute per input.

## Method

1. **Teacher.** `microsoft/unixcoder-base-nine` (12 layers) fine-tuned on all 214,586 training rows
   with binary cross-entropy (`configs/T0_teacher12.json`, 4-GPU DDP, best of 10 epochs).
2. **Input** (a function of `Function` only):
   - whitespace collapsed (16% fewer tokens; rows truncated at 512 tokens drop from 3.8% to 0.5%);
   - a flag token if `TODO`, `FIXME`, `HACK` or `XXX` occurs: `<satd_c>` inside a comment, `<satd_w>`
     elsewhere. It is computed on the full text, so truncation cannot hide it;
   - UniXcoder's encoder-only format, at most 512 tokens.
3. **Student.** Teacher layers 1, 3, 5, 7, 9 and 11, with each feed-forward block pruned to the 1,536
   most important of its 3,072 neurons (mean |activation| x output-weight norm on 4,096 training rows).
   One classification head after every layer.
4. **Training.** Weighted sum over the six exits of `0.5 * BCE(labels) + 0.5 * T^2 * BCE(teacher / T)`
   (T = 2), deeper exits weighted more; best of 10 epochs on the validation split.
5. **Early exit and calibration.** Each exit gets its own per-label thresholds. A row leaves the batch
   after layer *k* when both labels are a margin away from those thresholds. Margins are searched to
   maximize the official score, using a FLOPs simulator that matches the profiler within 0.02%.
6. **Evaluation.** The reference measurement loop, with dynamic padding per batch after sorting rows
   by length (see "Differences from the baseline notebook").

Data use follows the guide: the 214,586 training rows train every model; the public test split
(`ds["test"]`, 30,655 rows), which the guide designates for "model tuning, selection, and validation
reporting", selects checkpoints and calibrates thresholds. The hidden test set is the organizers'.

## Results

Validation = public test split, 30,655 rows. GFLOPs are totals for one pass over the set, measured
with the reference profiler loop.

**Submitted model, Colab T4** (`notebooks/NLBSE27_Submission_Colab_T4.ipynb`, batch 600, sorted
dynamic padding, 10 repetitions; `results/colab_t4_R8.json`):

| Model | Vuln P / R / F1 | MAT P / R / F1 | Mean P / R / F1 | GFLOPs | Runtime (s) | Score |
|---|---|---|---|---:|---:|---:|
| CodeBERT baseline (official, T4) | 0.9427 / 0.9353 / 0.9390 | 0.9783 / 0.8020 / 0.8814 | 0.9605 / 0.8687 / 0.9102 | 2,666,510 | 192.10 | 0.7299 |
| **Student R8 (T4)** | 0.9468 / 0.9390 / 0.9429 | 0.9956 / 0.8871 / 0.9382 | 0.9712 / 0.9131 / **0.9406** | **109,169** | **12.61** | **0.9545** |

The other rows in this README (teacher, ablations) were measured during training on an NVIDIA L40S.
Their F1 and GFLOPs are exact; runtime and score are **estimates**: L40S runtime x 5.32, the ratio
between the baseline's official T4 runtime and the same baseline cell on the L40S (which reproduced
its 2,666,510 GFLOPs and F1 0.9102 exactly). For R8 this estimate was 15.1 s and 0.9540, against
12.61 s and 0.9545 measured on the T4, so the estimates are slightly conservative.

| Model | Vuln P / R / F1 | MAT P / R / F1 | Mean F1 | GFLOPs | Runtime (s, est.) | Score (est.) |
|---|---|---|---:|---:|---:|---:|
| Teacher, 12 layers | 0.9592 / 0.9455 / 0.9523 | 0.9890 / 0.8931 / 0.9386 | 0.9455 | 834,264 | ~62.9 | ~0.899 |
| Student R8 (L40S) | 0.9468 / 0.9390 / 0.9429 | 0.9956 / 0.8871 / 0.9382 | 0.9406 | 109,167 | ~15.1 | ~0.954 |

R8 has 68.7M parameters (teacher 125.3M). Rows exit after layer 1-6:
7,864 / 11,671 / 8,872 / 1,697 / 149 / 402, i.e. 2.2 layers on average.

### Ablations (all students distilled from the same teacher)

| Run | Change | Mean F1 | GFLOPs | Score (est.) |
|---|---|---:|---:|---:|
| R0_4L_plain | 4 layers (2, 5, 8, 11), raw input, no exits | 0.9353 | 324,306 | 0.9346 |
| R1_4L_compact_flag | + whitespace compaction + flag tokens | 0.9392 | 278,093 | 0.9407 |
| R2_4L_exits | + early exit | 0.9295 | 136,878 | 0.9459 |
| R3_4L_exits_noKD | R2 without distillation | 0.9307 | 139,567 | 0.9464 |
| R6_4L_exits_lines | R2 with newline-preserving compaction | 0.9328 | 157,777 | 0.9461 |
| R4_3L_exits | 3 layers (3, 7, 11) | 0.9234 | 131,203 | 0.9429 |
| R5_4L_ffn1536_exits | R2 with FFN pruned to 1,536 | 0.9289 | 99,523 | 0.9482 |
| R7_6L_exits | 6 layers (1, 3, ..., 11) | 0.9388 | 147,520 | 0.9504 |
| R9_6L_ffn2048_exits | R7 with FFN 2,048 | 0.9384 | 136,520 | 0.9506 |
| **R8_6L_ffn1536_exits** | R7 with FFN 1,536 | **0.9406** | **109,167** | **0.9540** |
| R10_6L_exits_e14 | R7, 14 epochs | 0.9368 | 139,549 | 0.9497 |
| R11_6L_ffn1536_exits_e14 | R8, 14 epochs | 0.9395 | 113,725 | 0.9531 |

What they show:
- **Early exit** is the largest gain (+0.010 to +0.022 score for the same network); it costs about
  0.002-0.004 F1.
- **Compaction + flags** add 0.004 F1 (MAT 0.928 to 0.938) and remove 14% of the GFLOPs.
- **FFN pruning** keeps F1 and removes ~26% of the GFLOPs.
- **Six layers with early exit** beat four: most rows still exit after one or two layers.
- **Distillation adds nothing measurable** (R3 vs R2): the teacher fits the training set almost
  perfectly (99.94% sign agreement), so its soft targets carry little beyond the labels. The teacher
  matters through the weights the student is initialized from.
- **MAT F1 is capped near 0.94**: about 13% of MAT-positive functions contain no tag anywhere in
  `Function`.
- Threshold/margin tuning is not optimistic beyond ~0.002 F1: a 2-fold split of the validation set
  (tune on one half, evaluate on the other) is in each `results/runs/*/results.json` (`holdout`).

### Padding ablation

The reference notebook pads the whole evaluation set to one length (`padding=True` on all rows at
once), so sorting alone cannot reduce padding. We therefore measured R8 with the same reference loop
(batch 600; 3 repetitions on an RTX 4060 Laptop GPU, so runtimes are not T4 numbers) under three
padding modes (`python -m nlbse27ee.padding_ablation`, `results/padding_ablation_R8_rtx4060_b600.json`):

| Padding | Padded tokens | Mean F1 | GFLOPs | Runtime (s), RTX 4060 Laptop |
|---|---:|---:|---:|---:|
| `global`: whole set padded to one length (as the baseline) | 15,695,360 | 0.9406 | 324,741 | 26.15 |
| `dynamic`: each batch padded to its longest row, original order | 15,588,005 | 0.9406 | 323,727 | 23.97 |
| `sorted`: rows sorted by length, then `dynamic` (used for all reported results) | 4,910,360 | 0.9406 | 109,167 | 8.26 |

The real token count is 4,764,112. Dynamic padding alone barely helps at batch 600: almost every
batch of 600 rows in the original order contains a 512-token row. Sorting by length is what makes
per-batch padding effective: padded tokens drop to 3% above the real count and GFLOPs by 3.0x, with
identical F1. Of R8's 24.4x GFLOPs reduction relative to the baseline (2,666,510 to 109,167), the model
(depth and width pruning, early exit, input compaction) accounts for 8.2x under the baseline's own
padding, and sorted dynamic padding for the remaining 3.0x. The `sorted` GFLOPs match the L40S
measurement exactly, as GFLOPs do not depend on the GPU.

## Differences from the baseline notebook

The measurement is the reference loop: for each batch of 600, the forward pass runs 10 times inside
`torch.profiler.profile(with_flops=True)` and `time.time()`; time and `sum(k.flops) / 1e9` are summed
over batches and divided by 10. Tokenization and the GPU transfer are outside the timed region.
Everything else below is the baseline's own configuration, not a competition requirement:

| Setting | Baseline notebook | This submission |
|---|---|---|
| Model | CodeBERT, 12 layers | UniXcoder-based student, 6 layers, FFN 1,536, early exit |
| Input | `Function`, RoBERTa tokenizer, max 512 | `Function` only; compaction + flag token; UniXcoder encoder-only format; max 512 |
| Padding | whole set tokenized with `padding=True` (one global length) | each batch padded to its own longest row, rows sorted by length; predictions written back in the original order |
| Batch size | 600 | 600 |
| Precision | fp32 weights, CUDA autocast | fp16 weights, CUDA autocast |
| Decision | `sigmoid(logits) > 0.5` | `sigmoid(logits) > 0.5` on logits shifted by per-label thresholds chosen on validation |

## Reproducing

Hardware used: one machine with 4x NVIDIA L40S (48 GB). Total about 2.5 h of wall time: teacher
~45 min (DDP on 4 GPUs), first grid ~1.3 h, second grid ~1 h. A single 24 GB GPU works with
`NPROC=1` (about 4x longer for the teacher).

```bash
git clone https://github.com/khuongbk/nlbse2027-efficient-multitask-classification
cd nlbse2027-efficient-multitask-classification
python -m venv --system-site-packages venv && . venv/bin/activate   # keeps an existing CUDA torch
pip install -r requirements.txt

bash scripts/run_all.sh      # teacher -> calibration + teacher logits -> grid R0-R7 -> summary
bash scripts/run_round2.sh   # 6-layer grid R8-R11
python -m nlbse27ee.summarize runs
```

Or step by step:

```bash
# teacher (4 GPUs); writes teacher/unixcoder12_teacher.pt
NCCL_P2P_DISABLE=1 python -m torch.distributed.run --nproc_per_node 4 -m nlbse27ee.train --config configs/T0_teacher12.json
# one student on one GPU; writes runs/R8_6L_ffn1536_exits/{best.pt,results.json,export/}
python -m nlbse27ee.train --config configs/R8_6L_ffn1536_exits.json --gpu 0
# padding ablation of an exported model
python -m nlbse27ee.padding_ablation --model runs/R8_6L_ffn1536_exits/export --out results/padding.json
```

Notes:
- Every command resumes from `runs/<name>/last.pt` when re-run; encodings and teacher logits are cached in `cache/`.
- Use `python -m torch.distributed.run`, not `torchrun`, which may belong to another Python. If DDP hangs with
  100% GPU load and almost no memory, the container blocks GPU peer-to-peer copies; `NCCL_P2P_DISABLE=1`
  (set by `run_all.sh`) fixes it.
- `notebooks/NLBSE27_EarlyExit_UniXcoder.ipynb` runs one student step by step (data checks, the
  equivalence test of our encoder against `transformers`' `RobertaModel`, training, calibration, export).

### Evaluating on Colab T4

`runs/<name>/export/` is self-contained (fp16 weights, tokenizer, exit policy, and a copy of `nlbse27ee`).
Zip it (or upload it to the Hugging Face Hub), open `notebooks/NLBSE27_Submission_Colab_T4.ipynb` on a
T4 runtime, set `MODEL_ZIP` or `MODEL_SOURCE`, and run all. It prints per-label P/R/F1, their means,
runtime, GFLOPs and the submission score; `RUN_PADDING_ABLATION = True` also measures all padding modes.

## Repository layout

```
nlbse27ee/        preprocessing, model, training, calibration, evaluation, export
configs/          T0 (teacher) and R0-R11 (students)
scripts/          run_all.sh (teacher + grid 1), run_round2.sh (grid 2)
notebooks/        training walkthrough; Colab T4 evaluation
results/          results.json / history.json / config.json of every run, logs, summary, padding ablation
docs/             COMPLIANCE.md: each competition requirement and where it is met
```

## License

MIT, see `LICENSE`.
