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

## Result

Public test split (`ds["test"]`, 30,655 rows), reference measurement loop (batch 600, fp16 autocast,
profiler FLOPs, 10 repetitions):

| Model | Vuln P / R / F1 | MAT P / R / F1 | Mean P / R / F1 | GFLOPs | Runtime (s) | Score |
|---|---|---|---|---:|---:|---:|
| CodeBERT baseline (official, T4) | 0.9427 / 0.9353 / 0.9390 | 0.9783 / 0.8020 / 0.8814 | 0.9605 / 0.8687 / 0.9102 | 2,666,510 | 192.10 | 0.7299 |
| **Submission** (`A2_R11_all_full`) | 0.9483 / 0.9390 / 0.9436 | 0.9955 / 0.8812 / 0.9349 | 0.9719 / 0.9101 / **0.9392** | **55,443** | **8.16** (T4) | **0.9582** (T4) |
| Same recipe, held out (`A2_R11_all`) | 0.9494 / 0.9276 / 0.9384 | 0.9910 / 0.8733 / 0.9284 | 0.9702 / 0.9004 / 0.9334 | 53,214 | 8.33 (T4) | 0.9548 (T4) |

The submission is trained on all training rows and calibrates its thresholds on the public test (see
"Data use and model selection"); the held-out row is the same recipe trained on 92% of the training
rows and never tuned on the public test, so it is the unbiased estimate and the one comparable to the
baseline, which also never consults the public test before its single evaluation. Both rows were measured
on a Colab Tesla T4 with `notebooks/NLBSE27_Submission_Colab_T4.ipynb`
(`results/stage_b/colab_t4_A2_R11_all_full.json`, `results/stage_a2/colab_t4_A2_R11_all.json`). The model has 68.7M parameters (teacher 125.3M)
and a row runs 1.07 of its 6 transformer layers on average.

## Method

1. **Teacher.** `microsoft/unixcoder-base-nine` (12 layers) fine-tuned with binary cross-entropy
   (`configs/T0_teacher12.json`, 4-GPU DDP).
2. **Input**, a function of `Function` only:
   - whitespace collapsed (16% fewer tokens; rows truncated at 512 tokens drop from 3.8% to 0.5%);
   - a flag token if `TODO`, `FIXME`, `HACK` or `XXX` occurs: `<satd_c>` inside a comment, `<satd_w>`
     elsewhere, computed on the full text so truncation cannot hide it;
   - UniXcoder's encoder-only format, at most 512 tokens.
3. **Student.** Teacher layers 1, 3, 5, 7, 9 and 11; each feed-forward block pruned to the 1,536 most
   important of its 3,072 neurons. Seven exits: one on the embeddings (**exit 0**, before any
   transformer layer) and one after every layer, each with **attention pooling** over tokens.
4. **Training.** Weighted sum over exits of `0.5 * BCE(labels) + 0.5 * T^2 * BCE(soft targets / T)`
   (T = 2), with **out-of-fold soft targets** from four fold teachers and an **EMA** of the weights.
5. **Early exit, calibrated on the official score.** Each exit has per-label thresholds; a row leaves
   the batch at exit *k* when both labels are a margin away from them. Margins are searched to
   maximize the official score, using a FLOPs simulator that matches the profiler within 0.02%.
6. **Evaluation.** The reference measurement loop, with dynamic padding per batch after sorting rows
   by length (see "Differences from the baseline notebook").

## Data use and model selection

| Rows | Used for |
|---|---|
| 197,419 training rows (92%) | training teachers and students in the research stages |
| 17,167 training rows (8%, stratified on both labels, seed 42) | selecting epochs, thresholds, exit margins and the winning configuration |
| 30,655 public-test rows (`ds["test"]`) | measured **once** per research model, after all choices were made (held out) |
| all 214,586 training rows | the final retraining of the selected recipe |

The work ran in three stages, all with the same split and code:

- **Stage A** (`scripts/run_stage_a.sh`): the teacher and 12 students (R0-R11); the winner on the
  selection split is R11 (6 layers, FFN 1,536, early exit).
- **Stage A2** (`scripts/run_stage_a2.sh`): four fold teachers, then four improvements on R11 (EMA;
  exit 0 + attention pooling; out-of-fold distillation; all three). The winner on the selection
  split is "all three" (`A2_R11_all`).
- **Stage B** (`scripts/run_stage_b.sh`): the teacher and the winner are retrained on all 214,586
  training rows, each stopped at its selected epoch, and the thresholds and exit margins are calibrated
  on the public test split, which the guide designates for "model tuning, selection, and validation
  reporting". This retrained model is the submission.

Public-test numbers of the research models (stages A and A2) are therefore held-out estimates. The
submission's public-test numbers use the public test for threshold calibration only; its held-out
counterpart is `A2_R11_all`. An earlier exploratory round that selected models on the public test is
kept in `results/exploratory_public_selection/`; it reached the same conclusions, but its numbers are
optimistic (for example its teacher scored F1 0.9455 on the public test versus 0.9428 held out).

## Results

GFLOPs are totals for one pass over the public test, measured with the reference profiler loop and
independent of the GPU. Runtimes were measured on an NVIDIA L40S and converted with the factor 5.32
(the baseline's official T4 runtime divided by the same baseline cell on the L40S, which reproduced its
2,666,510 GFLOPs and F1 0.9102 exactly); they are **estimates** except where marked T4. In an earlier
check the estimate was conservative (15.1 s estimated, 12.6 s measured on a T4), and for the
submission too (11.6 s estimated, 8.16 s measured on a T4).

### Stage A: architecture and input (`results/stage_a/`)

| Run | Change | Selection score | Public F1 (held out) | GFLOPs | Public score (est.) |
|---|---|---:|---:|---:|---:|
| R0_4L_plain | 4 layers (2, 5, 8, 11), raw input, no exits | 0.9344 | 0.9297 | 324,306 | 0.9312 |
| R1_4L_compact_flag | + whitespace compaction + flag tokens | 0.9393 | 0.9342 | 278,093 | 0.9377 |
| R2_4L_exits | + early exit | 0.9448 | 0.9205 | 136,672 | 0.9404 |
| R3_4L_exits_noKD | R2 without distillation | 0.9460 | 0.9257 | 138,527 | 0.9434 |
| R6_4L_exits_lines | R2 with newline-preserving compaction | 0.9464 | 0.9249 | 146,347 | 0.9423 |
| R4_3L_exits | 3 layers (3, 7, 11) | 0.9439 | 0.9182 | 130,205 | 0.9398 |
| R5_4L_ffn1536_exits | R2 with FFN pruned to 1,536 | 0.9480 | 0.9239 | 95,931 | 0.9455 |
| R7_6L_exits | 6 layers (1, 3, ..., 11) | 0.9518 | 0.9364 | 162,032 | 0.9476 |
| R9_6L_ffn2048_exits | R7 with FFN 2,048 | 0.9540 | 0.9356 | 124,545 | 0.9499 |
| R8_6L_ffn1536_exits | R7 with FFN 1,536 | 0.9555 | 0.9368 | 110,750 | 0.9517 |
| R10_6L_exits_e14 | R7 with a 14-epoch schedule | 0.9533 | 0.9348 | 146,581 | 0.9479 |
| **R11_6L_ffn1536_exits_e14** | R8 with a 14-epoch schedule (stage-A winner) | **0.9557** | 0.9348 | 111,121 | 0.9503 |
| Teacher, 12 layers | | | 0.9428 | 834,264 | 0.8975 |

"Selection score" is the official score estimated on the 17,167 selection rows (runtime and GFLOPs
scaled to the public test's size); runs are chosen by it.

### Stage A2: improvements on R11 (`results/stage_a2/`)

| Run | Change from R11 | Selection score | Public F1 (held out) | GFLOPs | Public score (est.) |
|---|---|---:|---:|---:|---:|
| R11 | | 0.9557 | 0.9348 | 111,121 | 0.9503 |
| A2_R11_ema | + EMA of the weights | 0.9567 | 0.9356 | 115,143 | 0.9506 |
| A2_R11_oof_ema | + EMA + out-of-fold distillation | 0.9551 | 0.9367 | 101,284 | 0.9524 |
| A2_R11_ema_exit0_attn | + EMA + exit 0 + attention pooling | 0.9588 | 0.9361 | 63,689 | 0.9548 |
| **A2_R11_all** | + all three (winner) | **0.9604** | 0.9334 | **53,214** | 0.9541 (T4: 0.9548) |

### What the ablations show

- **Compaction + flag tokens** (R0 to R1): +0.0045 public F1 (MAT +0.012) and 14% fewer GFLOPs.
- **Early exit** halves the GFLOPs of the same network.
- **Depth and width:** six layers beat four and three once early exit is on; pruning the FFN from
  3,072 to 1,536 neurons keeps F1 and removes about a third of the cost.
- **Exit 0 + attention pooling is the largest single gain**: GFLOPs fall from 111k to 54-64k, because
  35% of the rows are decided on the embeddings alone (exit 0 reaches mean F1 0.84 on its own) and
  later exits decide more rows. In the submission, 24% of the rows exit on the embeddings and another 52% after the first layer.
- **Distilling from a teacher that fits its training set does not help** (R3 without distillation
  matches or beats R2): the teacher's logits have the sign of the hard label on 99.94% (Vulnerable) and
  99.99% (MAT) of its training rows. **Out-of-fold soft targets** (each row scored by a fold teacher
  that never saw it) disagree with the label on 3.37% of the rows (96.63% sign agreement), so they do
  carry information beyond the labels; their effect on F1 is small and not consistent across
  settings (+0.0019 public F1 alone, -0.0027 when combined with exit 0 and attention pooling).
- **EMA** adds about +0.001 on the selection split.
- **Public test vs. selection split:** every model scores lower on the public test than on the
  selection rows (about 0.005-0.008 F1), so held-out reporting matters.

### MAT error analysis (submission, public test)

MAT is highly imbalanced: 505 of 30,655 test functions (1.65%) are positive. The submission makes
445 true positives, 2 false positives and 60 false negatives (precision 0.9955, recall 0.8812). By the
flag token each function receives:

| Group | Functions | MAT-positive | Predicted positive | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| tag inside a comment (`<satd_c>`) | 444 | 443 | 444 | 443 | 1 | 0 |
| tag elsewhere (`<satd_w>`) | 7 | 1 | 2 | 1 | 1 | 0 |
| no tag | 30,204 | 61 | 1 | 1 | 0 | 60 |

All 60 false negatives are functions whose `Function` text contains no `TODO`/`FIXME`/`HACK`/`XXX`;
on the 451 functions with a tag, the model misses none. MAT recall is bounded by what `Function`
contains (61 of 505 positives, 12.1%, have no tag), not by the model.

### Early-exit policy of the submission

Per-exit thresholds `tau[k, label]` (logit space) and margins `m[k, label]`, calibrated on the public
test; "F1 alone" is the mean F1 if every row stopped at that exit.

| Exit | 0 (embeddings) | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tau` Vulnerable / MAT | -0.882 / 0.788 | 0.302 / -2.589 | 0.484 / -0.719 | 0.785 / 5.857 | 0.276 / 0.342 | 0.147 / 0.518 | 0.011 / -1.254 |
| `m` Vulnerable / MAT | 7.0 / 2.0 | 7.0 / 4.0 | 3.5 / 2.5 | 1.0 / 12.0 | 0.0 / 6.0 | 0.5 / 0.5 | (final) |
| F1 alone | 0.8448 | 0.9196 | 0.9336 | 0.9390 | 0.9408 | 0.9405 | 0.9415 |
| rows exiting here | 7,423 | 15,894 | 5,660 | 1,156 | 486 | 28 | 8 |

A row exits at exit *k* when `|z[k, label] - tau[k, label]| >= m[k, label]` for both labels. A margin of
0 makes that label's condition always true, so at exit 4 the decision depends on MAT alone; at exit 3
the MAT margin is the strictest (12.0), sending rows with an uncertain MAT logit deeper.

### Padding ablation

The reference notebook pads the whole evaluation set to one length (`padding=True` on all rows at
once), so sorting alone cannot reduce padding. The same reference loop was run under three padding
modes (`python -m nlbse27ee.padding_ablation`):

| Padding | Padded tokens | Mean F1 | GFLOPs | Runtime (s), L40S, 3 repetitions |
|---|---:|---:|---:|---:|
| `global`: whole set padded to one length (as the baseline) | 15,695,360 | 0.9392 | 156,277 | 4.64 |
| `dynamic`: each batch padded to its longest row, original order | 15,588,005 | 0.9392 | 156,277 | 4.66 |
| `sorted`: rows sorted by length, then `dynamic` (used for all reported results) | 4,910,360 | 0.9392 | 55,443 | 2.39 |

Submission, `results/stage_b/padding_ablation_A2_R11_all_full_l40s.json`; the real token count is
4,764,112. Dynamic padding alone changes nothing at batch 600: almost every batch of 600 rows in the
original order contains a 512-token row, and in all modes the model trims a batch to its remaining
rows' longest row after each exit. Sorting by length is what makes per-batch padding effective. Of
the 48x GFLOPs reduction relative to the baseline (2,666,510 to 55,443), the model accounts for 17x
under the baseline's own padding and sorted dynamic padding for the remaining 2.8x; F1 is identical.

## Differences from the baseline notebook

The measurement is the reference loop: for each batch of 600, the forward pass runs 10 times inside
`torch.profiler.profile(with_flops=True)` and `time.time()`; time and `sum(k.flops) / 1e9` are summed
over batches and divided by 10. Tokenization and the GPU transfer are outside the timed region.
Everything else below is the baseline's own configuration, not a competition requirement:

| Setting | Baseline notebook | This submission |
|---|---|---|
| Model | CodeBERT, 12 layers | UniXcoder-based student, 6 layers, FFN 1,536, 7 exits |
| Input | `Function`, RoBERTa tokenizer, max 512 | `Function` only; compaction + flag token; UniXcoder encoder-only format; max 512 |
| Padding | whole set tokenized with `padding=True` (one global length) | each batch padded to its own longest row, rows sorted by length; predictions written back in the original order |
| Batch size | 600 | 600 |
| Precision | fp32 weights, CUDA autocast | fp16 weights, CUDA autocast |
| Decision | `sigmoid(logits) > 0.5` | `sigmoid(logits) > 0.5` on logits shifted by the per-label thresholds of the exit used |

## Training and calibration details

| Item | Value |
|---|---|
| Exit heads | attention pooling: a `Linear(768, 1)` score per token (initialized to zero, i.e. starting as mean pooling), softmax over the real tokens, weighted sum, dropout 0.1 (training only), `Linear(768, 2)`. Exit 0 reads the embedding output. All heads start from the teacher's classifier. (Stage-A runs use masked mean pooling and no exit 0.) |
| Teacher head | masked mean pooling, `Linear(768, 2)` |
| Student loss | `sum_k w_k [ (1 - a) BCE(z_k, y) + a T^2 BCE(z_k / T, sigmoid(t / T)) ]`, `a = 0.5`, `T = 2`, `w_k` proportional to `k` (exit 0 smallest, last exit largest) |
| Soft targets `t` | out-of-fold: the 197,419 selection-stage training rows are split into 4 stratified folds; fold teacher *f* (`configs/T0_fold*.json`, 10 epochs max, selected on the 8% split) is trained without fold *f* and scores it; the 8% rows get the mean of the 4 fold teachers. Stage A uses the single teacher's logits |
| EMA | decay 0.999 per optimizer update; evaluation and checkpoints use the EMA weights |
| Optimizer | AdamW (betas 0.9/0.999, eps 1e-8); weight decay 0.01 on weight matrices, 0 on biases, LayerNorm and embeddings; heads use 10x the base learning rate |
| Student schedule | lr 4e-5 (heads 4e-4), effective batch 128 (16 x 8 accumulation), linear warmup over 6% of the updates of a 14-epoch schedule, then linear decay; gradient clipping 1.0; fp16 AMP; early stopping on the last exit's selection-split mean F1, patience 3 |
| Teacher schedule | lr 3e-5 (head 3e-4), effective batch 64 (16 x 4 GPUs, DDP), 10-epoch schedule; best epoch 8 |
| Final retraining | same schedules, all 214,586 rows, stopped at the selected epoch (teacher 8, student 13) |
| Batching | training: length-bucketed (chunks of 64 batches sorted by length, batches shuffled); inference: sorted dynamic padding |
| FFN pruning | per layer, keep the 1,536 of 3,072 neurons with the largest `mean |GELU(fc1(x))|` over the real tokens of 4,096 training rows, times the L2 norm of the neuron's `fc2` column |
| Thresholds `tau` | for each exit and label independently: the F1-maximizing threshold (precision-recall curve), placed midway to the next lower score |
| Margins `m` | coordinate ascent over the non-final exits; at each exit both labels' margins are searched jointly over {0, 0.5, ..., 4, 5, 6, 7, 8, 10, 12, inf}^2; up to 3 passes. Objective: the official score, with GFLOPs from a simulator of the profiler and runtime proportional to GFLOPs (calibrated on a measured run without early exit) |
| Decision | label = 1 when `z[k, label] > tau[k, label]` at the exit *k* where the row leaves |

## Reproducing

Hardware used: one machine with 4x NVIDIA L40S (48 GB). Wall time: stage A about 2.6 h (teacher
40 min with 4-GPU DDP, then 12 students on 4 GPUs), stage A2 about 2 h (fold teachers 50 min, one
per GPU, then four students), each stage B about 1.2 h.

```bash
git clone <repository-url> efficient-multitask-code-classification
cd efficient-multitask-code-classification
python -m venv --system-site-packages venv && . venv/bin/activate   # keeps an existing CUDA torch
pip install -r requirements.txt

bash scripts/run_stage_a.sh                    # teacher + R0-R11 on the 92% split, held-out public test
bash scripts/run_stage_a2.sh                   # fold teachers + improvements on R11
WINNER=A2_R11_all bash scripts/run_stage_b.sh  # retrain teacher + winner on all rows (default: best selection score)
python -m nlbse27ee.summarize runs
```

Single steps:

```bash
NCCL_P2P_DISABLE=1 python -m torch.distributed.run --nproc_per_node 4 -m nlbse27ee.train --config configs/T0_teacher12.json
python -m nlbse27ee.train --config configs/A2_R11_all.json --gpu 0
python -m nlbse27ee.padding_ablation --model runs/A2_R11_all/export --out results/padding.json
```

Notes:
- Every command resumes from `runs/<name>/last.pt` when re-run; encodings, teacher logits and
  out-of-fold logits are cached in `cache/`. Each `last.pt` holds weights, optimizer and EMA state
  (about 1 GB); all three stages need about 45 GB of disk unless finished runs' `last.pt` files are deleted.
- Use `python -m torch.distributed.run`, not `torchrun`, which may belong to another Python. If DDP
  hangs with 100% GPU load and almost no memory, the container blocks GPU peer-to-peer copies;
  `NCCL_P2P_DISABLE=1` (set by the scripts) fixes it.
- `notebooks/NLBSE27_EarlyExit_UniXcoder.ipynb` runs one student step by step (data checks, the
  equivalence test of our encoder against `transformers`' `RobertaModel`, training, calibration, export).

### Evaluating on Colab T4

`runs/<name>/export/` is self-contained (fp16 weights, tokenizer, exit policy, and a copy of
`nlbse27ee`). Zip it, open `notebooks/NLBSE27_Submission_Colab_T4.ipynb` on a T4 runtime, set
`MODEL_ZIP` (or `MODEL_SOURCE`), and run all. It prints per-label P/R/F1, their means, runtime,
GFLOPs and the submission score; `RUN_PADDING_ABLATION = True` also measures all padding modes.

## Repository layout

```
nlbse27ee/        preprocessing, model, training, calibration, evaluation, export
configs/          T0 (teacher), T0_fold0-3 (fold teachers), R0-R11, A2_R11_* (students); final/ (stage B)
scripts/          run_stage_a.sh, run_stage_a2.sh, run_stage_b.sh
notebooks/        training walkthrough; Colab T4 evaluation
results/          stage_a/, stage_a2/, stage_b/: config, history and results of every run, logs, summaries;
                  exploratory_public_selection/: the earlier round that selected on the public test
docs/             COMPLIANCE.md: each competition requirement and where it is met;
                  SOLUTION_STORY.md: the solution told as a story (Vietnamese);
                  MODEL_EXPLAINED.md: teacher, student and exits in detail (English and Vietnamese)
```

## License

MIT, see `LICENSE`.
