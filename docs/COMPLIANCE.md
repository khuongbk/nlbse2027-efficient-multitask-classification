# Competition requirements and where they are met

Requirements from the NLBSE'27 tool-competition page and the organizers' reference notebook
(`CodeBERT_baseline.ipynb`). Status as of the last update of this file.

| # | Requirement | Status | Where |
|---|---|---|---|
| 1 | One model predicting both labels jointly, in `[Vulnerable, MAT]` order | Done | `nlbse27ee/model.py` (`EarlyExitUniXcoder`): one network, output `[batch, 2]`. The per-layer exit heads are internal to that network; there are no per-label models. Labels are read from the dataset's `Label` column in its order |
| 2 | Only `Function` as input; no external data; pretrained models allowed if fine-tuned on the provided training set | Done | `nlbse27ee/preprocess.py`: the encoder reads `Function` only; whitespace compaction and the SATD flag token are functions of that text. The teacher starts from `microsoft/unixcoder-base-nine` and is fine-tuned on the 214,586 training rows (`configs/T0_teacher12.json`); students are built from the teacher and fine-tuned on the same rows. `Projectname`/`Filepath` are never read |
| 3 | Dataset `NLBSE/nlbse27-code-comment-classification`; evaluation on `ds["test"]` (30,655 public rows) | Done | `nlbse27ee/train.py` asserts 214,586 / 30,655 rows; the Colab notebook asserts `len(ds["test"]) == 30655` |
| 4 | Every row predicted, in the original order; no sampling, rebalancing, under- or oversampling of the evaluation set | Done | Predictions start at `-1`, are written back by row index, and the notebook asserts all 30,655 rows are filled. `quick` subsampling exists only for smoke tests and is `0` in every reported run (`results/*/runs/*/config.json`) |
| 5 | Evaluation on Google Colab T4 | Met | `notebooks/NLBSE27_Submission_Colab_T4.ipynb` on a Tesla T4: the submission scores mean F1 0.9392, 55,443 GFLOPs, 8.16 s, **score 0.9582** (`results/stage_b/colab_t4_A2_R11_all_full.json`). The exploratory model R8 was measured on a Tesla T4 (mean F1 0.9406, 109,169 GFLOPs, 12.61 s, score 0.9545; `results/exploratory_public_selection/colab_t4_R8.json`). Other runtimes are L40S-based estimates (see below) |
| 6 | Measurement as in the notebook: each batch forwarded 10 times, time and FLOPs summed over batches, then divided by 10 (not 10 independent pipeline runs) | Done | Colab notebook section 3 and `nlbse27ee/evaluate.py` (`organizer_protocol`) use the reference loop verbatim |
| 7 | FLOPs via `torch.profiler.profile(with_flops=True)`, `sum(k.flops) / 1e9` | Done | same places |
| 8 | Scored GFLOPs = total for one pass over the evaluation set, not per sample | Done | all reported GFLOPs are totals (e.g. 53,214 for A2_R11_all, 2,666,510 for the baseline) |
| 9 | Timed region as in the notebook: tokenization and the GPU transfer outside, the profiled forward loop inside | Done | Colab notebook section 3: encoding happens in section 2; `pad_rows` and `.to(device)` precede `begin = time.time()` |
| 10 | `model.eval()`, `torch.no_grad()`, CUDA autocast at inference | Done | `load_model` calls `.eval()`; the loop uses `torch.no_grad()` and `torch.amp.autocast("cuda")`. Difference: our weights are stored in fp16 |
| 11 | State the differences from the baseline (batch 600, max length 512, `sigmoid`, `probs > 0.5` are baseline settings, not requirements) | Done | table at the top of the Colab notebook and in the README |
| 12 | Per-label precision, recall, F1 and their arithmetic means | Done | Colab notebook section 4 (reference code); `results/*/runs/*/results.json` |
| 13 | Score with runtime cap 1,000 s, compute cap 3,000,000 GFLOPs, weights 0.60 / 0.20 / 0.20 | Done | Colab notebook section 4 (reference `score`), `nlbse27ee/exits.py` (`official_score`). Reproduces the baseline's 0.7299 |
| 14 | Public training/tuning/evaluation code with enough documentation to reproduce; the paper links the code | Code done; **link and weights pending** | this repository (README "Reproducing"). Still to do: push, host the exported model (e.g. Hugging Face), cite both URLs in the paper |

## Model selection without the public test

Checkpoints, thresholds, exit margins and the winning configuration are selected on a stratified 8%
split of the training set (17,167 rows, seed 42); research models train on the other 92%. Each
research model is measured on the public test once, after all of its choices were made, so those
numbers are held-out estimates. The submission retrains the selected recipe on all training rows
with the epoch count fixed in advance and calibrates thresholds and exit margins on the public test,
the split the guide designates for "model tuning, selection, and validation reporting". The public
test labels never produce gradients. An earlier round that selected models on the public test is
kept, labelled as exploratory, in `results/exploratory_public_selection/`.

## Padding: what is and is not claimed

The reference notebook tokenizes the whole evaluation set at once with `padding=True`, so every
batch has the same (global) length. Under that scheme, sorting rows by length changes nothing. Our
evaluation pads each batch to its own longest row (dynamic padding) after sorting by length; this is
a deliberate change to the evaluation-side preprocessing and is stated as such. Its effect is
measured, not assumed: `python -m nlbse27ee.padding_ablation` (and section 5 of the Colab notebook)
runs the same reference loop with `global`, `dynamic` and `sorted` padding. Results are in
`results/` and the README.

## Runtime numbers other than the T4 run

GFLOPs and F1 do not depend on the GPU. Runtime does: training-time estimates multiply the runtime
measured on an NVIDIA L40S by 5.32, the ratio between the baseline's official T4 runtime (192.1 s)
and the same baseline cell run on the L40S (36.1 s; it reproduced the official 2,666,510 GFLOPs and
F1 0.9102 exactly). For the exploratory model R8 the estimate was 15.1 s / score 0.9540 and the T4
measured 12.61 s / 0.9545; for the submission the estimate was 11.6 s / 0.9575 and the T4 measured
8.16 s / 0.9582, so these estimates tend to be conservative.

## Open items

| Item | Who | Notes |
|---|---|---|
| Push this repository (public) | team | the paper must link it |
| Share the exported final model (about 130 MB) and the teacher checkpoints (501 MB each) | team | a Google Drive link with "anyone with the link" access, the Hugging Face Hub, or a GitHub Release (git rejects files over 100 MB) |
| Cite repository and model URLs in the paper | team | acceptance criterion |
| Optional: padding ablation on the T4 (`RUN_PADDING_ABLATION = True`, ~15-20 min) | team | replaces the RTX 4060 runtimes in the README's padding table with T4 runtimes |
| Paper (2-4 pages, ICSE/NLBSE format): architecture, preprocessing, tuning on the training set, validation results | team | README "Method", "Results", "Differences" and the ablation tables are the source material |
