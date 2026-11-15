# Five-seed A2_R11_all student sweep

This sweep runs the latest `A2_R11_all` student recipe five times with seeds
42–46. All runs use the same 92/8 stratified split (split seed 42), the same
fine-tuned 12-layer teacher, and the existing four fold-teacher OOF logits.
The student seed changes shuffle/dropout and the activation-based FFN neuron
selection sample. Each run independently selects its checkpoint, thresholds,
and exit margins on the same inner validation split. Public-test results are
aggregated only after each run; no seed is selected on the public test.

The teacher and OOF fold teachers are fixed across these runs. This measures
student-seed variability, not variability from retraining the teacher ensemble.
V100 runtimes from concurrent seed jobs are preliminary; if runtime contributes
to the reported average, remeasure each exported model sequentially on an idle
GPU or run each Colab T4 bundle.

Seed 42 here is a new run, not the stage-A2 model `A2_R11_all`: GPU training is not bitwise
reproducible across hardware (L40S there, V100 here), and the margin search used an assumed
runtime factor of 2.0 instead of the measured 5.32, which weights runtime less and lets more rows
reach deeper exits (about 58k instead of 53k GFLOPs). Report F1 and GFLOPs from this sweep as
mean ± SD; the estimated score depends on the assumed factor.

```bash
GPUS="0 1 2" bash scripts/run_seed_sweep.sh          # configs, five runs, summary
python scripts/remeasure_seed_sweep.py --gpu 0        # runtimes on an idle GPU, then:
python scripts/summarize_seed_sweep.py
```

Checkpoints and per-run results go to `runs/seed_sweep/A2_R11_all_seed*/`, logs to
`logs/seed_sweep/`.

Re-running requires the existing `teacher/unixcoder12_teacher.pt` and
`cache/oof/fold0.npz`–`fold3.npz` artifacts, plus the environment in
`requirements.txt`. These large model/cache files are not committed here.
Set `PYTHON` to the training environment's Python executable if needed.
