"""Distill the fine-tuned UniXcoder teacher into an early-exit, pruned student.

Steps (each usable from the notebook): setup -> encode_inputs -> teacher_targets
-> build_student -> fit -> calibrate -> export. ``run(cfg)`` does all of them;
``python -m nlbse27ee.train --config configs/R2.json --gpu 0`` runs one grid cell.
The teacher itself is trained with the same code (``configs/T0_teacher12.json``:
pretrained init, 12 layers, no distillation), usually on several GPUs:
``torchrun --nproc_per_node 4 -m nlbse27ee.train --config configs/T0_teacher12.json``.

Data use follows the competition guide: all 214,586 training rows train the model;
the public test split (30,655 rows) is the official tuning/selection/validation set.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from . import exits as ex
from .data import BucketBatchSampler, IndexDataset, Packed, TrainCollator, pad_rows
from .evaluate import baseline_protocol, collect_exit_logits, organizer_protocol
from .export import save_model
from .model import (build_from_teacher, ffn_importance, full_teacher, load_teacher_state,
                    pretrained_state, to_teacher_state)
from .preprocess import TAG_SEEDS, Encoder, add_flag_tokens, pack

TRAIN_ROWS, VALIDATION_ROWS = 214_586, 30_655


@dataclass
class TrainConfig:
    name: str = "R2_4L_exits"
    work_root: str = "runs"
    cache_dir: str = "cache"
    teacher_checkpoint: str = "teacher/unixcoder12_teacher.pt"
    teacher_compact_mode: str = "full"    # input format the teacher was trained with
    teacher_use_flags: bool = True
    init_from_pretrained: bool = False    # teacher training: start from the HF model, no teacher
    export_teacher_to: str | None = None  # teacher training: write a teacher checkpoint here
    tokenizer_id: str = "microsoft/unixcoder-base-nine"
    tokenizer_revision: str | None = "refs/pr/3"
    dataset_id: str = "NLBSE/nlbse27-code-comment-classification"
    gpu: int = 0
    seed: int = 42
    # architecture
    teacher_layers: list[int] = field(default_factory=lambda: [2, 5, 8, 11])
    ffn_size: int = 3072
    dropout: float = 0.1
    # input
    max_length: int = 512
    compact_mode: str = "full"
    use_flags: bool = True
    # objective and optimization
    use_exits: bool = True
    kd_alpha: float = 0.5
    kd_temperature: float = 2.0
    epochs: int = 10
    patience: int = 3
    micro_batch: int = 32
    effective_batch: int = 128
    lr: float = 4e-5
    head_lr: float = 4e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    # bookkeeping
    calib_rows: int = 4096
    teacher_batch: int = 64
    eval_batch: int = 600
    num_workers: int = 2
    log_every: int = 200
    calibrate_runtime_with_baseline: bool = True
    t4_runtime_factor: float | None = None
    quick: int = 0          # >0: subsample train to this many rows (smoke tests only)
    max_steps: int = 0      # >0: stop each epoch after this many micro-batches (smoke tests)

    @classmethod
    def from_json(cls, path, **overrides) -> "TrainConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


def rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def is_main() -> bool:
    return rank_world()[0] == 0


def barrier() -> None:
    if rank_world()[1] > 1:
        dist.barrier()


def log(*parts) -> None:
    if is_main():
        print(datetime.now().strftime("%H:%M:%S"), *parts, flush=True)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and math.isinf(value):
        return None
    raise TypeError(type(value))


def save_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=_json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------

def setup(cfg: TrainConfig) -> dict:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if "LOCAL_RANK" in os.environ and not dist.is_initialized():   # launched by torchrun
        dist.init_process_group("nccl")
        cfg.gpu = int(os.environ["LOCAL_RANK"])
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(f"cuda:{cfg.gpu}")
    torch.cuda.set_device(device)
    work_dir = Path(cfg.work_root) / cfg.name
    cache_dir = Path(cfg.cache_dir)
    if is_main():
        work_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        save_json(work_dir / "config.json", {**asdict(cfg), "gpu": None})
    barrier()

    source = load_dataset(cfg.dataset_id)
    train, validation = source["train"], source["test"]
    assert (len(train), len(validation)) == (TRAIN_ROWS, VALIDATION_ROWS), "dataset changed?"
    if cfg.quick:
        rng = np.random.default_rng(cfg.seed)
        train = train.select(np.sort(rng.choice(len(train), cfg.quick, replace=False)).tolist())
        validation = validation.select(
            np.sort(rng.choice(len(validation), max(cfg.quick // 4, 1000), replace=False)).tolist())
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_id, revision=cfg.tokenizer_revision,
                                              use_fast=True)
    base_vocab = len(tokenizer)
    flag_ids = add_flag_tokens(tokenizer) if cfg.use_flags or cfg.teacher_use_flags else []
    log("device", torch.cuda.get_device_name(device), "x", rank_world()[1], "| train", len(train),
        "| validation", len(validation), "| vocab", base_vocab, "->", len(tokenizer))
    teacher_sha = (_key(cfg.tokenizer_id, cfg.tokenizer_revision, "pretrained") if cfg.init_from_pretrained
                   else file_sha256(cfg.teacher_checkpoint))
    return {"cfg": cfg, "device": device, "work_dir": work_dir, "cache_dir": cache_dir,
            "train": train, "validation": validation, "tokenizer": tokenizer,
            "base_vocab": base_vocab, "flag_ids": flag_ids, "teacher_sha": teacher_sha,
            "fingerprints": [source["train"]._fingerprint, source["test"]._fingerprint],
            "train_labels": np.asarray(train["Label"], dtype=np.int64),
            "val_labels": np.asarray(validation["Label"], dtype=np.int64)}


def _encoded(ctx: dict, split: str, encoder: Encoder) -> Packed:
    cfg = ctx["cfg"]
    path = ctx["cache_dir"] / (f"ids_{split}_" + _key(ctx["fingerprints"], split, cfg.quick, cfg.seed,
                                                     encoder.describe(), len(ctx["tokenizer"])) + ".npz")
    if not path.exists() and is_main():
        start = time.time()
        packed = Packed(*pack(encoder(ctx[split]["Function"])))
        tmp = path.with_name(path.stem + ".tmp.npz")
        packed.save(tmp)
        tmp.replace(path)
        log(f"encoded {split} with {encoder.describe()} in {time.time() - start:.0f}s;",
            f"mean tokens {packed.lengths.mean():.1f}, at max length {(packed.lengths >= encoder.max_length).mean():.4f}")
    barrier()
    return Packed.load(path)


def encode_inputs(ctx: dict) -> None:
    cfg = ctx["cfg"]
    student = Encoder(ctx["tokenizer"], cfg.max_length, cfg.compact_mode, cfg.use_flags)
    teacher = Encoder(ctx["tokenizer"], 512, cfg.teacher_compact_mode, cfg.teacher_use_flags)
    ctx["student_encoder"] = student
    ctx["train_ids"] = _encoded(ctx, "train", student)
    ctx["val_ids"] = _encoded(ctx, "validation", student)
    ctx["teacher_train_ids"] = _encoded(ctx, "train", teacher)


def teacher_targets(ctx: dict) -> None:
    """Teacher logits on every training row, in the teacher's own input format."""
    cfg, device = ctx["cfg"], ctx["device"]
    if cfg.kd_alpha == 0:
        ctx["teacher_logits"] = None
        return
    path = ctx["cache_dir"] / ("teacher_logits_" + _key(ctx["teacher_sha"], ctx["fingerprints"],
                                                        cfg.quick, cfg.seed, cfg.teacher_compact_mode,
                                                        cfg.teacher_use_flags) + ".npy")
    if not is_main():
        barrier()
        ctx["teacher_logits"] = np.load(path)
        return
    if path.exists():
        ctx["teacher_logits"] = np.load(path)
        barrier()
        return
    state = load_teacher_state(cfg.teacher_checkpoint)
    teacher = full_teacher(state).to(device).eval()
    packed, pad = ctx["teacher_train_ids"], ctx["tokenizer"].pad_token_id
    logits = np.zeros((len(packed), 2), dtype=np.float32)
    order = np.argsort(packed.lengths, kind="stable")
    start = time.time()
    with torch.no_grad():
        for i in range(0, len(order), cfg.teacher_batch):
            rows = order[i:i + cfg.teacher_batch]
            input_ids, attention_mask = pad_rows(packed, rows, pad)
            with torch.autocast("cuda", dtype=torch.float16):
                out = teacher.all_exits(input_ids.to(device), attention_mask.to(device))[-1]
            logits[rows] = out.float().cpu().numpy()
    assert np.isfinite(logits).all()
    np.save(path, logits)
    agreement = ((logits > 0) == (ctx["train_labels"] == 1)).mean(0)
    log(f"teacher logits in {time.time() - start:.0f}s; sign agreement with labels {agreement.round(4)}")
    ctx["teacher_logits"] = logits
    del teacher
    torch.cuda.empty_cache()
    barrier()


def build_student(ctx: dict):
    cfg, device = ctx["cfg"], ctx["device"]
    state = (pretrained_state(cfg.tokenizer_id, cfg.tokenizer_revision, cfg.seed) if cfg.init_from_pretrained
             else load_teacher_state(cfg.teacher_checkpoint))
    keep = [None] * len(cfg.teacher_layers)
    full = state["backbone.encoder.layer.0.intermediate.dense.weight"].shape[0]
    if cfg.ffn_size < full:
        teacher = full_teacher(state)
        packed, pad = ctx["teacher_train_ids"], ctx["tokenizer"].pad_token_id
        rows = np.random.default_rng(cfg.seed).choice(len(packed), min(cfg.calib_rows, len(packed)),
                                                      replace=False)
        batches = (pad_rows(packed, rows[i:i + 32], pad) for i in range(0, len(rows), 32))
        importance = ffn_importance(teacher, batches, device)
        keep = [torch.topk(importance[layer], cfg.ffn_size).indices.sort().values.cpu()
                for layer in cfg.teacher_layers]
        del teacher
        torch.cuda.empty_cache()
    init = {}
    if ctx["flag_ids"]:
        seeds = ctx["tokenizer"]([" " + tag for tag in TAG_SEEDS] + list(TAG_SEEDS),
                                 add_special_tokens=False)["input_ids"]
        sources = sorted({token for ids in seeds for token in ids})
        init = {flag: sources for flag in ctx["flag_ids"]}
    model = build_from_teacher(state, cfg.teacher_layers, keep, vocab_size=len(ctx["tokenizer"]),
                               new_token_init=init, dropout=cfg.dropout)
    log("student:", model.cfg.layers, "layers from teacher", cfg.teacher_layers, "| FFN",
        model.cfg.ffn_sizes, "| parameters", sum(p.numel() for p in model.parameters()))
    return model


def _per_exit_metrics(all_logits: np.ndarray, labels: np.ndarray):
    thresholds = ex.exit_thresholds(all_logits, labels)
    per_exit = [ex.label_metrics(labels, (all_logits[k] - thresholds[k] > 0).astype(np.int64))
                for k in range(all_logits.shape[0])]
    return thresholds, per_exit


class _AllExits(nn.Module):
    """DDP needs the training computation behind ``forward``."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model.all_exits(input_ids, attention_mask)


def _comparable(cfg_dict: dict) -> dict:
    return {k: v for k, v in cfg_dict.items() if k != "gpu"}


def fit(ctx: dict, model) -> dict:
    cfg, device, work_dir = ctx["cfg"], ctx["device"], ctx["work_dir"]
    rank, world = rank_world()
    pad = ctx["tokenizer"].pad_token_id
    model.to(device)
    train_ids = ctx["train_ids"]
    sampler = BucketBatchSampler(train_ids.lengths, cfg.micro_batch, cfg.seed, rank=rank, world=world)
    loader = DataLoader(IndexDataset(len(train_ids)), batch_sampler=sampler,
                        collate_fn=TrainCollator(train_ids, ctx["train_labels"],
                                                 ctx["teacher_logits"], pad),
                        num_workers=cfg.num_workers, pin_memory=True,
                        persistent_workers=cfg.num_workers > 0)
    heads = [p for n, p in model.named_parameters() if n.startswith("exits.")]
    decay = [p for n, p in model.named_parameters()
             if not n.startswith("exits.") and p.ndim >= 2 and not n.startswith(("position", "word"))]
    no_decay = [p for n, p in model.named_parameters()
                if not n.startswith("exits.") and not (p.ndim >= 2 and not n.startswith(("position", "word")))]
    optimizer = torch.optim.AdamW([
        {"params": decay, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "lr": cfg.lr, "weight_decay": 0.0},
        {"params": heads, "lr": cfg.head_lr, "weight_decay": 0.0}])
    accumulation = max(1, cfg.effective_batch // (cfg.micro_batch * world))
    steps_per_epoch = min(len(loader), cfg.max_steps) if cfg.max_steps else len(loader)
    total_updates = math.ceil(steps_per_epoch / accumulation) * cfg.epochs
    warmup = max(1, int(total_updates * cfg.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup,
                                    max(0.0, (total_updates - step) / max(1, total_updates - warmup))))
    scaler = torch.amp.GradScaler("cuda")
    layers = model.cfg.layers
    weights = (torch.arange(1, layers + 1, dtype=torch.float32) if cfg.use_exits
               else torch.eye(layers)[-1]).to(device)
    weights = weights / weights.sum()
    best_path, last_path = work_dir / "best.pt", work_dir / "last.pt"
    state = {"epoch": 0, "best_f1": -1.0, "best_epoch": 0, "bad_epochs": 0, "history": []}
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        assert _comparable(saved["config"]) == _comparable(asdict(cfg)), "config changed; use a new run name"
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        state = saved["state"]
        log("resumed after epoch", state["epoch"])
    net = _AllExits(model)
    if world > 1:
        from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
        net = DistributedDataParallel(net, device_ids=[device.index])
        net.register_comm_hook(None, fp16_compress_hook)   # halves all-reduce traffic
    log(f"{steps_per_epoch} micro-batches/epoch/GPU, {world} GPU(s), accumulation {accumulation},",
        f"effective batch {cfg.micro_batch * world * accumulation}, {total_updates} updates")
    temperature, alpha = cfg.kd_temperature, cfg.kd_alpha
    for epoch in range(state["epoch"] + 1, cfg.epochs + 1):
        if state["bad_epochs"] >= cfg.patience:
            break
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        start, running = time.time(), []
        for step, batch in enumerate(loader):
            if step >= steps_per_epoch:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True).expand(layers, -1, -1)
            group_start = (step // accumulation) * accumulation
            group = min(accumulation, steps_per_epoch - group_start)
            boundary = (step + 1) % accumulation == 0 or step + 1 == steps_per_epoch
            sync = net.no_sync() if world > 1 and not boundary else contextlib.nullcontext()
            with sync:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = net(input_ids, attention_mask)
                supervised = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean((1, 2))
                per_exit = supervised
                if alpha > 0:
                    soft = torch.sigmoid(batch["teacher"].to(device) / temperature).expand(layers, -1, -1)
                    distilled = F.binary_cross_entropy_with_logits(
                        logits / temperature, soft, reduction="none").mean((1, 2)) * temperature ** 2
                    per_exit = (1 - alpha) * supervised + alpha * distilled
                loss = (weights * per_exit).sum()
                scaler.scale(loss / group).backward()
            if boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running.append(per_exit.detach().float().cpu().numpy())
            if step % cfg.log_every == 0:
                log(f"epoch {epoch} step {step}/{steps_per_epoch} per-exit loss",
                    np.mean(running[-cfg.log_every:], axis=0).round(4))
        if is_main():
            all_logits = collect_exit_logits(model, ctx["val_ids"], pad, device, cfg.eval_batch)
            thresholds, per_exit_metrics = _per_exit_metrics(all_logits, ctx["val_labels"])
            final = per_exit_metrics[-1]
            improved = final["mean_f1"] > state["best_f1"] + 1e-6
            row = {"epoch": epoch, "seconds": time.time() - start,
                   "train_loss_per_exit": np.mean(running, axis=0).tolist(),
                   "val_mean_f1_per_exit": [m["mean_f1"] for m in per_exit_metrics],
                   "val_f1_vulnerability": final["Vulnerability"]["f1"], "val_f1_MAT": final["MAT"]["f1"],
                   "val_mean_f1": final["mean_f1"], "improved": improved}
            state["history"].append(row)
            state["epoch"] = epoch
            if improved:
                state.update(best_f1=final["mean_f1"], best_epoch=epoch, bad_epochs=0)
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, best_path)
            else:
                state["bad_epochs"] += 1
            torch.save({"config": asdict(cfg), "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(), "state": state}, last_path)
            save_json(work_dir / "history.json", state["history"])
            log(f"epoch {epoch}: val mean F1 per exit", np.round(row["val_mean_f1_per_exit"], 4),
                f"(vuln {row['val_f1_vulnerability']:.4f}, MAT {row['val_f1_MAT']:.4f})",
                "*" if improved else "", f"{row['seconds']:.0f}s")
        if world > 1:
            shared = [state]
            dist.broadcast_object_list(shared, src=0)
            state = shared[0]
    barrier()
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    return state


def baseline_calibration(ctx: dict) -> dict | None:
    """V100 -> T4 runtime factor from the organizers' baseline cell (cached per GPU type)."""
    cfg, device = ctx["cfg"], ctx["device"]
    if cfg.t4_runtime_factor:
        return {"t4_runtime_factor": cfg.t4_runtime_factor, "source": "config"}
    gpu = torch.cuda.get_device_name(device).replace(" ", "_")
    path = ctx["cache_dir"] / f"baseline_protocol_{gpu}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if not cfg.calibrate_runtime_with_baseline or cfg.quick:
        return None
    log("running the organizers' baseline protocol once to calibrate runtime (cached)...")
    result = baseline_protocol(device, ctx["validation"]["Function"], ctx["val_labels"])
    save_json(path, result)
    log(f"baseline on {gpu}: {result['avg_runtime_s']:.1f}s, F1 {result['metrics']['mean_f1']:.4f},"
        f" T4 factor {result['t4_runtime_factor']:.3f}")
    return result


def calibrate(ctx: dict, model) -> dict:
    """Per-exit thresholds and score-optimal exit margins, then official-protocol measurement."""
    cfg, device = ctx["cfg"], ctx["device"]
    pad, labels, val_ids = ctx["tokenizer"].pad_token_id, ctx["val_labels"], ctx["val_ids"]
    model.to(device).half().eval()
    all_logits = collect_exit_logits(model, val_ids, pad, device, cfg.eval_batch)
    thresholds, per_exit = _per_exit_metrics(all_logits, labels)
    layers = model.cfg.layers
    model.set_exit_policy(thresholds, np.full((layers, 2), np.inf))
    no_exit = organizer_protocol(model, val_ids, labels, pad, device)
    flops = ex.FlopsModel(val_ids.lengths, model.cfg.ffn_sizes, model.cfg.hidden_size)
    simulated = flops.gflops(np.full(len(val_ids), layers - 1))
    calibration = baseline_calibration(ctx)
    factor = calibration["t4_runtime_factor"] if calibration else 2.0
    runtime_per_gflop = no_exit["avg_runtime_s"] * factor / no_exit["avg_gflops"]
    log(f"no-exit: F1 {no_exit['metrics']['mean_f1']:.4f}, {no_exit['avg_runtime_s']:.2f}s local, "
        f"{no_exit['avg_gflops']:.0f} GFLOPs (simulator {simulated:.0f}); T4 factor {factor:.3f}"
        + ("" if calibration else " (ASSUMED: run baseline calibration for a real factor)"))
    result = {"per_exit_val_mean_f1": [m["mean_f1"] for m in per_exit],
              "thresholds": thresholds, "no_exit": _summary(no_exit, factor),
              "flops_simulator_ratio": simulated / no_exit["avg_gflops"],
              "t4_runtime_factor": factor, "t4_factor_source": calibration and calibration.get("gpu")}
    final = no_exit
    if cfg.use_exits:
        search = ex.search_margins(all_logits, labels, thresholds, flops, runtime_per_gflop)
        result["search"] = {k: v for k, v in search.items() if k != "metrics"}
        result["search"]["mean_f1"] = search["metrics"]["mean_f1"]
        result["holdout"] = ex.holdout_check(
            all_logits, labels,
            lambda rows: ex.FlopsModel(val_ids.lengths[rows], model.cfg.ffn_sizes, model.cfg.hidden_size),
            runtime_per_gflop)
        model.set_exit_policy(thresholds, search["margins"])
        final = organizer_protocol(model, val_ids, labels, pad, device)
        log(f"early exit: margins {np.round(search['margins'], 2).tolist()}, exits {final['exit_histogram']}")
    result["final"] = _summary(final, factor)
    log(f"final: F1 {final['metrics']['mean_f1']:.4f}, {final['avg_gflops']:.0f} GFLOPs, "
        f"local {final['avg_runtime_s']:.2f}s -> est. T4 {final['avg_runtime_s'] * factor:.1f}s, "
        f"est. score {result['final']['estimated_t4_score']:.4f}")
    return result


def _summary(measured: dict, factor: float) -> dict:
    t4 = measured["avg_runtime_s"] * factor
    return {"metrics": measured["metrics"], "local_runtime_s": measured["avg_runtime_s"],
            "gflops": measured["avg_gflops"], "estimated_t4_runtime_s": t4,
            "estimated_t4_score": ex.official_score(measured["metrics"]["mean_f1"], t4,
                                                    measured["avg_gflops"]),
            "exit_histogram": measured["exit_histogram"], "gpu": measured["gpu"]}


def export(ctx: dict, model, calibration: dict, training: dict) -> Path:
    cfg = ctx["cfg"]
    metadata = {"run": cfg.name, "teacher_sha256": ctx["teacher_sha"], "best_epoch": training["best_epoch"],
                "validation": calibration["final"]["metrics"],
                "estimated_t4_score": calibration["final"]["estimated_t4_score"],
                "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out = save_model(ctx["work_dir"] / "export", model, ctx["tokenizer"], ctx["student_encoder"],
                     json.loads(json.dumps(metadata, default=_json_default)))
    save_json(ctx["work_dir"] / "results.json",
              {"config": asdict(cfg), "training": training, "calibration": calibration})
    log("exported", out)
    return out


def save_teacher(model, path) -> Path:
    """Write the trained model as a teacher checkpoint (HF RobertaModel naming)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model_state_dict": to_teacher_state(model), "model_id": "microsoft/unixcoder-base-nine"}, tmp)
    tmp.replace(path)
    log("teacher checkpoint written:", path, "sha256", file_sha256(path)[:16])
    return path


def run(cfg: TrainConfig) -> dict | None:
    ctx = setup(cfg)
    encode_inputs(ctx)
    teacher_targets(ctx)
    model = build_student(ctx)
    training = fit(ctx, model)
    main_process = is_main()
    if dist.is_initialized():      # the rest is single-GPU work on rank 0
        dist.destroy_process_group()
    if not main_process:
        return None
    if cfg.export_teacher_to:
        save_teacher(model, cfg.export_teacher_to)
    calibration = calibrate(ctx, model)
    export(ctx, model, calibration, training)
    return calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--quick", type=int)
    parser.add_argument("--max-steps", type=int, dest="max_steps")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--teacher-checkpoint", dest="teacher_checkpoint")
    parser.add_argument("--baseline-only", action="store_true",
                        help="only run the organizers' baseline protocol (runtime calibration)")
    args = parser.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k not in ("config", "baseline_only")}
    cfg = TrainConfig.from_json(args.config, **overrides)
    if args.baseline_only:
        ctx = setup(cfg)
        print(json.dumps(baseline_calibration(ctx), indent=2, default=_json_default))
        return
    run(cfg)


if __name__ == "__main__":
    main()
