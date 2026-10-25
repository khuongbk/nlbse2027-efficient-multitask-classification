"""Self-contained model folder: fp16 weights, tokenizer, exit policy, and this package's code."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .model import EarlyExitUniXcoder, ModelConfig
from .preprocess import Encoder

CONFIG_NAME = "nlbse27ee_config.json"
WEIGHTS_NAME = "model.safetensors"


def save_model(out_dir, model: EarlyExitUniXcoder, tokenizer, encoder: Encoder,
               metadata: dict | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().half().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(state, str(out_dir / WEIGHTS_NAME))
    tokenizer.save_pretrained(str(out_dir / "tokenizer"))
    config = {"model": model.cfg.to_dict(), "preprocess": encoder.describe(),
              "exit_thresholds": model.exit_thresholds.cpu().tolist(),
              "exit_margins": [[None if v == float("inf") else v for v in row]
                               for row in model.exit_margins.cpu().tolist()],
              "metadata": metadata or {}}
    (out_dir / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    package = Path(__file__).resolve().parent
    shutil.copytree(package, out_dir / "nlbse27ee", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    return out_dir


def load_model(source, device="cuda", dtype=torch.float16):
    """Load from a local folder or a Hugging Face repo id; returns (model, encoder)."""
    from transformers import AutoTokenizer

    path = Path(str(source))
    if not path.exists():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(str(source)))
    config = json.loads((path / CONFIG_NAME).read_text(encoding="utf-8"))
    model = EarlyExitUniXcoder(ModelConfig(**config["model"]))
    model.load_state_dict({k: v.float() for k, v in load_file(str(path / WEIGHTS_NAME)).items()},
                          strict=True)
    margins = [[float("inf") if v is None else v for v in row] for row in config["exit_margins"]]
    model.set_exit_policy(config["exit_thresholds"], margins)
    model.to(device=device, dtype=dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(path / "tokenizer"), use_fast=True)
    return model, Encoder(tokenizer, **config["preprocess"])
