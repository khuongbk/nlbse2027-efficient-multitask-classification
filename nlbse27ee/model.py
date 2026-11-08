"""Early-exit, depth/width-pruned UniXcoder.

A self-contained RoBERTa encoder (numerically equivalent to ``transformers``'
RobertaModel, see ``verify_against_transformers``) with one classification head
per layer. At inference, rows whose head is confident leave the batch, so later
layers process fewer rows. Inputs must be right-padded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    ffn_sizes: list[int]
    hidden_size: int = 768
    heads: int = 12
    max_positions: int = 1026
    pad_token_id: int = 1
    layer_norm_eps: float = 1e-5
    dropout: float = 0.1
    teacher_layers: list[int] = field(default_factory=list)
    exit0: bool = False     # an extra exit on the embeddings, before any transformer layer
    head: str = "mean"      # exit pooling: "mean" (masked mean) or "attn" (learned attention pooling)

    @property
    def layers(self) -> int:
        return len(self.ffn_sizes)

    @property
    def exits(self) -> int:
        return self.layers + int(self.exit0)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EEOutput:
    """``logits`` are shifted so that ``logits > 0`` is the decision at the exit used."""
    logits: torch.Tensor
    exit_layer: torch.Tensor


class Layer(nn.Module):
    def __init__(self, cfg: ModelConfig, ffn_size: int):
        super().__init__()
        hidden = cfg.hidden_size
        self.heads = cfg.heads
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out = nn.Linear(hidden, hidden)
        self.ln1 = nn.LayerNorm(hidden, eps=cfg.layer_norm_eps)
        self.fc1 = nn.Linear(hidden, ffn_size)
        self.fc2 = nn.Linear(ffn_size, hidden)
        self.ln2 = nn.LayerNorm(hidden, eps=cfg.layer_norm_eps)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, return_ffn: bool = False):
        batch, length, hidden = x.shape
        q, k, v = (self.qkv(x).view(batch, length, 3, self.heads, hidden // self.heads)
                   .permute(2, 0, 3, 1, 4).unbind(0))
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
        x = self.ln1(x + self.drop(self.out(attended.transpose(1, 2).reshape(batch, length, hidden))))
        activation = F.gelu(self.fc1(x))
        x = self.ln2(x + self.drop(self.fc2(activation)))
        return (x, activation) if return_ffn else x


class AttentionPoolHead(nn.Module):
    """Softmax-weighted token pooling, then a linear classifier.

    The score layer starts at zero, so the head starts as exact mean pooling.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.score = nn.Linear(hidden, 1)
        self.out = nn.Linear(hidden, 2)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, drop: nn.Module) -> torch.Tensor:
        scores = self.score(x).squeeze(-1).float().masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1).to(x.dtype)
        pooled = (x * weights).sum(1, dtype=torch.float32)
        return self.out(drop(pooled))


class EarlyExitUniXcoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.hidden_size
        self.word = nn.Embedding(cfg.vocab_size, hidden, padding_idx=cfg.pad_token_id)
        self.position = nn.Embedding(cfg.max_positions, hidden)
        self.type_bias = nn.Parameter(torch.zeros(hidden))  # token_type 0, folded in
        self.emb_ln = nn.LayerNorm(hidden, eps=cfg.layer_norm_eps)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(Layer(cfg, size) for size in cfg.ffn_sizes)
        make_head = (lambda: AttentionPoolHead(hidden)) if cfg.head == "attn" else (lambda: nn.Linear(hidden, 2))
        self.exits = nn.ModuleList(make_head() for _ in range(cfg.exits))
        self.set_exit_policy(torch.zeros(cfg.exits, 2), torch.full((cfg.exits, 2), float("inf")))

    def set_exit_policy(self, thresholds, margins) -> None:
        """Logit thresholds and confidence margins per exit, kept in float32."""
        self._policy = (torch.as_tensor(thresholds, dtype=torch.float32).cpu().clone(),
                        torch.as_tensor(margins, dtype=torch.float32).cpu().clone())
        device = self.type_bias.device
        self.register_buffer("exit_thresholds", self._policy[0].to(device), persistent=False)
        self.register_buffer("exit_margins", self._policy[1].to(device), persistent=False)

    def _apply(self, fn, recurse=True):
        # .half()/.to() must not round the decision thresholds; restore them in float32.
        super()._apply(fn, recurse)
        if hasattr(self, "_policy"):
            self.set_exit_policy(*self._policy)
        return self

    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pad = self.cfg.pad_token_id
        positions = torch.arange(input_ids.shape[1], device=input_ids.device) + pad + 1
        positions = torch.where(attention_mask.bool(), positions, pad)
        x = self.word(input_ids) + self.position(positions) + self.type_bias
        return self.drop(self.emb_ln(x))

    @staticmethod
    def pool(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # The masked product is exact in x's dtype; accumulate in float32 without a float32 copy of x.
        mask = attention_mask.unsqueeze(-1)
        summed = (x * mask.to(x.dtype)).sum(1, dtype=torch.float32)
        return summed / mask.sum(1).float().clamp_min(1.0)

    def exit_logits(self, k: int, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        head = self.exits[k]
        if isinstance(head, nn.Linear):
            return head(self.drop(self.pool(x, attention_mask))).float()
        return head(x, attention_mask, self.drop).float()

    def all_exits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Raw logits of every exit, shape [exits, batch, 2] (training/calibration)."""
        x = self.embed(input_ids, attention_mask)
        attn_mask = attention_mask.bool()[:, None, None, :]
        outputs = [self.exit_logits(0, x, attention_mask)] if self.cfg.exit0 else []
        for layer in self.layers:
            x = layer(x, attn_mask)
            outputs.append(self.exit_logits(len(outputs), x, attention_mask))
        return torch.stack(outputs)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> EEOutput:
        """Early-exit inference; ``sigmoid(logits) > 0.5`` gives the labels."""
        batch = input_ids.shape[0]
        last = self.cfg.exits - 1
        offset = int(self.cfg.exit0)   # exit k follows k - offset + 1 transformer layers
        logits = torch.zeros(batch, 2, dtype=torch.float32, device=input_ids.device)
        exit_layer = torch.full((batch,), last, dtype=torch.long, device=input_ids.device)
        index = torch.arange(batch, device=input_ids.device)
        mask = attention_mask
        x = self.embed(input_ids, mask)
        for k in range(self.cfg.exits):
            if k >= offset:
                x = self.layers[k - offset](x, mask.bool()[:, None, None, :])
            z = self.exit_logits(k, x, mask) - self.exit_thresholds[k]
            if k == last:
                logits[index] = z
                break
            done = (z.abs() >= self.exit_margins[k]).all(-1)
            if bool(done.any()):
                logits[index[done]] = z[done]
                exit_layer[index[done]] = k
                keep = ~done
                if not bool(keep.any()):
                    break
                x, mask, index = x[keep], mask[keep], index[keep]
                width = int(mask.sum(1).max())
                x, mask = x[:, :width], mask[:, :width]
        return EEOutput(logits=logits, exit_layer=exit_layer)


# ---------------------------------------------------------------------------
# Building from the fine-tuned teacher checkpoint (HF RobertaModel naming)
# ---------------------------------------------------------------------------

def load_teacher_state(path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    if not any(key.startswith("backbone.") for key in state):
        raise KeyError("Expected 'backbone.*' keys (UniXcoder teacher checkpoint)")
    state = {key: value.float() for key, value in state.items() if torch.is_tensor(value)}
    factor = "backbone.embeddings.word_embeddings."
    if factor + "token.weight" in state:
        # A rank-factorized student (distillation_unixcoder_avatar): rebuild the full table.
        state[factor + "weight"] = state.pop(factor + "token.weight") @ state.pop(factor + "projection.weight").T
    return state


def pretrained_state(model_id: str, revision: str | None, seed: int = 0) -> dict[str, torch.Tensor]:
    """Pretrained UniXcoder in teacher-checkpoint naming, with a fresh classifier (teacher training)."""
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(model_id, revision=revision, add_pooling_layer=False)
    state = {"backbone." + key: value.float() for key, value in backbone.state_dict().items()}
    hidden = backbone.config.hidden_size
    generator = torch.Generator().manual_seed(seed)
    state["classifier.weight"] = torch.randn(2, hidden, generator=generator) * 0.02
    state["classifier.bias"] = torch.zeros(2)
    return state


def to_teacher_state(model: EarlyExitUniXcoder, type_vocab_size: int = 10) -> dict[str, torch.Tensor]:
    """Inverse of ``build_from_teacher``: our weights in HF RobertaModel naming (last exit = classifier)."""
    s = {key: value.detach().float().cpu() for key, value in model.state_dict().items()}
    hidden = model.cfg.hidden_size
    token_type = torch.zeros(type_vocab_size, hidden)
    token_type[0] = s["type_bias"]
    out = {"backbone.embeddings.word_embeddings.weight": s["word.weight"],
           "backbone.embeddings.position_embeddings.weight": s["position.weight"],
           "backbone.embeddings.token_type_embeddings.weight": token_type,
           "backbone.embeddings.LayerNorm.weight": s["emb_ln.weight"],
           "backbone.embeddings.LayerNorm.bias": s["emb_ln.bias"]}
    for i in range(model.cfg.layers):
        p, q = f"backbone.encoder.layer.{i}.", f"layers.{i}."
        for name, weight, bias in zip(("query", "key", "value"), s[q + "qkv.weight"].chunk(3),
                                      s[q + "qkv.bias"].chunk(3)):
            out[p + f"attention.self.{name}.weight"] = weight.clone()
            out[p + f"attention.self.{name}.bias"] = bias.clone()
        for target, source in (("attention.output.dense", "out"), ("attention.output.LayerNorm", "ln1"),
                               ("intermediate.dense", "fc1"), ("output.dense", "fc2"),
                               ("output.LayerNorm", "ln2")):
            out[p + target + ".weight"] = s[q + source + ".weight"]
            out[p + target + ".bias"] = s[q + source + ".bias"]
    assert model.cfg.head == "mean", "teacher checkpoints use a plain linear head"
    last = model.cfg.exits - 1
    out["classifier.weight"] = s[f"exits.{last}.weight"]
    out["classifier.bias"] = s[f"exits.{last}.bias"]
    return out


def teacher_layer_count(state: dict) -> int:
    prefix = "backbone.encoder.layer."
    return 1 + max(int(key[len(prefix):].split(".")[0]) for key in state if key.startswith(prefix))


def build_from_teacher(state: dict, teacher_layers: list[int], ffn_keep: list[torch.Tensor | None],
                       vocab_size: int | None = None,
                       new_token_init: dict[int, list[int]] | None = None,
                       dropout: float = 0.1, exit0: bool = False, head: str = "mean") -> EarlyExitUniXcoder:
    """Copy selected teacher layers (optionally FFN-pruned) into a new model.

    ``ffn_keep[i]`` holds the FFN neuron indices kept in student layer ``i``
    (None keeps all). Every exit head starts from the teacher's classifier.
    Extra vocabulary rows are initialized as the mean of ``new_token_init`` rows.
    """
    word = state["backbone.embeddings.word_embeddings.weight"]
    hidden = word.shape[1]
    vocab_size = vocab_size or word.shape[0]
    full_ffn = state["backbone.encoder.layer.0.intermediate.dense.weight"].shape[0]
    ffn_sizes = [full_ffn if keep is None else int(len(keep)) for keep in ffn_keep]
    cfg = ModelConfig(vocab_size=vocab_size, ffn_sizes=ffn_sizes, hidden_size=hidden,
                      heads=hidden // 64,
                      max_positions=state["backbone.embeddings.position_embeddings.weight"].shape[0],
                      dropout=dropout, teacher_layers=list(teacher_layers), exit0=exit0, head=head)
    model = EarlyExitUniXcoder(cfg)
    target = {}
    table = torch.zeros(vocab_size, hidden)
    table[:word.shape[0]] = word
    for new_id, sources in (new_token_init or {}).items():
        if new_id >= word.shape[0]:   # rows the checkpoint already has keep their trained values
            table[new_id] = word[sources].mean(0)
    target["word.weight"] = table
    target["position.weight"] = state["backbone.embeddings.position_embeddings.weight"]
    target["type_bias"] = state["backbone.embeddings.token_type_embeddings.weight"][0]
    target["emb_ln.weight"] = state["backbone.embeddings.LayerNorm.weight"]
    target["emb_ln.bias"] = state["backbone.embeddings.LayerNorm.bias"]
    for i, (source, keep) in enumerate(zip(teacher_layers, ffn_keep)):
        p = f"backbone.encoder.layer.{source}."
        attention = [state[p + f"attention.self.{name}.weight"] for name in ("query", "key", "value")]
        target[f"layers.{i}.qkv.weight"] = torch.cat(attention)
        target[f"layers.{i}.qkv.bias"] = torch.cat(
            [state[p + f"attention.self.{name}.bias"] for name in ("query", "key", "value")])
        target[f"layers.{i}.out.weight"] = state[p + "attention.output.dense.weight"]
        target[f"layers.{i}.out.bias"] = state[p + "attention.output.dense.bias"]
        target[f"layers.{i}.ln1.weight"] = state[p + "attention.output.LayerNorm.weight"]
        target[f"layers.{i}.ln1.bias"] = state[p + "attention.output.LayerNorm.bias"]
        fc1_w, fc1_b = state[p + "intermediate.dense.weight"], state[p + "intermediate.dense.bias"]
        fc2_w = state[p + "output.dense.weight"]
        if keep is not None:
            keep = torch.as_tensor(keep, dtype=torch.long)
            fc1_w, fc1_b, fc2_w = fc1_w[keep], fc1_b[keep], fc2_w[:, keep]
        target[f"layers.{i}.fc1.weight"], target[f"layers.{i}.fc1.bias"] = fc1_w, fc1_b
        target[f"layers.{i}.fc2.weight"] = fc2_w
        target[f"layers.{i}.fc2.bias"] = state[p + "output.dense.bias"]
        target[f"layers.{i}.ln2.weight"] = state[p + "output.LayerNorm.weight"]
        target[f"layers.{i}.ln2.bias"] = state[p + "output.LayerNorm.bias"]
    for k in range(cfg.exits):   # every exit head starts from the teacher's classifier
        prefix = f"exits.{k}." + ("out." if head == "attn" else "")
        target[prefix + "weight"] = state["classifier.weight"]
        target[prefix + "bias"] = state["classifier.bias"]
        if head == "attn":
            target[f"exits.{k}.score.weight"] = torch.zeros(1, hidden)
            target[f"exits.{k}.score.bias"] = torch.zeros(1)
    model.load_state_dict({key: value.clone() for key, value in target.items()}, strict=True)
    return model


def full_teacher(state: dict) -> EarlyExitUniXcoder:
    """The whole teacher in this implementation; its last exit is the teacher head."""
    layers = teacher_layer_count(state)
    return build_from_teacher(state, list(range(layers)), [None] * layers)


@torch.no_grad()
def ffn_importance(teacher: EarlyExitUniXcoder, batches, device) -> list[torch.Tensor]:
    """Per-neuron importance: mean |GELU activation| over real tokens x ||fc2 column||."""
    teacher.eval().to(device)
    sums = [torch.zeros(layer.fc1.out_features, device=device) for layer in teacher.layers]
    tokens = 0.0
    for input_ids, attention_mask in batches:
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        mask = attention_mask.unsqueeze(-1).float()
        tokens += float(mask.sum())
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            x = teacher.embed(input_ids, attention_mask)
            attn_mask = attention_mask.bool()[:, None, None, :]
            for i, layer in enumerate(teacher.layers):
                x, activation = layer(x, attn_mask, return_ffn=True)
                sums[i] += (activation.float().abs() * mask).sum((0, 1))
    return [(total / tokens) * layer.fc2.weight.float().norm(dim=0)
            for total, layer in zip(sums, teacher.layers)]


def verify_against_transformers(state: dict, model_id: str, revision: str | None,
                                device, atol: float = 2e-3) -> float:
    """Max |logit difference| between this implementation and HF RobertaModel (fp32)."""
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_id, revision=revision)
    config.num_hidden_layers = teacher_layer_count(state)
    config.vocab_size = state["backbone.embeddings.word_embeddings.weight"].shape[0]
    reference = AutoModel.from_config(config, add_pooling_layer=False)
    reference.load_state_dict({key[len("backbone."):]: value for key, value in state.items()
                               if key.startswith("backbone.")}, strict=False)
    reference.to(device).eval()
    ours = full_teacher(state).to(device).eval()
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(5, 50000, (4, 96), generator=generator)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, 60:] = 0
    attention_mask[3, 17:] = 0
    input_ids[attention_mask == 0] = ours.cfg.pad_token_id
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    with torch.no_grad():
        hidden = reference(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        expected = F.linear(ours.pool(hidden, attention_mask),
                            state["classifier.weight"].to(device), state["classifier.bias"].to(device))
        actual = ours.all_exits(input_ids, attention_mask)[-1]
    difference = float((expected - actual).abs().max())
    if difference > atol:
        raise AssertionError(f"Implementation mismatch vs transformers: {difference:.2e}")
    return difference
