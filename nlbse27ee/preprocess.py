"""Input preprocessing shared by training, calibration, and submission.

Everything here is a deterministic function of the ``Function`` text, which is the
only input the competition rules allow.
"""

from __future__ import annotations

import re

import numpy as np

MODE_TOKEN = "<encoder-only>"
FLAG_COMMENT = "<satd_c>"  # SATD tag inside a comment: ~99.7% precise on train
FLAG_WORD = "<satd_w>"     # SATD tag elsewhere (identifiers, strings, macros)
FLAG_TOKENS = (FLAG_COMMENT, FLAG_WORD)
TAG_SEEDS = ("TODO", "FIXME", "HACK", "XXX")

_TAG = re.compile(r"\b(?:TODO|FIXME|HACK|XXX)\b", re.IGNORECASE)
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/|#[^\n]*", re.DOTALL)
_HSPACE = re.compile(r"[ \t\f\v]+")
_NEWLINES = re.compile(r" ?\n[ \n]*")
_SPACE = re.compile(r"\s+")


def satd_flag(text: str) -> str | None:
    """Which SATD flag token to prepend, computed on the full, untruncated text."""
    if not _TAG.search(text):
        return None
    if any(_TAG.search(comment) for comment in _COMMENT.findall(text)):
        return FLAG_COMMENT
    return FLAG_WORD


def compact(text: str, mode: str) -> str:
    """Remove formatting whitespace; 'full' saves ~16% tokens, 'lines' ~8.5%."""
    if mode == "none":
        return text
    if mode == "lines":
        return _NEWLINES.sub("\n", _HSPACE.sub(" ", text)).strip()
    if mode == "full":
        return _SPACE.sub(" ", text).strip()
    raise ValueError(f"Unknown compact mode: {mode}")


def add_flag_tokens(tokenizer) -> list[int]:
    """Register the flag tokens (idempotent) and return their ids."""
    missing = [token for token in FLAG_TOKENS if token not in tokenizer.get_vocab()]
    if missing:
        tokenizer.add_tokens(missing, special_tokens=True)
    return [tokenizer.convert_tokens_to_ids(token) for token in FLAG_TOKENS]


class Encoder:
    """Text -> UniXcoder encoder-only ids: ``<s> <encoder-only> </s> [flag] code </s>``."""

    def __init__(self, tokenizer, max_length: int = 512, compact_mode: str = "full",
                 use_flags: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.compact_mode = compact_mode
        self.use_flags = use_flags
        self.prefix = [tokenizer.cls_token_id, tokenizer.convert_tokens_to_ids(MODE_TOKEN),
                       tokenizer.sep_token_id]
        assert self.prefix[1] != tokenizer.unk_token_id, "tokenizer lacks <encoder-only>"
        self.flag_ids = {}
        if use_flags:
            ids = [tokenizer.convert_tokens_to_ids(token) for token in FLAG_TOKENS]
            assert tokenizer.unk_token_id not in ids, "call add_flag_tokens(tokenizer) first"
            self.flag_ids = dict(zip(FLAG_TOKENS, ids))

    def describe(self) -> dict:
        return {"max_length": self.max_length, "compact_mode": self.compact_mode,
                "use_flags": self.use_flags}

    def __call__(self, texts, batch_size: int = 4096) -> list[list[int]]:
        out: list[list[int]] = []
        texts = [str(text or "") for text in texts]
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            flags = [satd_flag(text) if self.use_flags else None for text in chunk]
            bodies = self.tokenizer([compact(text, self.compact_mode) for text in chunk],
                                    add_special_tokens=False)["input_ids"]
            for flag, body in zip(flags, bodies):
                head = self.prefix + ([self.flag_ids[flag]] if flag else [])
                budget = self.max_length - len(head) - 1
                out.append(head + body[:budget] + [self.tokenizer.sep_token_id])
        return out


def pack(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Ragged ids -> (flat int32 array, int64 offsets) for compact storage."""
    lengths = np.fromiter((len(row) for row in rows), dtype=np.int64, count=len(rows))
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    flat = np.fromiter((token for row in rows for token in row), dtype=np.int32,
                       count=int(offsets[-1]))
    return flat, offsets
