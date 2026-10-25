"""Packed token storage, length-bucketed batching, and right-padded collation."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Sampler


class Packed:
    """Ragged token ids stored as one flat int32 array plus offsets."""

    def __init__(self, flat: np.ndarray, offsets: np.ndarray):
        self.flat, self.offsets = flat, offsets
        self.lengths = np.diff(offsets)

    def __len__(self) -> int:
        return len(self.lengths)

    def row(self, index: int) -> np.ndarray:
        return self.flat[self.offsets[index]:self.offsets[index + 1]]

    def save(self, path) -> None:
        np.savez(path, flat=self.flat, offsets=self.offsets)

    @classmethod
    def load(cls, path) -> "Packed":
        data = np.load(path)
        return cls(data["flat"], data["offsets"])


def pad_rows(packed: Packed, indices, pad_id: int, multiple: int = 1, width: int | None = None):
    """Right-pad the selected rows to the longest one (or to ``width``); returns (input_ids, attention_mask)."""
    indices = np.asarray(indices)
    width = int(width or packed.lengths[indices].max())
    if multiple > 1:
        width = -(-width // multiple) * multiple
    input_ids = np.full((len(indices), width), pad_id, dtype=np.int64)
    attention_mask = np.zeros((len(indices), width), dtype=np.int64)
    for i, index in enumerate(indices):
        row = packed.row(index)
        input_ids[i, :len(row)] = row
        attention_mask[i, :len(row)] = 1
    return torch.from_numpy(input_ids), torch.from_numpy(attention_mask)


def sorted_batches(lengths: np.ndarray, batch_size: int) -> list[np.ndarray]:
    """Deterministic length-sorted batches (inference); results are re-ordered by index."""
    order = np.argsort(lengths, kind="stable")
    return [order[start:start + batch_size] for start in range(0, len(order), batch_size)]


PADDING_MODES = ("sorted", "dynamic", "global")


def inference_batches(lengths: np.ndarray, batch_size: int, padding: str):
    """Batches and the padded width for each evaluation padding mode.

    - ``global``: original order, every row padded to the longest row of the whole set
      (what the reference notebook does by tokenizing everything with ``padding=True``).
    - ``dynamic``: original order, each batch padded to its own longest row.
    - ``sorted``: rows sorted by length, then batched and padded like ``dynamic``.
    Returns a list of (row indices, width or None); predictions are written back by index.
    """
    if padding not in PADDING_MODES:
        raise ValueError(f"padding must be one of {PADDING_MODES}")
    if padding == "sorted":
        return [(rows, None) for rows in sorted_batches(lengths, batch_size)]
    order = np.arange(len(lengths))
    width = int(lengths.max()) if padding == "global" else None
    return [(order[start:start + batch_size], width) for start in range(0, len(order), batch_size)]


class BucketBatchSampler(Sampler):
    """Shuffled batches of similar length: random mega-chunks, sorted inside.

    With ``world > 1`` every rank builds the same batch list and takes every
    ``world``-th batch (the tail is dropped so all ranks run the same step count).
    """

    def __init__(self, lengths: np.ndarray, batch_size: int, seed: int, chunk_batches: int = 64,
                 rank: int = 0, world: int = 1):
        self.lengths = lengths
        self.batch_size = batch_size
        self.seed = seed
        self.chunk = batch_size * chunk_batches
        self.rank, self.world = rank, world
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.lengths))
        batches = []
        for start in range(0, len(order), self.chunk):
            chunk = order[start:start + self.chunk]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            batches.extend(chunk[i:i + self.batch_size] for i in range(0, len(chunk), self.batch_size))
        shuffled = rng.permutation(len(batches))[:len(self) * self.world]
        for index in shuffled[self.rank::self.world]:
            yield batches[index].tolist()

    def __len__(self) -> int:
        return -(-len(self.lengths) // self.batch_size) // self.world


class TrainCollator:
    def __init__(self, packed: Packed, labels: np.ndarray, teacher_logits: np.ndarray | None,
                 pad_id: int):
        self.packed, self.labels, self.teacher = packed, labels, teacher_logits
        self.pad_id = pad_id

    def __call__(self, indices: list[int]) -> dict:
        input_ids, attention_mask = pad_rows(self.packed, indices, self.pad_id, multiple=8)
        batch = {"input_ids": input_ids, "attention_mask": attention_mask,
                 "labels": torch.from_numpy(self.labels[indices].astype(np.float32))}
        if self.teacher is not None:
            batch["teacher"] = torch.from_numpy(np.asarray(self.teacher[indices], dtype=np.float32))
        return batch


class IndexDataset(torch.utils.data.Dataset):
    """The batch sampler yields index lists; the collator does the gathering."""

    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index
