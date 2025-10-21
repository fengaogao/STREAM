"""Utility helpers for STREAM."""
from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch


_LOGGER: logging.Logger | None = None


def get_logger(name: str = "stream") -> logging.Logger:
    """Return a module-level logger with a standard configuration."""

    global _LOGGER
    if _LOGGER is None:
        logging.basicConfig(
            level=os.environ.get("STREAM_LOGLEVEL", "INFO"),
            format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        )
        _LOGGER = logging.getLogger(name)
    return _LOGGER


def set_seed(seed: int) -> None:
    """Set the global RNG seed across Python, NumPy and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class AverageMeter:
    """Tracks a running average of scalar values."""

    value: float = 0.0
    count: int = 0
    history: List[float] = field(default_factory=list)

    def update(self, new_value: float, weight: int = 1) -> None:
        self.value += new_value * weight
        self.count += weight
        self.history.append(new_value)

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.value / self.count


def moving_average(values: Sequence[float], window: int) -> List[float]:
    """Compute a simple moving average."""

    if window <= 0:
        raise ValueError("window must be positive")
    if not values:
        return []
    result: List[float] = []
    acc = 0.0
    for i, val in enumerate(values):
        acc += val
        if i >= window:
            acc -= values[i - window]
        denom = min(i + 1, window)
        result.append(acc / denom)
    return result


def topk_indices(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the top-*k* indices and values along the last dimension."""

    if k <= 0:
        raise ValueError("k must be positive")
    values, indices = torch.topk(logits, k=k, dim=-1)
    return values, indices


def batched(iterable: Iterable, batch_size: int) -> Iterable[List]:
    """Yield lists of size ``batch_size`` from *iterable*."""

    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def cosine_similarity_matrix(x: torch.Tensor) -> torch.Tensor:
    """Return pairwise cosine similarities between rows of ``x``."""

    x_norm = torch.nn.functional.normalize(x, dim=-1)
    return x_norm @ x_norm.t()


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return the entropy of a categorical distribution parameterised by logits."""

    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


__all__ = [
    "AverageMeter",
    "batched",
    "cosine_similarity_matrix",
    "entropy_from_logits",
    "get_logger",
    "moving_average",
    "set_seed",
    "topk_indices",
]
