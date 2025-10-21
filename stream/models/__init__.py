"""Model wrappers integrating STREAM with base recommenders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn


@dataclass
class StreamOutputs:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    base_logits: torch.Tensor


class BaseStreamModel(nn.Module):
    """Abstract interface implemented by STREAM-capable models."""

    num_items: int

    def stream_hidden_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def stream_positive_gradients(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def stream_base_logits(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, batch: Dict[str, torch.Tensor]) -> StreamOutputs:  # type: ignore[override]
        logits = self.stream_base_logits(batch)
        hidden = self.stream_hidden_states(batch)
        return StreamOutputs(logits=logits, hidden_states=hidden, base_logits=logits)


__all__ = ["BaseStreamModel", "StreamOutputs"]
