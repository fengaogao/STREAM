"""State adapter overlay for STREAM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


class RegionStateBank:
    """Maintains per-region latent states with rollback support."""

    def __init__(self, num_regions: int, rank: int, device: torch.device) -> None:
        self.device = device
        self.rank = rank
        self.states = torch.zeros(num_regions, rank, device=device)
        self._history: Dict[int, list[torch.Tensor]] = {i: [] for i in range(num_regions)}

    def get_state(self, region_id: int) -> torch.Tensor:
        return self.states[region_id]

    def apply_delta(self, region_id: int, delta: torch.Tensor) -> None:
        delta = delta.to(self.device)
        prev = self.states[region_id].clone()
        self._history[region_id].append(prev)
        self.states[region_id] = prev + delta

    def snapshot(self) -> torch.Tensor:
        return self.states.clone()

    def rollback_last(self, region_id: int) -> None:
        history = self._history.get(region_id)
        if history and len(history) > 0:
            self.states[region_id] = history.pop()
        else:
            self.states[region_id].zero_()

    def load_state(self, tensor: torch.Tensor) -> None:
        if tensor.shape != self.states.shape:
            raise ValueError("Snapshot shape mismatch")
        self.states.copy_(tensor.to(self.device))
        for history in self._history.values():
            history.clear()


@dataclass
class ItemHeadInit:
    U: torch.Tensor
    item_embeddings: Optional[torch.Tensor] = None
    lambda_l2: float = 1e-3


class ItemHead(nn.Module):
    """Low-rank item head operating inside the STREAM subspace."""

    def __init__(self, rank: int, num_items: int, device: torch.device) -> None:
        super().__init__()
        self.rank = rank
        self.num_items = num_items
        self.W = nn.Parameter(torch.zeros(rank, num_items, device=device))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state @ self.W

    @torch.no_grad()
    def initialise(self, init: ItemHeadInit) -> None:
        if init.item_embeddings is not None:
            proj = init.U.t().mm(init.item_embeddings.to(init.U.device))
            proj = proj.to(self.W.device)
            if proj.shape != self.W.shape:
                raise ValueError("Projected embeddings mismatch")
            self.W.copy_(proj)
        else:
            nn.init.normal_(self.W, std=0.02)

    def ridge_fit(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        lambda_l2: float,
    ) -> None:
        """Fit W using ridge regression in closed form."""

        # features: [N, rank], targets: [N, num_items]
        xtx = features.t().mm(features) + lambda_l2 * torch.eye(self.rank, device=features.device)
        xty = features.t().mm(targets)
        solution = torch.linalg.solve(xtx, xty)
        self.W.data.copy_(solution)


def apply_overlay(base_logits: torch.Tensor, state: torch.Tensor, item_head: ItemHead) -> torch.Tensor:
    """Combine base logits with STREAM state contribution."""

    overlay = item_head(state)
    return base_logits + overlay


__all__ = [
    "ItemHead",
    "ItemHeadInit",
    "RegionStateBank",
    "apply_overlay",
]
