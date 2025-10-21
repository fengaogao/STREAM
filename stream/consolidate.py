"""Periodic consolidation utilities for STREAM."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

import torch

from .state_adapter import ItemHead
from .utils import get_logger

LOGGER = get_logger(__name__)


def distill_item_head(
    item_head: ItemHead,
    deltas: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    eta: float = 0.1,
) -> None:
    """Aggregate recent state deltas into the shared item head."""

    deltas = list(deltas)
    if not deltas:
        return
    delta_stack = torch.stack([d[0] for d in deltas], dim=0)
    vector_stack = torch.stack([d[1] for d in deltas], dim=0)
    mean_delta = delta_stack.mean(dim=0).unsqueeze(1)
    mean_vector = vector_stack.mean(dim=0).unsqueeze(0)
    update = eta * mean_delta @ mean_vector
    item_head.W.data.add_(update.to(item_head.W.device))
    LOGGER.info("Distilled %d deltas into item head", len(deltas))


def maybe_apply_lora_merge(model, policy_path: Path, subspace: torch.Tensor) -> bool:
    """Placeholder implementation for optional LoRA merge policy."""

    if not policy_path.exists():
        LOGGER.info("No LoRA policy found at %s", policy_path)
        return False
    with policy_path.open("r", encoding="utf-8") as f:
        policy = json.load(f)
    target_modules: List[str] = policy.get("target_modules", [])
    if not target_modules:
        LOGGER.info("LoRA policy has no target modules; skipping merge")
        return False
    LOGGER.info(
        "LoRA merge requested for modules: %s (placeholder implementation)",
        ", ".join(target_modules),
    )
    # Actual LoRA merge would run in a shadow copy; we simply acknowledge here.
    return False


__all__ = ["distill_item_head", "maybe_apply_lora_merge"]
