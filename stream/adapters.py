"""Adaptation utilities for lightweight fine-tuning."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn


@dataclass(frozen=True)
class LoRAConfig:
    """Configuration for inserting LoRA adapters into linear layers."""

    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: Iterable[str] | None = None


class LoRALinear(nn.Module):
    """Wrap an ``nn.Linear`` with a trainable low-rank residual branch."""

    def __init__(self, base: nn.Linear, *, r: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.scaling = self.alpha / max(self.r, 1)
        device = self.base.weight.device
        dtype = self.base.weight.dtype

        if self.r > 0:
            self.lora_a = nn.Linear(base.in_features, self.r, bias=False)
            self.lora_b = nn.Linear(self.r, base.out_features, bias=False)
            self.lora_a.to(device=device, dtype=dtype)
            self.lora_b.to(device=device, dtype=dtype)
            self.reset_parameters()
        else:
            self.register_module("lora_a", nn.Identity())
            self.register_module("lora_b", nn.Identity())

        for param in self.base.parameters():
            param.requires_grad_(False)

    def reset_parameters(self) -> None:
        if self.r <= 0:
            return
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        result = self.base(x)
        if self.r <= 0:
            return result
        update = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return result + update


def _resolve_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for attr in parts[:-1]:
        parent = getattr(parent, attr)
    return parent, parts[-1]


def apply_lora_adapters(model: nn.Module, config: LoRAConfig) -> List[str]:
    """Insert LoRA adapters into modules whose qualified name matches ``config``.

    Returns the list of module names that were replaced.
    """

    if config.r <= 0:
        return []

    keywords = list(config.target_modules or [])
    if not keywords:
        # Default to the most common projection and feed-forward names across
        # HuggingFace encoder/decoder architectures.
        keywords = [
            "query",
            "key",
            "value",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "out_proj",
            "fc1",
            "fc2",
            "dense",
            "c_attn",
            "c_proj",
            "mlp",
            "lm_head",
        ]

    replaced: List[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(keyword in name for keyword in keywords):
            continue
        parent, attr = _resolve_parent_module(model, name)
        wrapped = LoRALinear(module, r=config.r, alpha=config.alpha, dropout=config.dropout)
        setattr(parent, attr, wrapped)
        replaced.append(name)
    return replaced


def freeze_module_parameters(module: nn.Module, *, trainable_names: Iterable[str] | None = None) -> None:
    """Freeze all parameters except those whose names contain any keyword."""

    keywords = list(trainable_names or [])
    for name, param in module.named_parameters():
        if not keywords:
            param.requires_grad = False
            continue
        param.requires_grad = any(keyword in name for keyword in keywords)


def set_trainable_layers(transformer: nn.Module, *, keep_last: int) -> None:
    """Freeze all but the final ``keep_last`` transformer blocks if available."""

    if keep_last <= 0:
        for param in transformer.parameters():
            param.requires_grad = False
        return

    # Common attribute names across encoder/decoder stacks
    possible_attrs = [
        "encoder.layer",
        "transformer.h",
        "model.decoder.layers",
        "gpt_neox.layers",
        "layers",
    ]
    for attr_path in possible_attrs:
        current = transformer
        try:
            for part in attr_path.split("."):
                current = getattr(current, part)
        except AttributeError:
            continue

        if not isinstance(current, (nn.ModuleList, list, tuple)):
            continue

        total_layers = len(current)
        for idx, layer in enumerate(current):
            allow_train = idx >= total_layers - keep_last
            for param in layer.parameters():
                param.requires_grad = allow_train
        return

    # Fallback: if we reach here simply leave parameters trainable.


__all__ = ["LoRAConfig", "LoRALinear", "apply_lora_adapters", "freeze_module_parameters", "set_trainable_layers"]
