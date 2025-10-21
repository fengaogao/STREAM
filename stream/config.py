"""Configuration utilities for STREAM."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict

import torch


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class StreamConfig:
    """Top level configuration for STREAM components.

    The defaults are aligned with the reference description in the project
    specification. The configuration is serialisable to and from JSON for ease
    of checkpointing.
    """

    rank_r: int = 32
    epsilon_kl: float = 0.01
    lambda_l2: float = 1e-3
    window_m: int = 500
    glr_threshold: float = 8.0
    conformal_alpha: float = 0.10
    conformal_q: int = 3
    temperature: float = 1.0
    router_k: int = 16
    device: str = _default_device()

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""

        return {
            "rank_r": self.rank_r,
            "epsilon_kl": self.epsilon_kl,
            "lambda_l2": self.lambda_l2,
            "window_m": self.window_m,
            "glr_threshold": self.glr_threshold,
            "conformal_alpha": self.conformal_alpha,
            "conformal_q": self.conformal_q,
            "temperature": self.temperature,
            "router_k": self.router_k,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamConfig":
        """Create a :class:`StreamConfig` from a dictionary."""

        return cls(**data)

    def to_json(self, path: Path) -> None:
        """Persist the configuration to *path* in JSON format."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "StreamConfig":
        """Load configuration from a JSON file."""

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


__all__ = ["StreamConfig"]
