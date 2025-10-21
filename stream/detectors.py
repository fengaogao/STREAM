"""Drift detection utilities for STREAM."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable

import numpy as np


def _bernoulli_glr(probs: Iterable[float], clicks: Iterable[int]) -> float:
    probs = np.asarray(list(probs))
    clicks = np.asarray(list(clicks))
    if probs.size == 0:
        return 0.0
    p_old = float(np.clip(np.mean(probs), 1e-6, 1 - 1e-6))
    p_new = float(np.clip(np.mean(clicks), 1e-6, 1 - 1e-6))
    llr = clicks * np.log(p_new / p_old) + (1 - clicks) * np.log((1 - p_new) / (1 - p_old))
    return float(np.sum(llr))


@dataclass
class Observation:
    probs: np.ndarray
    click: int


class DriftDetector:
    """Combines GLR and conformal triggers."""

    def __init__(
        self,
        window_m: int,
        glr_threshold: float,
        conformal_alpha: float,
        conformal_q: int,
        max_scores: int = 5000,
    ) -> None:
        self.window_m = window_m
        self.glr_threshold = glr_threshold
        self.conformal_alpha = conformal_alpha
        self.conformal_q = conformal_q
        self.window: Deque[Observation] = deque(maxlen=window_m)
        self.scores: Deque[float] = deque(maxlen=max_scores)

    def _update_conformal(self, residual: float) -> bool:
        triggered = False
        if len(self.scores) >= max(self.conformal_q, 1):
            quantile = float(np.quantile(list(self.scores), 1 - self.conformal_alpha))
            if residual > quantile:
                triggered = True
        self.scores.append(residual)
        return triggered

    def _compute_glr(self) -> bool:
        if len(self.window) < 2:
            return False
        probs = [float(obs.probs.max()) for obs in self.window]
        clicks = [int(obs.click) for obs in self.window]
        glr = _bernoulli_glr(probs, clicks)
        return glr > self.glr_threshold

    def update(self, obs: Dict) -> bool:
        """Update detector with a new observation and return whether drift fired."""

        probs = obs.get("probs")
        if probs is None:
            prob_val = float(obs.get("prob", 0.0))
            probs = np.asarray([prob_val], dtype=np.float32)
        else:
            probs = np.asarray(probs, dtype=np.float32)
        click = int(obs.get("click", 0))
        self.window.append(Observation(probs=probs, click=click))
        residual = abs(float(probs.max()) - click)
        conformal_trigger = self._update_conformal(residual)
        glr_trigger = self._compute_glr()
        return conformal_trigger or glr_trigger


__all__ = ["DriftDetector"]
