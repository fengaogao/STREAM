"""Trust-region solver for STREAM state updates."""
from __future__ import annotations

import torch


def solve_delta_s(g: torch.Tensor, F: torch.Tensor, lam: float, eps: float) -> torch.Tensor:
    """Solve the quadratic sub-problem with a KL constraint."""

    if g.ndim != 1:
        raise ValueError("g must be a 1-D tensor")
    if F.shape[0] != F.shape[1] or F.shape[0] != g.shape[0]:
        raise ValueError("Shape mismatch between g and F")
    device = g.device
    rank = g.shape[0]
    eye = torch.eye(rank, device=device, dtype=F.dtype)
    A = F + lam * eye
    try:
        sol = torch.linalg.solve(A, g)
    except RuntimeError:
        jitter = 1e-6
        A = A + jitter * eye
        sol = torch.linalg.solve(A, g)
    quad = sol @ (F @ sol)
    denom = float(quad.item()) + 1e-12
    eta = min(1.0, float((2.0 * eps / denom) ** 0.5))
    delta = -eta * sol
    return delta


__all__ = ["solve_delta_s"]
