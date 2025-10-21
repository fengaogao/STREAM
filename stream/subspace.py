"""Subspace extraction utilities for STREAM."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch


SubspaceMode = Literal["gradcov", "pca"]


@dataclass
class SubspaceResult:
    basis: torch.Tensor
    mode: SubspaceMode
    meta: dict

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "basis": self.basis,
                "mode": self.mode,
                "meta": self.meta,
            },
            out_dir / "subspace_U.pt",
        )


def _normalise_basis(basis: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(basis)
    return q


def compute_subspace(
    model,
    dataloader,
    rank: int,
    mode: SubspaceMode = "gradcov",
    device: Optional[torch.device] = None,
) -> SubspaceResult:
    """Compute an orthonormal subspace basis using gradients or PCA."""

    if device is None:
        device = next(model.parameters()).device  # type: ignore[call-arg]
    model.eval()
    feature_dim: Optional[int] = None
    if mode == "gradcov":
        cov: Optional[torch.Tensor] = None
        count = 0
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            grads = model.stream_positive_gradients(batch)
            if grads is None:
                raise RuntimeError("Model did not return gradients for gradcov mode")
            grads = grads.detach()
            if cov is None:
                feature_dim = grads.size(-1)
                cov = torch.zeros(feature_dim, feature_dim, device=device)
            cov = cov + grads.t().mm(grads)
            count += grads.size(0)
        if cov is None or count == 0:
            raise RuntimeError("No data available to compute gradient covariance")
        cov = cov / float(count)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        top = eigvecs[:, -rank:]
        basis = _normalise_basis(top)
        feature_dim = basis.size(0)
        meta = {"eigvals": eigvals.detach().cpu().tolist()[-rank:], "feature_dim": feature_dim}
    else:
        hidden_accum: list[torch.Tensor] = []
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            h = model.stream_hidden_states(batch)
            hidden_accum.append(h.detach())
        if not hidden_accum:
            raise RuntimeError("No hidden states collected for PCA")
        hidden = torch.cat(hidden_accum, dim=0)
        feature_dim = hidden.size(-1)
        hidden = hidden - hidden.mean(dim=0, keepdim=True)
        _, _, v = torch.linalg.svd(hidden, full_matrices=False)
        top = v[:rank].t()
        basis = _normalise_basis(top)
        meta = {"feature_dim": feature_dim}
    return SubspaceResult(basis=basis.cpu(), mode=mode, meta=meta)


__all__ = ["SubspaceResult", "compute_subspace"]
