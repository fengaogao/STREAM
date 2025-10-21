from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from stream.state_adapter import ItemHead
from stream.subspace import compute_subspace
from stream.trust_region import solve_delta_s


class DummyModel:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1))

    def eval(self):
        pass

    def stream_hidden_states(self, batch):
        return torch.stack(batch["hidden"], dim=0)

    def stream_positive_gradients(self, batch):
        return torch.stack(batch["grads"], dim=0)

    def stream_base_logits(self, batch):
        return torch.zeros(len(batch["hidden"]), 4)


def test_subspace_orthonormal():
    dim = 8
    rank = 3
    model = DummyModel(dim)
    batch = {
        "hidden": [torch.randn(dim) for _ in range(10)],
        "grads": [torch.randn(dim) for _ in range(10)],
    }
    dataloader = [batch]
    result = compute_subspace(model, dataloader, rank=rank, mode="pca")
    basis = result.basis
    assert basis.shape == (dim, rank)
    gram = basis.t() @ basis
    eye = torch.eye(rank)
    assert torch.allclose(gram, eye, atol=1e-5)


def test_trust_region_shape():
    rank = 4
    g = torch.randn(rank)
    W = torch.randn(rank, rank)
    F = W @ W.t()
    delta = solve_delta_s(g, F, lam=1e-3, eps=0.01)
    assert delta.shape == (rank,)


def test_overlay_shapes():
    rank = 5
    num_items = 7
    item_head = ItemHead(rank=rank, num_items=num_items, device=torch.device("cpu"))
    base_logits = torch.zeros(1, num_items)
    state = torch.randn(rank)
    output = base_logits + item_head(state)
    assert output.shape == (1, num_items)
