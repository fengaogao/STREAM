"""Evaluation metrics for recommendation quality and safety."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np


def recall_at_k(ranking: Sequence[int], positives: Sequence[int], k: int) -> float:
    """Compute Recall@K for a single ranking."""

    if k <= 0:
        raise ValueError("k must be positive")
    topk = ranking[:k]
    positives_set = set(positives)
    if not positives_set:
        return 0.0
    hits = sum(1 for item in topk if item in positives_set)
    return hits / len(positives_set)


def dcg_at_k(ranking: Sequence[int], positives: Sequence[int], k: int) -> float:
    """Discounted cumulative gain at K."""

    positives_set = set(positives)
    score = 0.0
    for i, item in enumerate(ranking[:k]):
        if item in positives_set:
            score += 1.0 / np.log2(i + 2)
    return score


def ndcg_at_k(ranking: Sequence[int], positives: Sequence[int], k: int) -> float:
    """Normalised discounted cumulative gain at K."""

    ideal_ranking = list(positives)[:k]
    ideal_dcg = dcg_at_k(ideal_ranking, positives, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(ranking, positives, k) / ideal_dcg


def mean_reciprocal_rank(ranking: Sequence[int], positives: Sequence[int]) -> float:
    """Mean reciprocal rank for a single ranking."""

    positives_set = set(positives)
    for idx, item in enumerate(ranking):
        if item in positives_set:
            return 1.0 / (idx + 1)
    return 0.0


def average_kl_divergence(old_probs: np.ndarray, new_probs: np.ndarray) -> float:
    """Average KL divergence ``KL(old || new)`` across rows."""

    eps = 1e-12
    old_probs = np.clip(old_probs, eps, 1.0)
    new_probs = np.clip(new_probs, eps, 1.0)
    ratio = old_probs / new_probs
    kl = (old_probs * np.log(ratio)).sum(axis=-1)
    return float(np.mean(kl))


def uplift_variance(uplifts: Sequence[float]) -> float:
    """Return the variance of uplift values across contexts."""

    if not uplifts:
        return 0.0
    return float(np.var(np.asarray(uplifts)))


def evaluate_rankings(
    rankings: Sequence[Sequence[int]],
    positives: Sequence[Sequence[int]],
    ks: Iterable[int] = (5, 10),
) -> Dict[str, float]:
    """Compute Recall@K, NDCG@K and MRR over a batch of rankings."""

    metrics: Dict[str, float] = {}
    ks = list(ks)
    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    mrrs: List[float] = []
    for ranking, pos in zip(rankings, positives):
        for k in ks:
            recalls[k].append(recall_at_k(ranking, pos, k))
            ndcgs[k].append(ndcg_at_k(ranking, pos, k))
        mrrs.append(mean_reciprocal_rank(ranking, pos))
    for k in ks:
        metrics[f"recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
    metrics["mrr"] = float(np.mean(mrrs)) if mrrs else 0.0
    return metrics


__all__ = [
    "average_kl_divergence",
    "dcg_at_k",
    "evaluate_rankings",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "recall_at_k",
    "uplift_variance",
]
