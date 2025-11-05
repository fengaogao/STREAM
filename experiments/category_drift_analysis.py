"""Analysis script for studying category preference drift in STREAM."""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stream.dataio import ItemVocab, build_dataloader, load_all_splits
from stream.models.bert_stream import BertStreamModel
from stream.models.causal_lm_stream import CausalLMStreamModel
from stream.train_offline import (
    compute_category_semantic_subspace,
    extract_targets_from_batch,
    load_item_categories,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class UserSegmentDistributions:
    """Container for per-user category distributions."""

    early: Optional[np.ndarray]
    late: Optional[np.ndarray]


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_user_category_distributions(
    splits: Mapping[str, Sequence[Mapping]],
    category_map: Mapping[int, Sequence[str]],
    category_order: Sequence[str],
    *,
    max_users: Optional[int] = None,
) -> Tuple[Dict[str, UserSegmentDistributions], List[str], Dict[str, Dict[str, Dict[str, float]]]]:
    """Compute per-user early and late category distributions."""

    categories = list(category_order)
    category_index = {cat: idx for idx, cat in enumerate(categories)}

    early_counts: Dict[str, Counter] = defaultdict(Counter)
    late_counts: Dict[str, Counter] = defaultdict(Counter)

    def _accumulate(records: Iterable[Mapping], dest: MutableMapping[str, Counter]) -> None:
        for rec in records:
            user_raw = rec.get("user")
            if user_raw is None:
                continue
            user_id = str(user_raw)
            items: Sequence[int] = rec.get("items", [])  # type: ignore[assignment]
            if not items:
                continue
            for item_idx in items:
                cats = category_map.get(int(item_idx), [])
                if not cats:
                    continue
                weight = 1.0 / float(len(cats))
                for cat in cats:
                    dest[user_id][cat] += weight

    _accumulate(splits.get("original", []), early_counts)
    late_records = list(splits.get("finetune", [])) + list(splits.get("test", []))
    _accumulate(late_records, late_counts)

    all_users = sorted(set(early_counts.keys()) | set(late_counts.keys()), key=lambda x: int(x))
    if max_users is not None:
        all_users = all_users[: max_users]

    user_distributions: Dict[str, UserSegmentDistributions] = {}
    user_prob_json: Dict[str, Dict[str, Dict[str, float]]] = {}

    for user_id in all_users:
        early_counter = early_counts.get(user_id, Counter())
        late_counter = late_counts.get(user_id, Counter())

        def _to_distribution(counter: Counter) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
            total = float(sum(counter.values()))
            probs = {cat: float(counter.get(cat, 0.0) / total) if total > 0 else 0.0 for cat in categories}
            if total <= 0:
                return None, probs
            vec = np.zeros(len(categories), dtype=np.float32)
            for cat, prob in probs.items():
                idx = category_index.get(cat)
                if idx is not None:
                    vec[idx] = prob
            return vec, probs

        early_vec, early_probs = _to_distribution(early_counter)
        late_vec, late_probs = _to_distribution(late_counter)
        user_distributions[user_id] = UserSegmentDistributions(early=early_vec, late=late_vec)
        user_prob_json[user_id] = {"early": early_probs, "late": late_probs}

    return user_distributions, categories, user_prob_json


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""

    if p.shape != q.shape:
        raise ValueError("Distributions must have the same shape for JSD")
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    js = 0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m)))
    return float(max(js, 0.0))


def compute_user_category_drift(
    user_distributions: Mapping[str, UserSegmentDistributions],
) -> Dict[str, Dict[str, float]]:
    """Compute drift metrics (L1 and JSD) for users."""

    drift: Dict[str, Dict[str, float]] = {}
    for user_id, dist in user_distributions.items():
        early = dist.early
        late = dist.late
        if early is None or late is None:
            continue
        if early.sum() <= 0 or late.sum() <= 0:
            continue
        l1 = 0.5 * float(np.abs(early - late).sum())
        jsd = js_divergence(early, late)
        drift[user_id] = {"l1": l1, "jsd": jsd}
    return drift


def save_json(data: Mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def plot_jsd_histogram(jsd_values: Sequence[float], out_path: Path, bins: int = 40) -> None:
    if not jsd_values:
        LOGGER.warning("No JSD values available for histogram plot")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(jsd_values, bins=bins, color="tab:blue", alpha=0.75)
    ax.set_xlabel("Jensen-Shannon divergence")
    ax.set_ylabel("Number of users")
    ax.set_title("Distribution of user category drift (JSD)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_top_drift_users_heatmap(
    top_users: Sequence[str],
    user_distributions: Mapping[str, UserSegmentDistributions],
    category_order: Sequence[str],
    out_path: Path,
    *,
    max_categories: int = 20,
) -> None:
    if not top_users:
        LOGGER.warning("No top drift users available for heatmap plot")
        return
    num_categories = min(len(category_order), max_categories)
    categories = list(category_order[:num_categories])
    early_matrix: List[np.ndarray] = []
    late_matrix: List[np.ndarray] = []
    for user_id in top_users:
        dist = user_distributions.get(user_id)
        if not dist or dist.early is None or dist.late is None:
            continue
        early_matrix.append(dist.early[:num_categories])
        late_matrix.append(dist.late[:num_categories])
    if not early_matrix:
        LOGGER.warning("No valid distributions for top drift users heatmap")
        return
    early_stack = np.stack(early_matrix, axis=0)
    late_stack = np.stack(late_matrix, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(max(10, num_categories * 0.6), len(top_users) * 0.4 + 3), sharey=True)
    for ax, data, title in zip(axes, [early_stack, late_stack], ["Early segment", "Late segment"]):
        im = ax.imshow(data, aspect="auto", cmap="viridis")
        ax.set_xticks(range(num_categories))
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.set_yticks(range(len(top_users)))
        ax.set_yticklabels(top_users)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Category distributions for top drift users")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_user_probability_maps(
    user_prob_json: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    user_maps: Dict[str, Dict[str, Dict[str, float]]] = {}
    for user_id, segments in user_prob_json.items():
        user_maps[user_id] = {
            "early": dict(segments.get("early", {})),
            "late": dict(segments.get("late", {})),
        }
    return user_maps


def define_drift_bucket(
    user_id: str,
    item_idx: int,
    user_prob_maps: Mapping[str, Mapping[str, Mapping[str, float]]],
    category_map: Mapping[int, Sequence[str]],
    *,
    top_k: int = 3,
    early_threshold: float = 0.2,
    drift_low_threshold: float = 0.05,
    late_threshold: float = 0.2,
) -> Optional[str]:
    user_entry = user_prob_maps.get(user_id)
    if not user_entry:
        return None
    early_probs = user_entry.get("early")
    late_probs = user_entry.get("late")
    if not early_probs or not late_probs:
        return None
    categories = category_map.get(int(item_idx), [])
    if not categories:
        return None

    sorted_cats = sorted(early_probs.items(), key=lambda kv: kv[1], reverse=True)
    top_categories = [cat for cat, prob in sorted_cats[:top_k] if prob > 0]

    intersect_top = any(cat in top_categories for cat in categories)
    meets_threshold = any(early_probs.get(cat, 0.0) >= early_threshold for cat in categories)
    if intersect_top and meets_threshold:
        return "non_drift"

    low_early = all(early_probs.get(cat, 0.0) < drift_low_threshold for cat in categories)
    high_late = any(late_probs.get(cat, 0.0) >= late_threshold for cat in categories)
    if low_early and high_late:
        return "drift"
    return None


def compute_ranking_metrics(scores: torch.Tensor, target: int, ks: Sequence[int]) -> Dict[str, float]:
    sorted_indices = torch.argsort(scores, descending=True)
    target_positions = (sorted_indices == target).nonzero(as_tuple=True)
    if len(target_positions[0]) == 0:
        return {f"hit@{k}": 0.0 for k in ks} | {f"recall@{k}": 0.0 for k in ks} | {f"ndcg@{k}": 0.0 for k in ks}
    rank = int(target_positions[0][0].item()) + 1
    metrics: Dict[str, float] = {}
    for k in ks:
        hit = 1.0 if rank <= k else 0.0
        metrics[f"hit@{k}"] = hit
        metrics[f"recall@{k}"] = hit
        metrics[f"ndcg@{k}"] = 1.0 / math.log2(rank + 1) if rank <= k else 0.0
    return metrics


def evaluate_metrics_per_bucket(
    model,
    dataloader,
    user_prob_maps: Mapping[str, Mapping[str, Mapping[str, float]]],
    category_map: Mapping[int, Sequence[str]],
    *,
    sample_meta: Sequence[Mapping[str, object]],
    model_type: str,
    ks: Sequence[int],
    top_k: int,
    early_threshold: float,
    drift_low_threshold: float,
    late_threshold: float,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
    metrics_sum: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[str, int] = defaultdict(int)

    ptr = 0
    first_param = next(model.parameters())
    device = first_param.device
    for batch in tqdm(dataloader, desc="eval-drift-buckets"):
        batch_size = batch["input_ids"].size(0)
        meta_batch = sample_meta[ptr : ptr + batch_size]
        ptr += batch_size
        batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            logits = model.stream_base_logits(batch_device)
        logits = logits.detach().cpu()
        targets = extract_targets_from_batch(batch_device, model_type).tolist()
        for row, meta, target in zip(logits, meta_batch, targets):
            if target < 0:
                continue
            user_entry = meta.get("user")
            if user_entry is None:
                continue
            user_id = str(user_entry)
            if not user_id:
                continue
            target_item = int(target)
            bucket = define_drift_bucket(
                user_id,
                target_item,
                user_prob_maps,
                category_map,
                top_k=top_k,
                early_threshold=early_threshold,
                drift_low_threshold=drift_low_threshold,
                late_threshold=late_threshold,
            )
            if bucket is None:
                continue
            metrics = compute_ranking_metrics(row, target_item, ks)
            for name, value in metrics.items():
                metrics_sum[bucket][name] += float(value)
            counts[bucket] += 1

    averaged: Dict[str, Dict[str, float]] = {}
    for bucket, metric_map in metrics_sum.items():
        bucket_count = counts.get(bucket, 0)
        if bucket_count <= 0:
            continue
        averaged[bucket] = {name: val / bucket_count for name, val in metric_map.items()}
    return averaged, counts


def build_sample_metadata(
    records: Sequence[Mapping],
    dataset,
    model_type: str,
) -> List[Dict[str, object]]:
    sample_meta: List[Dict[str, object]] = []
    if model_type == "causal":
        sample_specs = getattr(dataset, "_sample_specs", [])
        record_users = getattr(dataset, "_record_users", [])
        record_items = getattr(dataset, "_record_items", [])
        for record_idx, start, target_idx in sample_specs:
            if record_idx >= len(record_users) or record_idx >= len(record_items):
                continue
            if target_idx >= len(record_items[record_idx]):
                continue
            user_raw = record_users[record_idx]
            sample_meta.append({"user": str(user_raw)})
    elif model_type == "bert":
        for rec in records:
            user_raw = rec.get("user")
            items: Sequence[int] = rec.get("items", [])  # type: ignore[assignment]
            if not items:
                continue
            user_val = str(user_raw) if user_raw is not None else None
            sample_meta.append({"user": user_val})
    else:
        raise ValueError(f"Unsupported model_type {model_type}")
    return sample_meta


def plot_bucket_metrics(
    metrics: Mapping[str, Mapping[str, float]],
    counts: Mapping[str, int],
    out_path: Path,
    *,
    metrics_to_plot: Sequence[str],
) -> None:
    if not metrics:
        LOGGER.warning("No metrics available for bucket comparison plot")
        return
    buckets = sorted(metrics.keys())
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(6, 4 * len(metrics_to_plot)))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, metric_name in zip(axes, metrics_to_plot):
        values = [metrics[bucket].get(metric_name, 0.0) for bucket in buckets]
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        ax.bar(buckets, values, color=colors[: len(buckets)])
        ax.set_ylabel(metric_name)
        ax.set_ylim(0.0, max(values + [0.01]) * 1.1)
        count_labels = [f"n={counts.get(bucket, 0)}" for bucket in buckets]
        for idx, (bucket, value, label) in enumerate(zip(buckets, values, count_labels)):
            ax.text(idx, value + 1e-3, label, ha="center", va="bottom", fontsize=10)
        ax.set_title(f"{metric_name} by drift bucket")
    axes[-1].set_xlabel("Bucket")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def gather_user_sequences(records: Sequence[Mapping]) -> Dict[str, List[int]]:
    user_map: Dict[str, List[int]] = defaultdict(list)
    for rec in records:
        user_raw = rec.get("user")
        if user_raw is None:
            continue
        items: Sequence[int] = rec.get("items", [])  # type: ignore[assignment]
        if not items:
            continue
        user_map[str(user_raw)].extend(int(item) for item in items)
    return user_map


def merge_user_sequences(
    original: Mapping[str, List[int]],
    finetune: Mapping[str, List[int]],
    test: Mapping[str, List[int]],
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    combined_records: List[Dict[str, object]] = []
    early_boundaries: Dict[str, int] = {}
    users = sorted(set(original.keys()) | set(finetune.keys()) | set(test.keys()), key=lambda x: int(x))
    for user_id in users:
        early_seq = list(original.get(user_id, []))
        late_seq = list(finetune.get(user_id, [])) + list(test.get(user_id, []))
        combined_seq = early_seq + late_seq
        if len(combined_seq) < 2:
            continue
        record = {"user": int(user_id), "items": combined_seq}
        combined_records.append(record)
        early_boundaries[user_id] = len(early_seq)
    return combined_records, early_boundaries


def filter_records_by_users(records: Sequence[Mapping], allowed_users: Optional[set[str]]) -> List[Mapping]:
    if allowed_users is None:
        return list(records)
    filtered: List[Mapping] = []
    for rec in records:
        user_raw = rec.get("user")
        if user_raw is None:
            continue
        if str(user_raw) in allowed_users:
            filtered.append(rec)
    return filtered


def compute_direction_subspace_drift(
    model: CausalLMStreamModel,
    dataset,
    dataloader,
    basis: torch.Tensor,
    early_boundaries: Mapping[str, int],
    focus_users: Sequence[str],
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, List[Dict[str, object]]]]:
    direction_dim = basis.size(1)
    basis_device = basis.to(model.model.lm_head.weight.device)

    user_sums: Dict[str, Dict[str, torch.Tensor]] = {}
    user_counts: Dict[str, Dict[str, int]] = {}
    trajectories: Dict[str, List[Dict[str, object]]] = {user: [] for user in focus_users}

    sample_meta: List[Dict[str, object]] = []
    for record_idx, start, target_idx in dataset._sample_specs:  # type: ignore[attr-defined]
        user_raw = dataset._record_users[record_idx]  # type: ignore[attr-defined]
        items = dataset._record_items[record_idx]  # type: ignore[attr-defined]
        if target_idx >= len(items):
            continue
        user_id = str(user_raw)
        target_item = int(items[target_idx])
        boundary = early_boundaries.get(user_id, 0)
        segment = "early" if target_idx < boundary else "late"
        sample_meta.append(
            {
                "user": user_id,
                "segment": segment,
                "time_index": int(target_idx),
                "target_item": target_item,
            }
        )

    ptr = 0
    device = model.model.lm_head.weight.device
    for batch in tqdm(dataloader, desc="direction-drift"):
        batch_size = batch["input_ids"].size(0)
        meta_batch = sample_meta[ptr : ptr + batch_size]
        ptr += batch_size
        batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            hidden = model.stream_hidden_states(batch_device)
        projections = hidden @ basis_device
        projections = projections.detach().cpu()
        for proj, meta in zip(projections, meta_batch):
            user_id = meta["user"]  # type: ignore[assignment]
            segment = meta["segment"]  # type: ignore[assignment]
            if segment not in ("early", "late"):
                continue
            if user_id not in user_sums:
                user_sums[user_id] = {
                    "early": torch.zeros(direction_dim),
                    "late": torch.zeros(direction_dim),
                }
                user_counts[user_id] = {"early": 0, "late": 0}
            user_sums[user_id][segment] += proj
            user_counts[user_id][segment] += 1
            if user_id in trajectories:
                trajectories[user_id].append(
                    {
                        "time_index": meta["time_index"],
                        "segment": segment,
                        "projection": proj.numpy().tolist(),
                    }
                )

    user_direction_drift: Dict[str, Dict[str, object]] = {}
    for user_id, sums in user_sums.items():
        counts = user_counts.get(user_id, {})
        early_count = counts.get("early", 0)
        late_count = counts.get("late", 0)
        if early_count <= 0 or late_count <= 0:
            continue
        early_mean = (sums["early"] / early_count).numpy()
        late_mean = (sums["late"] / late_count).numpy()
        delta = float(np.linalg.norm(early_mean - late_mean))
        user_direction_drift[user_id] = {
            "delta": delta,
            "early_mean": early_mean.tolist(),
            "late_mean": late_mean.tolist(),
            "early_count": early_count,
            "late_count": late_count,
        }

    return user_direction_drift, trajectories


def rankdata(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.zeros_like(arr, dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 2:
        return float("nan")
    x_mean = x_arr.mean()
    y_mean = y_arr.mean()
    x_std = x_arr.std()
    y_std = y_arr.std()
    if x_std == 0 or y_std == 0:
        return float("nan")
    return float(((x_arr - x_mean) * (y_arr - y_mean)).mean() / (x_std * y_std))


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson_correlation(rankdata(x), rankdata(y))


def plot_user_trajectories(
    user_id: str,
    trajectory: Sequence[Mapping[str, object]],
    direction_labels: Sequence[str],
    out_path: Path,
    *,
    boundary: int,
) -> None:
    if not trajectory:
        return
    sorted_traj = sorted(trajectory, key=lambda entry: entry["time_index"])
    times = [int(entry["time_index"]) for entry in sorted_traj]
    segments = [str(entry["segment"]) for entry in sorted_traj]
    projections = [np.asarray(entry["projection"], dtype=np.float32) for entry in sorted_traj]
    proj_matrix = np.stack(projections, axis=0)
    num_dirs = min(len(direction_labels), proj_matrix.shape[1])
    if num_dirs <= 0:
        return
    fig, axes = plt.subplots(num_dirs, 1, figsize=(8, 3 * num_dirs), sharex=True)
    if num_dirs == 1:
        axes = [axes]
    colors = {"early": "tab:blue", "late": "tab:orange"}
    for idx in range(num_dirs):
        ax = axes[idx]
        values = proj_matrix[:, idx]
        ax.plot(times, values, color="gray", alpha=0.4, linestyle="--")
        early_times = [t for t, seg in zip(times, segments) if seg == "early"]
        early_vals = [v for v, seg in zip(values, segments) if seg == "early"]
        late_times = [t for t, seg in zip(times, segments) if seg == "late"]
        late_vals = [v for v, seg in zip(values, segments) if seg == "late"]
        ax.scatter(early_times, early_vals, color=colors["early"], label="early" if idx == 0 else None)
        ax.scatter(late_times, late_vals, color=colors["late"], label="late" if idx == 0 else None)
        ax.axvline(boundary - 0.5, color="black", linestyle=":", label="boundary" if idx == 0 else None)
        ax.set_ylabel(direction_labels[idx])
    axes[-1].set_xlabel("Interaction index")
    axes[0].set_title(f"User {user_id} trajectory in category subspace")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Category drift analysis for STREAM")
    parser.add_argument("--data_dir", type=Path, required=True, help="Dataset directory")
    parser.add_argument("--model_dir", type=Path, required=True, help="Trained model directory")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for analysis artifacts")
    parser.add_argument("--max_users", type=int, default=None, help="Limit analysis to the first N users")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hist_bins", type=int, default=40, help="Number of bins for JSD histogram")
    parser.add_argument("--top_k_users", type=int, default=20, help="Number of top drift users to visualize")
    parser.add_argument("--heatmap_categories", type=int, default=20, help="Maximum categories in heatmap plot")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers")
    parser.add_argument("--device", type=str, default=None, help="Device for model execution (e.g. cuda)" )
    parser.add_argument(
        "--model_type",
        type=str,
        default="causal",
        choices=["causal", "bert"],
        help="Model architecture used for analysis",
    )
    parser.add_argument("--drift_top_k", type=int, default=3, help="Top-k early categories considered non-drift")
    parser.add_argument("--non_drift_threshold", type=float, default=0.2, help="Probability threshold for non-drift")
    parser.add_argument("--drift_low_threshold", type=float, default=0.05, help="Maximum early probability for drift classification")
    parser.add_argument("--drift_late_threshold", type=float, default=0.2, help="Minimum late probability for drift classification")
    parser.add_argument("--metric_ks", type=int, nargs="*", default=[5, 10, 20], help="Ranking cutoffs")
    parser.add_argument("--metrics_plot", type=str, nargs="*", default=["ndcg@10", "recall@20"], help="Metrics to visualise in bar plot")
    parser.add_argument("--subspace_rank", type=int, default=8, help="Number of semantic directions to extract")
    parser.add_argument("--subspace_batch_size", type=int, default=32, help="Batch size when computing subspace")
    parser.add_argument("--min_category_samples", type=int, default=20, help="Minimum samples per category for subspace")
    parser.add_argument("--trajectory_directions", type=int, default=3, help="Number of directions to show in trajectories")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    set_random_seeds(args.seed)

    data_dir = args.data_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading metadata and splits from %s", data_dir)
    item_vocab = ItemVocab.from_metadata(data_dir)
    category_map, category_order = load_item_categories(data_dir, item_vocab, item_text_map=None)
    splits = load_all_splits(data_dir)

    if args.max_users is not None:
        LOGGER.info("Restricting analysis to first %d users", args.max_users)

    user_distributions, ordered_categories, user_prob_json = compute_user_category_distributions(
        splits,
        category_map,
        category_order,
        max_users=args.max_users,
    )
    drift_metrics = compute_user_category_drift(user_distributions)

    allowed_users = set(user_distributions.keys())
    filtered_splits = {}
    for split_name, records in splits.items():
        filtered_splits[split_name] = filter_records_by_users(records, allowed_users if allowed_users else None)
    splits = filtered_splits

    save_json(
        {"category_order": ordered_categories, "users": user_prob_json},
        out_dir / "user_category_distributions.json",
    )
    save_json(drift_metrics, out_dir / "user_category_drift.json")

    jsd_values = [metrics["jsd"] for metrics in drift_metrics.values() if not math.isnan(metrics["jsd"])]
    plot_jsd_histogram(jsd_values, out_dir / "jsd_hist.png", bins=args.hist_bins)

    top_users = [user for user, _ in sorted(drift_metrics.items(), key=lambda kv: kv[1]["jsd"], reverse=True)[: args.top_k_users]]
    plot_top_drift_users_heatmap(
        top_users,
        user_distributions,
        ordered_categories,
        out_dir / "top_drift_users_categories.png",
        max_categories=args.heatmap_categories,
    )

    LOGGER.info("Loading trained STREAM model (%s) from %s", args.model_type, model_dir)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type == "bert":
        model = BertStreamModel(item_vocab, device)
        pretrained = model.model.__class__.from_pretrained(str(model_dir))  # type: ignore[attr-defined]
        pretrained.to(device)
        model.model = pretrained  # type: ignore[assignment]
        tokenizer = None
    else:
        model = CausalLMStreamModel(
            pretrained_name_or_path=str(model_dir),
            item_vocab=item_vocab,
            device=device,
        )
        tokenizer = model.tokenizer
    model.eval()
    if hasattr(model, "model"):
        model.model.eval()  # type: ignore[operator]

    if args.model_type != "causal":
        tokenizer = None

    LOGGER.info("Preparing dataloader for test split")
    test_records = splits.get("test", [])
    dataset_test, dataloader_test = build_dataloader(
        test_records,
        model_type=args.model_type,
        batch_size=args.eval_batch_size,
        shuffle=False,
        item_vocab=item_vocab,
        tokenizer=tokenizer,
        num_workers=args.num_workers,
    )

    sample_meta_test = build_sample_metadata(test_records, dataset_test, args.model_type)
    if len(sample_meta_test) != len(dataset_test):
        LOGGER.warning(
            "Sample metadata length (%d) does not match dataset size (%d); evaluation results may be unreliable",
            len(sample_meta_test),
            len(dataset_test),
        )

    user_prob_maps = build_user_probability_maps(user_prob_json)
    bucket_metrics, bucket_counts = evaluate_metrics_per_bucket(
        model,
        dataloader_test,
        user_prob_maps,
        category_map,
        sample_meta=sample_meta_test,
        model_type=args.model_type,
        ks=args.metric_ks,
        top_k=args.drift_top_k,
        early_threshold=args.non_drift_threshold,
        drift_low_threshold=args.drift_low_threshold,
        late_threshold=args.drift_late_threshold,
    )
    save_json(bucket_metrics, out_dir / "drift_vs_non_drift_metrics.json")
    save_json(bucket_counts, out_dir / "drift_vs_non_drift_counts.json")
    plot_bucket_metrics(bucket_metrics, bucket_counts, out_dir / "drift_vs_non_drift_metrics.png", metrics_to_plot=args.metrics_plot)

    LOGGER.info("Computing category semantic subspace")
    original_records = splits.get("original", [])
    dataset_original, dataloader_original = build_dataloader(
        original_records,
        model_type=args.model_type,
        batch_size=args.subspace_batch_size,
        shuffle=False,
        item_vocab=item_vocab,
        tokenizer=tokenizer,
        num_workers=args.num_workers,
    )

    subspace = compute_category_semantic_subspace(
        model,
        dataloader_original,
        category_map,
        category_order,
        max_categories=args.subspace_rank,
        device=device,
        model_type=args.model_type,
        fallback_mode="pca",
        min_samples_per_category=args.min_category_samples,
    )
    torch.save(
        {
            "basis": subspace.basis,
            "meta": subspace.meta,
        },
        out_dir / "category_subspace.pt",
    )

    if args.model_type == "causal":
        LOGGER.info("Collecting hidden state trajectories across full timelines")
        original_sequences = gather_user_sequences(original_records)
        finetune_sequences = gather_user_sequences(splits.get("finetune", []))
        test_sequences = gather_user_sequences(test_records)
        combined_records, early_boundaries = merge_user_sequences(
            original_sequences,
            finetune_sequences,
            test_sequences,
        )

        if not combined_records:
            LOGGER.warning("No combined user records available for trajectory analysis")
            LOGGER.info("Analysis complete. Artifacts saved to %s", out_dir)
            return

        dataset_combined, dataloader_combined = build_dataloader(
            combined_records,
            model_type="causal",
            batch_size=args.eval_batch_size,
            shuffle=False,
            item_vocab=item_vocab,
            tokenizer=tokenizer,
            num_workers=args.num_workers,
        )

        user_direction_drift, trajectories = compute_direction_subspace_drift(
            model,
            dataset_combined,
            dataloader_combined,
            subspace.basis.to(device),
            early_boundaries,
            top_users,
        )
        save_json(user_direction_drift, out_dir / "user_direction_drift.json")

        common_users = [user for user in user_direction_drift.keys() if user in drift_metrics]
        jsd_list = [drift_metrics[user]["jsd"] for user in common_users]
        direction_list = [user_direction_drift[user]["delta"] for user in common_users]

        pearson = pearson_correlation(jsd_list, direction_list)
        spearman = spearman_correlation(jsd_list, direction_list)
        correlation_summary = {
            "users": len(common_users),
            "pearson_jsd_direction": pearson,
            "spearman_jsd_direction": spearman,
        }
        save_json(correlation_summary, out_dir / "drift_correlation.json")

        direction_labels: List[str] = []
        meta_categories = subspace.meta.get("categories") if isinstance(subspace.meta, dict) else None
        if isinstance(meta_categories, list):
            for entry in meta_categories:
                label = entry.get("category") if isinstance(entry, dict) else None
                if label:
                    direction_labels.append(str(label))
        while len(direction_labels) < subspace.basis.size(1):
            direction_labels.append(f"direction_{len(direction_labels)}")
        direction_labels = direction_labels[: args.trajectory_directions]

        for user_id in top_users:
            trajectory = trajectories.get(user_id, [])
            if not trajectory:
                continue
            boundary = early_boundaries.get(user_id, 0)
            plot_user_trajectories(
                user_id,
                trajectory,
                direction_labels,
                out_dir / f"user_trajectory_user{user_id}.png",
                boundary=boundary,
            )
    else:
        LOGGER.info(
            "Skipping hidden state trajectory analysis for model type '%s'", args.model_type
        )

    LOGGER.info("Analysis complete. Artifacts saved to %s", out_dir)


if __name__ == "__main__":
    main()
