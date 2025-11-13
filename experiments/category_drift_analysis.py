"""Analysis script for studying category preference drift in STREAM."""
from __future__ import annotations
import os
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
import re
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


@dataclass
class UserInteractionStats:
    """Store interaction counts for each user segment."""

    early: int
    late: int

    @property
    def total(self) -> int:
        return self.early + self.late


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_hf_model_dir(model_root: Path) -> Path:
    # 单文件权重的典型文件名
    single_weight_files = {
        "model.safetensors",
        "pytorch_model.bin",
        "flax_model.msgpack",
        "model.ckpt.index",
        "tf_model.h5",
    }

    def _has_config(path: Path) -> bool:
        return path.is_dir() and (path / "config.json").exists()

    def _looks_like_checkpoint(path: Path) -> bool:
        if not _has_config(path):
            return False

        if any((path / f).exists() for f in single_weight_files):
            return True

        index_ok = (path / "model.safetensors.index.json").exists()
        shard_ok = any(re.match(r"model-\d{5}-of-\d{5}\.safetensors$", p.name)
                       for p in path.glob("model-*-of-*.safetensors"))
        return index_ok and shard_ok

    if _looks_like_checkpoint(model_root):
        return model_root

    candidate = model_root / "model"
    if _looks_like_checkpoint(candidate):
        LOGGER.info("Detected Hugging Face checkpoint inside %s", candidate)
        return candidate

    for child in model_root.iterdir():
        if _looks_like_checkpoint(child):
            LOGGER.info("Detected Hugging Face checkpoint inside %s", child)
            return child

    expected = " or ".join(sorted(list(single_weight_files) + ["model.safetensors.index.json + shards"]))
    raise FileNotFoundError(
        f"Could not find a Hugging Face checkpoint in '{model_root}'. "
        f"Expected config.json and one of [{expected}]."
    )


def compute_user_category_distributions(
    splits: Mapping[str, Sequence[Mapping]],
    category_map: Mapping[int, Sequence[str]],
    category_order: Sequence[str],
    *,
    max_users: Optional[int] = None,
) -> Tuple[
    Dict[str, UserSegmentDistributions],
    List[str],
    Dict[str, Dict[str, Dict[str, float]]],
    Dict[str, UserInteractionStats],
]:
    """Compute per-user early and late category distributions."""

    categories = list(category_order)
    category_index = {cat: idx for idx, cat in enumerate(categories)}

    early_counts: Dict[str, Counter] = defaultdict(Counter)
    late_counts: Dict[str, Counter] = defaultdict(Counter)

    early_interactions: Dict[str, int] = defaultdict(int)
    late_interactions: Dict[str, int] = defaultdict(int)

    def _accumulate(
        records: Iterable[Mapping],
        dest: MutableMapping[str, Counter],
        interaction_dest: MutableMapping[str, int],
    ) -> None:
        for rec in records:
            user_raw = rec.get("user")
            if user_raw is None:
                continue
            user_id = str(user_raw)
            items: Sequence[int] = rec.get("items", [])  # type: ignore[assignment]
            if not items:
                continue
            for item_idx in items:
                interaction_dest[user_id] += 1
                cats = category_map.get(int(item_idx), [])
                if not cats:
                    continue
                weight = 1.0 / float(len(cats))
                for cat in cats:
                    dest[user_id][cat] += weight

    _accumulate(splits.get("original", []), early_counts, early_interactions)
    late_records = list(splits.get("finetune", [])) + list(splits.get("test", []))
    _accumulate(late_records, late_counts, late_interactions)

    all_users = sorted(set(early_counts.keys()) | set(late_counts.keys()), key=lambda x: int(x))
    if max_users is not None:
        all_users = all_users[: max_users]

    user_distributions: Dict[str, UserSegmentDistributions] = {}
    user_prob_json: Dict[str, Dict[str, Dict[str, float]]] = {}
    user_interaction_stats: Dict[str, UserInteractionStats] = {}

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
        early_interactions_count = early_interactions.get(user_id, 0)
        late_interactions_count = late_interactions.get(user_id, 0)
        user_interaction_stats[user_id] = UserInteractionStats(
            early=early_interactions_count,
            late=late_interactions_count,
        )

    return user_distributions, categories, user_prob_json, user_interaction_stats


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


def _items_to_distribution(
    items: Sequence[int],
    category_map: Mapping[int, Sequence[str]],
    category_order: Sequence[str],
) -> Optional[np.ndarray]:
    if not items:
        return None
    category_index = {cat: idx for idx, cat in enumerate(category_order)}
    counts = np.zeros(len(category_order), dtype=np.float64)
    total = 0.0
    for item_idx in items:
        cats = category_map.get(int(item_idx), [])
        if not cats:
            continue
        weight = 1.0 / float(len(cats))
        total += weight
        for cat in cats:
            idx = category_index.get(cat)
            if idx is not None:
                counts[idx] += weight
    if total <= 0:
        return None
    return (counts / total).astype(np.float32)


def compute_randomized_jsd(
    user_interaction_stats: Mapping[str, UserInteractionStats],
    original_sequences: Mapping[str, Sequence[int]],
    finetune_sequences: Mapping[str, Sequence[int]],
    test_sequences: Mapping[str, Sequence[int]],
    category_map: Mapping[int, Sequence[str]],
    category_order: Sequence[str],
    *,
    num_samples: int = 50,
    seed: Optional[int] = None,
) -> List[float]:
    rng = np.random.default_rng(seed)
    randomized_jsd: List[float] = []
    for user_id, stats in user_interaction_stats.items():
        early_len = stats.early
        late_len = stats.late
        if early_len <= 0 or late_len <= 0:
            continue
        combined = (
            list(original_sequences.get(user_id, []))
            + list(finetune_sequences.get(user_id, []))
            + list(test_sequences.get(user_id, []))
        )
        if len(combined) < early_len + late_len:
            continue
        for _ in range(max(1, num_samples)):
            shuffled = combined.copy()
            rng.shuffle(shuffled)
            early_items = shuffled[:early_len]
            late_items = shuffled[early_len : early_len + late_len]
            early_dist = _items_to_distribution(early_items, category_map, category_order)
            late_dist = _items_to_distribution(late_items, category_map, category_order)
            if early_dist is None or late_dist is None:
                continue
            randomized_jsd.append(js_divergence(early_dist, late_dist))
    return randomized_jsd


def summarize_jsd_statistics(
    jsd_map: Mapping[str, float],
    user_interaction_stats: Mapping[str, UserInteractionStats],
    interaction_bins: Sequence[int],
) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    values = [val for val in jsd_map.values() if not math.isnan(val)]
    if values:
        summary["overall"] = {
            "count": float(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
        }
    thresholds = sorted(set(interaction_bins))
    if 0 not in thresholds:
        thresholds.insert(0, 0)
    bins: List[Tuple[int, Optional[int]]] = []
    for idx, lower in enumerate(thresholds):
        upper: Optional[int]
        if idx + 1 < len(thresholds):
            upper = thresholds[idx + 1]
        else:
            upper = None
        bins.append((lower, upper))
    for lower, upper in bins:
        label = f">={lower}" if upper is None else f"{lower}-{upper}"
        bucket_values: List[float] = []
        for user_id, stats in user_interaction_stats.items():
            total = stats.total
            if total < lower:
                continue
            if upper is not None and total >= upper:
                continue
            jsd_val = jsd_map.get(user_id)
            if jsd_val is None or math.isnan(jsd_val):
                continue
            bucket_values.append(jsd_val)
        if not bucket_values:
            continue
        summary[label] = {
            "count": float(len(bucket_values)),
            "mean": float(np.mean(bucket_values)),
            "median": float(np.median(bucket_values)),
            "p95": float(np.percentile(bucket_values, 95)),
        }
    return summary


def save_json(data: Mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def plot_jsd_histograms(
    jsd_map: Mapping[str, float],
    randomized_jsd: Sequence[float],
    user_interaction_stats: Mapping[str, UserInteractionStats],
    out_dir: Path,
    *,
    hist_bins: int = 40,
    interaction_bins: Sequence[int] = (),
    jsd_hist_max: Optional[float] = None,   # <<-- 可用 main() 里 args.jsd_hist_max 传入
) -> None:
    jsd_values = [val for val in jsd_map.values() if not math.isnan(val)]
    if not jsd_values:
        LOGGER.warning("No JSD values available for histogram plot")
        return

    # ===== 计算共享的 bin edges（用于所有图，保证横轴一致）=====
    max_obs = max(jsd_values) if jsd_values else 0.0
    max_base = max(randomized_jsd) if randomized_jsd else 0.0
    x_max = jsd_hist_max if jsd_hist_max is not None else max(max_obs, max_base)
    if x_max <= 0:
        x_max = 0.5  # fallback
    shared_edges = np.linspace(0.0, x_max, hist_bins + 1)

    # ===== 图A：observed vs. baseline =====
    figA, axA = plt.subplots(figsize=(8, 6))
    axA.hist(jsd_values, bins=shared_edges, density=True,
             alpha=0.6, label="observed")
    if randomized_jsd:
        axA.hist(randomized_jsd, bins=shared_edges, density=True,
                 alpha=0.45, label="time-shuffled baseline")
    median_val = float(np.median(jsd_values))
    axA.axvline(median_val, color="tab:green", linestyle="--", linewidth=1.5, label="median")
    axA.set_xlim(0.0, x_max)
    axA.set_xlabel("Jensen-Shannon divergence")
    axA.set_ylabel("Density")
    axA.set_title("User category drift vs. randomized baseline")
    axA.legend(loc="upper right")
    (out_dir / "jsd_histograms").mkdir(parents=True, exist_ok=True)
    figA.tight_layout()
    figA.savefig(out_dir / "jsd_histograms/jsd_vs_baseline_hist.png", dpi=200)
    plt.close(figA)

    # ===== 图B：仅“不同交互数量”分桶；不叠加 baseline，不包含 overall =====
    thresholds = sorted(set(interaction_bins))
    if thresholds and 0 not in thresholds:
        thresholds.insert(0, 0)

    interaction_ranges: List[Tuple[int, Optional[int]]] = []
    if thresholds:
        for idx, lower in enumerate(thresholds):
            upper: Optional[int] = thresholds[idx + 1] if idx + 1 < len(thresholds) else None
            interaction_ranges.append((lower, upper))

    if not interaction_ranges:
        return

    figB, axes = plt.subplots(len(interaction_ranges), 1, figsize=(8, 4 * len(interaction_ranges)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, (lower, upper) in zip(axes, interaction_ranges):
        label = f">={lower}" if upper is None else f"{lower}-{upper}"
        values = []
        for user_id, stats in user_interaction_stats.items():
            total = stats.total
            if total < lower:
                continue
            if upper is not None and total >= upper:
                continue
            jsd_val = jsd_map.get(user_id)
            if jsd_val is None or math.isnan(jsd_val):
                continue
            values.append(jsd_val)
        if not values:
            ax.set_visible(False)
            continue
        ax.hist(values, bins=shared_edges, alpha=0.75)
        ax.set_ylabel("Number of users")
        ax.set_title(f"Users with total interactions in [{label})")

    axes[-1].set_xlim(0.0, x_max)  # 统一横轴范围
    axes[-1].set_xlabel("Jensen-Shannon divergence")
    figB.tight_layout()
    figB.savefig(out_dir / "jsd_histograms/jsd_by_interaction_bins.png", dpi=200)
    plt.close(figB)



def plot_user_category_heatmaps(
    selected_users: Sequence[str],
    user_distributions: Mapping[str, UserSegmentDistributions],
    category_order: Sequence[str],
    out_path: Path,
    *,
    max_categories: int = 20,
    title: str = "Category distributions",
) -> None:
    if not selected_users:
        LOGGER.warning("No users available for heatmap plot")
        return
    num_categories = min(len(category_order), max_categories)
    categories = list(category_order[:num_categories])
    early_matrix: List[np.ndarray] = []
    late_matrix: List[np.ndarray] = []
    valid_users: List[str] = []
    for user_id in selected_users:
        dist = user_distributions.get(user_id)
        if not dist or dist.early is None or dist.late is None:
            continue
        early_matrix.append(dist.early[:num_categories])
        late_matrix.append(dist.late[:num_categories])
        valid_users.append(user_id)
    if not early_matrix:
        LOGGER.warning("No valid distributions for heatmap")
        return
    early_stack = np.stack(early_matrix, axis=0)
    late_stack = np.stack(late_matrix, axis=0)

    fig_width = max(12, num_categories * 0.7)
    fig_height = len(valid_users) * 0.45 + 3
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), sharey=True)
    matrices = [early_stack, late_stack, late_stack - early_stack]
    titles = ["Early segment", "Late segment", "Late - Early"]
    cmaps = ["viridis", "viridis", "coolwarm"]
    for ax, data, subplot_title, cmap in zip(axes, matrices, titles, cmaps):
        if subplot_title == "Late - Early":
            vmax = np.percentile(np.abs(data), 98) if np.any(data) else 0.0
            if vmax <= 0:
                vmax = 0.1
            im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_xticks(range(num_categories))
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.set_yticks(range(len(valid_users)))
        ax.set_yticklabels(valid_users)
        ax.set_title(subplot_title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def select_users_by_jsd(
    drift_metrics: Mapping[str, Mapping[str, float]],
    user_interaction_stats: Mapping[str, UserInteractionStats],
    *,
    count: int,
    strategy: str = "top",
    min_total_interactions: int = 0,
) -> List[str]:
    if count <= 0:
        return []
    candidates: List[Tuple[str, float]] = []
    for user_id, metrics in drift_metrics.items():
        stats = user_interaction_stats.get(user_id)
        if not stats or stats.total < min_total_interactions or stats.early <= 0 or stats.late <= 0:
            continue
        jsd_val = float(metrics.get("jsd", float("nan")))
        if math.isnan(jsd_val):
            continue
        candidates.append((user_id, jsd_val))
    if not candidates:
        return []
    if strategy == "top":
        candidates.sort(key=lambda kv: kv[1], reverse=True)
    elif strategy == "median":
        median_val = float(np.median([val for _, val in candidates]))
        candidates.sort(key=lambda kv: abs(kv[1] - median_val))
    else:
        raise ValueError(f"Unsupported strategy '{strategy}'")
    return [user for user, _ in candidates[:count]]


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


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    num_samples: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        val = float(values[0])
        return val, val
    generator = rng or np.random.default_rng()
    means = np.empty(num_samples, dtype=np.float64)
    for idx in range(num_samples):
        sample_indices = generator.integers(0, values.size, size=values.size)
        means[idx] = float(values[sample_indices].mean())
    lower = float(np.percentile(means, 100 * (alpha / 2.0)))
    upper = float(np.percentile(means, 100 * (1.0 - alpha / 2.0)))
    return lower, upper


def collect_sample_metrics(
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
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []

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
            metrics = compute_ranking_metrics(row, target_item, ks)
            results.append(
                {
                    "user": user_id,
                    "target_item": target_item,
                    "metrics": metrics,
                    "heuristic_bucket": bucket,
                }
            )

    return results


def aggregate_metrics_by_bucket(
    samples: Sequence[Mapping[str, object]],
    metrics_to_plot: Sequence[str],
    bucket_getter,
    *,
    balance_users: bool = False,
    bootstrap_samples: int = 1000,
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, Tuple[float, float]]],
    Dict[str, int],
    Dict[str, int],
]:
    bucket_metric_user_values: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    bucket_user_sets: Dict[str, set[str]] = defaultdict(set)
    bucket_sample_counts: Dict[str, int] = defaultdict(int)

    for sample in samples:
        bucket = bucket_getter(sample)
        if bucket is None:
            continue
        user = str(sample.get("user"))
        if not user:
            continue
        metrics_map = sample.get("metrics")
        if not isinstance(metrics_map, dict):
            continue
        bucket_sample_counts[bucket] += 1
        bucket_user_sets[bucket].add(user)
        for metric_name, value in metrics_map.items():
            bucket_metric_user_values[bucket][metric_name][user].append(float(value))

    if not bucket_metric_user_values:
        return {}, {}, {}, {}

    rng = np.random.default_rng(seed)
    user_counts = {bucket: len(users) for bucket, users in bucket_user_sets.items()}
    min_user_count = (
        min((count for count in user_counts.values() if count > 0)) if balance_users and user_counts else None
    )

    mean_metrics: Dict[str, Dict[str, float]] = {}
    ci_metrics: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for bucket, metric_map in bucket_metric_user_values.items():
        mean_metrics[bucket] = {}
        ci_metrics[bucket] = {}
        for metric_name in metrics_to_plot:
            user_values = metric_map.get(metric_name)
            if not user_values:
                continue
            per_user_means = np.array(
                [np.mean(values) for values in user_values.values()], dtype=np.float64
            )
            if per_user_means.size == 0:
                continue
            if min_user_count is not None and per_user_means.size > min_user_count:
                indices = rng.choice(per_user_means.size, size=min_user_count, replace=False)
                per_user_means = per_user_means[indices]
            mean_metrics[bucket][metric_name] = float(per_user_means.mean())
            lower, upper = bootstrap_mean_ci(
                per_user_means,
                num_samples=bootstrap_samples,
                alpha=alpha,
                rng=rng,
            )
            ci_metrics[bucket][metric_name] = (lower, upper)

    return mean_metrics, ci_metrics, user_counts, dict(bucket_sample_counts)


def assign_jsd_quantile_buckets(
    drift_metrics: Mapping[str, Mapping[str, float]],
    *,
    num_bins: int,
) -> Tuple[Dict[str, str], List[float], List[str]]:
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    jsd_entries: List[Tuple[str, float]] = []
    for user_id, metrics in drift_metrics.items():
        jsd_val = float(metrics.get("jsd", float("nan")))
        if math.isnan(jsd_val):
            continue
        jsd_entries.append((user_id, jsd_val))
    if not jsd_entries:
        return {}, [], []
    values = np.array([val for _, val in jsd_entries], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(values, quantiles)
    edges[0] = float(values.min())
    edges[-1] = float(values.max())
    for idx in range(1, len(edges)):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = edges[idx - 1] + 1e-6
    labels = [f"Q{i + 1} [{edges[i]:.3f}, {edges[i + 1]:.3f}]" for i in range(num_bins)]
    user_buckets: Dict[str, str] = {}
    for user_id, jsd_val in jsd_entries:
        bucket_idx = int(np.searchsorted(edges, jsd_val, side="right") - 1)
        bucket_idx = min(max(bucket_idx, 0), num_bins - 1)
        user_buckets[user_id] = labels[bucket_idx]
    return user_buckets, edges.tolist(), labels


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
    user_counts: Mapping[str, int],
    out_path: Path,
    *,
    metrics_to_plot: Sequence[str],
    ci: Optional[Mapping[str, Mapping[str, Tuple[float, float]]]] = None,
    sample_counts: Optional[Mapping[str, int]] = None,
) -> None:
    if not metrics:
        LOGGER.warning("No metrics available for bucket comparison plot")
        return
    buckets = sorted(metrics.keys())
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(7, 4 * len(metrics_to_plot)))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    colors = plt.cm.tab10.colors
    for ax, metric_name in zip(axes, metrics_to_plot):
        values = [metrics[bucket].get(metric_name, 0.0) for bucket in buckets]
        bars = ax.bar(range(len(buckets)), values, color=[colors[idx % len(colors)] for idx in range(len(buckets))])
        ax.set_xticks(range(len(buckets)))
        ax.set_xticklabels(buckets)
        ax.set_ylabel(metric_name)
        ylim_top = max(values + [0.01]) * 1.1
        ax.set_ylim(0.0, ylim_top)
        for idx, bucket in enumerate(buckets):
            label_parts = [f"users={user_counts.get(bucket, 0)}"]
            if sample_counts:
                label_parts.append(f"samples={sample_counts.get(bucket, 0)}")
            label = ", ".join(label_parts)
            ax.text(idx, values[idx] + ylim_top * 0.02, label, ha="center", va="bottom", fontsize=10)
            if ci:
                interval = ci.get(bucket, {}).get(metric_name)
                if interval:
                    lower, upper = interval
                    if not math.isnan(lower) and not math.isnan(upper):
                        err = np.array([[values[idx] - lower], [upper - values[idx]]])
                        ax.errorbar(
                            idx,
                            values[idx],
                            yerr=err,
                            fmt="none",
                            ecolor="black",
                            elinewidth=1.2,
                            capsize=4,
                        )
        ax.set_title(f"{metric_name} by bucket")
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
        if basis_device.dtype != hidden.dtype:
            basis_device = basis_device.to(hidden.dtype)
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
                        "projection": proj.to(torch.float32).numpy().tolist(),
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
    parser.add_argument("--data_dir", type=Path, default="/home/zj/code/STREAM/ml-1m")
    parser.add_argument("--model_dir", type=Path, default="/home/zj/code/STREAM/ml-1m/bert")
    parser.add_argument("--tokenizer_dir", type=Path, default="/home/zj/code/STREAM/ml-1m/causal/tokenizer",
                        help="Directory that has tokenizer.json/tokenizer.model when model_dir lacks them")
    parser.add_argument("--out_dir", type=Path, default="/home/zj/code/STREAM/ml-1m/bert")
    parser.add_argument("--max_users", type=int, default=None, help="Limit analysis to the first N users")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hist_bins", type=int, default=100, help="Number of bins for JSD histogram")
    parser.add_argument(
        "--interaction_bins",
        type=int,
        nargs="*",
        default=[0, 20, 50, 100],
        help="Interaction count thresholds used to stratify JSD histograms",
    )
    parser.add_argument("--top_k_users", type=int, default=20, help="Number of top drift users to visualize")
    parser.add_argument(
        "--median_k_users",
        type=int,
        default=10,
        help="Number of median-drift users to visualize",
    )
    parser.add_argument("--heatmap_categories", type=int, default=30, help="Maximum categories in heatmap plot")
    parser.add_argument(
        "--heatmap_min_interactions",
        type=int,
        default=20,
        help="Minimum total interactions required for heatmap visualizations",
    )
    parser.add_argument(
        "--random_baseline_samples",
        type=int,
        default=300,
        help="Number of shuffles per user when building the randomized JSD baseline",
    )
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device for model execution (e.g. cuda)")
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert",
        choices=["causal", "bert"],
        help="Model architecture used for analysis",
    )
    parser.add_argument("--drift_top_k", type=int, default=3, help="Top-k early categories considered non-drift")
    parser.add_argument("--non_drift_threshold", type=float, default=0.2, help="Probability threshold for non-drift")
    parser.add_argument("--drift_low_threshold", type=float, default=0.05, help="Maximum early probability for drift classification")
    parser.add_argument("--drift_late_threshold", type=float, default=0.2, help="Minimum late probability for drift classification")
    parser.add_argument("--metric_ks", type=int, nargs="*", default=[5, 10, 20], help="Ranking cutoffs")
    parser.add_argument("--metrics_plot", type=str, nargs="*", default=["ndcg@20", "recall@20"], help="Metrics to visualise in bar plot")
    parser.add_argument(
        "--jsd_quantile_bins",
        type=int,
        default=4,
        help="Number of quantile buckets for JSD-based metric analysis",
    )
    parser.add_argument(
        "--balance_bucket_users",
        action="store_true",
        help="Balance buckets by downsampling to the same number of users before averaging metrics",
    )
    parser.add_argument(
        "--metric_bootstrap_samples",
        type=int,
        default=500,
        help="Number of bootstrap resamples for confidence intervals",
    )
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

    (
        user_distributions,
        ordered_categories,
        user_prob_json,
        user_interaction_stats,
    ) = compute_user_category_distributions(
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

    original_records = splits.get("original", [])
    finetune_records = splits.get("finetune", [])
    test_records = splits.get("test", [])
    original_sequences = gather_user_sequences(original_records)
    finetune_sequences = gather_user_sequences(finetune_records)
    test_sequences = gather_user_sequences(test_records)

    save_json(
        {"category_order": ordered_categories, "users": user_prob_json},
        out_dir / "user_category_distributions.json",
    )
    save_json(drift_metrics, out_dir / "user_category_drift.json")

    jsd_map = {
        user_id: metrics["jsd"]
        for user_id, metrics in drift_metrics.items()
        if not math.isnan(metrics.get("jsd", float("nan")))
    }
    randomized_jsd = compute_randomized_jsd(
        user_interaction_stats,
        original_sequences,
        finetune_sequences,
        test_sequences,
        category_map,
        ordered_categories,
        num_samples=args.random_baseline_samples,
        seed=args.seed,
    )
    plot_jsd_histograms(
        jsd_map, randomized_jsd, user_interaction_stats, out_dir,
        hist_bins=args.hist_bins,
        interaction_bins=args.interaction_bins,
    )

    jsd_summary = summarize_jsd_statistics(jsd_map, user_interaction_stats, args.interaction_bins)
    save_json(jsd_summary, out_dir / "jsd_summary.json")

    top_users = select_users_by_jsd(
        drift_metrics,
        user_interaction_stats,
        count=args.top_k_users,
        strategy="top",
        min_total_interactions=args.heatmap_min_interactions,
    )
    if top_users:
        plot_user_category_heatmaps(
            top_users,
            user_distributions,
            ordered_categories,
            out_dir / "top_drift_users_categories.png",
            max_categories=args.heatmap_categories,
            title="Top drift users (JSD)",
        )
    else:
        LOGGER.warning("No users met the criteria for top-drift heatmap")

    median_users = select_users_by_jsd(
        drift_metrics,
        user_interaction_stats,
        count=args.median_k_users,
        strategy="median",
        min_total_interactions=args.heatmap_min_interactions,
    )
    if median_users:
        plot_user_category_heatmaps(
            median_users,
            user_distributions,
            ordered_categories,
            out_dir / "median_drift_users_categories.png",
            max_categories=args.heatmap_categories,
            title="Median drift users (JSD)",
        )
    else:
        LOGGER.warning("No users met the criteria for median-drift heatmap")

    LOGGER.info("Loading trained STREAM model (%s) from %s", args.model_type, model_dir)
    hf_model_dir = _resolve_hf_model_dir(model_dir)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type == "bert":
        model = BertStreamModel(item_vocab, device)
        pretrained = model.model.__class__.from_pretrained(str(hf_model_dir))  # type: ignore[attr-defined]
        pretrained.to(device)
        model.model = pretrained  # type: ignore[assignment]
        tokenizer = None
    else:
        model = CausalLMStreamModel(
            pretrained_name_or_path=str(hf_model_dir),
            tokenizer_name_or_path = args.tokenizer_dir,
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
    sample_results = collect_sample_metrics(
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

    heuristic_means, heuristic_ci, heuristic_user_counts, heuristic_sample_counts = aggregate_metrics_by_bucket(
        sample_results,
        args.metrics_plot,
        lambda sample: sample.get("heuristic_bucket"),
        balance_users=args.balance_bucket_users,
        bootstrap_samples=args.metric_bootstrap_samples,
        seed=args.seed,
    )

    def _format_ci(ci_dict: Mapping[str, Mapping[str, Tuple[float, float]]]) -> Dict[str, Dict[str, List[float]]]:
        formatted: Dict[str, Dict[str, List[float]]] = {}
        for bucket, metric_map in ci_dict.items():
            formatted[bucket] = {
                metric_name: [float(interval[0]), float(interval[1])] for metric_name, interval in metric_map.items()
            }
        return formatted

    if heuristic_means:
        save_json(
            {
                "means": heuristic_means,
                "confidence_intervals": _format_ci(heuristic_ci),
                "user_counts": heuristic_user_counts,
                "sample_counts": heuristic_sample_counts,
            },
            out_dir / "drift_vs_non_drift_metrics.json",
        )
        save_json(
            {
                "users": heuristic_user_counts,
                "samples": heuristic_sample_counts,
            },
            out_dir / "drift_vs_non_drift_counts.json",
        )
        plot_bucket_metrics(
            heuristic_means,
            heuristic_user_counts,
            out_dir / "drift_vs_non_drift_metrics.png",
            metrics_to_plot=args.metrics_plot,
            ci=heuristic_ci,
            sample_counts=heuristic_sample_counts,
        )
    else:
        LOGGER.warning("No samples assigned to heuristic drift buckets")

    user_quantile_buckets, quantile_edges, quantile_labels = assign_jsd_quantile_buckets(
        drift_metrics,
        num_bins=args.jsd_quantile_bins,
    )
    quantile_means, quantile_ci, quantile_user_counts, quantile_sample_counts = aggregate_metrics_by_bucket(
        sample_results,
        args.metrics_plot,
        lambda sample: user_quantile_buckets.get(str(sample.get("user"))),
        balance_users=args.balance_bucket_users,
        bootstrap_samples=args.metric_bootstrap_samples,
        seed=args.seed,
    )
    if quantile_means:
        save_json(
            {
                "means": quantile_means,
                "confidence_intervals": _format_ci(quantile_ci),
                "user_counts": quantile_user_counts,
                "sample_counts": quantile_sample_counts,
                "quantile_edges": quantile_edges,
                "labels": quantile_labels,
            },
            out_dir / "jsd_quantile_metrics.json",
        )
        plot_bucket_metrics(
            quantile_means,
            quantile_user_counts,
            out_dir / "jsd_quantile_metrics.png",
            metrics_to_plot=args.metrics_plot,
            ci=quantile_ci,
            sample_counts=quantile_sample_counts,
        )
    else:
        LOGGER.warning("Unable to compute JSD quantile bucket metrics")

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
