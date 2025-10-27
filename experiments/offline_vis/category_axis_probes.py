"""Alignment and controllability visualisations for STREAM offline checkpoints."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

from stream.dataio import ItemVocab, build_dataloader, load_all_splits
from stream.models.bert_stream import BertStreamModel
from stream.models.causal_lm_stream import CausalLMStreamModel
from stream.state_adapter import ItemHead
from stream.subspace import SubspaceResult
from stream.train_offline import (
    extract_targets_from_batch,
    load_item_categories,
    move_batch_to_device,
)
from stream.utils import set_seed

sns.set_context("talk")


@dataclass
class AlignmentResult:
    cos_matrix: torch.Tensor
    row_labels: List[str]
    col_labels: List[str]
    metrics: Dict[str, float]
    counts: Dict[str, int]


@dataclass
class ControlResult:
    alphas: List[float]
    delta_logit_target: List[float]
    delta_logit_other: List[float]
    delta_ndcg_target: List[float]
    delta_ndcg_other: List[float]
    hit_rates: Dict[float, Dict[int, Dict[str, float]]]


# ---------------------------------------------------------------------------
# Artifact loading helpers
# ---------------------------------------------------------------------------


def load_subspace(artifacts_dir: Path, device: torch.device) -> Tuple[SubspaceResult, torch.Tensor]:
    """Load the stored subspace basis from ``artifacts_dir``."""

    subspace_path = artifacts_dir / "subspace" / "subspace_U.pt"
    payload = torch.load(subspace_path, map_location=device)
    basis = payload["basis"].to(device)
    result = SubspaceResult(basis=basis.cpu(), mode=payload.get("mode", "gradcov"), meta=payload.get("meta", {}))
    return result, basis


def load_item_head(artifacts_dir: Path, device: torch.device) -> ItemHead:
    """Load the learned item head weights."""

    head_path = artifacts_dir / "item_head.pt"
    payload = torch.load(head_path, map_location=device)
    rank = int(payload.get("rank"))
    num_items = int(payload.get("num_items"))
    head = ItemHead(rank=rank, num_items=num_items, device=device)
    head.load_state_dict(payload["W"])
    head.eval()
    return head


def load_trained_model(
    artifacts_dir: Path,
    model_type: str,
    item_vocab: ItemVocab,
    device: torch.device,
) -> Tuple[torch.nn.Module, Optional[object]]:
    """Instantiate and load the trained base model."""

    if model_type == "bert":
        model = BertStreamModel(item_vocab, device)
        model_dir = artifacts_dir / "model"
        model.model.from_pretrained(model_dir)  # type: ignore[operator]
        model.model.to(device)
        tokenizer = None
    else:
        model_dir = artifacts_dir / "model"
        tokenizer_dir = artifacts_dir / "tokenizer"
        tokenizer_path = str(tokenizer_dir) if tokenizer_dir.exists() else None
        model = CausalLMStreamModel(
            pretrained_name_or_path=str(model_dir),
            item_vocab=item_vocab,
            device=device,
            tokenizer_name_or_path=tokenizer_path,
        )
        tokenizer = model.tokenizer
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Experiment A: alignment heatmap
# ---------------------------------------------------------------------------


def collect_category_gradients(
    model,
    dataloader,
    category_map: Mapping[int, Sequence[str]],
    model_type: str,
    device: torch.device,
    max_batches: Optional[int] = None,
    min_samples: int = 20,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Aggregate whitened category gradients aligned with the training objective."""

    category_sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    processed = 0
    feature_dim: Optional[int] = None
    total_sum: Optional[torch.Tensor] = None
    total_sq_sum: Optional[torch.Tensor] = None
    total_count = 0

    for batch in tqdm(dataloader, desc="collect-gradients", leave=False):
        batch_device = move_batch_to_device(batch, device)
        grads = model.stream_positive_gradients(batch_device)
        targets = extract_targets_from_batch(batch_device, model_type)
        if grads is None:
            continue
        grads = grads.detach().to(device)

        if feature_dim is None:
            feature_dim = grads.size(-1)
            total_sum = torch.zeros(feature_dim, device=device)
            total_sq_sum = torch.zeros(feature_dim, device=device)
        assert total_sum is not None and total_sq_sum is not None

        total_sum += grads.sum(dim=0)
        total_sq_sum += (grads * grads).sum(dim=0)
        total_count += grads.size(0)

        for grad, target_idx in zip(grads, targets.tolist()):
            if target_idx < 0:
                continue
            categories = category_map.get(int(target_idx), [])
            if not categories:
                continue
            for category in categories:
                if category not in category_sums:
                    category_sums[category] = torch.zeros_like(grad)
                    counts[category] = 0
                category_sums[category] += grad
                counts[category] += 1
        processed += 1
        if max_batches is not None and processed >= max_batches:
            break

    if total_count == 0 or feature_dim is None or total_sum is None or total_sq_sum is None:
        return {}, {}

    # whitening statistics
    global_mean = total_sum / float(total_count)
    global_var = total_sq_sum / float(total_count) - global_mean * global_mean
    global_std = torch.sqrt(torch.clamp(global_var, min=1e-6))

    # build whitened category deltas: E[g|c] - E[g|~c], then / std
    category_deltas: Dict[str, torch.Tensor] = {}
    for category, total in category_sums.items():
        cnt = counts[category]
        if cnt < max(min_samples, 1):
            continue
        mean_grad = total / float(cnt)
        rest_cnt = total_count - cnt
        if rest_cnt <= 0:
            continue
        rest_mean = (total_sum - total) / float(rest_cnt)
        delta = mean_grad - rest_mean
        whitened = delta / global_std
        category_deltas[category] = whitened

    filtered_counts = {c: counts[c] for c in category_deltas.keys()}
    return category_deltas, filtered_counts

def compute_alignment(
    basis: torch.Tensor,
    subspace_meta: Mapping[str, object],
    category_vectors: Mapping[str, torch.Tensor],
    category_counts: Mapping[str, int],
) -> AlignmentResult:
    """Project category vectors (whitened deltas) onto the learned basis."""

    directions = [basis[:, i] for i in range(basis.size(1))]
    direction_labels: List[str] = []
    meta_categories = subspace_meta.get("categories") if subspace_meta else None
    if isinstance(meta_categories, list) and meta_categories:
        for idx, entry in enumerate(meta_categories):
            label = str(entry.get("category", f"dir_{idx+1}"))
            direction_labels.append(label)
    else:
        direction_labels = [f"dir_{i+1}" for i in range(len(directions))]

    category_list = direction_labels if direction_labels else sorted(category_vectors.keys())
    category_list = [c for c in category_list if c in category_vectors]
    if not category_list:
        category_list = sorted(category_vectors.keys())

    row_labels = [
        direction_labels[i] if i < len(direction_labels) else f"dir_{i+1}" for i in range(len(directions))
    ]
    cos_matrix = torch.zeros(len(directions), len(category_list))
    for i, direction in enumerate(directions):
        direction = direction / direction.norm().clamp_min(1e-8)
        for j, category in enumerate(category_list):
            vec = category_vectors[category]
            vec = vec / vec.norm().clamp_min(1e-8)
            cos_matrix[i, j] = torch.dot(direction, vec).item()

    # diagnostics
    limit = min(cos_matrix.size(0), cos_matrix.size(1))
    diag = torch.diagonal(cos_matrix[:limit, :limit])
    energy_total = float(torch.sum(cos_matrix ** 2).item())
    energy_diag = float(torch.sum(diag ** 2).item())
    diag_ratio = energy_diag / energy_total if energy_total > 0 else float("nan")
    off_diag = cos_matrix.clone()
    for idx in range(limit):
        off_diag[idx, idx] = 0.0
    max_cross = float(torch.max(off_diag.abs()).item()) if off_diag.numel() else 0.0
    mean_cross = float(off_diag.abs().mean().item()) if off_diag.numel() else 0.0
    mean_diag = float(diag.mean().item()) if diag.numel() else float("nan")
    mean_abs_diag = float(diag.abs().mean().item()) if diag.numel() else float("nan")
    mean_abs_off = float(off_diag.abs().mean().item()) if off_diag.numel() else float("nan")

    metrics = {
        "diag_ratio": diag_ratio,
        "max_cross": max_cross,
        "mean_cross": mean_cross,
        "mean_diag": mean_diag,
        "mean_abs_diag": mean_abs_diag,
        "mean_abs_offdiag": mean_abs_off,
        "num_directions": int(cos_matrix.size(0)),
        "num_categories": int(cos_matrix.size(1)),
    }
    return AlignmentResult(
        cos_matrix=cos_matrix,
        row_labels=row_labels,
        col_labels=category_list,
        metrics=metrics,
        counts={c: int(category_counts.get(c, 0)) for c in category_list},
    )

def plot_alignment_heatmap(result: AlignmentResult, output_path: Path) -> None:
    plt.figure(figsize=(0.8 * len(result.col_labels) + 4, 0.8 * len(result.row_labels) + 4))
    sns.heatmap(
        result.cos_matrix.cpu().numpy(),
        xticklabels=result.col_labels,
        yticklabels=result.row_labels,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
    )
    plt.title("Experiment A: Direction vs Category Cosine")
    plt.xlabel("Category")
    plt.ylabel("Direction")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Experiment B: controllability curves
# ---------------------------------------------------------------------------


def build_category_lookup(category_map: Mapping[int, Sequence[str]]) -> Dict[int, List[str]]:
    return {int(idx): list(categories) for idx, categories in category_map.items()}


def evaluate_direction_control(
    model,
    item_head: ItemHead,
    direction_index: int,
    alphas: Sequence[float],
    topk: Sequence[int],
    dataloader,
    category_lookup: Mapping[int, Sequence[str]],
    target_category: str,
    model_type: str,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> ControlResult:
    direction_vec = item_head.W[direction_index].detach().to(device)
    topk = sorted({int(k) for k in topk})
    alphas = sorted({float(a) for a in alphas})
    if 0.0 not in alphas:
        alphas = [0.0] + alphas
    groups = ("target", "other")
    sums_logit: Dict[float, Dict[str, float]] = {alpha: {g: 0.0 for g in groups} for alpha in alphas}
    sums_ndcg: Dict[float, Dict[str, float]] = {alpha: {g: 0.0 for g in groups} for alpha in alphas}
    hit_sums: Dict[float, Dict[int, Dict[str, float]]] = {
        alpha: {k: {g: 0.0 for g in groups} for k in topk} for alpha in alphas
    }
    counts: Dict[str, int] = {g: 0 for g in groups}
    processed = 0
    for batch in tqdm(dataloader, desc="experiment-b", leave=False):
        batch_device = move_batch_to_device(batch, device)
        with torch.no_grad():
            base_logits = model.stream_base_logits(batch_device)
        targets = extract_targets_from_batch(batch_device, model_type)
        valid_mask = targets >= 0
        if valid_mask.sum().item() == 0:
            processed += 1
            if max_batches is not None and processed >= max_batches:
                break
            continue
        base_logits = base_logits[valid_mask]
        targets = targets[valid_mask].to(device)
        batch_size = targets.size(0)
        batch_indices = torch.arange(batch_size, device=device)
        base_scores = base_logits[batch_indices, targets]
        base_ranks = (base_logits >= base_scores.unsqueeze(1)).sum(dim=1)
        base_ndcg = 1.0 / torch.log2(base_ranks.float() + 1.0)
        category_flags = torch.tensor(
            [target_category in category_lookup.get(int(idx), []) for idx in targets.cpu().tolist()],
            device=device,
            dtype=torch.bool,
        )
        if category_flags.any():
            counts["target"] += int(category_flags.sum().item())
        if (~category_flags).any():
            counts["other"] += int((~category_flags).sum().item())
        direction_targets = direction_vec[targets]
        for alpha in alphas:
            offset = direction_vec * alpha
            perturbed = base_logits + offset
            target_scores = base_scores + alpha * direction_targets
            ranks = (perturbed >= target_scores.unsqueeze(1)).sum(dim=1)
            ndcg = 1.0 / torch.log2(ranks.float() + 1.0)
            delta_ndcg = ndcg - base_ndcg
            delta_logit = alpha * direction_targets
            hits = {k: (ranks <= k) for k in topk}
            for group_name, mask in [("target", category_flags), ("other", ~category_flags)]:
                num = int(mask.sum().item())
                if num == 0:
                    continue
                sums_logit[alpha][group_name] += float(delta_logit[mask].sum().item())
                sums_ndcg[alpha][group_name] += float(delta_ndcg[mask].sum().item())
                for k in topk:
                    hit_sums[alpha][k][group_name] += float(hits[k][mask].float().sum().item())
        processed += 1
        if max_batches is not None and processed >= max_batches:
            break
    if counts["target"] == 0 or counts["other"] == 0:
        raise RuntimeError("Insufficient samples for target or comparison categories.")
    delta_logit_target = [sums_logit[alpha]["target"] / counts["target"] for alpha in alphas]
    delta_logit_other = [sums_logit[alpha]["other"] / counts["other"] for alpha in alphas]
    delta_ndcg_target = [sums_ndcg[alpha]["target"] / counts["target"] for alpha in alphas]
    delta_ndcg_other = [sums_ndcg[alpha]["other"] / counts["other"] for alpha in alphas]
    hit_rates: Dict[float, Dict[int, Dict[str, float]]] = {}
    for alpha in alphas:
        hit_rates[alpha] = {}
        for k in topk:
            hit_rates[alpha][k] = {
                "target": hit_sums[alpha][k]["target"] / counts["target"],
                "other": hit_sums[alpha][k]["other"] / counts["other"],
            }
    return ControlResult(
        alphas=list(alphas),
        delta_logit_target=delta_logit_target,
        delta_logit_other=delta_logit_other,
        delta_ndcg_target=delta_ndcg_target,
        delta_ndcg_other=delta_ndcg_other,
        hit_rates=hit_rates,
    )


def plot_dose_response(result: ControlResult, target_category: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax1.plot(result.alphas, result.delta_logit_target, marker="o", label=f"{target_category} Δlogit", color="tab:blue")
    ax1.plot(result.alphas, result.delta_logit_other, marker="o", linestyle="--", label="Other Δlogit", color="tab:blue")
    ax1.set_xlabel("Perturbation strength α")
    ax1.set_ylabel("Δlogit", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(result.alphas, result.delta_ndcg_target, marker="s", label=f"{target_category} ΔNDCG", color="tab:orange")
    ax2.plot(result.alphas, result.delta_ndcg_other, marker="s", linestyle="--", label="Other ΔNDCG", color="tab:orange")
    ax2.set_ylabel("ΔNDCG", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    plt.title(f"Experiment B: Dose–response for {target_category}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_topk_gain(result: ControlResult, target_category: str, topk: Sequence[int], output_path: Path) -> None:
    baseline_alpha = result.alphas[0]
    reference_alpha = result.alphas[-1]
    baseline_hits = result.hit_rates[baseline_alpha]
    reference_hits = result.hit_rates[reference_alpha]
    improvements = [reference_hits[k]["target"] - baseline_hits[k]["target"] for k in topk]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=[str(k) for k in topk], y=improvements, color="tab:green")
    plt.xlabel("Top-K")
    plt.ylabel("Hit-rate uplift")
    plt.title(f"Experiment B: Top-K gain for {target_category} (α={reference_alpha:.2f})")
    plt.axhline(0.0, color="black", linewidth=0.8)
    for idx, val in enumerate(improvements):
        offset = 0.001 if val >= 0 else -0.001
        plt.text(idx, val + offset, f"{val:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def run_experiments(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    set_seed(args.seed)
    item_vocab = ItemVocab.from_metadata(args.data_dir)
    category_map, _ = load_item_categories(args.data_dir, item_vocab)
    model, tokenizer = load_trained_model(args.artifacts_dir, args.model_type, item_vocab, device)
    subspace, basis = load_subspace(args.artifacts_dir, device)
    item_head = load_item_head(args.artifacts_dir, device)

    splits = load_all_splits(args.data_dir)
    _, eval_loader = build_dataloader(
        splits["original"],
        model_type=args.model_type,
        batch_size=args.batch_size,
        shuffle=False,
        item_vocab=item_vocab,
        tokenizer=tokenizer,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    category_means, counts = collect_category_gradients(
        model,
        eval_loader,
        category_map,
        args.model_type,
        device,
        max_batches=args.alignment_batches,
    )
    if not category_means:
        raise RuntimeError("No category gradients were collected; check data and metadata availability.")

    alignment = compute_alignment(basis, subspace.meta, category_means, counts)

    heatmap_path = output_dir / "experiment_a_heatmap.png"
    plot_alignment_heatmap(alignment, heatmap_path)
    metrics_path = output_dir / "experiment_a_metrics.json"
    summary_a = {
        "metrics": alignment.metrics,
        "category_counts": {cat: alignment.counts.get(cat, 0) for cat in alignment.col_labels},
        "direction_labels": alignment.row_labels,
        "category_labels": alignment.col_labels,
    }
    metrics_path.write_text(json.dumps(summary_a, indent=2), encoding="utf-8")

    direction_labels = alignment.col_labels
    if args.target_category not in direction_labels:
        raise ValueError(
            f"Target category '{args.target_category}' not present in alignment heatmap columns: {direction_labels}"
        )
    direction_index = direction_labels.index(args.target_category)
    topk_values = sorted({int(k) for k in args.topk})
    control = evaluate_direction_control(
        model=model,
        item_head=item_head,
        direction_index=direction_index,
        alphas=args.alphas,
        topk=topk_values,
        dataloader=eval_loader,
        category_lookup=build_category_lookup(category_map),
        target_category=args.target_category,
        model_type=args.model_type,
        device=device,
        max_batches=args.evaluation_batches,
    )
    dose_response_path = output_dir / f"experiment_b_dose_response_{args.target_category}.png"
    plot_dose_response(control, args.target_category, dose_response_path)
    topk_path = output_dir / f"experiment_b_topk_gain_{args.target_category}.png"
    plot_topk_gain(control, args.target_category, topk_values, topk_path)
    control_summary = {
        "alphas": control.alphas,
        "delta_logit_target": control.delta_logit_target,
        "delta_logit_other": control.delta_logit_other,
        "delta_ndcg_target": control.delta_ndcg_target,
        "delta_ndcg_other": control.delta_ndcg_other,
        "topk": topk_values,
        "hit_rates": {
            str(alpha): {
                str(k): {group: float(value) for group, value in rates.items()} for k, rates in k_dict.items()
            }
            for alpha, k_dict in control.hit_rates.items()
        },
        "target_category": args.target_category,
        "direction_index": direction_index,
    }
    summary_path = output_dir / f"experiment_b_summary_{args.target_category}.json"
    summary_path.write_text(json.dumps(control_summary, indent=2), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STREAM category-axis probes")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to ml-10M100K directory")
    parser.add_argument("--artifacts_dir", type=Path, required=True, help="Offline training output directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="Where to store experiment artefacts")
    parser.add_argument("--model_type", choices=["bert", "causal"], default="bert")
    parser.add_argument("--target_category", type=str, required=True)
    parser.add_argument("--alphas", type=float, nargs="*", default=[0.0, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--topk", type=int, nargs="*", default=[5, 10, 20])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--alignment_batches", type=int, default=None, help="Limit batches for Experiment A")
    parser.add_argument("--evaluation_batches", type=int, default=None, help="Limit batches for Experiment B")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_experiments(args)


if __name__ == "__main__":
    main()
