"""Evaluate how category-aligned directions affect ranking metrics across many samples."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Mapping, Optional, Sequence

import torch

from stream.dataio import ItemVocab, build_dataloader, load_all_splits
from stream.models.bert_stream import BertStreamModel
from stream.models.causal_lm_stream import CausalLMStreamModel
from stream.train_offline import (
    extract_targets_from_batch,
    load_item_categories,
    move_batch_to_device,
)
from stream.utils import set_seed


@dataclass
class CategoryDirection:
    vector: torch.Tensor
    index: int
    name: str
    meta: Mapping[str, object]


@dataclass
class StatsAccumulator:
    samples: int = 0
    category_mass_sum: float = 0.0
    valid_total: int = 0
    valid_in: int = 0
    valid_out: int = 0
    hits_total: float = 0.0
    hits_in: float = 0.0
    hits_out: float = 0.0
    target_prob_sum_in: float = 0.0
    target_prob_sum_out: float = 0.0
    target_logit_sum_in: float = 0.0
    target_logit_sum_out: float = 0.0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        category_mask: torch.Tensor,
        top_k: int,
    ) -> None:
        self.samples += int(logits.size(0))
        probs = torch.softmax(logits, dim=-1)
        self.category_mass_sum += float(probs[:, category_mask].sum().item())

        valid_mask = targets >= 0
        if not torch.any(valid_mask):
            return

        logits_valid = logits[valid_mask]
        probs_valid = probs[valid_mask]
        targets_valid = targets[valid_mask]
        batch_indices = torch.arange(targets_valid.size(0), device=logits.device)

        actual_top_k = min(top_k, logits.size(1))
        topk_indices = torch.topk(logits_valid, k=actual_top_k, dim=1).indices
        hits = torch.any(topk_indices == targets_valid.unsqueeze(1), dim=1)

        target_probs = probs_valid[batch_indices, targets_valid]
        target_logits = logits_valid[batch_indices, targets_valid]
        in_mask = category_mask[targets_valid]
        out_mask = ~in_mask

        self.valid_total += int(targets_valid.size(0))
        self.hits_total += float(hits.float().sum().item())

        if torch.any(in_mask):
            self.valid_in += int(in_mask.sum().item())
            self.hits_in += float(hits[in_mask].float().sum().item())
            self.target_prob_sum_in += float(target_probs[in_mask].sum().item())
            self.target_logit_sum_in += float(target_logits[in_mask].sum().item())
        if torch.any(out_mask):
            self.valid_out += int(out_mask.sum().item())
            self.hits_out += float(hits[out_mask].float().sum().item())
            self.target_prob_sum_out += float(target_probs[out_mask].sum().item())
            self.target_logit_sum_out += float(target_logits[out_mask].sum().item())


def _load_model(
    artifacts_dir: Path,
    model_type: str,
    item_vocab: ItemVocab,
    device: torch.device,
):
    if model_type == "bert":
        model = BertStreamModel(item_vocab, device)
        model_dir = artifacts_dir / "model"
        model.model.from_pretrained(model_dir)  # type: ignore[operator]
        model.model.to(device)
        tokenizer = None
    elif model_type == "causal":
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
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    model.eval()
    return model, tokenizer


def _load_category_direction(
    artifacts_dir: Path,
    requested_category: Optional[str],
    direction_index: Optional[int],
    device: torch.device,
) -> CategoryDirection:
    payload = torch.load(artifacts_dir / "subspace_U.pt", map_location=device)
    basis: torch.Tensor = payload["basis"].to(device)
    meta: Mapping[str, object] = payload.get("meta", {})
    categories_meta: List[Mapping[str, object]] = list(meta.get("categories", []))  # type: ignore[arg-type]

    name = None
    index: Optional[int] = None
    if requested_category and categories_meta:
        name_to_idx = {
            str(entry.get("category")): idx for idx, entry in enumerate(categories_meta) if "category" in entry
        }
        index = name_to_idx.get(requested_category)
        if index is None:
            raise ValueError(
                f"Category '{requested_category}' not found in subspace metadata. "
                f"Available categories: {sorted(name_to_idx.keys())}"
            )
        name = requested_category
    if index is None and direction_index is not None:
        index = int(direction_index)
    if index is None:
        index = 0
    if index < 0 or index >= basis.size(1):
        raise ValueError(f"Direction index {index} is out of range for basis with shape {tuple(basis.shape)}")
    if name is None:
        if categories_meta and index < len(categories_meta):
            name = str(categories_meta[index].get("category", f"direction_{index}"))
        else:
            name = f"direction_{index}"
    meta_entry: Mapping[str, object]
    if categories_meta and index < len(categories_meta):
        meta_entry = categories_meta[index]
    else:
        meta_entry = {"category": name}
    vector = basis[:, index]
    return CategoryDirection(vector=vector, index=index, name=name, meta=meta_entry)


def _load_item_linear_params(model) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(model, CausalLMStreamModel):
        weight = model.lm_head_weight[model.item_token_ids].detach()
        bias = None
    elif isinstance(model, BertStreamModel):
        weight = model.decoder_weight.detach()
        bias = model.decoder_bias.detach()
    else:
        raise TypeError(f"Unsupported model class {type(model)}")
    return weight, bias


def _compute_logits(hidden: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    logits = hidden @ weight.t()
    if bias is not None:
        logits = logits + bias
    return logits


def _select_category_items(
    category_map: Mapping[int, Sequence[str]],
    category_name: str,
    *,
    max_items: Optional[int] = None,
) -> List[int]:
    matched = [idx for idx, categories in category_map.items() if category_name in categories]
    if not matched:
        raise ValueError(f"No items found for category '{category_name}'")
    if max_items is not None:
        matched = matched[:max_items]
    return matched


def _summarise(stats: StatsAccumulator) -> Dict[str, Optional[float]]:
    def safe_div(numerator: float, denominator: int) -> Optional[float]:
        return numerator / denominator if denominator > 0 else None

    summary: Dict[str, Optional[float]] = {}
    summary["category_mass"] = stats.category_mass_sum / stats.samples if stats.samples else None
    summary["recall_all"] = safe_div(stats.hits_total, stats.valid_total)
    summary["recall_in"] = safe_div(stats.hits_in, stats.valid_in)
    summary["recall_out"] = safe_div(stats.hits_out, stats.valid_out)
    summary["target_prob_in"] = safe_div(stats.target_prob_sum_in, stats.valid_in)
    summary["target_prob_out"] = safe_div(stats.target_prob_sum_out, stats.valid_out)
    total_prob_sum = stats.target_prob_sum_in + stats.target_prob_sum_out
    summary["target_prob_all"] = safe_div(total_prob_sum, stats.valid_total)
    summary["target_logit_in"] = safe_div(stats.target_logit_sum_in, stats.valid_in)
    summary["target_logit_out"] = safe_div(stats.target_logit_sum_out, stats.valid_out)
    total_logit_sum = stats.target_logit_sum_in + stats.target_logit_sum_out
    summary["target_logit_all"] = safe_div(total_logit_sum, stats.valid_total)
    return summary


def _format_value(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "   n/a"
    return f"{value:8.4f}"


def _format_random(values: List[Optional[float]]) -> str:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return "   n/a"
    if len(filtered) == 1:
        return f"{filtered[0]:8.4f} ± 0.0000"
    return f"{mean(filtered):8.4f} ± {pstdev(filtered):6.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantitatively compare category directions against random baselines across many samples",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/zj/code/STREAM/ml-1m"),
        help="Dataset directory used for offline training",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=Path,
        default=Path("/home/zj/code/STREAM/ml-1m/bert"),
        help="Directory containing offline training artifacts",
    )
    parser.add_argument("--model_type", choices=["causal", "bert"], default="bert", help="Base model architecture")
    parser.add_argument("--category", type=str, required=True, help="Category name to evaluate")
    parser.add_argument(
        "--direction_index",
        type=int,
        default=None,
        help="Fallback direction index if category metadata is unavailable",
    )
    parser.add_argument("--split", type=str, default="original", help="Dataset split to draw samples from")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for the evaluation dataloader")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1024,
        help="Number of samples to aggregate over (set to 0 to evaluate the full split)",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Top-k threshold for recall computation")
    parser.add_argument("--alpha", type=float, default=0.8, help="Scaling applied to the category direction")
    parser.add_argument(
        "--random_trials",
        type=int,
        default=5,
        help="Number of random directions to compare against",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to run on")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("top_k must be positive")
    if args.random_trials < 0:
        raise ValueError("random_trials must be non-negative")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    set_seed(args.seed)

    item_vocab = ItemVocab.from_metadata(args.data_dir)
    model, tokenizer = _load_model(args.artifacts_dir, args.model_type, item_vocab, device)

    splits = load_all_splits(args.data_dir)
    if args.split not in splits:
        raise ValueError(f"Split '{args.split}' not found. Available splits: {sorted(splits.keys())}")
    records = splits[args.split]

    _, dataloader = build_dataloader(
        records,
        model_type=args.model_type,
        batch_size=args.batch_size,
        shuffle=False,
        item_vocab=item_vocab,
        tokenizer=tokenizer,
        num_workers=0,
    )

    category_map, _ = load_item_categories(args.data_dir, item_vocab)
    category_direction = _load_category_direction(
        args.artifacts_dir,
        args.category,
        args.direction_index,
        device,
    )

    category_items = _select_category_items(category_map, category_direction.name)
    category_mask = torch.zeros(item_vocab.num_items, dtype=torch.bool, device=device)
    category_mask[torch.tensor(category_items, device=device)] = True

    weight, bias = _load_item_linear_params(model)
    weight = weight.to(device)
    bias = bias.to(device) if bias is not None else None

    direction_vec = category_direction.vector.to(device, dtype=weight.dtype)
    direction_logits = direction_vec @ weight.t()
    direction_shift = (args.alpha * direction_logits).unsqueeze(0)

    random_shifts: List[torch.Tensor] = []
    direction_norm = direction_vec.norm().clamp_min(1e-6)
    for _ in range(args.random_trials):
        random_vec = torch.randn_like(direction_vec)
        random_vec = random_vec / random_vec.norm().clamp_min(1e-6) * direction_norm
        random_logits = random_vec @ weight.t()
        random_shifts.append((args.alpha * random_logits).unsqueeze(0))

    base_stats = StatsAccumulator()
    direction_stats = StatsAccumulator()
    random_stats = [StatsAccumulator() for _ in range(args.random_trials)]

    processed = 0
    max_samples = None if args.num_samples <= 0 else args.num_samples

    for batch in dataloader:
        batch_device = move_batch_to_device(batch, device)
        hidden = model.stream_hidden_states(batch_device)  # [B, D]
        logits_base = _compute_logits(hidden, weight, bias)
        targets = extract_targets_from_batch(batch, args.model_type).to(device)

        batch_size = logits_base.size(0)
        if max_samples is not None:
            remaining = max_samples - processed
            if remaining <= 0:
                break
            if remaining < batch_size:
                logits_base = logits_base[:remaining]
                targets = targets[:remaining]
                hidden = hidden[:remaining]
                batch_size = remaining
        processed += batch_size

        base_stats.update(logits_base, targets, category_mask, args.top_k)
        logits_direction = logits_base + direction_shift
        direction_stats.update(logits_direction, targets, category_mask, args.top_k)

        for trial_shift, stats in zip(random_shifts, random_stats):
            logits_random = logits_base + trial_shift
            stats.update(logits_random, targets, category_mask, args.top_k)

        if max_samples is not None and processed >= max_samples:
            break

    if base_stats.samples == 0:
        raise RuntimeError("No samples were processed. Check the split name or num_samples setting.")

    base_summary = _summarise(base_stats)
    direction_summary = _summarise(direction_stats)
    random_summaries = [_summarise(stats) for stats in random_stats]

    direction_logits = direction_logits.detach().cpu()
    category_mask_cpu = category_mask.detach().cpu()
    in_logits = direction_logits[category_mask_cpu]
    out_logits = direction_logits[~category_mask_cpu]

    print("=== Category direction comparison ===")
    print(f"Category: {category_direction.name} (index={category_direction.index})")
    print(f"Total samples processed: {base_stats.samples}")
    print(
        "Valid targets: "
        f"{base_stats.valid_total} (in-category={base_stats.valid_in}, out-of-category={base_stats.valid_out})"
    )
    print(f"Direction α: {args.alpha}")
    print("-- Direction logit shift statistics --")
    print(f"Items in category: {len(in_logits)}")
    print(f"  Mean shift: {in_logits.mean().item():.4f}")
    print(f"  Median shift: {in_logits.median().item():.4f}")
    print(f"  Positive ratio: {(in_logits > 0).float().mean().item():.4f}")
    if out_logits.numel() > 0:
        print(f"Items outside category: {len(out_logits)}")
        print(f"  Mean shift: {out_logits.mean().item():.4f}")
        print(f"  Median shift: {out_logits.median().item():.4f}")
        print(f"  Positive ratio: {(out_logits > 0).float().mean().item():.4f}")

    metrics = [
        (f"Recall@{args.top_k} (targets in category)", "recall_in"),
        (f"Recall@{args.top_k} (targets outside)", "recall_out"),
        ("Recall@{} (all targets)".format(args.top_k), "recall_all"),
        ("Target prob (in-category targets)", "target_prob_in"),
        ("Target prob (out-of-category targets)", "target_prob_out"),
        ("Target prob (all targets)", "target_prob_all"),
        ("Target logit (in-category targets)", "target_logit_in"),
        ("Target logit (out-of-category targets)", "target_logit_out"),
        ("Target logit (all targets)", "target_logit_all"),
        ("Category probability mass", "category_mass"),
    ]

    print("\n-- Aggregated metrics --")
    header = "{:<40} {:>12} {:>12} {:>12} {:>24}".format("Metric", "Baseline", "Category", "Δ", "Random baseline")
    print(header)
    print("-" * len(header))
    for label, key in metrics:
        base_value = base_summary.get(key)
        dir_value = direction_summary.get(key)
        delta: Optional[float]
        if base_value is None or dir_value is None:
            delta = None
        else:
            delta = dir_value - base_value
        random_values = [summary.get(key) for summary in random_summaries]
        formatted_random = _format_random(random_values)
        print(
            "{:<40} {:>12} {:>12} {:>12} {:>24}".format(
                label,
                _format_value(base_value),
                _format_value(dir_value),
                _format_value(delta),
                formatted_random,
            )
        )


if __name__ == "__main__":
    main()
