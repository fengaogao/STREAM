"""Visualise the effect of category-aligned directions on hidden states."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

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


def _load_item_linear_params(model) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(model, CausalLMStreamModel):
        weight = model.lm_head_weight[model.item_token_ids].detach()
        bias = None
    elif isinstance(model, BertStreamModel):
        weight = model.decoder_weight.detach()
        bias = model.decoder_bias.detach()
    else:
        raise TypeError(f"Unsupported model class {type(model)}")
    return weight, bias


def _slice_batch(batch: Mapping[str, torch.Tensor], index: int) -> Dict[str, torch.Tensor]:
    sliced: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            sliced[key] = value[index : index + 1]
    return sliced


def _compute_logits(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    logits = torch.mv(weight, hidden)
    if bias is not None:
        logits = logits + bias
    return logits


def _optimise_hidden(
    base_hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    target_idx: int,
    *,
    lr: float,
    steps: int,
    l2_weight: float,
    match_norm: bool,
) -> torch.Tensor:
    param = base_hidden.detach().clone().requires_grad_(True)
    optimiser = torch.optim.Adam([param], lr=lr)
    target_tensor = torch.tensor(target_idx, device=param.device)
    base_hidden_detached = base_hidden.detach()
    for _ in range(steps):
        optimiser.zero_grad()
        logits = _compute_logits(param, weight, bias)
        target_logit = logits[target_tensor]
        l2_term = (param - base_hidden_detached).pow(2).sum()
        loss = -target_logit + l2_weight * l2_term
        loss.backward()
        optimiser.step()
        if match_norm:
            with torch.no_grad():
                target_norm = base_hidden_detached.norm().clamp_min(1e-6)
                current_norm = param.norm().clamp_min(1e-6)
                param.mul_(target_norm / current_norm)
    return param.detach()


def _project_vectors(vectors: Iterable[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    stacked = torch.stack(list(vectors), dim=0)
    mean = stacked.mean(dim=0, keepdim=True)
    centered = stacked - mean
    if centered.size(0) < 2:
        basis = torch.eye(2, centered.size(1), device=centered.device)
    else:
        cov = centered.t().mm(centered)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        basis = eigvecs[:, -2:] if eigvecs.size(1) >= 2 else eigvecs
    projected = centered @ basis
    return projected, basis


def _format_metrics(value: float) -> str:
    return f"{value:.4f}"


def _select_sample(
    dataloader,
    model_type: str,
    category_map: Mapping[int, List[str]],
    desired_category: Optional[str],
    sample_index: Optional[int],
    max_batches: Optional[int] = None,
) -> Tuple[int, Dict[str, torch.Tensor], int]:
    seen = 0
    for batch_idx, batch in enumerate(dataloader):
        targets = extract_targets_from_batch(batch, model_type)
        batch_size = targets.size(0)
        for local_idx in range(batch_size):
            global_idx = seen + local_idx
            target = int(targets[local_idx].item())
            if sample_index is not None and global_idx != sample_index:
                continue
            if desired_category is not None:
                categories = category_map.get(target, [])
                if desired_category not in categories:
                    continue
            sliced = _slice_batch(batch, local_idx)
            return global_idx, sliced, target
        seen += batch_size
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break
    raise RuntimeError("Unable to locate a sample matching the specified criteria")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise category-aligned hidden state manipulations")
    parser.add_argument("--data_dir", type=Path, default="/home/zj/code/STREAM/ml-1m", help="Dataset directory used for offline training")
    parser.add_argument("--artifacts_dir", type=Path, default="/home/zj/code/STREAM/ml-1m/bert", help="Directory containing offline training artifacts")
    parser.add_argument("--model_type", choices=["causal", "bert"], default="bert", help="Base model architecture")
    parser.add_argument("--output", type=Path, default=Path("category_direction_visualisation.png"), help="Output image path")
    parser.add_argument("--category", type=str, default=None, help="Category name to visualise")
    parser.add_argument("--direction_index", type=int, default=None, help="Fallback direction index if category is not provided")
    parser.add_argument("--split", type=str, default="original", help="Dataset split to draw samples from")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for the evaluation dataloader")
    parser.add_argument("--sample_index", type=int, default=None, help="Select the N-th sample in iteration order (after filtering)")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit the number of dataloader batches to scan")
    parser.add_argument("--alpha", type=float, default=1, help="Scaling applied to the category direction")
    parser.add_argument("--optim_steps", type=int, default=200, help="Gradient ascent steps for the optimal hidden state")
    parser.add_argument("--optim_lr", type=float, default=0.001, help="Learning rate for hidden state optimisation")
    parser.add_argument("--l2_weight", type=float, default=1e-2, help="Regularisation weight keeping h close to original")
    parser.add_argument("--match_norm", action="store_true", help="Project optimised hidden state to match the original norm")
    parser.add_argument("--arrow_scale", type=float, default=1.0, help="Extra scale applied when drawing the category direction arrow")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to run on")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for reproducibility")
    args = parser.parse_args()

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
        args.artifacts_dir, args.category, args.direction_index, device
    )

    sample_idx, raw_batch, target_item = _select_sample(
        dataloader,
        args.model_type,
        category_map,
        category_direction.name if args.category else None,
        args.sample_index,
        max_batches=args.max_batches,
    )

    batch_device = move_batch_to_device(raw_batch, device)
    hidden = model.stream_hidden_states(batch_device)[0]
    direction_vec = category_direction.vector.to(hidden.device, dtype=hidden.dtype)
    manipulated = hidden + args.alpha * direction_vec

    weight, bias = _load_item_linear_params(model)
    weight = weight.to(hidden.device, dtype=hidden.dtype)
    bias = bias.to(hidden.device, dtype=hidden.dtype) if bias is not None else None

    optimal = _optimise_hidden(
        hidden,
        weight,
        bias,
        target_item,
        lr=args.optim_lr,
        steps=args.optim_steps,
        l2_weight=args.l2_weight,
        match_norm=args.match_norm,
    )

    logits_original = _compute_logits(hidden, weight, bias)
    logits_shifted = _compute_logits(manipulated, weight, bias)
    logits_optimal = _compute_logits(optimal, weight, bias)
    probs_original = torch.softmax(logits_original, dim=0)[target_item].item()
    probs_shifted = torch.softmax(logits_shifted, dim=0)[target_item].item()
    probs_optimal = torch.softmax(logits_optimal, dim=0)[target_item].item()

    cos_original_shifted = F.cosine_similarity(hidden.unsqueeze(0), manipulated.unsqueeze(0)).item()
    cos_original_optimal = F.cosine_similarity(hidden.unsqueeze(0), optimal.unsqueeze(0)).item()
    cos_shifted_optimal = F.cosine_similarity(manipulated.unsqueeze(0), optimal.unsqueeze(0)).item()

    projected, basis = _project_vectors([hidden, manipulated, optimal])
    proj_hidden, proj_shifted, proj_optimal = projected.cpu().numpy()
    proj_direction = (direction_vec @ basis).cpu().numpy()
    arrow_dx = args.alpha * args.arrow_scale * proj_direction[0]
    arrow_dy = args.alpha * args.arrow_scale * proj_direction[1]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(proj_hidden[0], proj_hidden[1], color="#1f77b4", label="Original h", s=80)
    ax.scatter(proj_shifted[0], proj_shifted[1], color="#ff7f0e", label="h + category", s=80)
    ax.scatter(proj_optimal[0], proj_optimal[1], color="#2ca02c", label="Optimised h*", s=80)
    ax.plot([proj_hidden[0], proj_optimal[0]], [proj_hidden[1], proj_optimal[1]], "k--", linewidth=1.0, label="h → h*")
    ax.arrow(
        proj_hidden[0],
        proj_hidden[1],
        arrow_dx,
        arrow_dy,
        width=0.002,
        head_width=0.05,
        length_includes_head=True,
        color="#d62728",
        label=f"Category direction ({category_direction.name})",
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        "Category direction effect\n"
        f"Sample #{sample_idx}, target item {target_item}, direction '{category_direction.name}'"
    )
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    # text_lines = [
    #     f"p(target): original={_format_metrics(probs_original)}",
    #     f"            shifted={_format_metrics(probs_shifted)}",
    #     f"            optimal={_format_metrics(probs_optimal)}",
    #     f"cos(h, h+Δ)={_format_metrics(cos_original_shifted)}",
    #     f"cos(h, h*)={_format_metrics(cos_original_optimal)}",
    #     f"cos(h+Δ, h*)={_format_metrics(cos_shifted_optimal)}",
    # ]
    # if category_direction.meta:
    #     meta_str = ", ".join(f"{k}={v}" for k, v in category_direction.meta.items() if k != "category")
    #     if meta_str:
    #         text_lines.append(f"direction meta: {meta_str}")
    # ax.text(0.02, 0.02, "\n".join(text_lines), transform=ax.transAxes, fontsize=10, va="bottom")

    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    plt.close(fig)

    print("=== Experiment summary ===")
    print(f"Sample index: {sample_idx}")
    print(f"Target item: {target_item}")
    print(f"Category direction: {category_direction.name} (index={category_direction.index})")
    print(f"Original probability: {probs_original:.6f}")
    print(f"Shifted probability: {probs_shifted:.6f}")
    print(f"Optimal probability: {probs_optimal:.6f}")
    print(f"cos(h, h+Δ): {cos_original_shifted:.6f}")
    print(f"cos(h, h*): {cos_original_optimal:.6f}")
    print(f"cos(h+Δ, h*): {cos_shifted_optimal:.6f}")
    print(f"Plot saved to: {args.output}")


if __name__ == "__main__":
    main()
