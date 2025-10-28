"""Offline training pipeline for STREAM."""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from sklearn.cluster import KMeans
from torch.optim import AdamW
from tqdm import tqdm

from stream.config import StreamConfig
from stream.dataio import ItemVocab, build_dataloader, load_all_splits
from stream.models.causal_lm_stream import CausalLMStreamModel
from stream.models.bert_stream import BertStreamModel
from stream.state_adapter import ItemHead, ItemHeadInit
from stream.subspace import SubspaceResult, compute_subspace
from stream.utils import get_logger, set_seed

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline training for STREAM")
    parser.add_argument("--data_dir", type=Path, default="/home/zj/code/STREAM/ml-10M100K")
    parser.add_argument("--out_dir", type=Path, default="/home/zj/code/STREAM/ml-10M100K/bert")
    parser.add_argument("--model_type", choices=["causal", "bert"], default="bert")
    parser.add_argument("--pretrained_name_or_path", type=str, default="/home/zj/model/Llama-2-7b-hf")
    parser.add_argument(
        "--num_category_directions",
        type=int,
        default=0,
        help="Number of category-aligned directions to keep (0 means use all available)",
    )
    parser.add_argument("--router_k", type=int, default=16)
    parser.add_argument("--subspace_mode", choices=["gradcov", "pca"], default="gradcov")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, device, model_type: str) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    for batch in tqdm(dataloader, desc="train", leave=False):
        optimizer.zero_grad()
        if model_type == "causal":
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            outputs = model.model(**inputs)
        else:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            outputs = model.model(**inputs)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
        steps += 1
    return total_loss / max(steps, 1)


def build_router(model, dataloader, router_k: int, device) -> Dict:
    hidden_vectors = []
    for batch in dataloader:
        with torch.no_grad():
            hidden = model.stream_hidden_states({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
        hidden_vectors.append(torch.nn.functional.normalize(hidden, dim=-1).cpu())
        if len(hidden_vectors) * hidden.shape[0] > 5000:
            break
    hidden_cat = torch.cat(hidden_vectors, dim=0)
    if router_k <= 1 or hidden_cat.size(0) < router_k:
        centers = hidden_cat.mean(dim=0, keepdim=True)
    else:
        kmeans = KMeans(n_clusters=router_k, random_state=0, n_init=10)
        kmeans.fit(hidden_cat.numpy())
        centers = torch.from_numpy(kmeans.cluster_centers_)
    return {"centers": centers}


def build_item_name_map(item_vocab: ItemVocab) -> dict[int, str]:
    m: dict[int, str] = {}
    for i in range(item_vocab.num_items):
        meta = item_vocab.meta_of(i) if hasattr(item_vocab, "meta_of") else {}
        name = ""
        if isinstance(meta, dict):
            name = meta.get("title") or meta.get("name") or ""
        m[i] = name
    return m


def load_item_categories(data_dir: Path, item_vocab: ItemVocab) -> Tuple[Dict[int, List[str]], List[str]]:
    item_text_path = data_dir / "item_text.json"
    category_map: Dict[int, List[str]] = {i: [] for i in range(item_vocab.num_items)}
    category_counter: Counter[str] = Counter()
    if not item_text_path.exists():
        LOGGER.warning("Item text metadata not found at %s", item_text_path)
        return category_map, []

    with item_text_path.open("r", encoding="utf-8") as f:
        item_text = json.load(f)

    marker = "Genres:"
    for idx_str, text in item_text.items():
        try:
            item_idx = int(idx_str)
        except ValueError:
            continue
        if item_idx >= item_vocab.num_items:
            continue
        categories: List[str] = []
        if marker in text:
            raw = text.split(marker, 1)[1]
            categories = [c.strip() for c in raw.split(",") if c.strip()]
        category_map[item_idx] = categories
        category_counter.update(categories)

    ordered_categories = [cat for cat, _ in category_counter.most_common()]
    return category_map, ordered_categories


def extract_targets_from_batch(batch: Dict[str, torch.Tensor], model_type: str) -> torch.Tensor:
    if model_type == "causal":
        targets = batch["target_item"]
        if targets.is_cuda:
            targets = targets.detach().cpu()
        return targets.long()

    labels = batch["labels"]
    if labels.is_cuda:
        labels = labels.detach().cpu()
    batch_targets = torch.full((labels.size(0),), -1, dtype=torch.long)
    for idx, row in enumerate(labels):
        positives = row[row >= 0]
        if positives.numel() > 0:
            batch_targets[idx] = int(positives[0].item())
    return batch_targets


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def compute_category_semantic_subspace(
    model,
    dataloader,
    category_map: Dict[int, List[str]],
    category_order: List[str],
    max_categories: int,
    device: torch.device,
    model_type: str,
    fallback_mode: str,
    min_samples_per_category: int = 20,
) -> SubspaceResult:
    if not category_order:
        LOGGER.warning("No category metadata detected, falling back to %s", fallback_mode)
        return compute_subspace(model, dataloader, rank=max_categories or 1, mode=fallback_mode, device=device)

    model.eval()
    category_sums: Dict[str, torch.Tensor] = {}
    category_counts: Dict[str, float] = {}
    total_sum: torch.Tensor | None = None
    total_sq_sum: torch.Tensor | None = None
    total_count = 0
    feature_dim: int | None = None

    for batch in tqdm(dataloader, desc="collect-directions", leave=False):
        batch_device = move_batch_to_device(batch, device)
        grads = model.stream_positive_gradients(batch_device)  # [B, D]
        targets = extract_targets_from_batch(batch_device, model_type)  # [B]
        grads = grads.detach().to(device)

        if feature_dim is None:
            feature_dim = grads.size(-1)
            total_sum = torch.zeros(feature_dim, device=device)
            total_sq_sum = torch.zeros(feature_dim, device=device)
        assert total_sum is not None and total_sq_sum is not None

        for grad, target in zip(grads, targets.tolist()):
            total_sum += grad
            total_sq_sum += grad * grad
            total_count += 1
            if target < 0 or target not in category_map:
                continue
            cats = category_map[target]
            if not cats:
                continue
            weight = 1.0 / float(len(cats))
            for c in cats:
                if c not in category_sums:
                    category_sums[c] = torch.zeros_like(grad)
                    category_counts[c] = 0.0
                category_sums[c] += grad * weight
                category_counts[c] += weight

    if total_count == 0 or feature_dim is None or total_sum is None or total_sq_sum is None:
        LOGGER.warning("No gradients collected, falling back to %s", fallback_mode)
        return compute_subspace(model, dataloader, rank=max_categories or 1, mode=fallback_mode, device=device)

    global_mean = total_sum / float(total_count)
    global_var = total_sq_sum / float(total_count) - global_mean * global_mean
    global_var = torch.clamp(global_var, min=1e-6)
    global_std = torch.sqrt(global_var)

    category_contrasts: Dict[str, Tuple[torch.Tensor, torch.Tensor, float]] = {}
    for c, grad_sum in category_sums.items():
        cnt = float(category_counts.get(c, 0.0))
        if cnt < min_samples_per_category:
            continue
        mean_grad = grad_sum / float(cnt)
        rest_cnt = total_count - cnt
        if rest_cnt <= 0:
            continue
        rest_mean = (total_sum - grad_sum) / float(rest_cnt)
        delta = mean_grad - rest_mean
        whitened_delta = delta / global_std
        category_contrasts[c] = (delta, whitened_delta, cnt)

    if not category_contrasts:
        LOGGER.warning("No category passed the minimum sample threshold, fallback to %s", fallback_mode)
        return compute_subspace(model, dataloader, rank=max_categories or 1, mode=fallback_mode, device=device)

    energy_map = {c: category_contrasts[c][1].norm().item() for c in category_contrasts}
    ordered = [c for c in category_order if c in category_contrasts]
    remaining = [c for c in category_contrasts.keys() if c not in ordered]
    ordered.extend(sorted(remaining, key=lambda x: energy_map[x], reverse=True))
    if max_categories > 0 and len(ordered) > max_categories:
        ordered = sorted(ordered, key=lambda x: energy_map[x], reverse=True)[:max_categories]

    whitened_matrix: List[torch.Tensor] = []
    deltas: List[torch.Tensor] = []
    meta_categories: List[Dict] = []

    for c in ordered:
        delta, wdelta, cnt = category_contrasts[c]
        whitened_matrix.append(wdelta)
        deltas.append(delta)
        meta_categories.append(
            {
                "category": c,
                "count": int(cnt),
                "share": float(cnt / total_count),
                "energy": float(wdelta.norm().item()),
            }
        )

    if not whitened_matrix:
        LOGGER.warning("Selected categories list is empty after filtering, fallback to %s", fallback_mode)
        return compute_subspace(model, dataloader, rank=max_categories or 1, mode=fallback_mode, device=device)

    whitened_stack = torch.stack(whitened_matrix, dim=1)  # [D, C]
    gram = whitened_stack.t() @ whitened_stack            # [C, C]
    stabiliser = 1e-4 * torch.eye(gram.size(0), device=gram.device)
    gram = gram + stabiliser
    shrinkage = 0.0
    if gram.size(0) > 1:
        diag = torch.diagonal(gram)
        diag_matrix = torch.diag_embed(diag)
        off_diag = gram - diag_matrix
        mean_diag = float(torch.clamp(diag.abs().mean(), min=1e-8).item())
        mean_off = float(off_diag.abs().mean().item())
        if mean_off > 0.0:
            shrinkage = min(0.5, mean_off / mean_diag)
            gram = (1.0 - shrinkage) * gram + shrinkage * diag_matrix

    identity = torch.eye(gram.size(0), device=gram.device)
    dual_projection = torch.linalg.solve(gram, identity)  # [C, C]
    clean_alignment = whitened_stack @ dual_projection     # [D, C]

    clean_gram = clean_alignment.t() @ clean_alignment
    clean_gram = (clean_gram + clean_gram.t()) * 0.5
    evals, evecs = torch.linalg.eigh(clean_gram)
    evals = torch.clamp(evals, min=1e-9)
    inv_sqrt = evecs @ torch.diag(evals.rsqrt()) @ evecs.t()
    orthonormal_whitened = clean_alignment @ inv_sqrt

    orthogonal_vectors: List[torch.Tensor] = []
    kept_meta: List[Dict] = []
    for idx in range(orthonormal_whitened.size(1)):
        whitened_vec = orthonormal_whitened[:, idx]
        raw_direction = whitened_vec * global_std
        norm = raw_direction.norm()
        if norm < 1e-6:
            continue
        raw_direction = raw_direction / norm
        alignment = torch.dot(raw_direction, deltas[idx])
        if alignment < 0:
            raw_direction = -raw_direction
            alignment = -alignment
            whitened_vec = -whitened_vec
            orthonormal_whitened[:, idx] = whitened_vec

        responses = torch.mv(whitened_stack.t(), whitened_vec)  # [C]
        responses_list = responses.tolist()
        response_self = float(responses_list[idx])
        response_off = [abs(float(r)) for j, r in enumerate(responses_list) if j != idx]
        max_cross = max(response_off) if response_off else 0.0
        mean_cross = float(sum(response_off) / len(response_off)) if response_off else 0.0

        meta_entry = dict(meta_categories[idx])
        meta_entry["sensitivity"] = float(alignment.item())
        meta_entry["response_self"] = response_self
        meta_entry["max_cross_response"] = max_cross
        meta_entry["mean_cross_response"] = mean_cross
        meta_entry["dual_weight_norm"] = float(dual_projection[:, idx].norm().item())

        kept_meta.append(meta_entry)
        orthogonal_vectors.append(raw_direction)

    if not orthogonal_vectors:
        LOGGER.warning("All semantic directions were filtered out after orthonormalisation, fallback to %s", fallback_mode)
        return compute_subspace(model, dataloader, rank=max_categories or 1, mode=fallback_mode, device=device)

    basis = torch.stack(orthogonal_vectors, dim=1)  # [D, R]
    category_rank = basis.size(1)

    meta = {
        "method": "category_dual_projection",
        "feature_dim": feature_dim,
        "total_samples": total_count,
        "requested_categories": max_categories,
        "effective_rank": int(category_rank),
        "categories": kept_meta,
        "category_overlap_shrinkage": float(shrinkage),
    }
    return SubspaceResult(basis=basis.detach().cpu(), mode="gradcov", meta=meta)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    item_vocab = ItemVocab.from_metadata(args.data_dir)
    splits = load_all_splits(args.data_dir)

    if args.model_type == "causal":
        model = CausalLMStreamModel(
            args.pretrained_name_or_path,
            item_vocab,
            device,
            tokenizer_name_or_path=None,
        )
        tokenizer = model.tokenizer
    else:
        model = BertStreamModel(item_vocab, device)
        tokenizer = None

    _, train_loader = build_dataloader(
        splits["original"],
        model_type=args.model_type,
        batch_size=args.batch_size,
        shuffle=True,
        item_vocab=item_vocab,
        tokenizer=tokenizer,
    )
    _, eval_loader = build_dataloader(
        splits["original"],
        model_type=args.model_type,
        batch_size=args.batch_size,
        shuffle=False,
        item_vocab=item_vocab,
        tokenizer=tokenizer,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, device, args.model_type)
        LOGGER.info("Epoch %d loss %.4f", epoch + 1, loss)

    category_map, category_order = load_item_categories(args.data_dir, item_vocab)
    category_budget = args.num_category_directions
    if args.subspace_mode == "pca":
        pca_rank = category_budget if category_budget > 0 else 32
        subspace = compute_subspace(model, eval_loader, rank=pca_rank, mode="pca", device=device)
    else:
        subspace = compute_category_semantic_subspace(
            model=model,
            dataloader=eval_loader,
            category_map=category_map,
            category_order=category_order,
            max_categories=category_budget,
            device=device,
            model_type=args.model_type,
            fallback_mode="gradcov",
        )

    effective_rank = subspace.basis.size(1)
    config = StreamConfig(rank_r=effective_rank, router_k=args.router_k)
    config.to_json(out_dir / "config.json")

    subspace.save(out_dir)

    U = subspace.basis.to(device)

    item_head = ItemHead(rank=effective_rank, num_items=item_vocab.num_items, device=device)
    if args.model_type == "causal":
        embeddings = model.lm_head_weight[model.item_token_ids].detach().t().to(device)
    else:
        embeddings = model.decoder_weight.detach().t().to(device)
    item_head.initialise(ItemHeadInit(U=U.cpu(), item_embeddings=embeddings.cpu(), lambda_l2=config.lambda_l2))
    torch.save({"W": item_head.state_dict(), "rank": effective_rank, "num_items": item_vocab.num_items}, out_dir / "item_head.pt")

    router = build_router(model, eval_loader, args.router_k, device)
    with (out_dir / "router.pkl").open("wb") as f:
        pickle.dump(router, f)

    model_dir = out_dir / "model"
    model_dir.mkdir(exist_ok=True)
    model.model.save_pretrained(model_dir)
    if args.model_type == "causal":
        tokenizer_dir = out_dir / "tokenizer"
        tokenizer.save_pretrained(tokenizer_dir)

    LOGGER.info("Training complete. Artifacts saved to %s", out_dir)


if __name__ == "__main__":
    main()
