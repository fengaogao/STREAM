"""Offline training pipeline for STREAM."""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import gzip
from collections import Counter
from functools import lru_cache
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from sklearn.cluster import KMeans
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from stream.config import StreamConfig
from stream.dataio import ItemVocab, build_dataloader, load_all_splits
from stream.models.causal_lm_stream import CausalLMStreamModel
from stream.models.bert_stream import BertStreamModel
from stream.state_adapter import ItemHead, ItemHeadInit
from stream.subspace import SubspaceResult, compute_subspace
from stream.utils import get_logger, set_seed
import torch.backends.cuda as cuda_backends
cuda_backends.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.amp import autocast, GradScaler
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--grad_accumulation_steps",
        type=int,
        default=1,
        help="Number of micro-batches to accumulate before each optimizer step",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping value (set <=0 to disable)",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--train_backbone",
        action="store_true",
        help="Unfreeze the full language-model backbone during fine-tuning (may require significantly more GPU memory)",
    )
    parser.add_argument(
        "--freeze_item_embeddings",
        action="store_true",
        help="Keep item token embeddings frozen when the backbone is frozen",
    )
    parser.add_argument(
        "--freeze_lm_head",
        action="store_true",
        help="Keep the LM head frozen when the backbone is frozen",
    )
    parser.add_argument(
        "--dataloader_workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes (set 0 to disable multiprocessing)",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Enable pinned host memory for DataLoader workers",
    )
    parser.add_argument(
        "--persistent_workers",
        action="store_true",
        help="Keep DataLoader workers alive between epochs (requires num_workers > 0)",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=None,
        help="Number of samples to prefetch per worker (requires num_workers > 0)",
    )
    parser.add_argument(
        "--pretokenize",
        action="store_true",
        help="Pre-tokenise the training split into an in-memory cache (may require substantial RAM)",
    )
    parser.add_argument(
        "--token_cache_dir",
        type=Path,
        default=None,
        help="Optional directory for cached tokenised samples",
    )
    parser.add_argument(
        "--token_cache_overwrite",
        action="store_true",
        help="Rebuild any existing token cache on disk",
    )
    parser.add_argument(
        "--materialize_token_cache",
        action="store_true",
        help="Eagerly precompute tokenised samples to disk before training starts",
    )
    parser.add_argument(
        "--disable_ddp_no_sync",
        action="store_true",
        help="Force gradient synchronisation on every micro-batch even when accumulating",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Optimizer weight decay")
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW (momentum term)",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.95,
        help="Beta2 for AdamW (variance term)",
    )
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="Epsilon for AdamW")
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Fraction of total steps used for linear LR warmup when --warmup_steps is not provided",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Explicit number of warmup steps (overrides --warmup_ratio)",
    )
    return parser.parse_args()


def _mask_parameter_rows(
    parameter: torch.nn.Parameter,
    row_indices: torch.Tensor,
    component_name: str,
) -> None:
    """Restrict gradients to the specified token rows for a parameter matrix."""

    if row_indices.numel() == 0:
        LOGGER.warning("No item token ids provided for %s; skipping fine-tuning", component_name)
        return

    with torch.no_grad():
        unique_ids = torch.unique(row_indices.detach().long())
        valid = unique_ids[(unique_ids >= 0) & (unique_ids < parameter.size(0))]
    if valid.numel() == 0:
        LOGGER.warning("No valid item token ids found for %s; skipping fine-tuning", component_name)
        return

    mask = torch.zeros(parameter.size(0), dtype=torch.bool, device=parameter.device)
    mask[valid] = True
    view_shape = (parameter.size(0),) + (1,) * (parameter.dim() - 1)
    mask = mask.view(view_shape)

    def _hook(grad: torch.Tensor) -> torch.Tensor:
        return grad * mask.to(grad.device, dtype=grad.dtype)

    parameter.requires_grad_(True)
    parameter.register_hook(_hook)
    LOGGER.info(
        "Unfreezing %s for %d item-specific rows (out of %d total)",
        component_name,
        int(valid.numel()),
        parameter.size(0),
    )


def _get_causal_backbone(model: CausalLMStreamModel):
    backbone = model.model
    if isinstance(backbone, torch.nn.DataParallel):
        backbone = backbone.module
    if isinstance(backbone, torch.nn.parallel.DistributedDataParallel):
        backbone = backbone.module
    return backbone


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    model_type: str,
    *,
    is_main_process: bool,
    distributed: bool,
    grad_accumulation_steps: int,
    max_grad_norm: float,
    enable_no_sync: bool,
    scaler: GradScaler,
    amp_dtype: torch.dtype,
    scheduler: Optional[LambdaLR],
) -> float:
    model.train()
    grad_accumulation_steps = max(grad_accumulation_steps, 1)
    if model_type not in {"causal", "bert"}:
        raise ValueError(f"Unsupported model_type {model_type}")
    total_loss = 0.0
    micro_steps = 0
    ddp_module = getattr(model, "model", None)
    use_no_sync = (
        distributed
        and enable_no_sync
        and grad_accumulation_steps > 1
        and hasattr(ddp_module, "no_sync")
    )
    progress = tqdm(dataloader, desc="train", leave=False, disable=not is_main_process)
    optimizer.zero_grad(set_to_none=True)

    def _has_nonfinite_grads():
        for p in model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                return True
        return False

    for step, batch in enumerate(progress, start=1):
        context = ddp_module.no_sync if (use_no_sync and step % grad_accumulation_steps != 0) else nullcontext
        with context():
            batch_on_device = move_batch_to_device(batch, device)
            with autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                outputs = model.model(
                    input_ids=batch_on_device["input_ids"],
                    attention_mask=batch_on_device["attention_mask"],
                    labels=batch_on_device["labels"],
                )
                loss = outputs.loss

            if not torch.isfinite(loss):
                if is_main_process:
                    LOGGER.warning(f"Non-finite loss detected: {loss.item()}. Skip this micro-step.")
                optimizer.zero_grad(set_to_none=True)
                continue
            if scaler.is_enabled():
                scaler.scale(loss / grad_accumulation_steps).backward()
            else:
                (loss / grad_accumulation_steps).backward()

        total_loss += float(loss.item())
        micro_steps += 1

        if step % grad_accumulation_steps == 0:
            if max_grad_norm > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if _has_nonfinite_grads():
                    if is_main_process:
                        LOGGER.warning("Non-finite grads detected. Zero and skip step.")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    remainder = micro_steps % grad_accumulation_steps
    if remainder != 0:
        if max_grad_norm > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            if _has_nonfinite_grads():
                if is_main_process:
                    LOGGER.warning("Non-finite grads detected on tail micro-batch. Skip optimizer step.")
                optimizer.zero_grad(set_to_none=True)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        if scaler.is_enabled():
            scaler.step(optimizer);
            scaler.update()
        else:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    metrics = torch.tensor([total_loss, float(micro_steps)], device=device)
    if distributed:
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    total_loss, micro_steps = metrics.tolist()
    return total_loss / max(int(micro_steps), 1)


def build_router(model, dataloader, router_k: int, device) -> Dict:
    hidden_vectors = []
    for batch in dataloader:
        with torch.no_grad():
            hidden = model.stream_hidden_states(
                {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            )
        hidden = torch.nn.functional.normalize(hidden.float(), dim=-1)
        hidden_vectors.append(hidden.cpu())

        if len(hidden_vectors) * hidden.shape[0] > 5000:
            break

    hidden_cat = torch.cat(hidden_vectors, dim=0).to(torch.float32)

    if router_k <= 1 or hidden_cat.size(0) < router_k:
        centers = hidden_cat.mean(dim=0, keepdim=True)
    else:
        kmeans = KMeans(n_clusters=router_k, random_state=0, n_init=10)
        kmeans.fit(hidden_cat.cpu().numpy())
        centers = torch.from_numpy(kmeans.cluster_centers_)

    return {"centers": centers}



def _load_item_id_mapping(data_dir: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    meta_path = data_dir / "item_ids.json"
    if not meta_path.exists():
        return mapping

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return mapping

    mid2idx = meta.get("mid2idx", {}) if isinstance(meta, dict) else {}
    for raw_id, raw_idx in mid2idx.items():
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        key = str(raw_id).strip()
        if not key:
            continue
        mapping[key] = idx
        lower = key.lower()
        upper = key.upper()
        mapping.setdefault(lower, idx)
        mapping.setdefault(upper, idx)
        if key.isdigit():
            mapping.setdefault(str(int(key)), idx)
    return mapping


def _clean_text(value: str, *, max_length: int = 512) -> str:
    text = " ".join(str(value).split())
    if max_length > 0 and len(text) > max_length:
        text = text[: max_length - 3].rstrip() + "..."
    return text


def _extract_amazon_categories(entry: Dict) -> List[str]:
    categories: set[str] = set()
    raw_categories = entry.get("categories")
    if isinstance(raw_categories, list):
        for item in raw_categories:
            if isinstance(item, list) and item:
                leaf = str(item[-1]).strip()
                if leaf:
                    categories.add(leaf)
            elif isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    categories.add(cleaned)
    single = entry.get("category") or entry.get("main_cat")
    if isinstance(single, str):
        cleaned = single.strip()
        if cleaned:
            categories.add(cleaned)
    return sorted(categories)


def _iter_amazon_metadata_file(path: Path):
    open_fn = gzip.open if path.suffix.endswith("gz") else open
    try:
        with open_fn(path, "rt", encoding="utf-8") as f:
            prefix = f.read(1)
            while prefix and prefix.isspace():
                prefix = f.read(1)
            f.seek(0)
            if prefix == "[":
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return
                if isinstance(data, list):
                    for entry in data:
                        if isinstance(entry, dict):
                            yield entry
                return
            for line in f:
                line = line.strip().rstrip(",")
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(entry, dict):
                    yield entry
    except OSError:
        return


@lru_cache(maxsize=8)
def _load_amazon_metadata_cached(data_dir_str: str) -> Tuple[Dict[int, str], Dict[int, List[str]]]:
    data_dir = Path(data_dir_str)
    id_mapping = _load_item_id_mapping(data_dir)
    if not id_mapping:
        return {}, {}

    meta_files: List[Path] = []
    patterns = [
        "meta_*.json",
        "meta_*.jsonl",
        "meta_*.json.gz",
        "meta_*.jsonl.gz",
        "item_meta.json",
        "item_meta.jsonl",
        "item_meta.json.gz",
        "item_meta.jsonl.gz",
        "metadata.json",
        "metadata.jsonl",
        "metadata.json.gz",
        "metadata.jsonl.gz",
    ]
    for pattern in patterns:
        meta_files.extend(sorted(data_dir.glob(pattern)))

    seen: set[Path] = set()
    unique_files: List[Path] = []
    for path in meta_files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_files.append(path)

    text_map: Dict[int, str] = {}
    category_map: Dict[int, List[str]] = {}

    for meta_path in unique_files:
        for entry in _iter_amazon_metadata_file(meta_path):
            raw_id = entry.get("asin") or entry.get("item_id") or entry.get("itemID") or entry.get("id")
            if not raw_id:
                continue
            key = str(raw_id).strip()
            if not key:
                continue
            idx = id_mapping.get(key) or id_mapping.get(key.lower()) or id_mapping.get(key.upper())
            if idx is None and key.isdigit():
                idx = id_mapping.get(str(int(key)))
            if idx is None:
                continue

            categories = _extract_amazon_categories(entry)
            if categories:
                category_map[idx] = categories

            parts: List[str] = []
            title = entry.get("title") or entry.get("Title")
            if isinstance(title, str) and title.strip():
                parts.append(_clean_text(title, max_length=256))
            brand = entry.get("brand")
            if isinstance(brand, str) and brand.strip():
                parts.append(f"Brand: {_clean_text(brand, max_length=128)}")
            price = entry.get("price")
            if isinstance(price, (int, float)):
                parts.append(f"Price: {price}")
            elif isinstance(price, str) and price.strip():
                parts.append(f"Price: {_clean_text(price, max_length=64)}")

            features = entry.get("feature") or entry.get("features")
            feature_text = ""
            if isinstance(features, list):
                feature_items = [
                    _clean_text(str(f), max_length=128)
                    for f in features
                    if isinstance(f, (str, int, float)) and str(f).strip()
                ]
                if feature_items:
                    feature_text = "; ".join(feature_items[:5])
            elif isinstance(features, str) and features.strip():
                feature_text = _clean_text(features, max_length=256)
            if feature_text:
                parts.append(f"Features: {feature_text}")

            description = entry.get("description")
            desc_text = ""
            if isinstance(description, list):
                desc_items = [
                    _clean_text(str(d), max_length=256)
                    for d in description
                    if isinstance(d, (str, int, float)) and str(d).strip()
                ]
                if desc_items:
                    desc_text = " ".join(desc_items)
            elif isinstance(description, str) and description.strip():
                desc_text = _clean_text(description, max_length=512)
            if desc_text:
                parts.append(desc_text)

            if categories:
                parts.append("Categories: " + ", ".join(categories))

            combined = " | ".join(p for p in parts if p)
            if combined:
                text_map[idx] = combined

    return text_map, category_map


def _load_amazon_metadata(data_dir: Path) -> Tuple[Dict[int, str], Dict[int, List[str]]]:
    return _load_amazon_metadata_cached(str(data_dir.resolve()))


def load_item_text_map(data_dir: Path, item_vocab: ItemVocab) -> dict[int, str]:
    text_map: dict[int, str] = {}

    def _fallback_name(idx: int) -> str:
        meta = item_vocab.meta_of(idx) if hasattr(item_vocab, "meta_of") else {}
        if isinstance(meta, dict):
            return str(meta.get("title") or meta.get("name") or "").strip()
        return ""

    for idx in range(item_vocab.num_items):
        text_map[idx] = _fallback_name(idx)

    item_text_path = data_dir / "item_text.json"
    if item_text_path.exists():
        with item_text_path.open("r", encoding="utf-8") as f:
            item_text = json.load(f)

        for idx_str, raw_text in item_text.items():
            try:
                item_idx = int(idx_str)
            except ValueError:
                continue
            if not (0 <= item_idx < item_vocab.num_items):
                continue
            cleaned = str(raw_text).strip()
            if cleaned:
                text_map[item_idx] = cleaned.replace(" | ", "; ")
    else:
        LOGGER.warning("Item text metadata not found at %s", item_text_path)

    amazon_text, _ = _load_amazon_metadata(data_dir)
    for idx, text in amazon_text.items():
        if 0 <= idx < item_vocab.num_items and text:
            text_map[idx] = text

    return text_map


def load_item_categories(
    data_dir: Path,
    item_vocab: ItemVocab,
    item_text_map: dict[int, str] | None = None,
) -> Tuple[Dict[int, List[str]], List[str]]:
    category_map: Dict[int, List[str]] = {i: [] for i in range(item_vocab.num_items)}
    category_counter: Counter[str] = Counter()

    _, amazon_categories = _load_amazon_metadata(data_dir)
    if amazon_categories:
        for item_idx, cats in amazon_categories.items():
            if not (0 <= item_idx < item_vocab.num_items):
                continue
            cleaned = [c for c in cats if c]
            if cleaned:
                category_map[item_idx] = cleaned
                category_counter.update(cleaned)

    if item_text_map is None:
        item_text_map = load_item_text_map(data_dir, item_vocab)

    marker = "Genres:"
    for item_idx, text in item_text_map.items():
        if not (0 <= item_idx < item_vocab.num_items):
            continue
        if category_map[item_idx]:
            continue
        categories: List[str] = []
        if text and marker in text:
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
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}

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


def _build_linear_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be > 0")

    num_warmup_steps = max(0, min(num_warmup_steps, num_training_steps))

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def main() -> None:
    args = parse_args()
    distributed = False
    rank = 0
    world_size = 1
    local_rank = 0

    if args.grad_accumulation_steps < 1:
        raise ValueError("--grad_accumulation_steps must be >= 1")
    if args.persistent_workers and args.dataloader_workers == 0:
        LOGGER.warning("persistent_workers requested but num_workers=0; disabling persistent workers")
        args.persistent_workers = False
    effective_prefetch = args.prefetch_factor
    if effective_prefetch is not None and args.dataloader_workers == 0:
        LOGGER.warning("prefetch_factor requires num_workers > 0; ignoring prefetch request")
        effective_prefetch = None

    default_device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == "causal" and dist.is_available():
        world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size_env > 1:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                device = torch.device("cuda", local_rank)
                init_kwargs = {"backend": backend}
            else:
                device = torch.device("cpu")
                init_kwargs = {"backend": backend}
            dist.init_process_group(**init_kwargs)
            distributed = True
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            device = default_device
    else:
        device = default_device

    is_main_process = rank == 0
    set_seed(args.seed + rank if distributed else args.seed)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    item_vocab = ItemVocab.from_metadata(args.data_dir)
    splits = load_all_splits(args.data_dir)

    item_text_map: dict[int, str] | None = None

    trainable_params: list[torch.nn.Parameter] | None = None
    trained_components: list[str] = []

    if args.model_type == "causal":
        item_text_map = load_item_text_map(args.data_dir, item_vocab)
        model = CausalLMStreamModel(
            args.pretrained_name_or_path,
            item_vocab,
            device,
            tokenizer_name_or_path=None,
            item_name_map=item_text_map,
        )
        backbone = model.model
        if args.train_backbone:
            for param in backbone.parameters():
                param.requires_grad_(True)
            trainable_params = list(backbone.parameters())
            if is_main_process:
                LOGGER.info(
                    "Training full causal LM backbone (%d parameter tensors)",
                    len(trainable_params),
                )
        else:
            for param in backbone.parameters():
                param.requires_grad_(False)

            trainable_set: dict[int, torch.nn.Parameter] = {}
            trainable_params = []

            if not args.freeze_item_embeddings:
                input_embeddings = backbone.get_input_embeddings()
                if input_embeddings is not None:
                    weight = input_embeddings.weight
                    _mask_parameter_rows(weight, model.item_token_ids, "item token embeddings")
                    if id(weight) not in trainable_set:
                        trainable_set[id(weight)] = weight
                        trainable_params.append(weight)
                        trained_components.append("item-embeddings(<item_i>)")

            if not args.freeze_lm_head:
                lm_head = getattr(backbone, "lm_head", None)
                if lm_head is not None:
                    for param in lm_head.parameters():
                        _mask_parameter_rows(param, model.item_token_ids, "LM head rows")
                        if id(param) not in trainable_set:
                            trainable_set[id(param)] = param
                            trainable_params.append(param)
                    trained_components.append("lm-head(<item_i>)")

            if not trainable_params:
                raise ValueError(
                    "No trainable parameters selected. Enable --train_backbone or unfreeze either the item embeddings or LM head."
                )

            if is_main_process:
                LOGGER.info(
                    "Training %d parameter group(s): %s",
                    len(trainable_params),
                    ", ".join(sorted(set(trained_components))),
                )
        if distributed:
            LOGGER.info(
                "Initialising DistributedDataParallel for causal LM on rank %d/%d (local rank %d)",
                rank,
                world_size,
                local_rank,
            )
            model.model = torch.nn.parallel.DistributedDataParallel(
                model.model,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
                broadcast_buffers=False,
            )
            backbone = _get_causal_backbone(model)
            if hasattr(backbone, "lm_head"):
                model.lm_head_weight = backbone.lm_head.weight  # type: ignore[attr-defined]
        tokenizer = model.tokenizer
    else:
        model = BertStreamModel(item_vocab, device)
        tokenizer = None
        trainable_params = list(model.parameters())

    cache_materialise_rank = args.materialize_token_cache and (not distributed or is_main_process)

    train_dataset, train_loader = build_dataloader(
        splits["original"],
        model_type=args.model_type,
        batch_size=args.batch_size,
        shuffle=True,
        item_vocab=item_vocab,
        tokenizer=tokenizer,
        num_workers=args.dataloader_workers,
        item_text_map=item_text_map if args.model_type == "causal" else None,
        pretokenize=args.pretokenize,
        token_cache_dir=args.token_cache_dir,
        cache_overwrite=args.token_cache_overwrite,
        materialize_cache=cache_materialise_rank,
        show_cache_progress=is_main_process and (args.pretokenize or cache_materialise_rank),
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=effective_prefetch,
    )
    _, eval_loader = build_dataloader(
        splits["original"],
        model_type=args.model_type,
        batch_size=args.batch_size,
        shuffle=False,
        item_vocab=item_vocab,
        tokenizer=tokenizer,
        num_workers=args.dataloader_workers,
        item_text_map=item_text_map if args.model_type == "causal" else None,
        token_cache_dir=args.token_cache_dir,
        cache_overwrite=args.token_cache_overwrite,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=effective_prefetch,
    )

    if distributed and args.materialize_token_cache and dist.is_initialized():
        dist.barrier()

    train_sampler: DistributedSampler | None = None
    if args.model_type == "causal" and distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        loader_kwargs = {
            "dataset": train_dataset,
            "batch_size": args.batch_size,
            "shuffle": False,
            "sampler": train_sampler,
            "num_workers": args.dataloader_workers,
            "collate_fn": train_loader.collate_fn,
            "pin_memory": args.pin_memory,
        }
        if args.dataloader_workers > 0:
            loader_kwargs["persistent_workers"] = args.persistent_workers
            if effective_prefetch is not None:
                loader_kwargs["prefetch_factor"] = effective_prefetch
        train_loader = DataLoader(**loader_kwargs)

    if args.model_type == "causal":
        if trainable_params is None:
            raise RuntimeError("Expected trainable parameters to be initialised for causal training")
        optimizer = AdamW(
            trainable_params,
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
    amp_enabled = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = GradScaler(device="cuda", enabled=amp_enabled and (amp_dtype == torch.float16))
    scheduler: Optional[LambdaLR] = None
    if len(train_loader) > 0:
        updates_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accumulation_steps))
        total_training_steps = updates_per_epoch * args.epochs
        warmup_steps = args.warmup_steps if args.warmup_steps is not None else int(total_training_steps * args.warmup_ratio)
        warmup_steps = min(warmup_steps, total_training_steps)
        if total_training_steps > 0 and warmup_steps >= 0:
            scheduler = _build_linear_scheduler(optimizer, warmup_steps, total_training_steps)

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.model_type,
            is_main_process=is_main_process,
            distributed=distributed,
            grad_accumulation_steps=args.grad_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            enable_no_sync=not args.disable_ddp_no_sync,
            scaler=scaler,
            amp_dtype=amp_dtype,
            scheduler=scheduler,
        )
        if is_main_process:
            LOGGER.info("Epoch %d loss %.4f", epoch + 1, loss)

    if distributed:
        dist.barrier()
    if distributed and not is_main_process:
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    if args.model_type == "causal" and distributed:
        backbone = _get_causal_backbone(model)
        model.model = backbone
        if hasattr(backbone, "lm_head"):
            model.lm_head_weight = backbone.lm_head.weight  # type: ignore[attr-defined]

    category_map, category_order = load_item_categories(
        args.data_dir,
        item_vocab,
        item_text_map=item_text_map,
    )
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

    if is_main_process:
        LOGGER.info("Training complete. Artifacts saved to %s", out_dir)

    if distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
