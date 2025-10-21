"""Offline training pipeline for STREAM."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict

import torch
from sklearn.cluster import KMeans
from torch.optim import AdamW
from tqdm import tqdm

from .config import StreamConfig
from .dataio import ItemVocab, build_dataloader, load_all_splits
from .models.causal_lm_stream import CausalLMStreamModel
from .models.bert_stream import BertStreamModel
from .state_adapter import ItemHead, ItemHeadInit
from .subspace import compute_subspace
from .utils import get_logger, set_seed

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline training for STREAM")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_type", choices=["causal", "bert"], required=True)
    parser.add_argument("--pretrained_name_or_path", type=str, default="gpt2")
    parser.add_argument("--rank_r", type=int, default=32)
    parser.add_argument("--router_k", type=int, default=16)
    parser.add_argument("--subspace_mode", choices=["gradcov", "pca"], default="gradcov")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default=None)
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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    item_vocab = ItemVocab.from_metadata(args.data_dir)
    splits = load_all_splits(args.data_dir)

    if args.model_type == "causal":
        model = CausalLMStreamModel(args.pretrained_name_or_path, item_vocab, device)
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

    config = StreamConfig(rank_r=args.rank_r, router_k=args.router_k)
    config.to_json(out_dir / "config.json")

    subspace = compute_subspace(model, eval_loader, rank=args.rank_r, mode=args.subspace_mode, device=device)
    subspace.save(out_dir)
    U = subspace.basis.to(device)

    item_head = ItemHead(rank=args.rank_r, num_items=item_vocab.num_items, device=device)
    if args.model_type == "causal":
        embeddings = model.lm_head_weight[model.item_token_ids].detach().t().to(device)
    else:
        embeddings = model.decoder_weight.detach().t().to(device)
    item_head.initialise(ItemHeadInit(U=U.cpu(), item_embeddings=embeddings.cpu(), lambda_l2=config.lambda_l2))
    torch.save({"W": item_head.state_dict(), "rank": args.rank_r, "num_items": item_vocab.num_items}, out_dir / "item_head.pt")

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
