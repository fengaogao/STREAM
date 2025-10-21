"""Online adaptation loop for STREAM."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import BertForMaskedLM

from .config import StreamConfig
from .dataio import ItemVocab, build_dataloader, load_all_splits
from .detectors import DriftDetector
from .models.bert_stream import BertStreamModel
from .models.causal_lm_stream import CausalLMStreamModel
from .state_adapter import ItemHead, apply_overlay, RegionStateBank
from .trust_region import solve_delta_s
from .utils import get_logger

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STREAM online loop")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--ckpt_dir", type=Path, required=True)
    parser.add_argument("--model_type", choices=["causal", "bert"], required=True)
    parser.add_argument("--epsilon_kl", type=float, default=0.01)
    parser.add_argument("--lambda_l2", type=float, default=1e-3)
    parser.add_argument("--window_m", type=int, default=500)
    parser.add_argument("--glr_threshold", type=float, default=8.0)
    parser.add_argument("--conformal_alpha", type=float, default=0.10)
    parser.add_argument("--conformal_q", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_artifacts(args, item_vocab: ItemVocab, device: torch.device):
    config_path = args.ckpt_dir / "config.json"
    config = StreamConfig.from_json(config_path)
    subspace = torch.load(args.ckpt_dir / "subspace_U.pt", map_location=device)
    item_head_state = torch.load(args.ckpt_dir / "item_head.pt", map_location=device)
    with (args.ckpt_dir / "router.pkl").open("rb") as f:
        router = pickle.load(f)
    item_head = ItemHead(rank=config.rank_r, num_items=item_vocab.num_items, device=device)
    item_head.load_state_dict(item_head_state["W"])
    centers = router.get("centers")
    num_regions = centers.shape[0]
    region_bank = RegionStateBank(num_regions=num_regions, rank=config.rank_r, device=device)
    return config, subspace, item_head, centers, region_bank


def route(hidden: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    hidden_norm = F.normalize(hidden, dim=-1)
    centers = centers.to(hidden.device)
    centers_norm = F.normalize(centers, dim=-1)
    sims = hidden_norm @ centers_norm.t()
    return sims.argmax(dim=1)


def extract_target(batch, model_type: str) -> torch.Tensor:
    if model_type == "causal":
        return batch["target_item"]
    labels = batch["labels"]
    targets = []
    for row in labels:
        positives = row[row >= 0]
        targets.append(positives[0] if positives.numel() > 0 else torch.tensor(0, device=row.device))
    return torch.stack(targets)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    item_vocab = ItemVocab.from_metadata(args.data_dir)
    config, subspace, item_head, centers, state_bank = load_artifacts(args, item_vocab, device)

    if args.model_type == "causal":
        tokenizer_path = args.ckpt_dir / "tokenizer"
        model = CausalLMStreamModel(
            str(args.ckpt_dir / "model"),
            item_vocab,
            device,
            tokenizer_name_or_path=str(tokenizer_path) if tokenizer_path.exists() else None,
        )
        tokenizer = model.tokenizer
    else:
        model = BertStreamModel(item_vocab, device)
        model.model = BertForMaskedLM.from_pretrained(args.ckpt_dir / "model").to(device)
        tokenizer = None

    splits = load_all_splits(args.data_dir)
    _, data_loader = build_dataloader(
        splits["finetune"],
        model_type=args.model_type,
        batch_size=1,
        shuffle=False,
        item_vocab=item_vocab,
        tokenizer=tokenizer,
    )

    detectors = [
        DriftDetector(
            window_m=args.window_m,
            glr_threshold=args.glr_threshold,
            conformal_alpha=args.conformal_alpha,
            conformal_q=args.conformal_q,
        )
        for _ in range(centers.shape[0])
    ]

    total = 0
    hits_base = 0
    hits_final = 0
    drift_events = 0

    for batch in data_loader:
        total += 1
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        hidden = model.stream_hidden_states(batch)
        region = route(hidden, centers.to(device))
        region_id = int(region.item())
        base_logits = model.stream_base_logits(batch)
        state = state_bank.get_state(region_id)
        overlay_logits = apply_overlay(base_logits, state, item_head)
        probs_base = F.softmax(base_logits / args.temperature, dim=-1)
        probs_final = F.softmax(overlay_logits / args.temperature, dim=-1)
        target = extract_target(batch, args.model_type)
        top_base = int(probs_base.argmax(dim=-1).item())
        top_final = int(probs_final.argmax(dim=-1).item())
        true_item = int(target.item())
        hits_base += int(top_base == true_item)
        hits_final += int(top_final == true_item)

        click = int(top_final == true_item)
        triggered = detectors[region_id].update({"probs": probs_final.squeeze().detach().cpu().numpy(), "click": click})
        if triggered:
            grads = model.stream_positive_gradients(batch).squeeze(0)
            g = grads
            probs = probs_final.squeeze(0)
            diag = probs * (1 - probs)
            W = item_head.W
            F_mat = (W * diag.unsqueeze(0)) @ W.t()
            delta = solve_delta_s(g, F_mat, args.lambda_l2, args.epsilon_kl)
            state_bank.apply_delta(region_id, delta)
            drift_events += 1

    LOGGER.info(
        "Processed %d events | base@1=%.3f final@1=%.3f | drifts=%d",
        total,
        hits_base / max(total, 1),
        hits_final / max(total, 1),
        drift_events,
    )


if __name__ == "__main__":
    main()
