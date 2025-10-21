"""Utilities for distilling STREAM state and applying optional merges."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch

from .consolidate import distill_item_head, maybe_apply_lora_merge
from .config import StreamConfig
from .state_adapter import ItemHead
from .utils import get_logger

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill STREAM state and optional merges")
    parser.add_argument("--ckpt_dir", type=Path, required=True)
    parser.add_argument("--updates_path", type=Path, default=None)
    parser.add_argument("--distill_every", type=int, default=None, help="Seconds between distillations")
    parser.add_argument("--lora_merge_policy", type=Path, default=None)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_deltas(path: Path | None, device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if path is None or not path.exists():
        LOGGER.info("No update file found for distillation")
        return []
    data = torch.load(path)
    deltas = []
    for item in data:
        delta_s = torch.tensor(item["delta_s"], device=device, dtype=torch.float32)
        item_vector = torch.tensor(item["item_vector"], device=device, dtype=torch.float32)
        deltas.append((delta_s, item_vector))
    return deltas


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = StreamConfig.from_json(args.ckpt_dir / "config.json")
    item_head_state = torch.load(args.ckpt_dir / "item_head.pt")
    item_head = ItemHead(rank=config.rank_r, num_items=item_head_state["num_items"], device=device)
    item_head.load_state_dict(item_head_state["W"])

    deltas = load_deltas(args.updates_path, device)
    if deltas:
        distill_item_head(item_head, deltas, eta=args.eta)
        torch.save(
            {"W": item_head.state_dict(), "num_items": item_head.num_items, "rank": config.rank_r},
            args.ckpt_dir / "item_head.pt",
        )
        LOGGER.info("Distillation applied to item head")

    if args.lora_merge_policy:
        maybe_apply_lora_merge(None, args.lora_merge_policy, torch.empty(0))


if __name__ == "__main__":
    main()
