"""Unified evaluation runner for distribution-shift protocols."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from stream.adapters import LoRAConfig, apply_lora_adapters, set_trainable_layers
from stream.dataio import ItemVocab, build_dataloader, load_all_splits
from stream.eval_protocols import (
    ProtocolConfig,
    ShiftEvalExample,
    build_protocol_examples,
    build_user_timelines,
    load_item_text_map,
)
from stream.metrics import ndcg_at_k, recall_at_k
from stream.models import BaseStreamModel
from stream.models.bert_stream import BertStreamModel
from stream.models.causal_lm_stream import CausalLMStreamModel
from stream.utils import get_logger, set_seed
from transformers import BertForMaskedLM


LOGGER = get_logger(__name__)

PROTOCOLS: Dict[str, ProtocolConfig] = {
    "O-Tail": ProtocolConfig(name="O-Tail", history_mode="no_concat", eval_split="original"),
    "T-NoConcat": ProtocolConfig(name="T-NoConcat", history_mode="no_concat", eval_split="test"),
    "T-Concat": ProtocolConfig(name="T-Concat", history_mode="concat", eval_split="test"),
    "T-FineTune": ProtocolConfig(
        name="T-FineTune", history_mode="no_concat", eval_split="test", requires_finetune=True
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate distribution-shift protocols")
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing original/finetune/test JSONL files")
    parser.add_argument("--model_dir", type=Path, required=True, help="Directory with the pretrained model weights")
    parser.add_argument("--tokenizer_dir", type=Path, default=None, help="Tokenizer directory (causal models)")
    parser.add_argument("--model_type", choices=["bert", "causal"], required=True)
    parser.add_argument("--protocol", choices=list(PROTOCOLS.keys()), required=True)
    parser.add_argument("--history_mode", choices=["no_concat", "concat"], default=None)
    parser.add_argument("--history_window", type=int, default=50)
    parser.add_argument("--metrics_k", nargs="*", type=int, default=[5, 10, 20])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_prompt_length", type=int, default=256, help="Max tokens for causal prompts")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory to store results")
    parser.add_argument("--finetune_steps", type=int, default=0)
    parser.add_argument("--finetune_lr", type=float, default=5e-5)
    parser.add_argument("--finetune_adapter", choices=["none", "lora"], default="none")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--subspace_rank", type=int, default=0, help="Number of top transformer blocks to keep trainable when adapter=none")
    parser.add_argument("--finetune_batch_size", type=int, default=None)
    parser.add_argument("--report_precision", type=int, default=4)
    return parser.parse_args()


class ShiftEvalDataset(Dataset):
    """Dataset that materialises protocol-specific rolling histories."""

    def __init__(
        self,
        examples: Sequence[ShiftEvalExample],
        *,
        model_type: str,
        item_vocab: ItemVocab,
        tokenizer=None,
        max_prompt_length: int = 256,
        item_text_map: Dict[int, str] | None = None,
    ) -> None:
        if model_type == "causal" and tokenizer is None:
            raise ValueError("tokenizer must be provided for causal evaluation")
        self.examples = list(examples)
        self.model_type = model_type
        self.item_vocab = item_vocab
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.item_text_map = item_text_map or {}
        if model_type == "causal":
            token = item_vocab.mask_token
            mask_id = tokenizer.convert_tokens_to_ids(token)
            if mask_id is None or mask_id < 0:
                raise ValueError(f"Tokenizer missing mask token {token}")
            self.mask_token_id = int(mask_id)
        else:
            self.mask_token_id = -1

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.examples)

    def _history_entry(self, item_idx: int) -> str:
        text = self.item_text_map.get(item_idx, "").strip()
        token = self.item_vocab.token_for(item_idx)
        if text:
            text = text.replace(" | ", "; ")
            return f"{token} ({text})"
        return token

    def _build_prompt(self, example: ShiftEvalExample) -> Dict[str, List[int]]:
        history_lines = [self._history_entry(idx) for idx in example.history]
        prompt = (
            f"The following are items that user {example.user} recently engaged with:\n"
            + "\n".join(history_lines)
            + "\n\nPredict the next item token only."
        )
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_prompt_length - 1,
        )
        input_ids = list(encoded["input_ids"]) + [self.mask_token_id]
        attention = list(encoded["attention_mask"]) + [1]
        labels = [-100] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}

    def __getitem__(self, idx: int) -> Dict[str, List[int] | int]:  # type: ignore[override]
        example = self.examples[idx]
        if self.model_type == "bert":
            history = list(example.history)
            attention = [1] * len(history)
            labels = [-100] * len(history)
            return {
                "input_ids": history,
                "attention_mask": attention,
                "labels": labels,
                "target_item": example.target,
            }
        data = self._build_prompt(example)
        data["target_item"] = example.target
        return data


def _pad_sequences(seqs: Sequence[Sequence[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in seqs)
    result = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for idx, seq in enumerate(seqs):
        result[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return result


def bert_eval_collate(batch: List[Dict[str, Sequence[int] | int]], pad_id: int) -> Dict[str, torch.Tensor]:
    input_ids = _pad_sequences([sample["input_ids"] for sample in batch], pad_id)
    attention = _pad_sequences([sample["attention_mask"] for sample in batch], 0)
    labels = _pad_sequences([sample["labels"] for sample in batch], -100)
    targets = torch.tensor([int(sample["target_item"]) for sample in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels, "target_item": targets}


def causal_eval_collate(batch: List[Dict[str, Sequence[int] | int]], pad_id: int) -> Dict[str, torch.Tensor]:
    input_ids = _pad_sequences([sample["input_ids"] for sample in batch], pad_id)
    attention = _pad_sequences([sample["attention_mask"] for sample in batch], 0)
    labels = _pad_sequences([sample["labels"] for sample in batch], -100)
    targets = torch.tensor([int(sample["target_item"]) for sample in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels, "target_item": targets}


def setup_model(
    args: argparse.Namespace, item_vocab: ItemVocab, device: torch.device
) -> tuple[BaseStreamModel, object | None]:
    if args.model_type == "bert":
        model = BertStreamModel(item_vocab, device)
        model.model = BertForMaskedLM.from_pretrained(str(args.model_dir)).to(device)
        return model, None
    tokenizer_path = args.tokenizer_dir or args.model_dir
    model = CausalLMStreamModel(
        pretrained_name_or_path=str(args.model_dir),
        item_vocab=item_vocab,
        device=device,
        tokenizer_name_or_path=str(tokenizer_path),
    )
    return model, model.tokenizer


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def compute_metrics(predictions: List[List[int]], positives: List[List[int]], ks: Sequence[int]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for k in ks:
        recalls = [recall_at_k(pred[:k], pos, k) for pred, pos in zip(predictions, positives)]
        ndcgs = [ndcg_at_k(pred[:k], pos, k) for pred, pos in zip(predictions, positives)]
        metrics[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
    return metrics


def run_evaluation(
    model: BaseStreamModel,
    dataloader: DataLoader,
    device: torch.device,
    metrics_k: Sequence[int],
) -> tuple[Dict[str, float], int]:
    model.eval()
    max_k = max(metrics_k)
    predictions: List[List[int]] = []
    positives: List[List[int]] = []
    with torch.no_grad():
        for batch in dataloader:
            tensor_batch = move_batch_to_device(batch, device)
            logits = model.stream_base_logits(tensor_batch)
            _, indices = torch.topk(logits, k=max_k, dim=-1)
            predictions.extend(indices.detach().cpu().tolist())
            positives.extend([[int(x)] for x in tensor_batch["target_item"].detach().cpu().tolist()])
    metrics = compute_metrics(predictions, positives, metrics_k)
    return metrics, len(predictions)


def _trainable_parameters(module: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def _ensure_head_trainable(model: BaseStreamModel) -> None:
    if hasattr(model, "model"):
        internal = getattr(model, "model")
        for attr in ["lm_head", "cls", "score"]:
            if hasattr(internal, attr):
                for param in getattr(internal, attr).parameters():
                    param.requires_grad = True


def fine_tune_if_needed(
    args: argparse.Namespace,
    config: ProtocolConfig,
    model: BaseStreamModel,
    tokenizer,
    item_vocab: ItemVocab,
    splits: Dict[str, List[Dict]],
    device: torch.device,
    *,
    batch_size: int,
) -> tuple[float, int]:
    if not config.requires_finetune or args.finetune_steps <= 0:
        return 0.0, 0

    finetune_records = splits.get("finetune", [])
    if not finetune_records:
        LOGGER.warning("Finetune split is empty; skipping parameter adaptation")
        return 0.0, 0

    dataset, dataloader = build_dataloader(
        finetune_records,
        model_type=args.model_type,
        batch_size=batch_size,
        shuffle=True,
        item_vocab=item_vocab,
        tokenizer=tokenizer,
        num_workers=args.num_workers,
    )
    if len(dataset) == 0:
        LOGGER.warning("Finetune dataset produced zero samples; skipping adaptation")
        return 0.0, 0

    internal_model = getattr(model, "model")
    if args.finetune_adapter == "lora":
        for param in internal_model.parameters():
            param.requires_grad = False
        config_lora = LoRAConfig(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
        replaced = apply_lora_adapters(internal_model, config_lora)
        _ensure_head_trainable(model)
        LOGGER.info("Inserted LoRA adapters into %d modules", len(replaced))
    else:
        if args.subspace_rank > 0:
            for param in internal_model.parameters():
                param.requires_grad = False
            set_trainable_layers(internal_model, keep_last=args.subspace_rank)
            _ensure_head_trainable(model)
            LOGGER.info("Unfroze last %d transformer blocks for finetuning", args.subspace_rank)
        else:
            for param in internal_model.parameters():
                param.requires_grad = True

    trainable = _trainable_parameters(internal_model)
    if not trainable:
        LOGGER.warning("No trainable parameters found; skipping adaptation")
        return 0.0, 0

    optimizer = torch.optim.AdamW(trainable, lr=args.finetune_lr)
    internal_model.train()
    iterator = iter(dataloader)
    steps = 0
    start = time.perf_counter()
    while steps < args.finetune_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        batch = move_batch_to_device(batch, device)
        inputs = {k: v for k, v in batch.items() if k in {"input_ids", "attention_mask", "labels"}}
        outputs = internal_model(**inputs, return_dict=True)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1
    if device.type == "cuda":  # type: ignore[attr-defined]
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    internal_model.eval()
    return elapsed, steps


def _format_seconds(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60.0)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours)}h{int(minutes)}m{secs:05.2f}s"
    if minutes >= 1:
        return f"{int(minutes)}m{secs:05.2f}s"
    return f"{secs:.2f}s"


def _collect_memory(device: torch.device) -> float:
    if device.type == "cuda":  # type: ignore[attr-defined]
        usage = torch.cuda.max_memory_allocated(device)
        return usage / (1024 ** 2)
    import resource
    import sys

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 ** 2)
    return usage / 1024.0


def _write_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _update_report(
    out_dir: Path,
    results: Dict[str, Dict],
    *,
    precision: int,
) -> None:
    order = ["O-Tail", "T-NoConcat", "T-Concat", "T-FineTune"]
    lines: List[str] = []
    for proto in order:
        entry = results.get(proto)
        if not entry:
            continue
        metrics = entry.get("metrics", {})
        metric_bits = [
            f"recall@{k}={metrics.get(f'recall@{k}', 0.0):.{precision}f}"
            for k in sorted({int(key.split("@")[-1]) for key in metrics if key.startswith("recall@")})
        ]
        metric_bits.extend(
            f"ndcg@{k}={metrics.get(f'ndcg@{k}', 0.0):.{precision}f}"
            for k in sorted({int(key.split("@")[-1]) for key in metrics if key.startswith("ndcg@")})
        )
        runtime = entry.get("evaluation_time", 0.0)
        memory = entry.get("peak_memory_mb", 0.0)
        lines.append(
            f"{proto}: {' | '.join(metric_bits)} | eval_time={_format_seconds(runtime)} | peak_mem={memory:.1f}MB"
        )
    report_path = out_dir / "report.txt"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _update_chart(out_dir: Path, results: Dict[str, Dict]) -> None:
    order = ["O-Tail", "T-NoConcat", "T-Concat", "T-FineTune"]
    if not all(proto in results for proto in order):
        return
    recalls = [results[proto]["metrics"].get("recall@10", 0.0) for proto in order]
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    bars = ax.bar(order, recalls, color=colors)
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Distribution Shift Comparison")
    for bar, value in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "recall10_comparison.png", dpi=300)
    plt.close(fig)


def update_results(
    out_dir: Path,
    protocol: str,
    result: Dict,
    *,
    precision: int,
) -> None:
    results_path = out_dir / "results.json"
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as handle:
            results = json.load(handle)
    else:
        results = {}
    results[protocol] = result
    _write_json(results_path, results)
    _update_report(out_dir, results, precision=precision)
    _update_chart(out_dir, results)


def main() -> None:
    args = parse_args()
    config = PROTOCOLS[args.protocol]
    if args.history_window <= 0:
        raise ValueError("history_window must be positive")
    metrics_k = sorted({int(k) for k in args.metrics_k if k > 0})
    if not metrics_k:
        raise ValueError("At least one positive metrics_k must be provided")
    if args.history_mode is None:
        args.history_mode = config.history_mode
    elif args.history_mode != config.history_mode:
        raise ValueError(
            f"Protocol {args.protocol} enforces history_mode={config.history_mode}; received {args.history_mode}"
        )

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    LOGGER.info("Using device %s", device)
    set_seed(args.seed)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_all_splits(args.data_dir)
    timelines = build_user_timelines(splits)
    examples = build_protocol_examples(args.protocol, timelines, args.history_window)
    if not examples:
        raise RuntimeError(f"Protocol {args.protocol} produced no evaluation examples")
    LOGGER.info("Constructed %d evaluation examples", len(examples))

    item_vocab = ItemVocab.from_metadata(args.data_dir)
    model, tokenizer = setup_model(args, item_vocab, device)
    internal_model = getattr(model, "model")
    internal_model.eval()

    item_text_map = load_item_text_map(args.data_dir)
    dataset = ShiftEvalDataset(
        examples,
        model_type=args.model_type,
        item_vocab=item_vocab,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        item_text_map=item_text_map,
    )
    if len(dataset) == 0:
        raise RuntimeError("Evaluation dataset is empty")

    if args.model_type == "bert":
        collate_fn = lambda batch: bert_eval_collate(batch, item_vocab.pad_id)
    else:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        collate_fn = lambda batch, pad_id=pad_id: causal_eval_collate(batch, pad_id)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    finetune_batch_size = args.finetune_batch_size or args.batch_size
    finetune_time, finetune_steps = fine_tune_if_needed(
        args,
        config,
        model,
        tokenizer,
        item_vocab,
        splits,
        device,
        batch_size=finetune_batch_size,
    )
    if finetune_steps:
        LOGGER.info("Completed %d finetune steps in %s", finetune_steps, _format_seconds(finetune_time))

    if device.type == "cuda":  # type: ignore[attr-defined]
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    metrics, total_examples = run_evaluation(model, dataloader, device, metrics_k)
    if device.type == "cuda":  # type: ignore[attr-defined]
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_memory = _collect_memory(device)

    LOGGER.info(
        "Protocol %s | examples=%d | %s | peak_mem=%.1fMB",
        args.protocol,
        total_examples,
        _format_seconds(elapsed),
        peak_memory,
    )

    result = {
        "protocol": args.protocol,
        "history_window": args.history_window,
        "metrics": metrics,
        "metrics_k": metrics_k,
        "evaluation_time": elapsed,
        "peak_memory_mb": peak_memory,
        "num_examples": total_examples,
        "model_type": args.model_type,
        "finetune_steps": finetune_steps,
        "finetune_time": finetune_time,
        "finetune_adapter": args.finetune_adapter,
    }

    protocol_path = out_dir / f"protocol_{args.protocol.replace('-', '_').lower()}.json"
    _write_json(protocol_path, result)
    update_results(out_dir, args.protocol, result, precision=args.report_precision)


if __name__ == "__main__":
    main()
