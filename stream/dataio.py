"""Data loading and preprocessing utilities for STREAM."""
from __future__ import annotations

import json
import random
from functools import partial
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


def load_jsonl_split(path: Path) -> List[Dict]:
    """Load a JSONL file into a list of dictionaries."""

    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


class ItemVocab:
    """Vocabulary helper that maps item ids to special tokens."""

    def __init__(self, num_items: int, prefix: str = "<item_{}>") -> None:
        self.num_items = int(num_items)
        self.prefix = prefix
        self.pad_token = "<pad>"
        self.mask_token = "<mask_item>"

    @classmethod
    def from_metadata(cls, data_dir: Path) -> "ItemVocab":
        meta_path = data_dir / "item_ids.json"
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        mid2idx = meta.get("mid2idx", {})
        if not mid2idx:
            raise ValueError("item_ids.json must contain a 'mid2idx' mapping")
        max_idx = max(int(idx) for idx in mid2idx.values())
        return cls(num_items=max_idx + 1)

    def token_for(self, item_idx: int) -> str:
        if not (0 <= item_idx < self.num_items):
            raise IndexError(f"item index {item_idx} out of range")
        return self.prefix.format(item_idx)

    def tokens(self) -> List[str]:
        return [self.token_for(i) for i in range(self.num_items)]

    @property
    def special_tokens(self) -> List[str]:
        return [self.pad_token, self.mask_token]

    @property
    def pad_id(self) -> int:
        return self.num_items  # appended after items

    @property
    def mask_id(self) -> int:
        return self.num_items + 1

    @property
    def bert_vocab_size(self) -> int:
        return self.num_items + 2  # pad + mask


def ensure_tokenizer_has_items(tokenizer, item_vocab: ItemVocab) -> None:
    """Extend a HuggingFace tokenizer with item tokens."""

    existing = set(tokenizer.get_vocab().keys())
    new_tokens = [tok for tok in item_vocab.tokens() + item_vocab.special_tokens if tok not in existing]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": item_vocab.pad_token})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token


class CausalLMDataset(Dataset):
    """Dataset that produces next-item prediction examples for causal LMs."""

    def __init__(
        self,
        records: Sequence[Dict],
        tokenizer,
        item_vocab: ItemVocab,
        max_history: int = 50,
        max_length: int = 128,
    ) -> None:
        self.tokenizer = tokenizer
        self.item_vocab = item_vocab
        self.max_history = max_history
        self.max_length = max_length
        self._sample_specs: List[Tuple[int, int, int]] = []
        self._record_items: List[Tuple[int, ...]] = []
        self._record_users: List[int] = []
        self._item_token_id_cache: Dict[int, int] = {}
        self._build(records)

    def _build(self, records: Sequence[Dict]) -> None:
        for rec in records:
            items: List[int] = rec.get("items", [])
            if len(items) < 2:
                continue
            user = int(rec.get("user", -1))
            record_idx = len(self._record_items)
            self._record_items.append(tuple(items))
            self._record_users.append(user)
            for idx in range(1, len(items)):
                start = max(0, idx - self.max_history)
                self._sample_specs.append((record_idx, start, idx))


    def _token_id_for_item(self, item_idx: int) -> int:
        token_id = self._item_token_id_cache.get(item_idx)
        if token_id is not None:
            return token_id
        token = self.item_vocab.token_for(item_idx)
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0:
            raise ValueError("Tokenizer missing target item token")
        self._item_token_id_cache[item_idx] = token_id
        return token_id

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._sample_specs)

    def __getitem__(self, idx: int) -> Dict[str, Sequence[int]]:  # type: ignore[override]
        record_idx, start_idx, target_pos = self._sample_specs[idx]
        items = self._record_items[record_idx]
        user = self._record_users[record_idx]
        history_items = items[start_idx:target_pos]
        target_item = items[target_pos]
        history_tokens = [self.item_vocab.token_for(i) for i in history_items]
        prompt = "User {} History: {} Next?".format(
            user,
            " ; ".join(history_tokens) if history_tokens else "<none>",
        )
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length - 1,
        )
        target_token_id = self._token_id_for_item(target_item)
        input_ids = encoded["input_ids"] + [target_token_id]
        attention_mask = encoded["attention_mask"] + [1]
        labels = [-100] * len(encoded["input_ids"]) + [target_token_id]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "target_item": target_item,
        }


def _pad_sequences(seqs: List[Sequence[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in seqs)
    padded = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return padded


def causal_lm_collate(batch: List[Dict[str, Sequence[int]]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    input_ids = _pad_sequences([sample["input_ids"] for sample in batch], pad_token_id)
    attention_mask = _pad_sequences([sample["attention_mask"] for sample in batch], 0)
    labels = _pad_sequences([sample["labels"] for sample in batch], -100)
    target_items = torch.tensor([sample["target_item"] for sample in batch], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "target_item": target_items,
    }


class BertSequenceDataset(Dataset):
    """Dataset for masked item prediction with a BERT-style model."""

    def __init__(
        self,
        records: Sequence[Dict],
        item_vocab: ItemVocab,
        max_length: int = 100,
        mask_prob: float = 0.15,
    ) -> None:
        self.item_vocab = item_vocab
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.samples: List[Dict[str, Sequence[int]]] = []
        self._build(records)

    def _apply_masking(self, sequence: List[int]) -> Tuple[List[int], List[int], List[int]]:
        length = len(sequence)
        labels = [-100] * length
        num_mask = max(1, int(round(length * self.mask_prob)))
        candidate_positions = list(range(length))
        random.shuffle(candidate_positions)
        mask_positions = candidate_positions[:num_mask]
        masked_items: List[int] = []
        for pos in mask_positions:
            original = sequence[pos]
            labels[pos] = original
            masked_items.append(original)
            rand = random.random()
            if rand < 0.8:
                sequence[pos] = self.item_vocab.mask_id
            elif rand < 0.9:
                sequence[pos] = random.randrange(self.item_vocab.num_items)
            else:
                pass  # keep original token
        return sequence, labels, mask_positions

    def _build(self, records: Sequence[Dict]) -> None:
        for rec in records:
            items: List[int] = rec.get("items", [])
            if len(items) < 1:
                continue
            sequence = items[-self.max_length :]
            sequence, labels, mask_positions = self._apply_masking(sequence[:])
            attention = [1] * len(sequence)
            self.samples.append(
                {
                    "input_ids": sequence,
                    "attention_mask": attention,
                    "labels": labels,
                    "masked_positions": mask_positions,
                }
            )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Sequence[int]]:  # type: ignore[override]
        return self.samples[idx]


def bert_collate(batch: List[Dict[str, Sequence[int]]], pad_id: int) -> Dict[str, torch.Tensor]:
    input_ids = _pad_sequences([sample["input_ids"] for sample in batch], pad_id)
    attention_mask = _pad_sequences([sample["attention_mask"] for sample in batch], 0)
    labels = _pad_sequences([sample["labels"] for sample in batch], -100)
    max_pos = max(len(sample.get("masked_positions", [])) for sample in batch)
    if max_pos == 0:
        masked_positions = torch.empty((len(batch), 0), dtype=torch.long)
    else:
        masked_positions = torch.full((len(batch), max_pos), -1, dtype=torch.long)
        for i, sample in enumerate(batch):
            positions = sample.get("masked_positions", [])
            if positions:
                masked_positions[i, : len(positions)] = torch.tensor(positions, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "masked_positions": masked_positions,
    }


def build_dataloader(
    records: Sequence[Dict],
    model_type: str,
    batch_size: int,
    shuffle: bool,
    item_vocab: ItemVocab,
    tokenizer=None,
    num_workers: int = 0,
) -> Tuple[Dataset, DataLoader]:
    """Construct a dataset and dataloader for the requested model type."""

    if model_type == "causal":
        if tokenizer is None:
            raise ValueError("tokenizer must be provided for causal LM datasets")
        dataset = CausalLMDataset(records, tokenizer=tokenizer, item_vocab=item_vocab)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        collate_fn = partial(causal_lm_collate, pad_token_id=pad_id)
    elif model_type == "bert":
        dataset = BertSequenceDataset(records, item_vocab=item_vocab)
        collate_fn = partial(bert_collate, pad_id=item_vocab.pad_id)
    else:
        raise ValueError(f"Unknown model_type {model_type}")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return dataset, dataloader


def load_all_splits(data_dir: Path) -> Dict[str, List[Dict]]:
    """Load the original, finetune and test splits from *data_dir*."""

    splits = {}
    for name in ["original", "finetune", "test"]:
        splits[name] = load_jsonl_split(data_dir / f"{name}.jsonl")
    return splits


__all__ = [
    "ItemVocab",
    "bert_collate",
    "build_dataloader",
    "causal_lm_collate",
    "ensure_tokenizer_has_items",
    "load_all_splits",
    "load_jsonl_split",
]
