"""Data loading and preprocessing utilities for STREAM."""
from __future__ import annotations

import json
import logging
import os
import random
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


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
        max_history: int = 15,
        max_length: int = 256,
        item_text_map: Dict[int, str] | None = None,
        *,
        pretokenize: bool = False,
        token_cache_dir: Path | None = None,
        cache_overwrite: bool = False,
        materialize_cache: bool = False,
        show_cache_progress: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.item_vocab = item_vocab
        self.max_history = max_history
        self.max_length = max_length
        self._sample_specs: List[Tuple[int, int, int]] = []
        self._record_items: List[Tuple[int, ...]] = []
        self._record_users: List[int] = []
        self._item_token_id_cache: Dict[int, int] = {}
        self._item_text_map = item_text_map or {}
        self._prompt_cache: Dict[int, str] = {}
        self._pretokenize = bool(pretokenize)
        self._cache_dir = token_cache_dir
        if self._cache_dir is not None:
            self._cache_dir = self._cache_dir.expanduser().resolve()
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_overwrite = cache_overwrite
        self._encoded_cache: List[Dict[str, Sequence[int]]] | None = [] if self._pretokenize else None
        self._build(records)
        if self._pretokenize:
            approx_tokens = len(self._sample_specs) * self.max_length * 3
            approx_bytes = approx_tokens * 4
            if approx_bytes > 0:
                LOGGER.info(
                    "Pre-tokenising %d samples into memory (~%.1f MB)",
                    len(self._sample_specs),
                    approx_bytes / (1024 * 1024),
                )
                if approx_bytes > 8 * 1024 * 1024 * 1024:
                    LOGGER.warning(
                        "In-memory cache may require %.1f GB; consider using --token_cache_dir to spill to disk",
                        approx_bytes / (1024 * 1024 * 1024),
                    )
        if self._pretokenize or materialize_cache:
            self._prepare_cache(materialize_disk=materialize_cache, show_progress=show_cache_progress)

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
        if self._encoded_cache is not None:
            return self._encoded_cache[idx]
        cache_path = self._cache_path(idx)
        if cache_path is not None and cache_path.exists() and not self._cache_overwrite:
            return torch.load(cache_path)
        sample = self._encode_sample(idx)
        if cache_path is not None:
            self._store_cache(cache_path, sample)
        return sample

    def _prepare_cache(self, *, materialize_disk: bool, show_progress: bool) -> None:
        if not self._sample_specs:
            return
        if materialize_disk and self._cache_dir is None and not self._pretokenize:
            LOGGER.warning(
                "Requested to materialise token cache without specifying --token_cache_dir; skipping disk cache warmup."
            )
            materialize_disk = False
        iterator = range(len(self._sample_specs))
        progress = tqdm(iterator, desc="token-cache", disable=not show_progress)
        encoded_cache: List[Dict[str, Sequence[int]]] | None = self._encoded_cache if self._pretokenize else None
        for idx in progress:
            if self._pretokenize:
                sample = self._ensure_cached(idx, want_sample=True)
                if encoded_cache is not None:
                    encoded_cache.append(sample)
            elif materialize_disk:
                self._ensure_cached(idx, want_sample=False)
        if encoded_cache is not None and len(encoded_cache) == len(self._sample_specs):
            self._encoded_cache = encoded_cache

    def _ensure_cached(self, idx: int, *, want_sample: bool) -> Dict[str, Sequence[int]]:
        cache_path = self._cache_path(idx)
        need_encode = True
        sample: Dict[str, Sequence[int]] | None = None
        if cache_path is not None and cache_path.exists() and not self._cache_overwrite:
            need_encode = False
        if need_encode:
            sample = self._encode_sample(idx)
            if cache_path is not None:
                self._store_cache(cache_path, sample)
        if want_sample:
            if sample is None:
                if cache_path is None:
                    sample = self._encode_sample(idx)
                else:
                    sample = torch.load(cache_path)
            return sample
        if sample is None and cache_path is not None and cache_path.exists():
            sample = torch.load(cache_path)
        return sample if sample is not None else self._encode_sample(idx)

    def _encode_sample(self, idx: int) -> Dict[str, Sequence[int]]:
        record_idx, start_idx, target_pos = self._sample_specs[idx]
        items = self._record_items[record_idx]
        user = self._record_users[record_idx]
        history_items = items[start_idx:target_pos]
        target_item = items[target_pos]
        history_tokens = [self._history_entry(i) for i in history_items]
        prompt = (
                f"The following are movies that user {user} has recently watched and enjoyed:\n"
                + "\n".join(
            f"{ht}" for ht in history_tokens
        )
                + "\n\nBased on this watching pattern and genre preferences, "
                  "please predict the next movie the user will most likely enjoy.\n"
                  "Answer with the next item token only."
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

    def _cache_path(self, idx: int) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"sample_{idx:08d}.pt"

    def _store_cache(self, path: Path, sample: Dict[str, Sequence[int]]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(sample, tmp_path)
        os.replace(tmp_path, path)

    def _history_entry(self, item_idx: int) -> str:
        cached = self._prompt_cache.get(item_idx)
        if cached is not None:
            return cached

        token = self.item_vocab.token_for(item_idx)
        text = self._item_text_map.get(item_idx, "").strip()
        if text:
            enriched = text.replace(" | ", "; ")
            formatted = f"{token} ({enriched})"
        else:
            formatted = token
        self._prompt_cache[item_idx] = formatted
        return formatted


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
    item_text_map: Dict[int, str] | None = None,
    sampler: Optional[torch.utils.data.Sampler] = None,
    *,
    pretokenize: bool = False,
    token_cache_dir: Path | None = None,
    cache_overwrite: bool = False,
    materialize_cache: bool = False,
    show_cache_progress: bool = False,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
) -> Tuple[Dataset, DataLoader]:
    """Construct a dataset and dataloader for the requested model type."""

    if model_type == "causal":
        if tokenizer is None:
            raise ValueError("tokenizer must be provided for causal LM datasets")
        dataset = CausalLMDataset(
            records,
            tokenizer=tokenizer,
            item_vocab=item_vocab,
            item_text_map=item_text_map,
            pretokenize=pretokenize,
            token_cache_dir=token_cache_dir,
            cache_overwrite=cache_overwrite,
            materialize_cache=materialize_cache,
            show_cache_progress=show_cache_progress,
        )
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        collate_fn = partial(causal_lm_collate, pad_token_id=pad_id)
    elif model_type == "bert":
        dataset = BertSequenceDataset(records, item_vocab=item_vocab)
        collate_fn = partial(bert_collate, pad_id=item_vocab.pad_id)
    else:
        raise ValueError(f"Unknown model_type {model_type}")
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=sampler,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    dataloader = DataLoader(**{"dataset": dataset, **loader_kwargs})
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
