"""Causal LM integration with STREAM."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from stream.dataio import ItemVocab, ensure_tokenizer_has_items
from stream.models import BaseStreamModel

def _init_item_token_from_text(
    model,
    tokenizer,
    token_str: str,
    name_text: str,
) -> None:
    ids = tokenizer(name_text, add_special_tokens=False).input_ids
    if len(ids) == 0:
        return

    base_emb = model.get_input_embeddings().weight.data         # [V, d]
    head_emb = model.lm_head.weight.data                        # [V, d]
    vec = base_emb[ids].mean(dim=0)                             # [d]

    tid = tokenizer.convert_tokens_to_ids(token_str)
    if tid is None or tid < 0:
        return
    base_emb[tid].copy_(vec)
    head_emb[tid].copy_(vec)


class CausalLMStreamModel(BaseStreamModel):
    def __init__(self,
        pretrained_name_or_path: str,
        item_vocab: ItemVocab,
        device: torch.device,
        tokenizer_name_or_path: str | None = None,
        torch_dtype: torch.dtype | None = torch.float16,
        device_map: str | None = "auto",
        item_name_map: dict[int, str] | None = None,
    ) -> None:
        super().__init__()
        self.device = device

        tokenizer_source = tokenizer_name_or_path or pretrained_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
        ensure_tokenizer_has_items(self.tokenizer, item_vocab)

        # 尝试使用 flash-attn；不可用则自动忽略该参数
        model_kwargs = {}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_name_or_path,
            **model_kwargs
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        if item_name_map:
            for i in range(self.num_items):
                tok = item_vocab.token_for(i)
                name_txt = item_name_map.get(i, "")
                if name_txt:
                    _init_item_token_from_text(self.model, self.tokenizer, tok, name_txt)

        self.model.to(device if device_map is None else self.model.device)

        self.item_vocab = item_vocab
        self.num_items = item_vocab.num_items
        self.item_token_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(item_vocab.token_for(i)) for i in range(self.num_items)],
            device=(device if device_map is None else self.model.device),
            dtype=torch.long,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.lm_head_weight: nn.Parameter = self.model.lm_head.weight  # type: ignore[attr-defined]

    def _last_hidden(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = {k: v.to(self.device) for k, v in batch.items() if k in {"input_ids", "attention_mask"}}
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        lengths = inputs["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_indices, lengths]
        return last_hidden

    def stream_hidden_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._last_hidden(batch).detach()

    def stream_base_logits(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self._last_hidden(batch)
        item_weights = self.lm_head_weight[self.item_token_ids]
        logits = hidden @ item_weights.t()
        return logits

    def stream_positive_gradients(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self._last_hidden(batch).detach().requires_grad_(True)
        item_weights = self.lm_head_weight[self.item_token_ids]
        logits = hidden @ item_weights.t()
        target = batch["target_item"].to(hidden.device)
        positive = logits.gather(1, target.unsqueeze(1)).squeeze(1)
        grads = torch.autograd.grad(positive.sum(), hidden)[0]
        return grads.detach()


__all__ = ["CausalLMStreamModel"]
