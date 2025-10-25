"""BERT-style masked item prediction model for STREAM."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from transformers import BertConfig, BertForMaskedLM

from stream.dataio import ItemVocab
from stream.models import BaseStreamModel


class BertStreamModel(BaseStreamModel):
    def __init__(self, item_vocab: ItemVocab, device: torch.device) -> None:
        super().__init__()
        config = BertConfig(
            vocab_size=item_vocab.bert_vocab_size,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=512,
            pad_token_id=item_vocab.pad_id,
        )
        self.model = BertForMaskedLM(config)
        self.model.to(device)
        self.device = device
        self.item_vocab = item_vocab
        self.num_items = item_vocab.num_items

    @property
    def decoder_weight(self) -> nn.Parameter:
        return self.model.cls.predictions.decoder.weight[: self.num_items]

    @property
    def decoder_bias(self) -> nn.Parameter:
        return self.model.cls.predictions.decoder.bias[: self.num_items]

    def _encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = {k: v.to(self.device) for k, v in batch.items() if k in {"input_ids", "attention_mask"}}
        outputs = self.model.bert(**inputs, return_dict=True)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (hidden * mask).sum(dim=1) / denom
        return pooled

    def stream_hidden_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._encode(batch).detach()

    def stream_base_logits(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pooled = self._encode(batch)
        logits = pooled @ self.decoder_weight.t() + self.decoder_bias
        return logits

    def _extract_targets(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        labels = batch["labels"].to(self.device)
        targets = []
        for row in labels:
            positives = row[row >= 0]
            if positives.numel() == 0:
                targets.append(torch.tensor(0, device=self.device))
            else:
                targets.append(positives[0])
        return torch.stack(targets)

    def stream_positive_gradients(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pooled = self._encode(batch).detach().requires_grad_(True)
        logits = pooled @ self.decoder_weight.t() + self.decoder_bias
        targets = self._extract_targets(batch)
        positive = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        grads = torch.autograd.grad(positive.sum(), pooled)[0]
        return grads.detach()


__all__ = ["BertStreamModel"]
