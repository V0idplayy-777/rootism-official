"""Rootism decoder-only language model.

Nothing here is loaded from an external model: embeddings, layers, norms,
rotary tables, cache and the LM head are all constructed from `ModelConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..config import ModelConfig
from .block import RootismDecoderLayer
from .kv_cache import RootismKVCache
from .losses import chunked_causal_lm_loss
from .normalization import RMSNorm
from .rotary import compute_rotary_embeddings


@dataclass
class RootismOutput:
    last_hidden_state: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    loss_sum: torch.Tensor | None = None
    num_valid_tokens: torch.Tensor | None = None
    past_key_values: RootismKVCache | None = None


def packed_position_ids(document_ids: torch.Tensor) -> torch.Tensor:
    """RoPE positions reset at every packed-document boundary."""
    bsz, seq_len = document_ids.shape
    idx = torch.arange(seq_len, device=document_ids.device, dtype=torch.long).expand(bsz, seq_len)
    change = torch.ones_like(document_ids, dtype=torch.bool)
    change[:, 1:] = document_ids[:, 1:] != document_ids[:, :-1]
    markers = torch.where(change, idx, torch.full_like(idx, -1))
    group_start, _ = torch.cummax(markers, dim=1)
    return (idx - group_start.clamp(min=0)).to(torch.long)


class RootismModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList(
            [RootismDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        past_key_values: RootismKVCache | None = None,
        use_cache: bool | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, RootismKVCache | None]:
        cfg = self.config
        use_cache = cfg.use_cache if use_cache is None else use_cache
        bsz, q_len = input_ids.shape
        device = input_ids.device

        if past_key_values is not None and document_ids is not None:
            raise ValueError("packed document_ids are only supported without a KV cache (training path)")

        if position_ids is None:
            if document_ids is not None:
                position_ids = packed_position_ids(document_ids)
            else:
                start = past_key_values.length if past_key_values is not None else 0
                position_ids = (
                    torch.arange(start, start + q_len, device=device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(bsz, -1)
                )
        position_ids = position_ids.to(torch.long)

        if past_key_values is not None:
            past_key_values.prepare(position_ids)
        kv_len = past_key_values._next_length if past_key_values is not None else q_len
        padding_mask = None
        if attention_mask is not None:
            padding_mask = attention_mask.to(torch.bool)
            if padding_mask.shape[1] < kv_len:
                # History in the cache is always valid; pad the left of the mask.
                prefix = torch.ones(
                    (bsz, kv_len - padding_mask.shape[1]),
                    dtype=torch.bool,
                    device=device,
                )
                padding_mask = torch.cat([prefix, padding_mask], dim=1)

        hidden_states = self.embed_tokens(input_ids)
        cos, sin, temperature = compute_rotary_embeddings(
            cfg, position_ids, hidden_states.dtype, hidden_states.device,
            seq_len=kv_len,
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cos=cos,
                sin=sin,
                temperature=temperature,
                config=cfg,
                position_ids=position_ids,
                cache=past_key_values if use_cache else None,
                padding_mask=padding_mask,
                document_ids=document_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                training=self.training,
            )
        hidden_states = self.norm(hidden_states)
        if past_key_values is not None:
            past_key_values.commit()
        return hidden_states, past_key_values if use_cache else None


class RootismForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = RootismModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            # With tying, the input embedding parameter object IS the output
            # projection. No second matrix is registered.
            self.lm_head = None

    def get_output_weight(self) -> torch.Tensor:
        if self.config.tie_word_embeddings:
            return self.model.get_input_embeddings().weight
        return self.lm_head.weight  # type: ignore[union-attr]

    def _logits_fn(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.get_output_weight())

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        past_key_values: RootismKVCache | None = None,
        use_cache: bool | None = None,
        labels: torch.Tensor | None = None,
        num_logits_to_keep: int = 1,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        **_: Any,
    ) -> RootismOutput:
        hidden_states, cache = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            document_ids=document_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        if labels is not None:
            # Causal shift: hidden[t] predicts labels[t+1].
            shifted_hidden = hidden_states[:, :-1, :]
            shifted_labels = labels[:, 1:].contiguous()
            loss_sum, valid = chunked_causal_lm_loss(
                shifted_hidden,
                shifted_labels,
                self._logits_fn,
                chunk_size=self.config.loss_chunk_size,
            )
            loss = loss_sum / valid.clamp(min=1).to(loss_sum.dtype)
            return RootismOutput(
                last_hidden_state=hidden_states,
                loss=loss,
                loss_sum=loss_sum,
                num_valid_tokens=valid,
                past_key_values=cache,
            )
        kept = hidden_states[:, -num_logits_to_keep:, :]
        return RootismOutput(
            last_hidden_state=hidden_states,
            logits=self._logits_fn(kept),
            past_key_values=cache,
        )
