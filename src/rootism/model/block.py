"""Pre-norm decoder layer with stable residual pathways (requirements 12, 18)."""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .attention import RootismAttention, attention_forward
from .feedforward import SwiGLUMLP
from .normalization import RMSNorm
from .rotary import apply_rotary_pos_emb


class RootismDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = RootismAttention(config, layer_idx)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        temperature: float,
        config: ModelConfig,
        position_ids: torch.Tensor,
        cache=None,
        padding_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        training: bool = False,
    ) -> torch.Tensor:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        q, k, v = self.self_attn.project(h)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v, position_ids)
        context = attention_forward(
            q, k, v,
            config=config,
            temperature=temperature,
            padding_mask=padding_mask,
            document_ids=document_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            query_chunk_size=config.attention_query_chunk_size,
            training=training,
        )
        hidden_states = residual + self.self_attn.output(context)

        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states
