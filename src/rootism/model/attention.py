"""Grouped-query attention with memory-efficient execution (requirements 13, 14).

Paths, in order of preference:

1. `flash_attn_varlen_func` (optional dependency) for packed sequences: O(T)
   memory, true block-diagonal causality, native GQA support.
2. PyTorch SDPA with `is_causal=True` and no mask (Flash/Memory-Efficient
   kernel selected by PyTorch) for ordinary unpadded prefill. No attention
   matrix is materialised.
3. Chunked SDPA with an explicitly constructed boolean mask for padded and/or
   packed sequences. Only a (chunk_q x kv_len) boolean mask is materialised,
   never a full attention score matrix; chunking bounds mask memory to
   O(chunk_q * kv_len) instead of O(kv_len^2) at 32K context.

Numerical stability: SDPA handles softmax stability internally; YaRN's
attention temperature is folded into the softmax scale.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from ..config import ModelConfig

try:  # optional acceleration, never a hard dependency
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as _flash_varlen_func
except Exception:  # pragma: no cover - depends on optional install
    _flash_varlen_func = None


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, KV_heads, T, D) -> (B, Q_heads, T, D) without copying when n_rep=1."""
    if n_rep == 1:
        return x
    bsz, kv_heads, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bsz, kv_heads, n_rep, seq_len, head_dim)
        .reshape(bsz, kv_heads * n_rep, seq_len, head_dim)
    )


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    is_causal: bool,
    scale: float,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    n_rep = q.shape[1] // k.shape[1]
    kwargs = dict(
        attn_mask=attn_mask,
        dropout_p=dropout_p if training else 0.0,
        is_causal=is_causal,
        scale=scale,
    )
    if n_rep > 1:
        # torch>=2.5 supports native GQA (no KV replication). Older builds or
        # backends that reject enable_gqa+mask fall back to a broadcast view.
        try:
            if "enable_gqa" in (F.scaled_dot_product_attention.__doc__ or ""):
                return F.scaled_dot_product_attention(q, k, v, enable_gqa=True, **kwargs)
        except (RuntimeError, TypeError):
            pass
        k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
    return F.scaled_dot_product_attention(q, k, v, **kwargs)


def attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    config: ModelConfig,
    temperature: float = 1.0,
    padding_mask: torch.Tensor | None = None,
    document_ids: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    query_chunk_size: int = 2_048,
    training: bool = False,
) -> torch.Tensor:
    """Dispatch attention. All tensors are (B, heads, T, head_dim).

    `document_ids` (B, kv_len) marks packed-document membership; tokens in
    different documents can never attend to one another, preserving causal
    semantics across sample boundaries (requirement 53).
    """
    bsz, _, q_len, head_dim = q.shape
    kv_len = k.shape[2]
    device = q.device
    q_start = kv_len - q_len  # queries occupy the last q_len slots of the packed axis
    scale = 1.0 / (math.sqrt(head_dim) * temperature)

    # ---- optional FlashAttention varlen path for packed training ----------
    if cu_seqlens is not None and _flash_varlen_func is None and document_ids is None:
        raise ValueError(
            "packed training with cu_seqlens needs flash-attn, or explicit "
            "document_ids so the chunked-mask fallback can isolate documents"
        )
    if cu_seqlens is not None and _flash_varlen_func is not None:
        q_f = q.transpose(1, 2).reshape(-1, q.shape[1], head_dim).to(q.dtype)
        k_f = k.transpose(1, 2).reshape(-1, k.shape[1], head_dim)
        v_f = v.transpose(1, 2).reshape(-1, v.shape[1], head_dim)
        if max_seqlen is None:
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        out = _flash_varlen_func(
            q_f, k_f, v_f,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            dropout_p=config.attention_dropout if training else 0.0,
            softmax_scale=scale, causal=True,
        )
        return out.reshape(bsz, q_len, q.shape[1], head_dim).transpose(1, 2)

    # ---- fast path: no explicit mask needed -------------------------------
    plain = padding_mask is None and document_ids is None
    if plain and q_len == kv_len:
        return _sdpa(
            q, k, v, attn_mask=None, is_causal=True, scale=scale,
            dropout_p=config.attention_dropout, training=training,
        )
    if plain and q_len == 1:
        # single-token decode: every cached key is in the past, so no mask
        return _sdpa(
            q, k, v, attn_mask=None, is_causal=False, scale=scale,
            dropout_p=config.attention_dropout, training=training,
        )

    # ---- chunked explicit mask: bounded memory, exact semantics -----------
    k_idx = torch.arange(kv_len, device=device)
    outputs = []
    for q0 in range(0, q_len, query_chunk_size):
        q1 = min(q0 + query_chunk_size, q_len)
        q_idx = torch.arange(q0, q1, device=device) + q_start
        allowed = k_idx[None, :] <= q_idx[:, None]  # (cq, kv)
        if document_ids is not None:
            q_docs = document_ids[:, q_start + q0 : q_start + q1]
            same_doc = q_docs[:, :, None] == document_ids[:, None, :]
            allowed = allowed[None, :, :] & same_doc  # (B, cq, kv)
        else:
            allowed = allowed.expand(bsz, -1, -1)
        if padding_mask is not None:
            allowed = allowed & padding_mask[:, None, :].to(torch.bool)
        mask = allowed[:, None, :, :]  # (B, 1, cq, kv); True = allowed
        outputs.append(
            _sdpa(
                q[:, :, q0:q1], k, v, attn_mask=mask, is_causal=False, scale=scale,
                dropout_p=config.attention_dropout, training=training,
            )
        )
    return torch.cat(outputs, dim=2)


class RootismAttention(nn.Module):
    """GQA projection block. Rotary and cache are supplied by the backbone."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_rep = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        if config.qk_norm:
            from .normalization import RMSNorm

            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = self.k_norm = None

    def project(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        return q, k, v

    def output(self, context: torch.Tensor) -> torch.Tensor:
        bsz, _, seq_len, _ = context.shape
        return self.o_proj(context.transpose(1, 2).reshape(bsz, seq_len, -1))
