"""
Rootism Grouped-Query Causal Attention System (Steps 13 & 14).

Key engineering features:
- 5:1 Grouped-Query Attention (20 query heads, 4 key/value heads, 128 head_dim in Rootism-3B),
  reducing KV-cache memory by 80% relative to Multi-Head Attention (2.50 GiB vs 12.50 GiB at 32K BF16).
- Optional per-head QK-RMSNorm (`q_norm`, `k_norm`) bounding pre-RoPE query/key norms for BF16/FP16 stability.
- Dispatches directly to `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True, enable_gqa=True)`
  when unmasked, avoiding O(S^2) attention matrix materialization on 32K–64K contexts.
- Supports `document_ids` block-diagonal causal masking for packed multi-sample training without cross-sample leakage.
- Supports incremental KV-caching (`KVCacheLayerState`) for streaming inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from rootism.config import RootismModelConfig
from rootism.model.norms import RootismRMSNorm
from rootism.model.rope import apply_rotary_pos_emb

def _probe_sdpa_enable_gqa() -> bool:
    try:
        q = torch.empty((1, 2, 2, 4), device="meta", dtype=torch.float32)
        kv = torch.empty((1, 1, 2, 4), device="meta", dtype=torch.float32)
        F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
        return True
    except TypeError:
        return False


_SDPA_SUPPORTS_ENABLE_GQA: bool = _probe_sdpa_enable_gqa()


@dataclass
class KVCacheLayerState:
    """Per-layer key/value cache state for O(1)-step autoregressive decoding."""

    key: torch.Tensor  # [batch, n_kv_heads, cached_len, head_dim]
    value: torch.Tensor  # [batch, n_kv_heads, cached_len, head_dim]

    @property
    def seq_length(self) -> int:
        return int(self.key.shape[2])

    def append(self, new_key: torch.Tensor, new_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.key = torch.cat([self.key, new_key], dim=2)
        self.value = torch.cat([self.value, new_value], dim=2)
        return self.key, self.value


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Zero-copy view expansion of KV heads from [B, n_kv_heads, S, D] to [B, n_heads, S, D]
    when needed by explicit reference math or older SDPA kernels.
    """
    if n_rep == 1:
        return hidden_states
    bsz, n_kv_heads, slen, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(bsz, n_kv_heads, n_rep, slen, head_dim)
    return expanded.reshape(bsz, n_kv_heads * n_rep, slen, head_dim)


def build_packed_causal_mask(
    q_len: int,
    kv_len: int,
    device: torch.device,
    past_len: int = 0,
    attention_mask: torch.Tensor | None = None,
    document_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Construct boolean SDPA mask (True = attend, False = masked out) for:
    - Chunked prefill with non-zero `past_len`
    - Padding masks (`attention_mask` of shape [B, kv_len] with 1=valid, 0=pad)
    - Packed multi-document causal isolation (`document_ids` of shape [B, q_len] where q_len == kv_len)
    """
    q_idx = torch.arange(past_len, past_len + q_len, device=device)[:, None]  # [q_len, 1]
    k_idx = torch.arange(0, kv_len, device=device)[None, :]  # [1, kv_len]
    causal = k_idx <= q_idx  # [q_len, kv_len]
    mask = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, kv_len]

    if document_ids is not None:
        if past_len != 0 or q_len != kv_len:
            raise ValueError("document_ids sequence packing requires past_len == 0 and q_len == kv_len")
        same_doc = document_ids[:, :, None] == document_ids[:, None, :]  # [B, q_len, kv_len]
        # Negative document_ids (e.g., -1 for trailing pad slots) never attend to other slots,
        # except self-diagonal to keep softmax denominator well-defined (avoiding 0/0 NaNs).
        valid_doc = (document_ids >= 0)[:, :, None] & (document_ids >= 0)[:, None, :]
        diag = torch.eye(q_len, dtype=torch.bool, device=device).unsqueeze(0)
        doc_mask = (same_doc & valid_doc) | diag
        mask = mask & doc_mask.unsqueeze(1)

    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape[1] != kv_len:
            raise ValueError(
                f"attention_mask must have shape [batch, {kv_len}], got {tuple(attention_mask.shape)}"
            )
        key_valid = attention_mask.to(torch.bool)[:, None, None, :]  # [B, 1, 1, kv_len]
        mask = mask & key_valid
        if past_len == 0 and q_len == kv_len:
            # Ensure padded query positions attend at least to themselves to prevent all-False row NaNs
            diag = torch.eye(q_len, dtype=torch.bool, device=device)[None, None, :, :]
            mask = mask | diag

    return mask


class RootismAttention(nn.Module):
    """Multi-head / Grouped-Query Causal Self-Attention with QK-RMSNorm and RoPE."""

    def __init__(self, config: RootismModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_key_value_groups
        self.dropout_p = config.attention_dropout
        self.scale = 1.0 / math.sqrt(float(self.head_dim))

        self.q_proj = nn.Linear(
            self.d_model, self.n_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.d_model, self.n_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.d_model, self.n_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, self.d_model, bias=config.attention_bias
        )

        if config.qk_norm:
            self.q_norm: RootismRMSNorm | None = RootismRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm: RootismRMSNorm | None = RootismRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        past_key_value: KVCacheLayerState | None = None,
        use_cache: bool = False,
        force_math_reference: bool = False,
    ) -> tuple[torch.Tensor, KVCacheLayerState | None]:
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = apply_rotary_pos_emb(q, k, cos=cos, sin=sin)

        past_len = 0
        if past_key_value is not None:
            past_len = past_key_value.seq_length
            k, v = past_key_value.append(k, v)
            next_cache: KVCacheLayerState | None = past_key_value if use_cache else None
        elif use_cache:
            next_cache = KVCacheLayerState(key=k, value=v)
        else:
            next_cache = None

        kv_len = k.shape[2]
        dropout_p = self.dropout_p if self.training else 0.0

        if force_math_reference:
            attn_output = self._reference_fp32_attention(
                q=q,
                k=k,
                v=v,
                past_len=past_len,
                attention_mask=attention_mask,
                document_ids=document_ids,
            )
        else:
            # Fast memory-efficient O(S) path whenever standard causal or single-step KV decode applies
            can_use_unmasked_fast_path = (
                attention_mask is None
                and document_ids is None
                and (past_len == 0 or q_len == 1)
            )
            if can_use_unmasked_fast_path:
                is_causal = q_len > 1 and past_len == 0
                sdpa_mask = None
            else:
                is_causal = False
                sdpa_mask = build_packed_causal_mask(
                    q_len=q_len,
                    kv_len=kv_len,
                    device=hidden_states.device,
                    past_len=past_len,
                    attention_mask=attention_mask,
                    document_ids=document_ids,
                )

            if _SDPA_SUPPORTS_ENABLE_GQA and self.num_key_value_groups > 1:
                attn_output = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=sdpa_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    enable_gqa=True,
                )
            else:
                k_exp = repeat_kv(k, self.num_key_value_groups)
                v_exp = repeat_kv(v, self.num_key_value_groups)
                attn_output = F.scaled_dot_product_attention(
                    q,
                    k_exp,
                    v_exp,
                    attn_mask=sdpa_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)
        projected = self.o_proj(attn_output)
        return projected, next_cache

    def _reference_fp32_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_len: int,
        attention_mask: torch.Tensor | None,
        document_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Exact FP32 scaled dot-product reference path used for numerical regression verification."""
        orig_dtype = q.dtype
        q_len = q.shape[2]
        kv_len = k.shape[2]
        k_exp = repeat_kv(k, self.num_key_value_groups).to(torch.float32)
        v_exp = repeat_kv(v, self.num_key_value_groups).to(torch.float32)
        q_fp32 = q.to(torch.float32)

        scores = torch.matmul(q_fp32, k_exp.transpose(-2, -1)) * self.scale
        bool_mask = build_packed_causal_mask(
            q_len=q_len,
            kv_len=kv_len,
            device=q.device,
            past_len=past_len,
            attention_mask=attention_mask,
            document_ids=document_ids,
        )
        scores = scores.masked_fill(~bool_mask, torch.finfo(torch.float32).min)
        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v_exp)
        return out.to(orig_dtype)
