"""
Rootism Rotary Positional Embeddings with YaRN / NTK-by-parts Context Scaling (Steps 17 & 19).

Engineering highlights:
- Position indices and inverse frequencies are strictly computed in `torch.float32`.
  (In BF16, integers > 256 lose unit precision; at position 32,768 BF16 step size is 256,
  which would corrupt token-level positional resolution if computed in low precision.)
- High base `rope_theta = 500,000.0` preserves long-range symbol discrimination across 32K tokens.
- Built-in YaRN / NTK-by-parts smooth wavelength ramp supports extending context to 65,536
  and 131,072 tokens without phase aliasing or high-frequency local-syntax degradation,
  including attention temperature compensation m = 0.1 * ln(factor) + 1.0.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from rootism.config import RopeScalingConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dimension into two halves and rotate: [-x2, x1]."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        q: [batch, n_heads, seq_len, head_dim]
        k: [batch, n_kv_heads, seq_len, head_dim]
        cos: [batch, 1, seq_len, head_dim] or [1, 1, seq_len, head_dim] (float32)
        sin: [batch, 1, seq_len, head_dim] or [1, 1, seq_len, head_dim] (float32)
    """
    q_dtype = q.dtype
    k_dtype = k.dtype
    q_fp32 = q.to(torch.float32)
    k_fp32 = k.to(torch.float32)

    q_rot = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
    k_rot = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
    return q_rot.to(q_dtype), k_rot.to(k_dtype)


class RootismRotaryEmbedding(nn.Module):
    """
    Independent RoPE + YaRN positional encoding module.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 32_768,
        rope_theta: float = 500_000.0,
        scaling: RopeScalingConfig | None = None,
    ) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(f"head_dim must be a positive even integer, got {head_dim}")
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = float(rope_theta)
        self.scaling = scaling or RopeScalingConfig()

        inv_freq, attn_scale = self._compute_inv_freq_and_mscale(device=torch.device("cpu"))
        self.attention_scaling: float = attn_scale
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_inv_freq_and_mscale(self, device: torch.device) -> tuple[torch.Tensor, float]:
        dim_indices = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
        base_inv_freq = 1.0 / (self.rope_theta ** (dim_indices / float(self.head_dim)))

        if self.scaling.scaling_type == "none" or self.scaling.factor == 1.0:
            return base_inv_freq, 1.0

        # YaRN / NTK-by-parts wavelength-aware interpolation:
        # High frequencies (short wavelengths: local syntax/indentation) are NOT stretched;
        # Low frequencies (long wavelengths >= original context) are linearly interpolated by 1/factor;
        # Intermediate wavelengths smoothly blend between the two.
        factor = float(self.scaling.factor)
        orig_ctx = float(self.scaling.original_max_position_embeddings)
        low_freq_wavelen = orig_ctx / float(self.scaling.low_freq_factor)
        high_freq_wavelen = orig_ctx / float(self.scaling.high_freq_factor)

        wavelens = (2.0 * math.pi) / base_inv_freq
        scaled_inv_freq = base_inv_freq / factor

        # Smooth ramp in [0, 1]: 0 -> pure interpolation, 1 -> keep original high frequency
        smooth = (orig_ctx / wavelens - float(self.scaling.low_freq_factor)) / (
            float(self.scaling.high_freq_factor) - float(self.scaling.low_freq_factor)
        )
        smooth = torch.clamp(smooth, 0.0, 1.0)
        blended = (1.0 - smooth) * scaled_inv_freq + smooth * base_inv_freq

        inv_freq = torch.where(
            wavelens < high_freq_wavelen,
            base_inv_freq,
            torch.where(wavelens > low_freq_wavelen, scaled_inv_freq, blended),
        )

        if self.scaling.attention_factor is not None:
            mscale = float(self.scaling.attention_factor)
        else:
            mscale = 0.1 * math.log(factor) + 1.0
        return inv_freq, mscale

    def forward(
        self,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute float32 rotary (cos, sin) tables for arbitrary `position_ids` of shape [batch, seq_len].
        Also supports packed sequences where `position_ids` resets to 0 at each packed document boundary!
        """
        if self.inv_freq.is_meta or self.inv_freq.device != position_ids.device:
            inv_freq, attn_scale = self._compute_inv_freq_and_mscale(device=position_ids.device)
            self.inv_freq = inv_freq
            self.attention_scaling = attn_scale

        # position_ids: [batch, seq_len] -> [batch, 1, seq_len] in float32
        pos_fp32 = position_ids.to(dtype=torch.float32)[:, None, :]
        # inv_freq: [head_dim // 2] -> [1, head_dim // 2, 1]
        inv_fp32 = self.inv_freq[None, :, None].to(dtype=torch.float32)

        # freqs: [batch, head_dim // 2, seq_len] -> transpose -> [batch, seq_len, head_dim // 2]
        freqs = (inv_fp32 @ pos_fp32).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # [batch, seq_len, head_dim]

        cos = (emb.cos() * self.attention_scaling).unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        sin = (emb.sin() * self.attention_scaling).unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        return cos, sin
