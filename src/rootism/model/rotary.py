"""Rotary positional embeddings and context extension (requirement 17).

Design decisions (documented in docs/MODEL.md):

* Plain RoPE with theta=100000 is used for the initial 32768-token training run.
* "linear", "dynamic_ntk" and "yarn" scaling are implemented for the optional
  65536-token extension phase.
* dynamic NTK is deliberately NOT the default: it recomputes frequencies from
  the *current* sequence length, so a KV cache built at short length becomes
  inconsistent once the length crosses the threshold. YaRN with a fixed factor
  is length-independent and therefore cache-safe.
* YaRN attention temperature is applied to the softmax scale, NOT by scaling
  cos/sin. Scaling cos/sin breaks the orthogonality of the rotation and changes
  activation norms; scaling logits is mathematically equivalent and keeps
  RoPE norm-preserving.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

from ..config import ModelConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """q/k: (B, heads, T, head_dim); cos/sin: (B, 1, T, head_dim) broadcastable."""
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out


def _base_inv_freq(theta: float, head_dim: int, device: torch.device) -> torch.Tensor:
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
    return 1.0 / (theta**exponents)


def _yarn_find_correction_dim(num_rotations: float, dim: int, base: float, max_pos: int) -> float:
    return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, max_pos: int
) -> Tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_pos))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_pos))
    return max(low, 0), min(high, dim - 1)


def _yarn_inv_freq(config: ModelConfig, device: torch.device) -> torch.Tensor:
    scaling = config.rope_scaling
    dim = config.head_dim
    theta = config.rope_theta
    original = scaling.original_max_position_embeddings
    exponents = torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim

    inv_freq_extrapolation = 1.0 / (theta**exponents)
    inv_freq_interpolation = 1.0 / (scaling.factor * theta**exponents)

    low, high = _yarn_find_correction_range(
        scaling.beta_fast, scaling.beta_slow, dim, theta, original
    )
    idx = torch.arange(dim // 2, dtype=torch.float32, device=device)
    ramp = ((idx - low) / max(high - low, 1e-3)).clamp(0.0, 1.0)
    # low-frequency dimensions extrapolate (unchanged), high-frequency interpolate
    extrapolation_mask = 1.0 - ramp
    return (
        inv_freq_interpolation * (1.0 - extrapolation_mask)
        + inv_freq_extrapolation * extrapolation_mask
    )


def compute_rotary_embeddings(
    config: ModelConfig,
    position_ids: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    *,
    seq_len: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Return (cos, sin, attention_temperature).

    cos/sin have shape (B, 1, T, head_dim) ready for broadcasting over heads.
    Computation happens in fp32 to keep long-context angles accurate, then is
    cast to the activation dtype once.
    """
    scaling = config.rope_scaling
    dim = config.head_dim
    temperature = 1.0
    pos_mult = 1.0

    if scaling.type == "none":
        inv_freq = _base_inv_freq(config.rope_theta, dim, device)
    elif scaling.type == "linear":
        inv_freq = _base_inv_freq(config.rope_theta, dim, device)
        pos_mult = 1.0 / scaling.factor
    elif scaling.type == "dynamic_ntk":
        length = seq_len if seq_len is not None else int(position_ids.max().item()) + 1
        theta = config.rope_theta
        if length > scaling.original_max_position_embeddings:
            exponent = dim / (dim - 2)
            theta = theta * (
                (scaling.factor * length / scaling.original_max_position_embeddings)
                - (scaling.factor - 1)
            ) ** exponent
        inv_freq = _base_inv_freq(theta, dim, device)
    elif scaling.type == "yarn":
        inv_freq = _yarn_inv_freq(config, device)
        temperature = (
            scaling.attention_factor
            if scaling.attention_factor is not None
            else 0.1 * math.log(scaling.factor) + 1.0
        )
    else:  # pragma: no cover - config.validate() prevents this
        raise ValueError(f"unsupported RoPE scaling {scaling.type!r}")

    positions = position_ids.to(torch.float32) * pos_mult
    freqs = positions.unsqueeze(-1) * inv_freq.view(1, 1, -1)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype).unsqueeze(1)
    sin = emb.sin().to(dtype).unsqueeze(1)
    return cos, sin, temperature


def rotary_frequency(config: ModelConfig, *, seq_len: int | None = None) -> torch.Tensor:
    """Expose inv_freq for numerical tests without allocating activations."""
    device = torch.device("cpu")
    scaling = config.rope_scaling
    if scaling.type == "yarn":
        return _yarn_inv_freq(config, device)
    theta = config.rope_theta
    if scaling.type == "dynamic_ntk" and seq_len and seq_len > scaling.original_max_position_embeddings:
        dim = config.head_dim
        theta = theta * (
            (scaling.factor * seq_len / scaling.original_max_position_embeddings)
            - (scaling.factor - 1)
        ) ** (dim / (dim - 2))
    return _base_inv_freq(theta, config.head_dim, device)
