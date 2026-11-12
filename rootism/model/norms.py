"""
Rootism Normalisation Primitives (Step 16).

Implements RootismRMSNorm with explicit FP32 variance accumulation to prevent
underflow/overflow during BF16 and FP16 mixed-precision training on long contexts.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RootismRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (bias-free, pre-norm safe).

    y = (x / sqrt(mean(x^2) + eps)) * weight
    Variance statistics are always computed in torch.float32 regardless of input dtype.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def reset_parameters(self) -> None:
        if not self.weight.is_meta:
            nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        inv_rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = x_fp32 * inv_rms
        return normed.to(orig_dtype) * self.weight.to(orig_dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"
