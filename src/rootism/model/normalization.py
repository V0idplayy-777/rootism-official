"""RMSNorm with fp32 accumulation for low-precision stability (requirement 16)."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Computation is performed in float32 regardless of the input dtype, then
    cast back. This avoids bf16/fp16 variance underflow on long activations.
    The learnable gain is a single vector of size `hidden_size`; there is no
    bias and no mean subtraction (cheaper and empirically no worse for
    decoder-only LMs at this scale).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x32 = x.to(torch.float32)
        variance = x32.pow(2).mean(dim=-1, keepdim=True)
        x32 = x32 * torch.rsqrt(variance + self.eps)
        # gain in fp32, cast once at the end
        out = x32 * self.weight.to(torch.float32)
        return out.to(input_dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"
