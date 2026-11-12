"""
Rootism Gated Feed-Forward Network — SwiGLU (Step 15).

FFN(x) = W_down( SiLU(W_gate(x)) * W_up(x) )

In Rootism-3B:
- d_model = 2,560 (10 * 256)
- intermediate_size = 6,656 (26 * 256, ~2.6x d_model)
- Bias-free linear projections for Tensor Core tile alignment and parameter efficiency.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rootism.config import RootismModelConfig


class RootismSwiGLU(nn.Module):
    """Gated SwiGLU feed-forward block."""

    def __init__(self, config: RootismModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.d_model, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.d_model, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.d_model, bias=config.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
