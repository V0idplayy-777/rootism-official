"""Gated feed-forward network (SwiGLU), requirement 15."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..config import ModelConfig


class SwiGLUMLP(nn.Module):
    """down( silu(gate(x)) * up(x) )

    Three matrices instead of two: the extra gate projection buys measurable
    quality at a given parameter budget. `intermediate_size` is a multiple of
    64 so tensor-core tile alignment is preserved on both NVIDIA and AMD GPUs.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        if config.hidden_act != "silu":
            raise ValueError(f"unsupported activation {config.hidden_act!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
