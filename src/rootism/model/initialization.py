"""Random initialization — Rootism never loads pretrained weights (requirement 03)."""

from __future__ import annotations

import math

import torch
from torch import nn

from ..config import ModelConfig
from .normalization import RMSNorm


def _init(model: nn.Module, config: ModelConfig) -> None:
    std = config.initializer_range
    # Depth-scaled residual initialisation: projections that write into the
    # residual stream are scaled by 1/sqrt(2L). With L=28 this keeps the
    # variance of the residual sum approximately constant at initialisation
    # and prevents activation blow-up before warmup completes.
    residual_std = std / math.sqrt(2.0 * config.num_hidden_layers) if config.depth_scaled_init else std

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            s = residual_std if (name.endswith("o_proj") or name.endswith("down_proj")) else std
            nn.init.normal_(module.weight, mean=0.0, std=s)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            with torch.no_grad():
                module.weight[config.pad_token_id].zero_()
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)


def initialize_weights(model: nn.Module, config: ModelConfig, *, seed: int | None = None) -> None:
    """Initialise every parameter from scratch.

    Parameters on the `meta` device are skipped so that `build_model(...,
    device="meta")` can be used for exact parameter counting with zero RAM.
    """
    if any(p.device.type == "meta" for p in model.parameters()):
        return
    if seed is not None:
        # fork_rng keeps this deterministic without clobbering the caller's RNG.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            _init(model, config)
    else:
        _init(model, config)
