"""Thin convenience layer so callers never hand-build configs inside model code."""

from __future__ import annotations

import torch

from ..config import ModelConfig
from .language_model import RootismForCausalLM


def model_config_from(data: dict) -> ModelConfig:
    return ModelConfig.from_dict(data)


def build_model(
    config: ModelConfig,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    init_weights: bool = True,
    seed: int | None = None,
) -> RootismForCausalLM:
    """Construct Rootism, optionally on the `meta` device for zero-RAM analysis.

    `device="meta"` + `init_weights=False` instantiates the full ~3B model
    without allocating memory; parameter shapes and counts are then exact.
    """
    dev = torch.device(device)
    if dev.type == "meta":
        with torch.device("meta"):
            model = RootismForCausalLM(config)
    else:
        model = RootismForCausalLM(config)
        if dtype is not None:
            model = model.to(dtype=dtype)
        model = model.to(device=dev)
    if init_weights and dev.type != "meta":
        from .initialization import initialize_weights

        initialize_weights(model, config, seed=seed)
    elif init_weights and dev.type == "meta":
        raise ValueError("cannot initialize weights on the meta device; pass init_weights=False")
    return model
