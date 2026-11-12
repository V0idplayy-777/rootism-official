"""Reproducibility helpers (requirement G)."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy and Torch RNGs.

    `deterministic=True` requests deterministic algorithms: this can reduce
    throughput and is not available for every kernel. Hardware-dependent
    nondeterminism (atomic reductions, some fused kernels) is documented in
    docs/REPRODUCIBILITY.md.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_dtype(name: str | Any) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float64": torch.float64,
    }
    key = str(name).lower()
    if key not in table:
        raise ValueError(f"unknown dtype {name!r}; supported: {sorted(table)}")
    return table[key]
