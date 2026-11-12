"""
Rootism — Independently Engineered 3B-Parameter AI Coding & Reasoning Platform.

Copyright (c) 2026 Dubnium (V0idplayy-777). All Rights Reserved.
"""

from __future__ import annotations

__project__ = "Rootism"
__version__ = "0.1.0"
__architecture_family__ = "rootism-decoder-v1"

from rootism.config import (
    RopeScalingConfig,
    RootismConfig,
    RootismDataConfig,
    RootismDistributedConfig,
    RootismMemoryConfig,
    RootismModelConfig,
    RootismOptimizerConfig,
    RootismTokenizerConfig,
    RootismTrainingConfig,
)

__all__ = [
    "__project__",
    "__version__",
    "__architecture_family__",
    "RopeScalingConfig",
    "RootismModelConfig",
    "RootismTokenizerConfig",
    "RootismDataConfig",
    "RootismOptimizerConfig",
    "RootismMemoryConfig",
    "RootismDistributedConfig",
    "RootismTrainingConfig",
    "RootismConfig",
]
