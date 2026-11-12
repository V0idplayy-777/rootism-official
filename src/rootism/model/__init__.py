"""Rootism model: independently implemented decoder-only transformer."""

from .config_bridge import build_model, model_config_from  # noqa: F401  (kept tiny on purpose)
from .kv_cache import RootismKVCache
from .language_model import RootismForCausalLM, RootismModel
from .parameters import analytic_breakdown, empirical_breakdown, format_breakdown

__all__ = [
    "RootismModel",
    "RootismForCausalLM",
    "RootismKVCache",
    "build_model",
    "model_config_from",
    "analytic_breakdown",
    "empirical_breakdown",
    "format_breakdown",
]
