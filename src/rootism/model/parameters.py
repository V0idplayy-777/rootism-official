"""Exact parameter accounting (requirements 04, 20).

`analytic_breakdown` derives counts from the config alone; `empirical_breakdown`
counts the real tensors in an instantiated module. Tests assert they agree
exactly for both the production config (on the meta device, so no 12 GB RAM is
needed) and small CPU configs.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from ..config import ModelConfig

GROUPS = (
    "embeddings",
    "attention",
    "feed_forward",
    "normalization",
    "output_projection",
)


def analytic_breakdown(config: ModelConfig) -> dict[str, int]:
    d = config.hidden_size
    v = config.vocab_size
    h = config.num_attention_heads
    kv = config.num_key_value_heads
    hd = config.head_dim
    ff = config.intermediate_size
    layers = config.num_hidden_layers

    q = d * (h * hd) + (h * hd if config.attention_bias else 0)
    k = d * (kv * hd) + (kv * hd if config.attention_bias else 0)
    vproj = d * (kv * hd) + (kv * hd if config.attention_bias else 0)
    o = (h * hd) * d + (d if config.attention_bias else 0)
    qk_norm = 2 * hd if config.qk_norm else 0
    attention_per_layer = q + k + vproj + o + qk_norm

    gate = d * ff + (ff if config.mlp_bias else 0)
    up = d * ff + (ff if config.mlp_bias else 0)
    down = ff * d + (d if config.mlp_bias else 0)
    ff_per_layer = gate + up + down
    norms_per_layer = 2 * d  # input_layernorm + post_attention_layernorm

    embeddings = v * d
    output_projection = 0 if config.tie_word_embeddings else v * d
    normalization = layers * norms_per_layer + d  # + final norm
    attention = layers * attention_per_layer
    feed_forward = layers * ff_per_layer
    total = embeddings + output_projection + normalization + attention + feed_forward
    return {
        "embeddings": embeddings,
        "attention": attention,
        "feed_forward": feed_forward,
        "normalization": normalization,
        "output_projection": output_projection,
        "total": total,
        "_per_layer_attention": attention_per_layer,
        "_per_layer_feed_forward": ff_per_layer,
        "_per_layer_norms": norms_per_layer,
    }


def empirical_breakdown(model: nn.Module) -> dict[str, int]:
    groups = {g: 0 for g in GROUPS}
    groups["total"] = 0
    for name, param in model.named_parameters():
        n = param.numel()
        lname = name.lower()
        # Check the owning submodule first: q_norm/k_norm live inside self_attn
        # and are counted as attention parameters in the analytic breakdown.
        if "self_attn" in lname:
            key = "attention"
        elif "norm" in lname:
            key = "normalization"
        elif "mlp" in lname:
            key = "feed_forward"
        elif "embed_tokens" in lname:
            key = "embeddings"
        elif "lm_head" in lname:
            key = "output_projection"
        else:  # pragma: no cover - guards against new params being silently ignored
            raise ValueError(f"unclassified parameter {name!r}")
        groups[key] += n
        groups["total"] += n
    return groups


def kv_cache_bytes_per_token(config: ModelConfig, dtype: torch.dtype = torch.bfloat16) -> int:
    elements = 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim
    return elements * torch.empty((), dtype=dtype).element_size()


def format_breakdown(breakdown: Mapping[str, int], *, title: str = "Rootism parameters") -> str:
    total = breakdown["total"]
    lines = [title, "=" * len(title)]
    for g in GROUPS:
        n = breakdown.get(g, 0)
        lines.append(f"  {g:<20s} {n:>16,}  ({100.0 * n / total:5.2f}%)")
    lines.append(f"  {'TOTAL':<20s} {total:>16,}  ({total / 1e9:.4f} B)")
    return "\n".join(lines)


def search_depth_for_target(
    *,
    vocab_size: int,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    intermediate_size: int,
    target: float = 3.0e9,
    max_layers: int = 48,
) -> list[dict[str, int | float]]:
    """Enumerate depths and report distance from the parameter target."""
    rows = []
    for layers in range(1, max_layers + 1):
        cfg = ModelConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
        )
        total = analytic_breakdown(cfg)["total"]
        rows.append({"layers": layers, "total": total, "abs_error": abs(total - target)})
    return rows
