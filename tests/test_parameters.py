from __future__ import annotations

import pytest
import torch

from rootism.config import ModelConfig, tiny_test_config
from rootism.model.config_bridge import build_model
from rootism.model.parameters import (
    analytic_breakdown,
    empirical_breakdown,
    format_breakdown,
    kv_cache_bytes_per_token,
    search_depth_for_target,
)

GROUPS = ("embeddings", "attention", "feed_forward", "normalization", "output_projection")


@pytest.mark.slow
def test_production_counts_on_meta_device_are_exact():
    """Instantiate the FULL 3B model on `meta` (zero RAM) and count real tensors."""
    cfg = ModelConfig()
    cfg.validate()
    analytic = analytic_breakdown(cfg)
    model = build_model(cfg, device="meta", init_weights=False)
    empirical = empirical_breakdown(model)
    for group in (*GROUPS, "total"):
        assert empirical[group] == analytic[group], group
    assert empirical["total"] == 3_120_737_280
    assert 2.85e9 <= empirical["total"] <= 3.25e9, "approximately 3B"
    assert sum(analytic[g] for g in GROUPS) == analytic["total"]
    assert format_breakdown(analytic)


def test_tiny_counts_match_exactly():
    cfg = tiny_test_config()
    analytic = analytic_breakdown(cfg)
    model = build_model(cfg, seed=0)
    empirical = empirical_breakdown(model)
    assert all(empirical[g] == analytic[g] for g in (*GROUPS, "total"))
    assert sum(p.numel() for p in model.parameters()) == analytic["total"]


def test_tied_embeddings_and_optional_features_are_counted():
    cfg = tiny_test_config(tie_word_embeddings=True, qk_norm=True, attention_bias=True, mlp_bias=True)
    analytic = analytic_breakdown(cfg)
    model = build_model(cfg, seed=0)
    empirical = empirical_breakdown(model)
    assert all(empirical[g] == analytic[g] for g in (*GROUPS, "total"))
    assert analytic["output_projection"] == 0


def test_kv_cache_bytes_match_formula():
    cfg = ModelConfig()
    expected = 2 * 28 * 8 * 128 * 2  # k+v * layers * kv heads * head_dim * bf16 bytes
    assert kv_cache_bytes_per_token(cfg, torch.bfloat16) == expected
    assert kv_cache_bytes_per_token(cfg, torch.bfloat16) / 1024 == pytest.approx(112.0)
    assert expected * 32_768 / 1024**3 == pytest.approx(3.5)


def test_depth_search_selects_28_layers_for_48k_vocab():
    rows = search_depth_for_target(
        vocab_size=49_152, hidden_size=3_072, num_attention_heads=24,
        num_key_value_heads=8, head_dim=128, intermediate_size=8_192,
    )
    best = min(rows, key=lambda r: r["abs_error"])
    # 27 layers is numerically closest to exactly 3.00B; we deliberately choose
    # 28 (3.12B, +4%) to buy an extra transformer block while staying "~3B".
    assert best["layers"] == 27
    chosen = next(r for r in rows if r["layers"] == 28)
    assert chosen["total"] == 3_120_737_280
    assert 0.95 * 3.0e9 <= chosen["total"] <= 1.05 * 3.0e9
