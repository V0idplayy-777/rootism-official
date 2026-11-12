from __future__ import annotations

import pytest
import torch

from rootism.config import ModelConfig, RopeScalingConfig, tiny_test_config
from rootism.model.rotary import apply_rotary_pos_emb, compute_rotary_embeddings, rotary_frequency


def test_rope_inner_product_depends_only_on_relative_position():
    cfg = tiny_test_config()
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, cfg.head_dim)
    k = torch.randn(1, 1, 1, cfg.head_dim)
    cos7, sin7, t7 = compute_rotary_embeddings(cfg, torch.tensor([[7]]), torch.float32, torch.device("cpu"))
    cos12, sin12, _ = compute_rotary_embeddings(cfg, torch.tensor([[12]]), torch.float32, torch.device("cpu"))
    cos0, sin0, _ = compute_rotary_embeddings(cfg, torch.tensor([[0]]), torch.float32, torch.device("cpu"))
    cos5, sin5, _ = compute_rotary_embeddings(cfg, torch.tensor([[5]]), torch.float32, torch.device("cpu"))
    q7, _ = apply_rotary_pos_emb(q, q, cos7, sin7)
    _, k12 = apply_rotary_pos_emb(k, k, cos12, sin12)
    q0, _ = apply_rotary_pos_emb(q, q, cos0, sin0)
    _, k5 = apply_rotary_pos_emb(k, k, cos5, sin5)
    assert t7 == 1.0
    assert torch.allclose((q7 * k12).sum(), (q0 * k5).sum(), atol=1e-5)


def test_rotation_preserves_norm():
    cfg = tiny_test_config()
    torch.manual_seed(1)
    x = torch.randn(1, 2, 4, cfg.head_dim)
    pos = torch.arange(4).unsqueeze(0)
    cos, sin, _ = compute_rotary_embeddings(cfg, pos, torch.float32, torch.device("cpu"))
    rotated, _ = apply_rotary_pos_emb(x, x.clone(), cos, sin)
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_yarn_factor_one_matches_plain_rope():
    cfg = tiny_test_config(rope_scaling=RopeScalingConfig(type="yarn", factor=1.0))
    plain = rotary_frequency(tiny_test_config())
    yarn = rotary_frequency(cfg)
    assert torch.allclose(plain, yarn, atol=1e-7)
    _, _, temperature = compute_rotary_embeddings(cfg, torch.tensor([[10]]), torch.float32, torch.device("cpu"))
    assert temperature == pytest.approx(1.0)


def test_dynamic_ntk_changes_frequency_beyond_original_context():
    cfg = tiny_test_config(
        rope_scaling=RopeScalingConfig(type="dynamic_ntk", factor=2.0, original_max_position_embeddings=64)
    )
    short = rotary_frequency(cfg, seq_len=64)
    long = rotary_frequency(cfg, seq_len=128)
    assert not torch.allclose(short, long)
    # This is exactly why dynamic NTK is NOT the inference default: the same
    # absolute position gets different angles as the sequence grows.
    assert (short - long).abs().max() > 0


@pytest.mark.parametrize("scaling", [
    RopeScalingConfig(type="none"),
    RopeScalingConfig(type="linear", factor=2.0),
    RopeScalingConfig(type="dynamic_ntk", factor=2.0),
    RopeScalingConfig(type="yarn", factor=2.0),
])
def test_long_positions_are_numerically_finite(scaling):
    cfg = ModelConfig(
        vocab_size=257,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        max_position_embeddings=65_536,
        rope_scaling=scaling,
    )
    cfg.validate()
    pos = torch.tensor([[65_000]])
    cos, sin, temperature = compute_rotary_embeddings(cfg, pos, torch.bfloat16, torch.device("cpu"), seq_len=65_001)
    assert torch.isfinite(cos).all() and torch.isfinite(sin).all()
    assert cos.shape == (1, 1, 1, cfg.head_dim)
    assert temperature >= 1.0
    if scaling.type == "yarn":
        assert temperature > 1.0
