from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from rootism.config import tiny_test_config
from rootism.model.config_bridge import build_model


def test_forward_shapes_and_backward(tiny_model, tiny_config):
    ids = torch.randint(4, tiny_config.vocab_size, (2, 9))
    out = tiny_model(ids, num_logits_to_keep=9)
    assert out.logits.shape == (2, 9, tiny_config.vocab_size)
    assert torch.isfinite(out.logits).all()
    out.logits.sum().backward()
    q = tiny_model.model.layers[0].self_attn.q_proj.weight
    assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0


def test_causal_shift_and_padding_exclusion(tiny_model, tiny_config):
    ids = torch.randint(4, tiny_config.vocab_size, (2, 7))
    labels = ids.clone()
    labels[:, 0] = -100
    labels[1, 4:] = -100
    out = tiny_model(ids, labels=labels)

    shifted_hidden = out.last_hidden_state[:, :-1, :]
    shifted_labels = labels[:, 1:].contiguous()
    ref_logits = tiny_model._logits_fn(shifted_hidden).float()
    ref_loss_sum = F.cross_entropy(
        ref_logits.reshape(-1, tiny_config.vocab_size),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    assert torch.allclose(out.loss_sum, ref_loss_sum, atol=1e-4)
    assert out.num_valid_tokens.item() == (shifted_labels != -100).sum().item()
    assert torch.allclose(out.loss, out.loss_sum / out.num_valid_tokens)
    out.loss.backward()
    assert torch.isfinite(tiny_model.model.layers[-1].mlp.down_proj.weight.grad).all()


def test_packed_and_padded_attention_paths_are_finite(tiny_model, tiny_config):
    ids = torch.randint(4, tiny_config.vocab_size, (2, 8))
    doc = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2], [0, 0, 1, 1, 1, 1, 1, 2]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0]])
    out = tiny_model(ids, document_ids=doc, attention_mask=mask, labels=ids)
    assert torch.isfinite(out.loss)
    out.loss.backward()


def test_bfloat16_forward_is_finite(tiny_config):
    model = build_model(tiny_config, seed=1).to(torch.bfloat16)
    ids = torch.randint(4, tiny_config.vocab_size, (1, 6))
    out = model(ids, labels=ids)
    assert torch.isfinite(out.loss)
    assert out.loss.dtype == torch.float32  # loss accumulation is fp32


def test_optional_architecture_variants(tiny_config):
    for overrides in ({"qk_norm": True}, {"tie_word_embeddings": True}, {"attention_bias": True, "mlp_bias": True}):
        cfg = tiny_test_config(**overrides)
        model = build_model(cfg, seed=2)
        ids = torch.randint(4, cfg.vocab_size, (1, 5))
        out = model(ids, labels=ids)
        assert torch.isfinite(out.loss), overrides
        out.loss.backward()


def test_depth_scaled_init_reduces_residual_projection_std(tiny_config):
    import math

    from rootism.utils.seed import set_seed

    cfg = tiny_config
    set_seed(5, deterministic=False)
    scaled = build_model(cfg, seed=5)
    unscaled_cfg = tiny_test_config(depth_scaled_init=False)
    unscaled = build_model(unscaled_cfg, seed=5)
    o_scaled = scaled.model.layers[0].self_attn.o_proj.weight.std().item()
    o_unscaled = unscaled.model.layers[0].self_attn.o_proj.weight.std().item()
    expected_ratio = 1 / math.sqrt(2 * cfg.num_hidden_layers)
    assert o_scaled / o_unscaled == pytest.approx(expected_ratio, rel=0.05)
