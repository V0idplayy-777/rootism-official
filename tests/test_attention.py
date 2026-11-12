from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from rootism.config import tiny_test_config
from rootism.model.attention import attention_forward, repeat_kv


def _reference(q, k, v, mask, head_dim):
    # `mask` is a 2-D (q_len, kv_len) boolean; broadcast over batch and heads.
    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=mask[None, None, :, :],
        enable_gqa=True,
        scale=1.0 / math.sqrt(head_dim),
    )


def test_chunked_causal_mask_matches_full_mask_reference():
    cfg = tiny_test_config(attention_query_chunk_size=2)
    torch.manual_seed(0)
    q = torch.randn(1, 4, 6, 16)
    k = torch.randn(1, 2, 6, 16)
    v = torch.randn(1, 2, 6, 16)
    doc = torch.zeros(1, 6, dtype=torch.long)  # forces the chunked mask path
    out = attention_forward(q, k, v, config=cfg, document_ids=doc)
    mask = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    ref = _reference(q, k, v, mask, 16)
    assert torch.allclose(out, ref, atol=1e-5)


def test_future_tokens_cannot_influence_earlier_outputs():
    cfg = tiny_test_config(attention_query_chunk_size=3)
    torch.manual_seed(1)
    q = torch.randn(1, 4, 6, 16)
    k = torch.randn(1, 2, 6, 16)
    v = torch.randn(1, 2, 6, 16)
    doc = torch.zeros(1, 6, dtype=torch.long)
    out = attention_forward(q, k, v, config=cfg, document_ids=doc)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 3:, :] = torch.randn_like(k2[:, :, 3:, :]) * 50
    v2[:, :, 3:, :] = torch.randn_like(v2[:, :, 3:, :]) * 50
    out2 = attention_forward(q, k2, v2, config=cfg, document_ids=doc)
    assert torch.allclose(out[:, :, :3, :], out2[:, :, :3, :], atol=1e-5)
    assert not torch.allclose(out[:, :, 3:, :], out2[:, :, 3:, :])


def test_packed_documents_are_isolated():
    cfg = tiny_test_config(attention_query_chunk_size=2)
    torch.manual_seed(2)
    q = torch.randn(1, 4, 6, 16)
    k = torch.randn(1, 2, 6, 16)
    v = torch.randn(1, 2, 6, 16)
    doc = torch.tensor([[0, 0, 0, 1, 1, 1]])
    out = attention_forward(q, k, v, config=cfg, document_ids=doc)
    causal = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    same = doc[0][:, None] == doc[0][None, :]
    mask = causal & same
    ref = _reference(q, k, v, mask, 16)
    assert torch.allclose(out, ref, atol=1e-5)
    # Deliberately compare against the WRONG (non-isolated) result to ensure
    # the test would catch a broken implementation.
    naive = attention_forward(q, k, v, config=cfg, document_ids=torch.zeros(1, 6, dtype=torch.long))
    assert not torch.allclose(out[:, :, 3:, :], naive[:, :, 3:, :], atol=1e-4)


def test_padding_mask_matches_reference():
    cfg = tiny_test_config(attention_query_chunk_size=2)
    torch.manual_seed(3)
    q = torch.randn(1, 4, 6, 16)
    k = torch.randn(1, 2, 6, 16)
    v = torch.randn(1, 2, 6, 16)
    padding = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    out = attention_forward(q, k, v, config=cfg, padding_mask=padding)
    causal = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    mask = causal & padding[0][None, :]
    ref = _reference(q, k, v, mask, 16)
    assert torch.allclose(out, ref, atol=1e-5)


def test_fast_causal_path_matches_explicit_causal_mask():
    cfg = tiny_test_config()
    torch.manual_seed(4)
    q = torch.randn(1, 4, 6, 16)
    k = torch.randn(1, 2, 6, 16)
    v = torch.randn(1, 2, 6, 16)
    out = attention_forward(q, k, v, config=cfg)
    mask = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    ref = _reference(q, k, v, mask, 16)
    assert torch.allclose(out, ref, atol=1e-5)


def test_repeat_kv_expands_without_copy_when_possible():
    x = torch.randn(2, 2, 5, 8)
    assert repeat_kv(x, 1) is x
    y = repeat_kv(x, 3)
    assert y.shape == (2, 6, 5, 8)
    assert torch.equal(y[:, 0], y[:, 1]) and torch.equal(y[:, 1], y[:, 2])


def test_packed_position_ids_reset_at_document_boundaries():
    from rootism.model.language_model import packed_position_ids

    doc = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2]])
    assert packed_position_ids(doc).tolist() == [[0, 1, 2, 0, 1, 0, 1, 2]]
