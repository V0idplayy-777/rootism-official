"""Automated Tests for Phase II: Model Architecture & Long-Context Verification (Steps 11–20)."""

from __future__ import annotations

import torch

from rootism.config import RopeScalingConfig, RootismModelConfig
from rootism.model.attention import RootismAttention
from rootism.model.inspector import evaluate_architecture_candidates, inspect_model_parameters
from rootism.model.norms import RootismRMSNorm
from rootism.model.rope import RootismRotaryEmbedding, apply_rotary_pos_emb
from rootism.model.transformer import IGNORE_INDEX, RootismDecoderLayer, RootismForCausalLM


def test_exact_3b_parameter_breakdown_and_tied_deduplication() -> None:
    report = inspect_model_parameters()
    assert report.total_trainable_parameters == 3_009_630_720
    assert report.embedding_parameters == 167_772_160
    assert report.attention_parameters == 629_155_840
    assert report.feed_forward_parameters == 2_044_723_200
    assert report.normalisation_parameters == 207_360
    assert report.output_projection_parameters == 167_772_160
    assert report.matches_analytical_formula is True
    assert report.kv_cache_bytes_per_token_bf16 == 81_920
    assert report.kv_cache_32k_gib_bf16 == 2.5
    assert report.kv_cache_64k_gib_bf16 == 5.0

    # Verify tied-embedding config counts shared matrix once
    tied_cfg = RootismModelConfig(tie_word_embeddings=True, n_layers=42)
    tied_report = inspect_model_parameters(tied_cfg)
    assert tied_report.output_projection_parameters == 0
    assert tied_report.matches_analytical_formula is True

    candidates = evaluate_architecture_candidates()
    assert len(candidates) == 3
    assert all(2_850_000_000 <= c["total_trainable_parameters"] <= 3_150_000_000 for c in candidates)


def test_rmsnorm_low_precision_bf16_numerical_stability() -> None:
    norm = RootismRMSNorm(128, eps=1e-6)
    # Large magnitude values (1e4) squared = 1e8; safe in FP32 variance accumulation under BF16 input
    x_bf16 = torch.randn(2, 16, 128, dtype=torch.bfloat16) * 5000.0
    out = norm(x_bf16)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    rms = out.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=5e-2)


def test_rope_relative_invariance_fp32_long_positions_and_yarn_scaling() -> None:
    rope = RootismRotaryEmbedding(head_dim=128, max_position_embeddings=32_768, rope_theta=500_000.0)
    q = torch.randn(1, 4, 2, 128, dtype=torch.float32)
    k = torch.randn(1, 4, 2, 128, dtype=torch.float32)

    # Relative offset invariance: <R(m)q, R(n)k> == <R(m+delta)q, R(n+delta)k>
    pos_a = torch.tensor([[10, 25]])
    pos_b = torch.tensor([[10_010, 10_025]])
    cos_a, sin_a = rope(pos_a)
    cos_b, sin_b = rope(pos_b)
    qa, ka = apply_rotary_pos_emb(q, k, cos_a, sin_a)
    qb, kb = apply_rotary_pos_emb(q, k, cos_b, sin_b)

    score_a = (qa[:, :, 1] * ka[:, :, 0]).sum(dim=-1)
    score_b = (qb[:, :, 1] * kb[:, :, 0]).sum(dim=-1)
    assert torch.allclose(score_a, score_b, rtol=1e-4, atol=1e-3)

    # Verify positions 32,767 and 32,768 are strictly distinct (no BF16 integer rounding!)
    pos_32k = torch.tensor([[32_767, 32_768]])
    cos_32k, sin_32k = rope(pos_32k)
    assert not torch.allclose(cos_32k[:, :, 0, :], cos_32k[:, :, 1, :])
    assert not torch.allclose(sin_32k[:, :, 0, :], sin_32k[:, :, 1, :])

    # YaRN 2x scaling to 65,536 tokens preserves high-frequency dim 0 while interpolating low-frequency tail
    yarn_cfg = RopeScalingConfig(
        scaling_type="yarn",
        factor=2.0,
        original_max_position_embeddings=32_768,
    )
    rope_yarn = RootismRotaryEmbedding(
        head_dim=128, max_position_embeddings=65_536, rope_theta=500_000.0, scaling=yarn_cfg
    )
    assert torch.isclose(rope_yarn.inv_freq[0], rope.inv_freq[0])
    assert torch.isclose(rope_yarn.inv_freq[-1], rope.inv_freq[-1] / 2.0)
    assert rope_yarn.attention_scaling > 1.0


def test_gqa_attention_sdpa_matches_fp32_reference_causality_kvcache_and_packing() -> None:
    cfg = RootismModelConfig.tiny_verification(
        d_model=128, n_heads=4, n_kv_heads=2, head_dim=32, n_layers=2
    )
    attn = RootismAttention(cfg, layer_idx=0).eval()
    rope = RootismRotaryEmbedding(head_dim=32, max_position_embeddings=2048)

    x = torch.randn(2, 12, 128)
    pos = torch.arange(12).unsqueeze(0).expand(2, 12)
    cos, sin = rope(pos)

    out_fast, _ = attn(x, cos, sin, force_math_reference=False)
    out_ref, _ = attn(x, cos, sin, force_math_reference=True)
    assert torch.allclose(out_fast, out_ref, atol=1e-5)

    # Strict causal non-leakage: mutating token 8 must NOT change outputs 0..7
    x_mut = x.clone()
    x_mut[:, 8:, :] += 50.0
    out_mut, _ = attn(x_mut, cos, sin)
    assert torch.allclose(out_fast[:, :8, :], out_mut[:, :8, :], atol=1e-6)

    # KV-cache incremental decoding equivalence
    prefill_out, cache = attn(x[:, :6, :], cos[:, :, :6, :], sin[:, :, :6, :], use_cache=True)
    step_outs = [prefill_out]
    for t in range(6, 12):
        step_o, cache = attn(
            x[:, t : t + 1, :],
            cos[:, :, t : t + 1, :],
            sin[:, :, t : t + 1, :],
            past_key_value=cache,
            use_cache=True,
        )
        step_outs.append(step_o)
    cat_cached = torch.cat(step_outs, dim=1)
    assert torch.allclose(out_fast, cat_cached, atol=1e-5)

    # Sequence-packed document isolation: doc 0 = [0..5], doc 1 = [6..11]
    doc_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
    pos_packed = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]])
    cos_p, sin_p = rope(pos_packed)
    x1 = x[:1].clone()
    out_p1, _ = attn(x1, cos_p, sin_p, document_ids=doc_ids)

    # Mutate document 0 heavily -> document 1 outputs [6..11] must be 100% invariant!
    x2 = x1.clone()
    x2[:, :6, :] = torch.randn_like(x2[:, :6, :]) * 25.0
    out_p2, _ = attn(x2, cos_p, sin_p, document_ids=doc_ids)
    assert torch.allclose(out_p1[:, 6:, :], out_p2[:, 6:, :], atol=1e-6)


def test_long_context_32768_and_65536_actual_execution() -> None:
    """
    Step 19 mandate: Long-context support (32,768 native and 65,536 YaRN) must be
    actually executed and verified rather than merely declared in configuration files.
    Uses head_dim=128 GQA decoder block with memory-efficient SDPA (O(S) memory).
    """
    long_cfg = RootismModelConfig(
        model_name="Rootism-LongCtx-Verify",
        verification_mode=True,
        vocab_size=512,
        d_model=256,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        head_dim=128,  # Exact production head_dim = 128!
        intermediate_size=512,
        max_position_embeddings=32_768,
        extended_context_length=65_536,
        rope_scaling=RopeScalingConfig(
            scaling_type="yarn",
            factor=2.0,
            original_max_position_embeddings=32_768,
        ),
    )
    layer = RootismDecoderLayer(long_cfg, layer_idx=0).to(torch.bfloat16).eval()
    rope = RootismRotaryEmbedding(
        head_dim=128,
        max_position_embeddings=32_768,
        rope_theta=500_000.0,
        scaling=long_cfg.rope_scaling,
    )

    with torch.no_grad():
        # 1. Native 32,768-token forward pass + tail causal check
        seq_32k = 32_768
        h_32k = torch.randn(1, seq_32k, 256, dtype=torch.bfloat16)
        pos_32k = torch.arange(seq_32k, dtype=torch.long).unsqueeze(0)
        cos_32k, sin_32k = rope(pos_32k)
        out_32k, _ = layer(h_32k, cos_32k, sin_32k)
        assert out_32k.shape == (1, 32_768, 256)
        assert torch.isfinite(out_32k).all()

        # Perturb final token 32,767 -> token 32,766 must remain unchanged
        h_32k_mut = h_32k.clone()
        h_32k_mut[:, -1, :] += 10.0
        out_32k_mut, _ = layer(h_32k_mut, cos_32k, sin_32k)
        assert torch.equal(out_32k[:, :-1, :], out_32k_mut[:, :-1, :])
        del h_32k, h_32k_mut, out_32k, out_32k_mut, cos_32k, sin_32k

        # 2. Extended 65,536-token YaRN forward pass
        seq_64k = 65_536
        h_64k = torch.randn(1, seq_64k, 256, dtype=torch.bfloat16)
        pos_64k = torch.arange(seq_64k, dtype=torch.long).unsqueeze(0)
        cos_64k, sin_64k = rope(pos_64k)
        out_64k, _ = layer(h_64k, cos_64k, sin_64k)
        assert out_64k.shape == (1, 65_536, 256)
        assert torch.isfinite(out_64k).all()


def test_chunked_cross_entropy_exact_loss_and_gradient_equivalence() -> None:
    cfg = RootismModelConfig.tiny_verification(vocab_size=512, d_model=64, n_layers=2, head_dim=16, n_heads=4, n_kv_heads=2, intermediate_size=128)
    torch.manual_seed(42)
    m_dense = RootismForCausalLM(cfg)
    m_chunk = RootismForCausalLM(cfg)
    m_chunk.load_state_dict(m_dense.state_dict())

    ids = torch.randint(4, 512, (2, 37))
    labels = ids.clone()
    labels[0, -5:] = IGNORE_INDEX

    out_dense = m_dense(input_ids=ids, labels=labels, return_logits=True)
    assert out_dense.loss is not None
    out_dense.loss.backward()

    out_chunk = m_chunk(input_ids=ids, labels=labels, return_logits=False, loss_chunk_size=11)
    assert out_chunk.loss is not None
    out_chunk.loss.backward()

    assert out_dense.active_loss_tokens == out_chunk.active_loss_tokens
    assert torch.allclose(out_dense.loss, out_chunk.loss, atol=1e-6)
    for (n1, p1), (n2, p2) in zip(
        m_dense.named_parameters(), m_chunk.named_parameters(), strict=True
    ):
        assert n1 == n2
        assert p1.grad is not None and p2.grad is not None
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5), f"Grad mismatch on {n1}"
