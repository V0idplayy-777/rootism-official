# Rootism Model Architecture

**Architecture ID:** `rootism-decoder-v1` · **Config ID (production preset):**
`cfg-32803900fe06bf79` · **Schema:** 1

All numbers below were produced by `python -m rootism.cli.params --verify-meta`,
which instantiates the **full** model on PyTorch's `meta` device (zero RAM) and
compares the analytic breakdown with the real parameter tensors. They are not
estimates.

## 1. Production configuration

| Field | Value |
|---|---|
| `vocab_size` | 49,152 |
| `hidden_size` (`d`) | 3,072 |
| `num_hidden_layers` (`L`) | 28 |
| `num_attention_heads` | 24 |
| `num_key_value_heads` | 8 (GQA ratio 3) |
| `head_dim` | 128 |
| `intermediate_size` | 8,192 |
| `max_position_embeddings` | 32,768 |
| `rope_theta` | 100,000 |
| `rope_scaling` | `none` (YaRN factor 2 available for a 65,536 extension phase) |
| `norm_type` / `hidden_act` | `rmsnorm` / `silu` (SwiGLU) |
| `tie_word_embeddings` | `false` |
| `initializer_range` | 0.02 |
| `depth_scaled_init` | `true` |
| `loss_chunk_size` | 512 |
| `attention_query_chunk_size` | 2,048 |
| `pad/bos/eos` | 0 / 1 / 2 |

`num_attention_heads × head_dim = hidden_size` is enforced by validation so the
parameter budget stays exact and `o_proj` is square.

## 2. Exact parameter breakdown (requirement 20)

| Group | Parameters | Share |
|---|---:|---:|
| Embeddings (`V×d`) | 150,994,944 | 4.84% |
| Attention (28 layers) | 704,643,072 | 22.58% |
| Feed-forward (28 layers) | 2,113,929,216 | 67.74% |
| Normalisation (56 layer norms + final norm) | 175,104 | 0.01% |
| Output projection (`V×d`, untied) | 150,994,944 | 4.84% |
| **Total trainable** | **3,120,737,280** | **3.1207 B** |

Per layer:

| Component | Formula | Parameters |
|---|---|---:|
| `q_proj` | `d × (24·128)` | 9,437,184 |
| `k_proj` | `d × (8·128)` | 3,145,728 |
| `v_proj` | `d × (8·128)` | 3,145,728 |
| `o_proj` | `(24·128) × d` | 9,437,184 |
| attention subtotal | | 25,165,824 |
| `gate_proj` / `up_proj` / `down_proj` | `3 × d × 8192` | 75,497,472 |
| two RMSNorm gains | `2 × d` | 6,144 |
| **layer total** | | **100,669,440** |

`28 × 100,669,440 = 2,818,744,320`, plus `2 × 150,994,944 + 3,072` gives
`3,120,737,280`. Analytic and empirical counts agree exactly (test:
`tests/test_parameters.py::test_production_counts_on_meta_device_are_exact`).

Optional config flags (`qk_norm`, biases, tied embeddings) are also accounted for
exactly and covered by tests.

### Depth / vocabulary search (executed)

`scripts/rootism_param_search.py` output:

| Vocab | Depth nearest 3.00B | Total | Emb+head | Transformer body |
|---:|---:|---:|---:|---:|
| 32,768 | 28 | 3,020,073,984 | 201,326,592 | 2,818,572,288 |
| 49,152 | 27 | 3,020,067,840 | 301,989,888 | 2,717,908,992 |
| 65,536 | 26 | 3,020,061,696 | 402,653,184 | 2,617,245,696 |

**Chosen:** 49,152 vocab, 28 layers → 3.1207 B (+4.0% over 3.00B). The extra
layer is worth ~100M body parameters; 64K would instead spend that budget on a
second embedding/head matrix and force depth back to 26. See `TOKENIZER.md` for
the compression analysis that fixes the vocabulary size.

## 3. KV cache (requirement 14)

```
bytes/token = 2 (K,V) × L × kv_heads × head_dim × bytes_per_element
            = 2 × 28 × 8 × 128 × 2 (bf16) = 114,688 B = 112 KiB/token
```

| Context | bf16 KV cache per sequence |
|---:|---:|
| 4,096 | 0.44 GiB |
| 8,192 | 0.88 GiB |
| 32,768 | **3.50 GiB** |
| 65,536 | 7.00 GiB (extension phase, unverified) |

`RootismKVCache` preallocates `(B, kv_heads, max_seq_len, head_dim)` per layer, so
decoding performs no reallocation. Cache equivalence with a full recomputation is
tested to `5.96e-08` logits difference (`test_kv_cache_matches_full_recompute`).

## 4. Attention paths (requirement 13)

1. **Plain prefill** — `F.scaled_dot_product_attention(..., is_causal=True)`,
   `enable_gqa=True`; PyTorch selects a Flash/Memory-Efficient kernel; no mask
   and no attention matrix is materialised.
2. **Single-token decode** — same call with `is_causal=False` (all cached keys
   are in the past).
3. **Padded or packed sequences** — explicit boolean mask built in query chunks
   of 2,048, enforcing `kv_index <= q_index`, optional `document_ids` equality and
   optional padding. Peak mask memory at 32K is ~64 MiB instead of ~1 GiB.
4. **Optional** `flash_attn_varlen_func` with `cu_seqlens` for packed GPU
   training. The hook exists; the path is **UNVERIFIED** because `flash-attn` is
   not installed in this sandbox. Without it, path 3 is used automatically.

Reference-mask equivalence, future-token isolation, document isolation and
padding correctness are all tested in `tests/test_attention.py`.

## 5. Positional encoding (requirement 17)

`cos/sin` are computed in fp32 from `position_ids` and cast once to the
activation dtype. `compute_rotary_embeddings` returns a third value,
`attention_temperature`, used as `scale = 1/(sqrt(head_dim) × temperature)` in
SDPA. YaRN sets `temperature = 0.1·ln(factor)+1`; plain/linear/dynamic-NTK use 1.

| `rope_scaling.type` | Behaviour | Default? |
|---|---|---|
| `none` | plain RoPE | **yes** (32K run) |
| `linear` | positions divided by `factor` | no |
| `dynamic_ntk` | base recomputed from current length | no — **cache-unsafe** |
| `yarn` | NTK-by-parts ramp + attention temperature | optional for 65K extension |

The RoPE relative-position property, norm preservation, YaRN factor-1 equivalence
to plain RoPE, dynamic-NTK frequency shifts and finiteness at position 65,000 are
tested in `tests/test_rope.py`. Long-context *training* behaviour is unverified.

## 6. Initialisation (requirement 18)

```
std            = 0.02
residual_std   = 0.02 / sqrt(2L) = 0.02 / sqrt(56) ≈ 0.002673
```

`o_proj` and `down_proj` use `residual_std`; all other Linear/Embedding weights use
`std`; biases start at zero; RMSNorm gains at one; the padding embedding row is
zeroed. `initialize_weights(..., seed=...)` uses `torch.random.fork_rng()` so the
caller's RNG state is preserved. **No pretrained weights exist anywhere in the
codebase.**

## 7. Loss and sequence packing (requirements 53, 60)

* Causal shift: `hidden[:, :-1]` predicts `labels[:, 1:]`.
* `chunked_causal_lm_loss` computes a **sum** reduction in fp32 over 512-token
  chunks; the test asserts exact equality with full-logit cross-entropy.
* `RootismForCausalLM.forward` also returns `num_valid_tokens` so a trainer can
  divide by the *global* non-padding count (correct under gradient accumulation).
* `document_ids` resets RoPE positions per packed document and blocks
  cross-document attention. `packed_position_ids` is tested against explicit
  boundaries.

## 8. Checkpoint binding

`save_model_checkpoint` stores `config`, `config_id`, `architecture_id`,
`tokenizer_id`, `step`, torch version and a SHA-256 digest. Loading is strict and
calls `compatibility_report()`. Changing the tokenizer vocabulary, the model
config or the architecture ID invalidates the checkpoint by design.

## 9. Known limitations

* 3.5 GiB KV cache per 32K sequence limits serving concurrency; KV quantisation
  is Stage 5 work and is unverified.
* Full 32K prefill has not been executed on a GPU in this environment.
* The 65,536-token extension requires continued pretraining; scaling alone does
  not confer long-context competence.
* No mixture-of-experts, no sliding-window attention, no sparse attention;
  quadratic attention cost remains.
* Model quality is entirely unmeasured — the weights are random.
