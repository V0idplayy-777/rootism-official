# Rootism — Central Architecture Specification

**Schema version:** `1` · **Model architecture ID:** `rootism-decoder-v1` ·
**Status:** authoritative for Stage 1. This document is maintained alongside
`STATUS.md`; architectural decisions recorded here are not silently reversed.

---

## 1. System shape

```
                       ┌──────────────────────────────────────────────┐
                       │  Web platform (Stage 7, NOT IMPLEMENTED)     │
                       └───────────────────┬──────────────────────────┘
                                           │ HTTPS / SSE (Stage 5)
   ┌──────────────────────┐        ┌──────▼───────────┐       ┌──────────────────┐
   │ Coding agent (S6)    │◄──────►│ Serving backend  │◄─────►│ Evaluation (S8) │
   │ NOT IMPLEMENTED      │ tools  │ (S5, NOT IMPL.)  │       │ NOT IMPLEMENTED  │
   └──────────────────────┘        └──────┬───────────┘       └──────────────────┘
                                          │
                       ┌──────────────────▼───────────────────┐
                       │ Inference runtime  (Stage 1 core)    │
                       │ Generator + KVCache + Sampling       │
                       └──────────────────┬───────────────────┘
                                          │ loads
   ┌──────────────────────┐    ┌─────────▼──────────┐    ┌──────────────────────┐
   │ Data pipeline (S2)   │───►│ Training (S3)      │───►│ Checkpoint + manifest│
   │ NOT IMPLEMENTED      │    │ NOT IMPLEMENTED    │    │ (Stage 1 core)       │
   └──────────────────────┘    └─────────┬──────────┘    └──────────────────────┘
                                          │ trains
                       ┌──────────────────▼───────────────────┐
                       │ RootismForCausalLM  (Stage 1, DONE)  │
                       │ RMSNorm · RoPE · GQA · SwiGLU        │
                       └──────────────────┬───────────────────┘
                                          │ token ids
                       ┌──────────────────▼───────────────────┐
                       │ RootismTokenizer (Stage 1, DONE)     │
                       │ byte-level BPE · 49 specials         │
                       └──────────────────────────────────────┘
```

A single `ModelConfig` object is the root of the identity chain:

```
ModelConfig ──config_id (sha256 of canonical JSON)──┐
TokenizerConfig ──tokenizer_id (sha256 of vocab)────┼──► checkpoint manifest
Dataset manifest (Stage 2) ──manifest_id────────────┘      (schema v1)
```

`compatibility_report()` in `src/rootism/version.py` fails closed if a checkpoint
is loaded with a different config, tokenizer or architecture ID. **Token mappings
may never change after training begins** (requirement 28).

---

## 2. Module map (Stage 1)

| Path | Responsibility |
|---|---|
| `src/rootism/config.py` | `ModelConfig`, `RopeScalingConfig`, `TokenizerConfig`, `TrainingTokenBudget`; validation, canonical serialisation, `config_id` |
| `src/rootism/version.py` | schema versions, canonical hashing, compatibility assertions |
| `src/rootism/model/normalization.py` | RMSNorm with fp32 accumulation |
| `src/rootism/model/rotary.py` | RoPE, linear / dynamic-NTK / YaRN extension, attention temperature |
| `src/rootism/model/attention.py` | GQA projections, SDPA dispatch, chunked boolean masks, optional FlashAttention varlen |
| `src/rootism/model/feedforward.py` | SwiGLU MLP |
| `src/rootism/model/block.py` | pre-norm decoder layer, residual stream |
| `src/rootism/model/language_model.py` | backbone + causal-LM head, packed position ids, label shifting |
| `src/rootism/model/losses.py` | chunked exact cross-entropy |
| `src/rootism/model/initialization.py` | depth-scaled random initialisation |
| `src/rootism/model/parameters.py` | analytic + empirical parameter breakdown |
| `src/rootism/model/kv_cache.py` | static preallocated KV cache |
| `src/rootism/model/generation.py` | streaming generation, sampling, stop conditions |
| `src/rootism/tokenizer/` | special tokens, pre-tokenizer, BPE trainer, encode/decode, training + evaluation |
| `src/rootism/io/checkpoint.py` | atomic writes, SHA-256 manifests, recovery points, strict model load |
| `src/rootism/accounting.py` | training-token ledger + budget enforcement |
| `src/rootism/hardware.py` | analytical hardware/storage/throughput estimates |
| `src/rootism/cli/` | `params`, `tokenizer`, `doctor` |
| `scripts/` | parameter search, vocab sweep, smoke validation |
| `tests/` | 76 executed tests |

Dependency direction is strictly downward: `model`/`tokenizer` import only
`config`, `version`, `utils`; `io` imports `config` + `model` lazily; `cli`
imports everything. No cycles.

---

## 3. Architectural decisions

For each: **selected**, **why**, **alternatives considered**, **expected
performance implication**, **known limitation**.

### D1 — Decoder-only pre-norm transformer
*Selected*: causal decoder-only stack, 28 layers, `d=3072`.
*Why*: causal LM training is data-efficient, supports packing, and the same
forward path serves chat, FIM completion and agent tool calls.
*Alternatives*: encoder–decoder (wasted encoder for pure generation),
MoE (router instability + 3B active-param accounting ambiguity), SSM/Mamba
hybrids (weaker tooling/long-context verification today), diffusion LMs
(not mature for code).
*Implication*: one code path for training/inference; simple KV cache.
*Limitation*: no bidirectional understanding; FIM is handled by token layout,
not by a bidirectional encoder.

### D2 — RMSNorm with fp32 accumulation (requirement 16)
*Selected*: pre-norm RMSNorm, `eps=1e-5`, gain only, computed in fp32 even in bf16.
*Alternatives*: LayerNorm (mean subtraction + bias; extra cost), post-norm
(unstable at depth), DeepNorm, reversible/residual-free variants.
*Implication*: cheap, stable under bf16; no mean/bias parameters.
*Limitation*: activation scale still needs depth-scaled init (D7) and warmup.

### D3 — RoPE, θ=100000, no scaling for the 32,768-token run (requirement 17)
*Selected*: plain RoPE for the initial context; YaRN (NTK-by-parts + attention
temperature) available with a fixed factor for a 65,536-token extension phase.
Linear and dynamic-NTK scaling are implemented but not enabled by default.
*Why*: plain RoPE is length-independent, so KV-cache entries remain valid as the
sequence grows. **Dynamic NTK recomputes frequencies from the current length;
a cache built at short length becomes inconsistent after the threshold** — a
real inference hazard, so it is not the default. YaRN with a fixed factor is
static and cache-safe.
*Alternatives*: ALiBi (weaker extrapolation for code), learned absolute positions
(no extrapolation), NoPE, xPos, CoPE.
*Implication*: `rope_theta=100000` gives usable resolution over 32K without
extension; YaRN temperature is applied to the softmax **scale**, not by scaling
cos/sin, so rotations remain norm-preserving.
*Limitation*: 65,536-token behaviour is **not validated by training**; it requires
a continued-pretraining phase. Tests only verify numerical finiteness at
position 65,000.

### D4 — Grouped-Query Attention, 24 query heads / 8 KV heads (requirement 14)
*Selected*: `head_dim=128`, GQA ratio 3.
*Alternatives*: MHA (3× KV cache), MQA (aggressive KV sharing, quality loss),
MLA/latent KV (more implementation risk), sliding-window attention (loses
global repo context).
*Implication*: **112 KiB/token bf16 KV cache = 3.5 GiB per sequence at 32,768
tokens**. Native SDPA GQA avoids replicating KV heads.
*Limitation*: 3.5 GiB/sequence is significant for high-concurrency serving;
KV-cache quantisation is deferred to Stage 5 and is *unverified*.

### D5 — SwiGLU with `intermediate_size=8192` (requirement 15)
*Selected*: three matrices, SiLU gate, `ff ≈ 2.67·d`, multiple of 64.
*Alternatives*: plain 2-matrix FFN (`4d`), GeGLU, ReLU², gated linear units with
larger width and fewer layers.
*Implication*: FFN holds 67.7% of parameters — the usual quality-per-FLOP optimum
at this scale.
*Limitation*: activation memory dominates; activation checkpointing is required
(Stage 3).

### D6 — Untied output projection
*Selected*: separate `lm_head` (`V×d`).
*Alternatives*: weight tying (saves 151M params, allows ~+1.5 layers).
*Implication*: more expressive output distribution for a code-heavy vocabulary.
*Limitation*: 9.7% of parameters sit in embedding+head rather than transformer
body. At 64K vocab this rises to 12.9% — one reason 48K was chosen.

### D7 — Depth-scaled residual initialisation (requirement 18)
*Selected*: residual-writing projections (`o_proj`, `down_proj`) initialised with
`std = initializer_range / sqrt(2L)`; all other weights `N(0, 0.02²)`; padding
row zeroed; norms set to ones.
*Alternatives*: plain scaled init, GPT-2 `1/sqrt(2L)` on all residuals, zero-init
of residual projections (μP-like), fixup initialisation.
*Implication*: residual variance stays approximately constant at step 0.
*Limitation*: warmup and gradient clipping (Stage 3) are still required.

### D8 — Memory-efficient attention dispatch (requirement 13)
*Selected*: three paths — (1) optional `flash_attn_varlen_func` for packed
training; (2) SDPA with `is_causal=True` and no mask for plain prefill;
(3) chunked SDPA with an explicit boolean mask for padded/packed sequences,
bounded to `chunk_q × kv_len` booleans (2,048 × 32,768 ≈ 64 MiB) rather than a
full 32K² mask. No attention score matrix is materialised by Rootism code.
*Alternatives*: always materialise a full `(B,T,T)` mask (≈1 GiB at 32K), custom
Triton kernel (portability cost), FlexAttention `BlockMask` (backend dependent).
*Limitation*: the chunked fallback still evaluates `O(T²)` attention FLOPs; it
bounds memory, not compute. Plain 32K prefill on GPU is validated in Stage 3/5,
**not in this CPU sandbox** (see STATUS.md).

### D9 — Sequence packing with document identity (requirement 53)
*Selected*: packed rows carry `document_ids`; RoPE positions reset per document
via `packed_position_ids`; attention masks enforce both `kv_idx <= q_idx` and
`doc equality`.
*Alternatives*: padding every sample (wasted compute), cross-document attention
with `<eos>` separators (leakage), position-id offsets only (does not block
cross-document attention).
*Implication*: near-zero padding waste with correct causal semantics.
*Limitation*: `document_ids` and KV cache are mutually exclusive in Stage 1;
packed generation is not supported (training-only path).

### D10 — Chunked exact cross-entropy
*Selected*: sum-reduction CE computed over `loss_chunk_size=512` token chunks.
*Alternatives*: materialise full `(B,T,V)` logits (≈6.4 GiB fp32 at 32K × 48K),
Liger/fused kernels (extra dependency), per-microbatch mean (wrong under
gradient accumulation).
*Implication*: mathematically identical loss with bounded peak memory; global
normalisation happens in the trainer (Stage 3).
*Limitation*: more kernel launches; small on GPU.

### D11 — Independent byte-level BPE with a code-aware pre-tokeniser (req. 21–24)
*Selected*: 256-byte alphabet, identity Unicode normalisation, whitespace runs
(indentation) emitted as their own pre-tokens, punctuation runs kept together
so `=>`, `===`, `::`, `<<=` can become single learned tokens, 48K vocabulary.
*Alternatives*: off-the-shelf pretrained tokenizer (**prohibited**), Unigram/SentencePiece
(weaker for code symbols), word-level regex BPE with leading spaces (destroys
indentation), token-level grammar/PL-specific lexer (brittle across 19 languages).
*Implication*: every UTF-8 string is encodable; indentation round-trips exactly.
*Limitation*: single-machine merge learning with a bounded pre-token table
(`max_unique_pretokens`); Stage 2 must shard frequency counting for the
production corpus.

### D12 — Injection-safe special tokens (requirement 25)
*Selected*: 49 fixed reserved IDs; `encode()` does **not** interpret special-token
strings found in raw text by default. Input builders opt in with
`allowed_special`. No NFKC normalisation.
*Alternatives*: parse specials everywhere (prompt-injection hazard via repository
files), dynamic special-token registration (breaks checkpoint binding).
*Implication*: untrusted code containing `<tool_call>` cannot fabricate a tool
call.
*Limitation*: pipelines must explicitly opt in when constructing structured
prompts; forgetting to do so yields literal text (safe but non-functional).

### D13 — Atomic, versioned checkpoints (requirements 55, 57, 28)
*Selected*: `torch.save` to a temp file → `fsync` → `os.replace` → canonical JSON
manifest with SHA-256, size, step, `config_id`, `tokenizer_id`,
`architecture_id`, torch version. Strict `load_state_dict`.
*Alternatives*: plain `torch.save` (torn writes on preemption), sharded-only
formats (harder single-file recovery), silent config migration (unsafe).
*Implication*: corrupted/incomplete checkpoints are detected and skipped by
`find_recovery_point()`.
*Limitation*: Stage 1 saves **model** state only; full optimiser/scheduler/RNG/
dataset-position resume arrives in Stage 3.

### D14 — Actual-token ledger (requirements 05, 60)
*Selected*: seven distinct counters with invariants; only `committed_tokens`
counts toward the target; preflight validation of the deduplicated corpus size;
`max_epochs=4`, `min_unique_fraction=0.25` (≥162.5B unique deduplicated tokens
for a 650B target) to prevent repeat-inflation.
*Alternatives*: counting padded tokens, counting epochs as tokens, trusting a
step counter.
*Implication*: the 600B floor cannot be met by looping a small corpus or by
padding.
*Limitation*: the ledger is only as honest as the data pipeline that feeds it;
Stage 2/3 must wire it to real batch accounting and verify against shard
manifests.

---

## 4. Persisted formats (schema v1)

| Artifact | Format | Identity |
|---|---|---|
| Model config | canonical JSON (`ModelConfig.to_dict`) | `config_id` = sha256 prefix |
| Tokenizer | canonical JSON with `token_strings`, `merges`, `corpus_sha256`, `fingerprint` | `tokenizer_id` = sha256 prefix |
| Checkpoint | `torch.save` dict + sibling `*.pt.manifest.json` | sha256 digest + manifest fields |
| Dataset manifest (Stage 2) | canonical JSON, licence + provenance per dataset | `manifest_id` |
| Evaluation report (Stage 8) | canonical JSON | checkpoint id + dataset version |

All JSON is written with sorted keys and no incidental whitespace; content hashes
are therefore reproducible across machines.

---

## 5. Known gaps (authoritative list in `STATUS.md`)

Stages 2–8 are **not implemented**. Concretely absent today: dataset registry and
licence verification, streaming corpus storage, deduplication, quality control,
contamination checks, the distributed trainer, optimiser/scheduler/LR warmup,
activation checkpointing, FSDP/ZeRO, stability monitors, full training resume,
profiling, reasoning-task generators, quantisation, HTTP API, coding agent and
workspace tools, sandbox execution, the web platform, benchmarks and CI.
No claim of production readiness is made.
