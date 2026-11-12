# Stage 1 Completion Report

**Date:** 2026-11-12 · **Branch:** `arena/b5e3aae0-rootism-official` ·
**Status:** ✅ COMPLETE (for Stage 1 only) · **Rootism overall:** NOT production-ready, NOT trained

---

## 1. Implementation plan (executed)

Stage 1 builds the foundation that every later stage depends on:

1. Central architecture + status + roadmap documentation and a single versioned
   configuration layer.
2. A from-scratch ~3B decoder-only transformer: config, RMSNorm, RoPE (+ scaling),
   GQA with memory-efficient attention, SwiGLU, pre-norm residual blocks, causal
   LM head, chunked exact loss, random depth-scaled initialisation.
3. Programmatic parameter accounting verified against real tensors on the `meta`
   device.
4. Independent byte-level BPE tokenizer: special tokens, code-aware
   pre-tokeniser, deterministic trainer with disk spill, encode/decode, FIM,
   evaluation, fingerprinting, injection-safe policy.
5. KV cache + streaming generation + sampling + stop conditions.
6. Atomic SHA-256-verified checkpoints bound to config and tokenizer identity.
7. Training-token ledger enforcing the >600B / 650B budget.
8. Analytical hardware/storage/duration estimates from explicit assumptions.
9. Automated tests and documentation.

No pretrained model, tokenizer, checkpoint or distillation was used.

## 2. Source code delivered

```
pyproject.toml, requirements*.txt, .gitignore, README.md
src/rootism/
  __init__.py, version.py, config.py, accounting.py, hardware.py
  utils/{hashing,seed}.py
  model/{config_bridge,normalization,rotary,attention,feedforward,block,
         language_model,losses,initialization,parameters,kv_cache,generation}.py
  tokenizer/{special_tokens,pretokenizer,bpe,tokenizer,train,evaluation,pilot_corpus}.py
  io/checkpoint.py
  cli/{params,tokenizer,doctor}.py
scripts/{rootism_param_search,rootism_vocab_sweep,rootism_smoke}.py
tests/ (11 test modules + conftest.py)
docs/ (11 files)
```

## 3. Integration

The identity chain is real and enforced, not decorative:

```
ModelConfig.config_id ─┐
Tokenizer.tokenizer_id ─┼──► checkpoint manifest ──► compatibility_report() ──► strict state_dict load
architecture_id ───────┘
```

End-to-end integration tests exercise: `tokenizer.encode → tensor → model
forward/loss/backward → checkpoint save/load → Generator.stream →
tokenizer.decode`. Packed sequences, padded batches and KV-cache decode all go
through the same `RootismForCausalLM` implementation. `TokenLedger` consumes the
model's `num_valid_tokens` semantics (sum reduction, `ignore_index=-100`).

## 4. Automated tests — executed

```
PYTHONPATH=src .venv/bin/python -m pytest -q
........................................................................ [ 94%]
....                                                                     [100%]
76 passed   (76 tests collected, 0 skipped, 0 failed)
```

| Test file | Tests | Coverage |
|---|---:|---|
| `test_config.py` | 8 | config validation, serialisation, budget enforcement, schema |
| `test_parameters.py` | 5 | meta-device 3B count, analytic==empirical, variants, KV formula |
| `test_model_forward.py` | 6 | forward/backward, exact CE equality, padding, packing, bf16, init |
| `test_rope.py` | 8 | relative-position property, norm preservation, 4 scaling modes (parametrised), 65K finiteness |
| `test_attention.py` | 7 | chunked masks vs reference, causality, doc isolation, padding, fast path |
| `test_kv_cache_generation.py` | 10 | cache equivalence, streaming, stops, determinism, cancellation, 32,767 decode |
| `test_checkpoint.py` | 7 | atomicity, corruption, recovery, exact restore, tokenizer/config mismatch |
| `test_tokenizer.py` | 13 | round-trip 19 languages, Unicode, injection safety, determinism, tampering |
| `test_accounting.py` | 6 | all counters, invariants, preflight, completion, anti-inflation |
| `test_integration.py` | 2 | tokenizer↔model↔generation, padded + packed training |
| `test_versions.py` | 4 | config identity, compatibility, hashing, schema |

Two tests are marked `slow` (full meta-device 3B instantiation; 32,768-slot cache
decode). All were executed, none skipped.

## 5. Validation results — measured

### 5.1 Parameter count (`python -m rootism.cli.params --verify-meta`)

```
embeddings                150,994,944  ( 4.84%)
attention                 704,643,072  (22.58%)
feed_forward            2,113,929,216  (67.74%)
normalization                 175,104  ( 0.01%)
output_projection         150,994,944  ( 4.84%)
TOTAL                   3,120,737,280  (3.1207 B)
KV cache (bf16): 112.0 KiB/token
meta-device verification: MATCH
```

### 5.2 Forward/backward and cache equivalence

* Tiny model loss finite, gradients finite and non-zero.
* Chunked CE `loss_sum` equals full-logit `F.cross_entropy(reduction="sum")` to
  `1e-4`.
* KV-cache logits vs full recompute: **0.0** max difference at prefill,
  **5.96e-08** at decode.
* bf16 forward produces fp32 finite loss.
* Packed/padded attention paths match explicit reference masks to `1e-5`.

### 5.3 Vocabulary sweep (`scripts/rootism_vocab_sweep.py --documents 12000`)

Pilot corpus: 12,022 original documents, 4,045,604 bytes. Measured
bytes/token: 1.3700 (512) → 1.5479 (1,024) → 1.5621 (2,048) → 1.8034
(4,096) → 1.9259 (8,192) → 2.5126 (15,948 achieved). Identifier fragmentation
fell 0.884 → 0.059. Power-law fit `R²=0.9134`; extrapolated 48K → 2.7464,
64K → 2.8745 (4.67% gain) at a cost of 100.7M params and 2 layers. **Selected
49,152.** Extrapolation limitations are documented, not hidden.

### 5.4 Smoke validation (`scripts/rootism_smoke.py`)

Six genuine AdamW steps on a 106,944-parameter random model:
losses `[5.57234, 5.62177, 5.54349, 5.49440, 5.44701, 5.53685]` (finite,
downward trend), 120 committed actual tokens out of 144 presented (24 masked),
checkpoint restored with **identical logits**, 5 generated tokens. This is a
pipeline test only, explicitly not evidence of learning.

### 5.5 Hardware report (`python -m rootism.cli.doctor`, CPU sandbox)

```
torch 2.15.0+cu132 | CUDA: False | GPUs: 0 | RAM: 3.85 GiB | BF16: False
static total 46.50 GiB, peak/shard 7.78 GiB (8 shards)
estimated duration 40.2 days on 8 H100-class GPUs (assumed 40% MFU)
WARNING: No CUDA device detected
```

Analytical estimates only; no throughput was measured.

### 5.6 Tokenizer integrity

Pilot evaluation: 100% Unicode round-trip, 100% indentation round-trip,
special-token injection safety pass. Fingerprint tampering rejected. Two
identical training runs produced identical `tokenizer_id`.

## 6. Documentation updates

`README.md`, `docs/ARCHITECTURE.md` (14 documented decisions with alternatives,
implications and limitations), `docs/STATUS.md` (all 100 requirements + acceptance
checklist), `docs/MODEL.md`, `docs/TOKENIZER.md`, `docs/HARDWARE.md`,
`docs/INSTALL.md`, `docs/LICENSING.md`, `docs/TRAINING_BUDGET.md`,
`docs/REPRODUCIBILITY.md`, `docs/ROADMAP.md`, and this report.

## 7. Completion report

**Done and verified:** requirements 02–04, 08–18, 20–22, 24–25, 27–28, 57,
and partial completion of 05, 19, 23, 26, 55, 60, 77–78, 97–99.

**Explicitly NOT done:** Stages 2–8 (data/licensing, distributed training,
reasoning corpora, serving API, coding agent, web platform, evaluation, CI,
security review). No dataset is registered; no commercial-use claim is made; the
model is untrained; no benchmark or throughput result is claimed.

**Deviations / honest caveats:**
* Full 32,768-token *prefill* and 65,536-token context behaviour are
  **UNVERIFIED** (no GPU). A real forward at position 32,767 with a 32,768-slot
  cache is tested; that is a decode-position test, not a full prefill.
* The FlashAttention varlen hook is implemented but **UNVERIFIED** (optional
  dependency absent); the chunked SDPA fallback is tested.
* Vocabulary selection at 48K uses extrapolation from a 4 MB pilot corpus; it
  must be re-validated on the licensed production corpus.
* `Tokenizer` merge learning is single-machine with a bounded pre-token table;
  production-scale sharded counting is Stage 2.
* Checkpointing currently covers model state only; full training resume is Stage 3.

**Exit criteria for Stage 1:** all met — plan, code, integration, 76 executed
tests, measured validation results, updated docs, this report.

**Next:** Stage 2 — data pipeline and licensing registry, as specified in
`ROADMAP.md`.
