# Rootism — Stage 1 Implementation Plan (Steps 01–50)

**Status:** Active Execution (Stage 1 of 2)  
**Scope:** Master Engineering Specification Phases I–V (Steps 01–46) and Phase VI Part 1 (Steps 47–50)  
**Deferred to Stage 2 (Next Turn):** Steps 51–100 (LR scheduling, sequence packing, checkpoint fault tolerance, full training loop & profiler, reasoning subsystem, coding agent, inference engine, FastAPI backend, HTML platform, evaluation suite, final production acceptance).

---

## 1. Objectives & Non-Negotiable Constraints

1. **100% Independent Model Implementation (Steps 01–03):**
   - Implement `RootismForCausalLM` from first principles in PyTorch (`rootism/model/`).
   - Zero HuggingFace `transformers`, `tokenizers`, `tiktoken`, or `sentencepiece` dependencies.
   - Enforce runtime provenance guardrails (`rootism/core/provenance.py`) rejecting pretrained external weights, distillation teachers, or fine-tuning adapters on non-Rootism checkpoints.
2. **Exact ~3.00B Parameter Target (Steps 04, 11–20):**
   - Canonical `Rootism-3B` configuration:
     - `vocab_size = 65,536` ($2^{16}$, `uint16`-aligned, 256-multiple for Tensor Cores)
     - `d_model = 2,560` ($10 \times 256$)
     - `n_layers = 40` (deep compositional stack; divisible by 2, 4, 5, 8, 10, 20)
     - `n_heads = 20`, `n_kv_heads = 4`, `head_dim = 128` (5:1 GQA; native FlashAttention-2/3 128-dim tile; 80 KiB/token BF16 KV cache = 2.50 GiB at 32K)
     - `intermediate_size = 6,656` ($26 \times 256$, bias-free SwiGLU)
     - `qk_norm = True` (`RootismRMSNorm(128)` per Q/K head for BF16/FP16 long-context logit stability)
     - `tie_word_embeddings = False` (untied input embedding and output projection)
     - **Exact Total Trainable Parameters:** **`3,009,630,720`** (~3.0096B, +0.32% from 3.0000B).
3. **650B Training Token Budget & 6-Category Accounting (Step 05):**
   - Production training configurations enforce `target_training_tokens >= 600,000,000,000` with default **`650,000,000,000`**.
   - `TokenAccountingLedger` tracks all 6 required counters distinctly without conflating padding, loss-masked tokens, or epoch repeats with net processed training tokens.
4. **Long-Context Support (Steps 17, 19):**
   - Native context length `32,768` tokens (`rope_theta = 500,000.0`, FP32 position math).
   - Built-in YaRN / NTK-by-parts rotary frequency scaling supporting `65,536` (2x) and `131,072` (4x) context windows with attention temperature compensation $m = 0.1 \ln(s) + 1.0$.
5. **Independent Byte-Level Code-Aware BPE Tokenizer (Steps 21–28):**
   - 64 structured special tokens (IDs `0..63`), 256 bijective UTF-8 byte tokens (IDs `64..319`), and learned BPE merges (`320..65,535`).
   - Preserves Python/YAML indentation runs, multi-character operators (`===`, `->`, `::`, `>=`), single-digit numeric boundaries, file paths, and arbitrary UTF-8 without `<unk>`.
   - Cryptographic `TokenizerVersionLock` permanently bound to model config and checkpoints.
   - Prompt-injection safe: literal special token strings in untrusted user/corpus text are never parsed into control token IDs unless explicitly authorized.
6. **Deliberate Programming Language & Ecosystem Engineering (Steps 29–34):**
   - 10 Primary languages (Python, JS, TS, HTML, CSS, SQL, C, C++, Rust, Go) + 9 Secondary languages (Java, C#, PHP, Bash, PowerShell, Lua, Kotlin, Swift, Ruby).
   - Syntax/compiler verification hooks, framework manifest detection, cross-language semantic equivalence engine, principled 650B token allocation, and out-of-scope language epistemic calibration.
7. **Commercial Data Provenance, Quality, Deduplication, Decontamination & Sharding (Steps 35–46):**
   - Strict SPDX commercial allowlist + file-header/submodule/vendored deep license scanner.
   - 13-field dataset provenance registry & auto-generated attribution ledger.
   - Verified original synthetic data generator (executes generated tests against reference solutions before admission).
   - Multi-stage quality & prompt-injection sanitizer, 5-level deduplication (SHA-256, MinHash+LSH, normalized-code, snippet, repo overlap), benchmark decontamination, curriculum mixture sampler, and O(1)-RAM `mmap` `uint16` `.rshard`/`.ridx` streaming storage.
8. **Distributed, Memory-Optimised Training Foundations & Optimiser (Steps 47–50):**
   - PyTorch hardware capability discovery & analytical memory/throughput planner.
   - DDP, FSDP (`SHARD_GRAD_OP` / `FULL_SHARD`), Tensor-Parallel specs, and Pipeline-Parallel stage partitioning.
   - BF16 / FP16 (`GradScaler`) / FP32 precision controller, selective activation checkpointing, and `no_sync()` gradient accumulation.
   - Decoupled AdamW with strict 2D-vs-1D weight-decay grouping and optional FP32 master-weight/moment updates.

---

## 2. Execution & Verification Sequence

1. **Core Config, Provenance & Token Accounting** (`rootism/config.py`, `rootism/core/`)
2. **Rootism-3B Neural Architecture & Inspector** (`rootism/model/`)
3. **Rootism Byte-Level BPE Tokenizer, Trainer & Evaluator** (`rootism/tokenizer/`)
4. **Programming Languages, Frameworks & Cross-Language Engine** (`rootism/languages/`)
5. **Commercial Data Licensing, Quality, Dedup, Decontamination, Synthetic & Sharding** (`rootism/data/`)
6. **Distributed Strategy, Memory Optimisation & Optimiser** (`rootism/training/`)
7. **Unified CLI, Production YAML Configs & Technical Documentation** (`rootism/cli.py`, `configs/`, `docs/`)
8. **Comprehensive Automated Test Suite (`pytest`) & Live Validation Benchmarks** (`tests/`)
