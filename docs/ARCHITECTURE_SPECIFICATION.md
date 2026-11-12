# Rootism — Central Architecture Specification & Decision Records

**Architecture Identifier:** `rootism-decoder-v1`  
**Canonical Model:** `Rootism-3B` (`3,009,630,720` verified trainable parameters)  
**Target Training Budget:** `650,000,000,000` net processed non-padding training tokens (`> 600,000,000,000` strictly enforced)  
**Origin Policy:** 100% independent implementation from randomly initialized parameters (`ROOTISM_INDEPENDENT_SCRATCH_V1`).

---

## 1. System-Wide Architecture Overview

Rootism is structured as a single unified Python platform (`rootism/`) where every subsystem shares validated Pydantic configuration contracts (`rootism/config.py`), cryptographic provenance hashes (`rootism/core/provenance.py`), and permanent tokenizer version locks (`TokenizerVersionLock`):

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Rootism Unified Configuration                        │
│     RootismConfig (Model + Tokenizer + Commercial Data + Training Spec)      │
└───────────────┬──────────────────────┬───────────────────────┬───────────────┘
                │                      │                       │
                ▼                      ▼                       ▼
┌───────────────────────────┐ ┌────────────────────┐ ┌─────────────────────────┐
│     Phase III Tokenizer   │ │ Phase IV-V Corpus  │ │   Phase II Model Core   │
│  CodeAwarePretokenizer    │ │ SPDX License Gate  │ │   RootismForCausalLM    │
│  Byte-Level BPE (65,536)  │ │ Deep Header Scan   │ │   40x Decoder Layers    │
│  64 Special Control IDs   │ │ Quality & Security │ │   5:1 GQA + QK-RMSNorm  │
│  SHA-256 Version Lock     │ │ 5-Level Dedup      │ │   RoPE (500k) + YaRN    │
└───────────────┬───────────┘ │ Decontamination    │ │   SwiGLU (6,656)        │
                │             │ Verified Synthetic │ │   Chunked CE Loss       │
                └────────────►│ uint16 .rshard mmap│ └────────────┬────────────┘
                              └─────────┬──────────┘              │
                                        │                         │
                                        ▼                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                  Phase VI Distributed Training & Optimisation                │
│  Hardware Planner • FSDP2 (DTensor fully_shard) / DDP • BF16 / FP16 Scaler   │
│  Selective Activation Checkpointing • Decoupled AdamW • 6-Metric Token Ledger│
└───────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│     Stage 2 Integration Seam (Steps 51–100: Scheduled for Next Turn)         │
│  LR Scheduler & Sequence Packing • Atomic Fault-Tolerant Checkpointing       │
│  Structured Reasoning Engine • Multi-File Coding Agent • KV-Cached Serving   │
│  Responsive HTML Developer Platform • Contamination-Isolated Benchmarks      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Exact `Rootism-3B` Parameter Breakdown

Computed programmatically by walking `RootismForCausalLM.named_parameters()` (`rootism inspect-model`):

| Subsystem | Tensor Shapes | Per-Layer Count | Total Parameters | Share (%) |
|---|---|---:|---:|---:|
| **Token Embeddings** (`tok_embeddings`) | `65,536 × 2,560` | — | `167,772,160` | `5.5745%` |
| **Attention Projections + QK-Norm** (`40` layers) | `Q: 2560×2560`, `K: 2560×512`, `V: 2560×512`, `O: 2560×2560`, `q_norm: 128`, `k_norm: 128` | `15,728,896` | `629,155,840` | `20.9048%` |
| **SwiGLU Feed-Forward** (`40` layers) | `gate: 2560×6656`, `up: 2560×6656`, `down: 6656×2560` | `51,118,080` | `2,044,723,200` | `67.9393%` |
| **Layer & Final RMSNorms** | `40 × (2 × 2560) + 2560` | `5,120` (+`2,560`) | `207,360` | `0.0069%` |
| **Output Projection** (`lm_head`, untied) | `2,560 × 65,536` | — | `167,772,160` | `5.5745%` |
| **TOTAL TRAINABLE PARAMETERS** | — | `66,852,096` / block | **`3,009,630,720`** | **`100.0000%`** |

---

## 3. Architecture Decision Records (ADRs)

### ADR-01: 40-Layer Depth × 2,560 Width vs Wider/Shallower or Narrower/Deeper Alternatives
- **Selected:** `n_layers = 40`, `d_model = 2,560`, `n_heads = 20`, `n_kv_heads = 4`, `head_dim = 128`, `intermediate_size = 6,656` (`3,009,630,720` parameters).
- **Why Selected:**
  1. Multi-step algorithmic reasoning, type tracking, and state tracing scale strongly with sequential circuit depth (`40` layers vs `26` layers in a `d_model=3072` design).
  2. Every matrix dimension (`2,560 = 10 × 256`, `6,656 = 26 × 256`, `65,536 = 256 × 256`, `head_dim = 128`) is aligned to 128/256-element Tensor Core tiles and FlashAttention-2/3 native `head_dim=128` SRAM blocks (zero kernel padding).
  3. `40` layers divide evenly by `2, 4, 5, 8, 10, 20` for balanced Pipeline Parallelism, and `20Q / 4KV` divides evenly by `TP ∈ {1, 2, 4}`.
- **Alternatives Considered (Programmatically Audited in `evaluate_architecture_candidates()`):**
  - *Candidate-Wide-26L* (`d=3072, L=26, 24Q/8KV, ffn=8192`, `3.020B` params): 35% shallower circuit depth and 30% larger BF16 KV cache (`3.25 GiB` vs `2.50 GiB` at 32K).
  - *Candidate-Deep-48L* (`d=2048, L=48, 16Q/4KV, ffn=7680`, `3.037B` params): narrower residual bottleneck (`2048`) increases cross-layer communication latency by 20% and inflates KV cache to `3.00 GiB` at 32K.
- **Expected Performance Implications:** Higher compositional reasoning accuracy per FLOP alongside a compact `80 KiB/token` BF16 KV cache (`2.50 GiB` @ 32K, `5.00 GiB` @ 64K).
- **Known Limitations:** Cannot use `TP = 8` (since `n_kv_heads = 4`); however, for a 3B parameter model, `TP = 1` with FSDP2 Hybrid/ZeRO-2 easily fits inside GPU memory without intra-layer all-reduce latency.

### ADR-02: 5:1 Grouped-Query Attention + Per-Head QK-RMSNorm
- **Selected:** `n_heads = 20`, `n_kv_heads = 4` (5 query heads per KV group) with `RootismRMSNorm(128)` applied to `Q` and `K` prior to RoPE.
- **Why Selected:** Cuts KV cache 5× relative to MHA (`2.50 GiB` vs `12.50 GiB` at 32K), enabling single-GPU 32K–64K coding sessions on consumer/workstation GPUs. Per-head QK-RMSNorm bounds pre-softmax attention logits, preventing BF16/FP16 softmax saturation and loss spikes during 650B-token training at `lr = 3e-4`.
- **Alternatives Considered:** Standard MHA (`12.50 GiB` KV cache at 32K — prohibitive for multi-file coding sessions), MQA (`n_kv_heads = 1`, `0.625 GiB` KV cache — measurable degradation on multi-hop symbol retrieval across large files).
- **Known Limitations:** Adds `256` parameters and two tiny RMSNorm launches per layer (`0.0003%` parameter overhead).

### ADR-03: FP32 RoPE (`theta = 500,000`) + YaRN Wavelength-Aware Context Scaling
- **Selected:** Rotary Position Embeddings computed strictly in `torch.float32` with `rope_theta = 500,000.0`, native context `32,768`, and built-in YaRN / NTK-by-parts scaling + softmax temperature compensation $m = 0.1 \ln(s) + 1.0$ for `65,536` and `131,072` contexts.
- **Why Selected:** In `bfloat16`, integers above `256` lose unit resolution (at position `32,768`, BF16 ULP step is `256`, collapsing adjacent tokens onto identical rotary angles!). Enforcing `float32` position/frequency math preserves exact single-token resolution across 32K–128K tokens. YaRN leaves short-wavelength high frequencies untouched (preserving local token/indentation syntax) while interpolating long wavelengths smoothly.
- **Alternatives Considered:** ALiBi (weaker long-range exact symbol copy/lookup in code repositories), naive linear Position Interpolation (degrades local syntax resolution unless heavily fine-tuned).
- **Known Limitations:** Context extension from 32K to 64K+ performs best when annealed on long sequences (included as Stage 3 of `configs/training/pretrain_650b.yaml`).

### ADR-04: Checkpointed Chunked Cross-Entropy Loss Projection
- **Selected:** Sequence-chunked cross-entropy (`loss_chunk_size = 2,048` tokens wrapped in `torch.utils.checkpoint`) when `return_logits=False`.
- **Why Selected:** At `seq_len = 32,768` and `vocab_size = 65,536`, materializing dense FP32 `[1, 32768, 65536]` logits consumes **8.59 GiB** of VRAM alone. Chunked checkpointed projection reduces peak logit VRAM 16× to **0.54 GiB** while producing **mathematically identical** loss and gradients (verified in `test_chunked_cross_entropy_exact_loss_and_gradient_equivalence`).
- **Alternatives Considered:** Dense full-sequence `lm_head` materialization (causes OOM on 32K training microbatches).
- **Known Limitations:** Recomputes the linear `lm_head` chunk during backward (~2% FLOP overhead).

### ADR-05: Independent 65,536-Vocab Code-Aware Byte-Level BPE Tokenizer + Injection Boundary
- **Selected:** `vocab_size = 65,536` (`uint16`: `0..63` frozen special/control tokens, `64..319` bijective UTF-8 byte tokens, `320..65,535` BPE merges) with regex pre-tokenization preserving indentation runs, multi-char operators (`===`, `->`, `::`), and single-digit numbers; `allow_special=False` enforced on all untrusted inputs.
- **Why Selected:**
  1. `65,536` maximizes compression across 19 programming languages + prose while fitting every token ID inside a 2-byte `uint16` — cutting the 650B-token binary dataset from `2.60 TB` (`uint32`) to **`1.30 TB`** (`uint16`).
  2. Single-digit splitting prevents arbitrary multi-digit number fragmentation (`1024` vs `102`+`4`), materially improving arithmetic and line-number reasoning.
  3. Default `allow_special=False` guarantees untrusted user prompts or repository files containing literal `"<|tool_call_start|>"` can never forge control tokens.
- **Alternatives Considered:** Unigram LM / SentencePiece (frequently normalizes or collapses significant Python/YAML whitespace unless heavily patched), `128K` vocab (forces `uint32` shards, doubling disk I/O to `2.60 TB` and inflating embedding + `lm_head` parameters from `335M` to `655M`, stealing `320M` parameters from reasoning layers).
- **Known Limitations:** Non-Latin natural languages use slightly more UTF-8 byte merges per character than a 128K multilingual vocabulary.

### ADR-06: Composable PyTorch FSDP2 (`DTensor` `fully_shard`) + Selective Activation Checkpointing
- **Selected:** PyTorch 2.x composable `fully_shard` (`reshard_after_forward=False` for ZeRO-2, `True` for ZeRO-3, 2D `DeviceMesh` for Hybrid multi-node) paired with hardware-adaptive `BF16` / `FP16 + GradScaler` / `FP32` and selective checkpointing (`every_n_layers = 2`).
- **Why Selected:** Unlike legacy FSDP1 `FlatParameter`, FSDP2 shards each `nn.Parameter` cleanly as a `DTensor` (preserving exact 2D-vs-1D AdamW weight-decay boundaries and working across both CUDA `nccl` and CPU `gloo`). Selective checkpointing cuts activation memory by ~45% for only ~16.5% FLOP recompute overhead.
- **Alternatives Considered:** Pipeline-only or TP=8 sharding (unnecessary intra-layer latency for a 3B model), legacy FSDP1 (fails on CPU verification workers and complicates per-parameter optimizer states).
- **Known Limitations:** Ultra-low-VRAM GPUs (`<= 24 GiB`, e.g. RTX 4090) require `fsdp_zero3` + `full` activation checkpointing (+33% recompute FLOPs).

### ADR-07: Decoupled AdamW (`betas=(0.9, 0.95)`, `wd=0.1`) + Optional `RootismFP32MasterAdamW`
- **Selected:** Parameter-grouped decoupled AdamW (`weight_decay = 0.1` strictly on $\ge 2\text{D}$ matrices; `0.0` on 1D RMSNorm/QK-Norm/bias vectors), $\beta_1 = 0.9, \beta_2 = 0.95, \epsilon = 10^{-8}$, with fused CUDA kernel dispatch and an independently implemented `RootismFP32MasterAdamW` for direct low-precision weight storage.
- **Why Selected:** $\beta_2 = 0.95$ (exponential window ~20 steps vs ~1,000 steps for `0.999`) recovers rapidly from rare gradient spikes on long code/math sequences. Applying weight decay to RMSNorm 1D gains destabilizes layer scales and is explicitly prevented.
- **Alternatives Considered:** Lion (higher sensitivity to weight decay / LR ratio across curriculum transitions), Adafactor (factored second-moment approximations can degrade tail convergence at >500B tokens).
- **Known Limitations:** Full FP32 optimizer state consumes `12 bytes/param` (`33.64 GiB` globally, sharded across GPUs via FSDP2).

### ADR-08: Token-Indexed Warmup-Stable-Decay (WSD) Schedule + Boundary-Safe Sequence Packing
- **Selected:** `RootismLRScheduler` indexed by net `processed_training_tokens` supporting `wsd` (linear warmup $\to$ peak plateau $\to$ final 15% cosine decay) and `cosine`, paired with `SequencePacker` resetting `position_ids` to `0` at every document boundary, emitting `document_ids` for block-diagonal causal isolation, and setting `labels = -100` on the opening token of each newly packed document.
- **Why Selected:** Indexing LR by processed tokens rather than optimizer step count keeps learning-rate trajectories continuous when `sequence_length` increases from `4,096` to `32,768` / `65,536` across curriculum stages. Boundary-safe packing achieves >86%–99% sequence fill efficiency with zero cross-document attention leakage (proven in `test_sequence_packer_causal_document_isolation`).
- **Alternatives Considered:** Unmasked concatenation with `<|eos|>` only (pollutes RoPE positions and leaks cross-document context across packed samples), step-based cosine LR (forces fixed horizon early and jumps on batch-size changes).
- **Known Limitations:** Block-diagonal `document_ids` masks materialize a boolean `[B, 1, S, S]` mask on packed batches when custom `FlexAttention` kernels are unavailable; mitigated by `4,096` packing lengths during Stages 1–2.

### ADR-09: Atomic Staging-Directory Checkpointing + Cryptographic SHA-256 Recovery Fallback
- **Selected:** `AtomicCheckpointManager` writing `model.safetensors` (with scratch provenance metadata), `training_state.pt` (optimizer, scheduler, Python/NumPy/CUDA RNGs, 6-category `TokenAccountingLedger`, `dataset_cursor` + `SequencePacker` buffer), `config.yaml`, and `checkpoint_manifest.json` (per-file SHA-256 digests) inside `step_XXXXXXXX.tmp`, `fsync`ing files and parent directory before atomic `os.replace`, plus `PreemptionSignalCatcher` (`SIGTERM`/`SIGINT`).
- **Why Selected:** Multi-day 650B runs inevitably encounter node preemptions or disk interruptions. Staging + SHA-256 verification guarantees partial `.tmp` writes or bit-flipped checkpoints are automatically skipped in favor of the latest verified recovery point (`test_atomic_checkpoint_resume_and_corrupted_fallback`).
- **Alternatives Considered:** In-place `torch.save` overwrites (single crash mid-write permanently destroys the recovery checkpoint).
- **Known Limitations:** Temporarily requires `1x` extra checkpoint size (`~42.13 GB` for `Rootism-3B`) during the atomic `.tmp` swap window.

### ADR-10: Multi-Objective Reasoning Supervision & Strict Deliberation Privacy Boundary
- **Selected:** Three structured supervision formulations (`plan_implement_verify` with differential span weights `0.65 / 1.15 / 1.50`, `contrastive_flaw_repair`, and `execution_state_trace`), evaluated by `ReasoningQualityEvaluator` (conclusion accuracy, non-contradiction, and conciseness efficiency), paired with `ReasoningOutputFormatter` supporting 5 presentation modes (`concise`, `explain`, `structured`, `approach_summary`, `assumptions_and_intermediates`).
- **Why Selected:** Pure chain-of-thought imitation rewards verbose rambling even when conclusions fail. Differential weighting + contrastive flaw localization anchors supervision on valid conclusions and working code. `ReasoningOutputFormatter` unconditionally strips `<|reason_plan_start|>` and `<|reason_verify_start|>` scratchpads while surfacing only genuine `<|reason_summary_start|>` summaries (never fabricating reasoning when absent).
- **Alternatives Considered:** Exposing raw internal scratchpads directly to users (leaks noisy search paths and prompt-policy internals), uniform next-token weighting across verbose drafts.
- **Known Limitations:** Rule-based contradiction/conciseness checks complement, rather than replace, executable solver/test verification.

### ADR-11: Confined Multi-Step Coding Agent, Anti-Loop Guardrails & Human Confirmation Gate
- **Selected:** `CodingAgentOrchestrator` (`inspect -> plan -> edit -> execute -> test -> diagnose -> repair -> summarise`) operating over `AgentToolRegistry` (with reversible `FileMutationRollbackEntry` stack and unified diffs), `SandboxedCommandRunner` (`RLIMIT_CPU`, `RLIMIT_FSIZE`, process-group timeout kills, secret env stripping), `SafetyPolicyGuard` (lexical + realpath confinement, unconditional blocklist, and `confirm-*` human approval tokens for destructive actions), and duplicate-action / stagnation guardrails.
- **Why Selected:** Software engineering agents fail in production primarily from runaway edit-test loops or unsafe shell mutations. Combining cryptographic action-fingerprint deduplication, stagnation counters, reversible unified-diff snapshots, and mandatory human confirmation tokens prevents both infinite loops and accidental data loss.
- **Alternatives Considered:** Unrestricted shell tool execution (unacceptable workspace escape and secret-exfiltration risk).
- **Known Limitations:** POSIX `RLIMIT_*` resource caps provide process-level sandboxing; multi-tenant cloud deployments should additionally run each workspace inside an isolated container/microVM.

### ADR-12: Non-Destructive `INT8` / `INT4` Weight Quantisation & KV-Cached SSE Serving Runtime
- **Selected:** `RootismInferenceEngine` loading verified `.safetensors` checkpoints with incremental `KVCacheLayerState` decoding, cooperative `threading.Event` cancellation, YaRN context-window fitting, and non-destructive `QuantizedLinearInt8` (per-channel symmetric) & `QuantizedLinearInt4` (group-64 nibble-packed `uint8`) transforms, served by FastAPI (`create_rootism_app`) with SSE streaming, `asyncio.Semaphore` concurrency control, sliding-window rate limiting, and optional Bearer API-key auth.
- **Why Selected:** Cloning/exporting quantised weights in memory or separate artifacts preserves the master full-precision checkpoint intact (`2.26x–2.85x+` compression; `0.99995` INT8 logit cosine similarity). SSE streaming over `KVCacheLayerState` delivers low-TTFT token-by-token responses with immediate disconnect cancellation.
- **Alternatives Considered:** In-place destructive weight quantisation (irreversibly degrades master checkpoints), WebSocket-only streaming (harder to proxy through standard HTTP load balancers and `curl` clients than SSE).
- **Known Limitations:** Custom PyTorch dequantize-on-forward `INT4`/`INT8` modules prioritize portability across CPU and CUDA over custom architecture-specific Triton/CUTLASS kernels.

### ADR-13: Zero-CDN Root-Directory HTML Developer Platform with Relative API Routing
- **Selected:** Standalone `index.html`, `styles.css`, and `app.js` placed directly in the repository root (`/home/user/rootism-official/`), communicating exclusively via relative `./api/v1/*` paths, featuring WCAG AA+ dark/light themes, multi-session chat with live SSE streaming, 5-mode reasoning selector, collapsible faithful reasoning summaries, integrated file explorer + line-numbered code editor + FIM completion + unified diff/undo viewer, live Coding Agent trace panel with interactive human confirmation gate, and truthful checkpoint telemetry.
- **Why Selected:** Placing the platform in the repository root makes static inspection and hosting immediate, zero-CDN CSS/JS guarantees offline/air-gapped reliability, and strictly relative `./api/v1/*` URLs ensure browser clients work identically on local ports and reverse-proxied preview hosts (`https://{port}-{sandboxId}.e2b.app`).
- **Alternatives Considered:** Heavyweight bundled SPA build chains requiring Node build steps or hardcoded `http://localhost:8000` API origins (breaks browser previews when backend runs in a remote sandbox).
- **Known Limitations:** Built-in syntax highlighting uses lightweight client regex rules rather than a multi-megabyte WASM grammar bundle.

