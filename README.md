# Rootism

**Rootism** is an independently engineered, commercial-grade **~3.01-billion-parameter (`3,009,630,720`)** AI coding, multi-step reasoning, and software engineering agent system built completely from scratch — with **zero** pretrained weights, **zero** external tokenizers, **zero** fine-tuning on third-party models, and **zero** distillation from external systems.

---

## Core Engineering Highlights

- **Scratch `Rootism-3B` Architecture (`rootism/model/`)**:
  - `40` pre-RMSNorm decoder layers, `d_model = 2560`, `20` query heads / `4` KV heads (`5:1` GQA with per-head QK-RMSNorm), `head_dim = 128`, SwiGLU FFN (`intermediate_size = 6656`), untied embeddings (`vocab_size = 65,536`) -> **`3,009,630,720` verified trainable parameters**.
  - `float32` Rotary Position Embeddings (`theta = 500,000`) + YaRN wavelength-aware scaling supporting **`32,768` native** and **`65,536+` extended** contexts, plus checkpointed chunked cross-entropy reducing 32K logit VRAM `16x` (`8.59 GiB -> 0.54 GiB`).
- **Custom 65,536 Byte-Level BPE Tokenizer (`rootism/tokenizer/`)**:
  - `64` frozen protocol/reasoning/tool/FIM control tokens + `256` bijective UTF-8 byte tokens + `65,216` code-aware BPE merges, single-digit numeric splitting, cryptographic SHA-256 `TokenizerVersionLock`, and strict `allow_special=False` prompt-injection isolation.
- **Commercial Provenance & `>600B` (`650B` Target) Data Infrastructure (`rootism/data/`, `rootism/languages/`)**:
  - 19-language engineering registry (`10` primary: Python, JS, TS, HTML, CSS, SQL, C, C++, Rust, Go; `9` secondary: Java, C#, PHP, Bash, PowerShell, Lua, Kotlin, Swift, Ruby) + honest unsupported-language calibration.
  - Strict SPDX commercial license gate, 13-field `DatasetProvenanceRegistry`, nested file-level license scanner, 5-level deduplicator (SHA-256, normalized text, AST/token-normalized code, MinHash-LSH near-dedup, repo cluster cap), 13-gram benchmark decontaminator, subprocess-verified synthetic task generator, and zero-copy `uint16` `.rshard` `mmap` streaming shards.
- **Distributed & Fault-Tolerant Pretraining Engine (`rootism/training/`)**:
  - Hardware-adaptive single-GPU / DDP / composable PyTorch `FSDP2` (`DTensor` `fully_shard` ZeRO-2, ZeRO-3, and Hybrid multi-node), `TokenAccountingLedger` enforcing `>600B` processed training tokens across 6 distinct categories, token-indexed `WSD`/`Cosine` LR scheduler, boundary-safe `SequencePacker`, `TrainingStabilityMonitor`, `TrainingThroughputProfiler`, and `AtomicCheckpointManager` (`.tmp` staging + `fsync` + SHA-256 manifest + automatic fallback recovery + `SIGTERM`/`SIGINT` preemption handler).
- **Structured Reasoning & Sandboxed Multi-Step Coding Agent (`rootism/reasoning/`, `rootism/agent/`)**:
  - Progressive multi-faculty reasoning curriculum, multi-objective differential-weight & contrastive supervision, `ReasoningQualityEvaluator`, and 5 user presentation modes (`concise`, `explain`, `structured`, `approach_summary`, `assumptions_and_intermediates`) with strict redaction of private scratchpads.
  - `CodingAgentOrchestrator` (`inspect -> plan -> edit -> execute -> test -> diagnose -> repair -> summarise`) with `SandboxedCommandRunner` (workspace realpath confinement, timeouts, `RLIMIT_*`, secret env stripping), unified-diff rollback stack, anti-loop guardrails, and mandatory human confirmation tokens (`confirm-*`) for destructive operations.
- **Inference Runtime, `INT8`/`INT4` Quantisation, FastAPI Server & Root Web Platform (`rootism/inference/`, `rootism/server/`, `/index.html`)**:
  - KV-cached `RootismInferenceEngine` with SSE token streaming, cooperative cancellation, FIM code completion, and non-destructive `INT8` / `INT4` weight quantisation.
  - Developer Studio placed directly in the **repository root** (`index.html`, `styles.css`, `app.js`) using strictly relative `./api/v1/*` routes, WCAG AA+ dark/light themes, multi-session streaming chat, integrated file explorer + line-numbered editor + unified diff/undo viewer, and live Coding Agent panel with interactive safety confirmation banner.

---

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 1. Run full 42-test automated verification suite (Phases I–XII)
pytest -v

# 2. Audit verified 3.01B architecture & 650B cluster hardware plans
rootism inspect-model --compare-candidates
rootism plan-hardware --gpu h100_80gb_sxm --num-gpus 64

# 3. Verify commercial dataset provenance, INT8/INT4 quantisation & evaluation harness
rootism verify-licenses
rootism quantize-eval
rootism evaluate

# 4. Launch the FastAPI Server + Root HTML Developer Platform on 0.0.0.0:8000
rootism serve --host 0.0.0.0 --port 8000
```

---

## Documentation Index

- [`docs/ARCHITECTURE_SPECIFICATION.md`](docs/ARCHITECTURE_SPECIFICATION.md) — Central Architecture Specification & ADRs `01–13`
- [`docs/HARDWARE_AND_SCALING_ESTIMATES.md`](docs/HARDWARE_AND_SCALING_ESTIMATES.md) — Explicit Formula-Based Memory, Storage & 650B Cluster Duration Matrix
- [`docs/LICENSING_AND_PROVENANCE.md`](docs/LICENSING_AND_PROVENANCE.md) & [`THIRD_PARTY_DATA_NOTICES.md`](THIRD_PARTY_DATA_NOTICES.md) — Commercial SPDX Governance & Dataset Attribution Ledger
- [`docs/OPERATOR_AND_DEPLOYMENT_GUIDE.md`](docs/OPERATOR_AND_DEPLOYMENT_GUIDE.md) — Operator Runbook (Tokenizer, Sharding, `torchrun` FSDP2, Recovery, Quantisation, Serving)
- [`docs/TRAINING_AND_WEIGHTS.md`](docs/TRAINING_AND_WEIGHTS.md) — Step-by-step: train a model, then serve those exact weights in the studio (standby → live switch)
- [`docs/FINAL_PRODUCTION_ACCEPTANCE_AUDIT.md`](docs/FINAL_PRODUCTION_ACCEPTANCE_AUDIT.md) — Step 100 Acceptance Audit & Honest Training Status Disclosure
- [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) — Complete 100-Step Engineering Tracker
