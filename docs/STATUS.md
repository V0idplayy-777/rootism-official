# Rootism — Project Status Register

**Branch:** `arena/b5e3aae0-rootism-official` · **Stage:** 1 of 8 complete ·
**Last updated:** 2026-11-12 · **Schema version:** 1

State meanings:

* **DONE (tested)** — implemented, integrated, and covered by executed tests.
* **PARTIAL** — real implementation exists but a required part is missing.
* **NOT IMPLEMENTED** — no implementation. No placeholder exists.
* **UNVERIFIED** — implementation exists but the specific claim has not been
  tested in this environment (usually requires GPU/large-scale compute).

Nothing below is marked DONE without tests in `tests/` that were actually run.

---

## 1. Subsystem register

| # | Subsystem | Requirements | State | Depends on | Outstanding work |
|---|---|---|---|---|---|
| S1 | Project identity & non-negotiables | 01–10 | **PARTIAL** | — | Legal review, full production acceptance (100) |
| S2 | Model architecture | 11–20 | **DONE (tested)** | config, torch | GPU long-context prefill **UNVERIFIED** |
| S3 | Parameter accounting | 04, 20 | **DONE (tested)** | S2 | — |
| S4 | Tokenizer | 21–28 | **PARTIAL** | S1 | Production-corpus training, cross-language eval at scale, sharded BPE counting |
| S5 | Programming-language scope | 29–34 | **NOT IMPLEMENTED** | S6 | Language mixture, framework examples, cross-language tasks |
| S6 | Data, licensing & provenance | 35–46 | **NOT IMPLEMENTED** | S4 | Registry, licence verification, QC, dedup, contamination, mixture, streaming shards, inspection |
| S7 | Training engine | 47–60 | **NOT IMPLEMENTED** | S6, S2 | Distributed strategy, AMP, optimiser, LR schedule, packing loader, stability, full resume, profiling |
| S8 | Reasoning & intelligence | 61–67 | **NOT IMPLEMENTED** | S6, S7 | Original reasoning corpora, objectives, user-visible summary controls |
| S9 | Coding agent | 68–76 | **NOT IMPLEMENTED** | S5, S10 | Orchestration, workspace, tools, sandbox, tests/debug, safety |
| S10 | Inference & serving | 77–80 | **PARTIAL** | S2, S7 | Quantisation, batching, HTTP API, auth, monitoring, concurrency |
| S11 | Web platform / UX | 81–90 | **NOT IMPLEMENTED** | S10, S9 | Frontend, chat, workspace UI, agent activity, settings, a11y |
| S12 | Evaluation & QA | 91–96 | **NOT IMPLEMENTED** | S7, S2 | Benchmarks, generalisation tests, regression gates, structured reports |
| S13 | Production engineering | 97–99 | **PARTIAL** | all | CI, security review, deployment docs, dependency licence audit |
| S14 | Final acceptance | 100 | **NOT MET** | all | Full checklist in §3 |
| S15 | Config / versioning / reproducibility | 09, 28, G | **DONE (tested)** | — | Dataset-manifest binding in Stage 2 |
| S16 | Checkpointing & recovery | 55, 57 | **PARTIAL** | S2 | Optimiser/scheduler/RNG/dataset-state resume (Stage 3) |
| S17 | Token accounting / budget | 05, 60 | **PARTIAL** | S7 | Wire ledger to real distributed steps; verify against shard manifests |
| S18 | Hardware estimates | E, 58, 59 | **PARTIAL** | S2 | Replace analytical estimates with measured profiler results |
| S19 | Automated tests | 98 | **PARTIAL** | all | Model/tokenizer/checkpoint/API/agent/security/frontend tests; 76 exist today |

---

## 2. Requirement-by-requirement state

### Phase I — Identity (01–10)
| Req | State | Note |
|---|---|---|
| 01 identity / commercial-grade intent | PARTIAL | Codebase organised as a product; not yet a commercial deployment |
| 02 independent architecture | **DONE (tested)** | `src/rootism/model/*` written from scratch; no model libraries |
| 03 no pretrained dependencies | **DONE (tested)** | random init only; tokenizer uses no pretrained vocab; test asserts no `transformers`/`tokenizers`/`sentencepiece` imports |
| 04 ~3B with programmatic breakdown | **DONE (tested)** | 3,120,737,280 verified on `meta` device |
| 05 >600B tokens, 650B target, distinct categories | **PARTIAL** | `TrainingTokenBudget` + `TokenLedger` enforce it; no trainer yet |
| 06 production quality / no fakes | **DONE (tested)** for Stage 1 | No simulation anywhere in Stage 1 |
| 07 no unnecessary shortcuts | DONE for Stage 1 | Chunked CE, chunked masks, exact CE equality tested |
| 08 documented technical decisions | **DONE** | `ARCHITECTURE.md` §3 (14 decisions w/ alternatives, implications, limitations) |
| 09 coherent architecture | **DONE (tested)** | Shared config/version/checkpoint identity chain |
| 10 engineering honesty | **DONE** | `UNVERIFIED` / `NOT IMPLEMENTED` used explicitly |

### Phase II — Model (11–20)
| Req | State | Note |
|---|---|---|
| 11 architecture research/evaluation | **DONE** | Documented in `ARCHITECTURE.md` §3; pilot tests executed |
| 12 core autoregressive transformer | **DONE (tested)** | forward/backward, bf16, variants |
| 13 efficient attention | **DONE (tested)** | SDPA fast path + chunked masks; flash-varlen hook implemented but **UNVERIFIED** (optional dep not installed) |
| 14 GQA + KV requirements | **DONE (tested)** | 24/8 heads, 112 KiB/token bf16, 3.5 GiB @ 32K |
| 15 SwiGLU | **DONE (tested)** | ff=8192 |
| 16 RMSNorm / low precision | **DONE (tested)** | fp32 accumulation; bf16 forward finite |
| 17 RoPE + scaling | **DONE (tested)** | plain/linear/dynamic-NTK/YaRN; finiteness at position 65,000 |
| 18 stable residuals | **DONE (tested)** | depth-scaled init verified numerically |
| 19 context ≥32,768, test long context | **PARTIAL / UNVERIFIED** | config=32,768; decode at position 32,767 with a 32,768-slot cache tested on CPU; **full 32K prefill and 65K training extension NOT VERIFIED** (requires GPU) |
| 20 precise parameter breakdown | **DONE (tested)** | analytic == empirical on `meta` |

### Phase III — Tokenizer (21–28)
| Req | State | Note |
|---|---|---|
| 21 independent tokenizer | **DONE (tested)** | own BPE trainer |
| 22 algorithm evaluation | **DONE** | byte-level BPE chosen; alternatives documented |
| 23 vocabulary selection by empirical analysis | **PARTIAL** | `scripts/rootism_vocab_sweep.py` executed (4 MB pilot); 48K selected; **extrapolated region must be re-run on the production corpus** |
| 24 programming-language representation | **DONE (tested)** | 19 languages, Unicode, paths, indentation, operators, strings, comments round-trip |
| 25 special tokens + semantics | **DONE (tested)** | 49 tokens with normative semantics |
| 26 tokenizer training pipeline | **PARTIAL** | streaming SQLite spill, deterministic merges, validation, serialisation; production-scale sharding missing |
| 27 tokenizer evaluation | **DONE (tested)** | compression, fragmentation, Unicode, NL, cross-language, injection safety |
| 28 tokenizer ↔ checkpoint versioning | **DONE (tested)** | fingerprint + `compatibility_report` mismatch test |

### Phase IV — Languages (29–34) — **NOT IMPLEMENTED**
Language scope, competence, frameworks, cross-language reasoning, prioritisation and
unsupported-language behaviour all require the data mixture (Stage 2). The pilot
corpus contains original samples in all 19 named languages **for tokenizer tests
only**; it is explicitly not training data.

### Phase V — Data & licensing (35–46) — **NOT IMPLEMENTED**
No external dataset is downloaded, registered or shipped. `docs/LICENSING.md`
defines the policy and registry schema; the registry itself is empty. **No
commercial-use claim is made.**

### Phase VI — Training engine (47–60) — **NOT IMPLEMENTED** except:
| Req | State | Note |
|---|---|---|
| 55 checkpointing | **PARTIAL** | model-state checkpointing done and tested; optimiser/scheduler/RNG/dataset state missing |
| 57 checkpoint validation | **DONE (tested)** | atomic writes, SHA-256, recovery points, corruption detection |
| 60 token accounting | **PARTIAL** | ledger implemented and tested, not yet wired to a trainer |

`scripts/rootism_smoke.py` performs a genuine 6-step gradient-based optimisation
on a 106,944-parameter randomly initialised model. It is a **pipeline
validation**, explicitly not evidence that Rootism has learned anything.

### Phase VII — Reasoning (61–67) — **NOT IMPLEMENTED**
`<reasoning>` / `<task>` special tokens and their "user-visible summary, not
private trace" semantics are defined (25); generation of such summaries and
reasoning training objectives are absent.

### Phase VIII — Coding agent (68–76) — **NOT IMPLEMENTED**

### Phase IX — Inference & serving (77–80)
| Req | State | Note |
|---|---|---|
| 77 inference loads real weights | **DONE (tested)** | `Generator` runs `RootismForCausalLM`; checkpoint loader restores exact outputs |
| 78 efficient generation | **DONE (tested)** | KV cache, sampling, temperature/top-k/top-p, repetition penalty, stop tokens/strings/sequences, streaming, cancellation, max length |
| 79 quantisation | **NOT IMPLEMENTED** | no 8-bit/4-bit path; unquantised checkpoint preserved |
| 80 production serving | **NOT IMPLEMENTED** | no API, concurrency control, auth, health, logging or monitoring |

### Phase X — Web platform (81–90) — **NOT IMPLEMENTED**

### Phase XI — Evaluation (91–96) — **NOT IMPLEMENTED**
The 76 tests are **software regression tests**, not model-quality benchmarks. No
benchmark result is reported.

### Phase XII — Production (97–99)
| Req | State | Note |
|---|---|---|
| 97 modular organisation | **DONE** | see `ARCHITECTURE.md` §2 |
| 98 comprehensive automated tests | **PARTIAL** | 76 model/tokenizer/checkpoint/accounting/integration tests pass; API/frontend/agent/security tests absent |
| 99 documentation | **PARTIAL** | Stage 1 docs complete; training/data/agent/serving/deployment docs pending their stages |

---

## 3. Final production acceptance (requirement 100) — **NOT MET**

| Criterion | State |
|---|---|
| Independently implemented architecture | ✅ tested |
| ~3B verified trainable parameters | ✅ 3,120,737,280, meta-device verified |
| No pretrained weights / fine-tuning | ✅ tested |
| Correct tokenizer integration | ✅ tested end-to-end |
| Working forward + backward | ✅ tested |
| Working checkpoint serialisation + restoration | ✅ tested (model state) |
| >600B configured actual tokens (650B initial) | ⚠️ configured + enforced, not yet consumable by a trainer |
| Commercially usable documented datasets | ❌ none registered |
| Verified licensing / provenance | ❌ not started |
| Scalable preprocessing | ❌ not implemented |
| Distributed training support | ❌ not implemented |
| Memory optimisation | ⚠️ chunked loss/masks + SDPA; AMP/FSDP/checkpointing absent |
| Fault recovery (full resume) | ⚠️ model state only |
| Accurate token accounting | ⚠️ ledger tested, trainer integration absent |
| Coding + NL + reasoning training objectives | ❌ not implemented |
| Generalisation / coding evaluations | ❌ not implemented |
| Coding agent (multi-file, tools, terminal, tests, permissions) | ❌ not implemented |
| HTML frontend + backend integration | ❌ not implemented |
| Streaming inference via API | ⚠️ generator streams in-process; no API |
| Conversation / workspace / agent activity UI | ❌ not implemented |
| Reasoning presentation controls | ❌ not implemented |
| Automated testing (full scope) | ⚠️ partial |
| Security review | ❌ not done |
| Reproducible builds | ⚠️ `pyproject.toml` + pinned schema; no lockfile yet |
| Monitoring / error handling / deployment docs | ❌ not implemented |

**Conclusion:** Rootism is **not production-ready and not trained**. Stage 1 is a
validated foundation built exactly so the user *can* perform the real training
run once Stages 2–3 land.

---

## 4. Explicitly unverified claims

* Full 32,768-token **prefill** attention on GPU (CPU sandbox cannot allocate it).
* 65,536-token context behaviour after YaRN extension (only numerical finiteness
  of RoPE at position 65,000 is tested).
* FlashAttention varlen path (optional dependency absent in this environment).
* Any throughput, GPU-utilisation, tokens/second or training-duration figure —
  `docs/HARDWARE.md` contains **analytical estimates only**, never measurements.
* Any model-quality, benchmark or generalisation result.
* Commercial legal clearance for any dataset or for the trained model.
