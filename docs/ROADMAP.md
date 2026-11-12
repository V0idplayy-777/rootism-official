# Rootism Implementation Roadmap

Per directive A, Rootism is developed in deliberate stages. A stage is only
complete when it has: (1) a plan, (2) source code, (3) integration with existing
components, (4) automated tests, (5) executed validation results, (6) updated
documentation, and (7) a completion report. **Stage 1 is the only complete
stage.** No stage is marked complete in advance.

| Stage | Scope | Requirements | Depends on | State |
|---|---|---|---|---|
| **1** | Foundation, model, tokenizer, checkpointing, accounting, estimates, docs, tests | 01–04, 09–28, 55/57 (model state), 60 (ledger), 97–99 (partial) | — | ✅ **COMPLETE** — `STAGE_01_REPORT.md` |
| **2** | Data pipeline & licensing: registry with licence verification, streaming shards, dedup (exact/near/code/repo), QC filters, contamination gate, mixture config, inspection tooling, dataset manifest versioning, unique/dedup token counting | 29–46 | 1 | ❌ NOT IMPLEMENTED |
| **3** | Training engine: DDP/FSDP + optional TP/PP, BF16/FP16, AdamW + fused, warmup/cosine, global/micro batch accounting, packing loader, activation checkpointing, stability monitors, full resumable checkpoints, fault recovery, profiler, hardware validation | 47–60 | 2 | ❌ NOT IMPLEMENTED |
| **4** | Reasoning & objectives: original reasoning-task generators (deduction, constraint satisfaction, algorithms, diagnosis), progressive curricula, structured-solution formats, faithful user-visible reasoning summaries, coding-integrated tasks | 61–67 | 2, 3 | ❌ NOT IMPLEMENTED |
| **5** | Inference & serving: quantisation (8/4-bit) with quality+memory measurement, request batching/scheduling, FastAPI backend, SSE streaming, auth, validation, health, logging, resource monitoring, concurrency limits | 77–80 | 1, 3 | ❌ NOT IMPLEMENTED |
| **6** | Coding agent: orchestration state machine, workspace model, safe file ops, structured tool calls with JSON-schema validation, isolated terminal with timeouts/resource limits, test runner + result classification, iterative repair, prompt-injection defences, confirmation gates | 68–76 | 2, 5 | ❌ NOT IMPLEMENTED |
| **7** | Web platform: professional dark/light UI, streaming chat, markdown + syntax highlighting, conversation management, regeneration/stop, reasoning presentation control, file explorer/editor/tabs, agent activity timeline, model settings, responsive + accessible | 81–90 | 5, 6 | ❌ NOT IMPLEMENTED |
| **8** | Evaluation, QA & production hardening: coding + reasoning benchmarks, generalisation/memorisation probes, regression gates, structured reports with uncertainty, CI, security review, reproducible builds/lockfile, deployment guide, final acceptance (100) | 91–100 | all | ❌ NOT IMPLEMENTED |

## Stage 2 exit criteria (next)

* Dataset registry populated with licence evidence for every entry; exclusion
  list explicit; no dataset with unverified commercial terms enters a mixture.
* Streaming shard format + manifest with exact source/version/hash and token
  counts for every shard.
* Exact + near + code + repository deduplication implemented with tests.
* QC filters for encoding, empties, spam, malicious/injection-like content, with
  measured false-positive checks.
* Benchmark contamination gate separating eval data from training shards.
* Configurable mixture with documented proportions and token accounting wired to
  `TokenLedger.set_corpus_counts()` / `preflight_budget()`.
* Inspection CLI: samples, distributions, licences, token counts, QC reports.
* Re-run of `scripts/rootism_vocab_sweep.py` on the licensed corpus; freeze the
  production tokenizer and bind its ID.

## Rules that apply to every later stage

* No pretrained weights, no fine-tuning, no distillation, no hardcoded model
  responses, no fake integrations.
* Every new subsystem adds executed tests and updates `STATUS.md`,
  `ARCHITECTURE.md` and the relevant docs.
* Benchmark/throughput/quality numbers are reported only when actually measured.
* Config, checkpoint, tokenizer and dataset-manifest identities remain bound.
* The 650B actual-token floor may not be relaxed.
