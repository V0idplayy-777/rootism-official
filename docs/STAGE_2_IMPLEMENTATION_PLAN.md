# Rootism — Stage 2 Implementation Plan (Steps 51–100)

**Status:** Active Execution (Stage 2 of 2)  
**Scope:** Master Engineering Specification Phase VI Part 2 (Steps 51–60) and Phases VII–XII (Steps 61–100), including the root-directory HTML developer platform (`/index.html`, `/styles.css`, `/app.js`).

---

## 1. Subsystem Implementation Sequence

1. **Phase VI Part 2 — Complete Training Engine (`rootism/training/`):**
   - `scheduler.py` (Step 51): Smooth token/step-indexed Warmup + Cosine & Warmup-Stable-Decay (WSD) LR schedules.
   - `batching.py` (Step 52): Distributed global-batch and microbatch reconciliation.
   - `packing.py` (Step 53): Document-isolated sequence packing (`position_ids` reset + `document_ids` causal mask + cross-document `-100` label boundary guard).
   - `stability.py` (Step 54): Loss-spike z-score detector, NaN/Inf guard, gradient explosion/vanishing tracker, layer activation distribution hooks, and optimizer state monitor.
   - `checkpointing.py` (Steps 55–57): Atomic temp-dir + SHA-256 manifest + `fsync` + rename checkpointing, corrupted-checkpoint fallback recovery, and preemption signal handling.
   - `profiler.py` (Step 58): Throughput, MFU, memory, data-stall, communication, and checkpoint I/O bottleneck profiler.
   - `trainer.py` (Steps 59–60): Full hardware-adaptive resumable pretraining loop enforcing the configured `>= 600B` (`650B` default) net processed token budget.

2. **Phase VII — Reasoning & Intelligence (`rootism/reasoning/`):**
   - `tasks.py` (Steps 61–63): Progressively difficult solver-verified tasks across deduction, induction, abstraction, CSP, causal debugging, math, and counterexample analysis.
   - `supervision.py` (Steps 64–65): Plan-Implement-Verify, Contrastive Flaw-Localization, and Execution-State supervision formats + non-verbosity-biased `ReasoningQualityEvaluator`.
   - `modes.py` (Steps 66–67): User-selectable reasoning modes (`concise`, `explain`, `structured`, `approach_summary`, `assumptions_and_intermediates`) with strict separation between internal scratchpads and faithful user-visible summaries.

3. **Phase VIII — Integrated Coding Agent (`rootism/agent/`):**
   - `safety.py` (Step 76): Prompt-injection scanner, path-jail verifier, command risk classifier, and explicit confirmation gate protocol.
   - `workspace.py` (Steps 70, 71, 75): Project tree indexer, cross-file import graph analyzer, and atomic rollback-capable file CRUD.
   - `sandbox.py` (Step 72): Isolated process-group command runner with `rlimit`, environment scrubbing, timeouts, and structured execution logs.
   - `tools.py` (Step 73): Pydantic JSON-schema validated tool definitions and `<|tool_call_start|>` protocol parser/serializer.
   - `testing.py` (Step 74): Test runner & structured parser distinguishing `passed`, `failed`, `skipped`, and `could_not_run` with root-cause file/line extraction.
   - `orchestrator.py` (Steps 68–69): Multi-step agent state machine connecting `RootismInferenceEngine`, workspace tools, test execution, and iterative repair.

4. **Phase IX — Inference Engine, Quantisation & Production Serving (`rootism/inference/`, `rootism/server/`):**
   - `engine.py` (Steps 77–78): Actual Rootism checkpoint loader + KV-cached streaming generator (`temperature`, `top_p`, `top_k`, `repetition_penalty`, `stop_sequences`, cancellation).
   - `quantization.py` (Step 79): Non-destructive `INT8` per-channel and `INT4` group-wise weight quantisation + perplexity/memory trade-off evaluator.
   - `schemas.py` & `app.py` (Steps 80, 89): FastAPI production backend serving `/api/v1/*` and root `/` platform assets with SSE streaming, concurrency control, optional API-key auth, conversation persistence, workspace APIs, and live agent endpoints.

5. **Phase X — Root-Directory HTML Platform (`/index.html`, `/styles.css`, `/app.js`, Steps 81–90):**
   - Responsive, accessible (WCAG AA+, semantic HTML, keyboard navigable) dark/light developer studio located directly in the repository root and served over relative `/api/v1/*` routes.

6. **Phases XI & XII — Evaluation, Regression, Documentation & Acceptance (`rootism/evaluation/`, `docs/`, `tests/`, Steps 91–100):**
   - Executable coding, reasoning, perturbation-generalisation, and checkpoint-regression benchmark framework with Wilson 95% confidence intervals.
   - Complete automated test suites + live preview verification.
