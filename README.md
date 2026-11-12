# Rootism

**Rootism** is an independently implemented, ~3-billion-parameter AI coding model
and the production platform around it: tokenizer, training system, inference
engine, coding agent, API, web UI and evaluation framework.

> **Current state: Stage 1 of 8 (foundation + model + tokenizer) is complete and tested.**
> The model has **randomly initialised weights** and has **not been trained**. The
> data pipeline, distributed trainer, serving backend, coding agent, web platform
> and evaluation harness are tracked as *not implemented* in
> [`docs/STATUS.md`](docs/STATUS.md). **Do not begin the 650B-token training run
> until Stages 2–3 are complete.** Nothing in this repository substitutes a
> pretrained model, simulates inference, or reports invented results.

This repository is being built in deliberate stages per the Master Engineering
Specification. Each stage ships source code, integration, executed tests,
validation results and documentation. See [`docs/ROADMAP.md`](docs/ROADMAP.md) and
[`docs/STAGE_01_REPORT.md`](docs/STAGE_01_REPORT.md).

## What exists today (Stage 1)

| Subsystem | Status | Where |
|---|---|---|
| Independently implemented decoder-only model (RMSNorm, RoPE, GQA, SwiGLU, depth-scaled init) | implemented + tested | `src/rootism/model/` |
| Exact parameter accounting on the `meta` device — **3,120,737,280** trainable params | verified | `src/rootism/model/parameters.py` |
| Forward + backward, bf16 stability, packed/padded attention, chunked loss | implemented + tested | `src/rootism/model/` |
| KV cache, streaming generation, sampling, stop conditions | implemented + tested | `src/rootism/model/kv_cache.py`, `generation.py` |
| Independent byte-level BPE tokenizer (own trainer, pre-tokenizer, 49 special tokens) | implemented + tested | `src/rootism/tokenizer/` |
| Atomic, SHA-256-verified checkpoints bound to config + tokenizer identity | implemented + tested | `src/rootism/io/` |
| 650B actual-token accounting with a hard >600B floor | implemented + tested | `src/rootism/accounting.py` |
| Hardware/resource estimates from explicit assumptions | implemented | `src/rootism/hardware.py` |
| 76 automated tests, all executed and passing | verified | `tests/` |

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
export PYTHONPATH=src

.venv/bin/python -m pytest -q                          # 76 tests
.venv/bin/python -m rootism.cli.params --verify-meta   # exact 3B parameter proof
.venv/bin/python -m rootism.cli.doctor                 # hardware + estimates
.venv/bin/python scripts/rootism_smoke.py              # real training step + generation
.venv/bin/python scripts/rootism_vocab_sweep.py        # tokenizer vocab analysis
.venv/bin/python -m rootism.cli.tokenizer pilot --output artifacts/pilot_tokenizer.json
```

Full setup, hardware guidance and architectural decisions:
[`docs/INSTALL.md`](docs/INSTALL.md), [`docs/HARDWARE.md`](docs/HARDWARE.md),
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/MODEL.md`](docs/MODEL.md),
[`docs/TOKENIZER.md`](docs/TOKENIZER.md).

## Non-negotiables

* No pretrained model, checkpoint, tokenizer or distillation is used anywhere.
* Every trainable parameter is created from `ModelConfig` and random initialisation.
* The configured training budget is **650,000,000,000 actual non-padding tokens**,
  enforced above a hard floor of **600,000,000,000**. Padding, masked tokens and
  failed steps never count.
* Components that are not implemented are labelled as such; untested claims are
  labelled *unverified*.

## Licence

Proprietary / all rights reserved — see [`LICENSE`](LICENSE). This is **not** a
claim of legal clearance for commercial model deployment. Dependency licences and
the (still empty) dataset provenance registry are tracked in
[`docs/LICENSING.md`](docs/LICENSING.md).
