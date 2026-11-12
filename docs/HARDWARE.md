# Hardware, Storage and Training-Time Estimates

> **All figures in this document are analytical estimates derived from explicit
> assumptions. They are NOT measurements.** No GPU was available in the Stage 1
> sandbox (CPU-only, 3.85 GiB RAM, torch 2.15.0+cu132, CUDA unavailable). The
> Stage 3 profiler replaces these numbers with measurements.

Regenerate with:

```bash
PYTHONPATH=src .venv/bin/python -m rootism.cli.doctor --json
PYTHONPATH=src .venv/bin/python -m rootism.cli.params --verify-meta --json
```

## 1. Assumptions (`src/rootism/hardware.py::ASSUMPTIONS`)

| Assumption | Value |
|---|---|
| BF16 parameter bytes | 2 |
| Gradient bytes | 2 |
| AdamW fp32 state (master + m + v) | 12 bytes/param |
| Stored source bytes/token | 3.5 (pilot measurement on UTF-8 code) |
| Tokenised uint16 bytes/token | 2 |
| Training FLOPs | `6 × N_body × D` |
| Reference GPU BF16 dense peak | 989 TFLOP/s (1× H100 SXM) |
| Reference MFU | 40% |
| Reference GPU VRAM | 80 GiB |
| Default shards | 8 |
| Default microbatch | 32,768 tokens |

The activation model is documented in `activation_bytes_per_token()`: without
checkpointing it retains per layer `4d + 2h + 2kv + 3ff + d` elements; with
checkpointing it retains `3d` (layer input + recompute staging). Flash attention
is assumed, so no `T²` score matrix is stored.

## 2. Model memory (production preset, 3,120,737,280 params)

| Item | Size |
|---|---:|
| BF16 weights | 5.81 GiB |
| Gradients | 5.81 GiB |
| AdamW fp32 states | 34.88 GiB |
| **Static total** | **46.50 GiB** |
| Static per shard (8-way FSDP) | 5.81 GiB |
| Activations per shard (32,768-token microbatch, checkpointing) | 1.97 GiB |
| **Estimated peak per shard** | **7.78 GiB** |
| BF16 model-only checkpoint | 5.81 GiB |
| Full resumable checkpoint (weights+grads+optimiser) | 46.50 GiB |

Without activation checkpointing the activation term would be roughly 20×
larger and would not fit 80 GiB GPUs at this microbatch. **Activation
checkpointing is mandatory**, not optional (Stage 3).

## 3. Inference memory

112 KiB per token (bf16 KV cache), i.e.:

| Context | Per sequence |
|---:|---:|
| 4,096 | 0.44 GiB |
| 8,192 | 0.88 GiB |
| 32,768 | 3.50 GiB |
| 65,536 | 7.00 GiB |

Plus 5.81 GiB of BF16 weights. Concurrency is therefore KV-bound at long
context: 8 × 32K sequences ≈ 28 GiB of KV cache. KV-cache quantisation (Stage 5)
is unverified.

## 4. Training compute and duration

`6 × 2,818,572,288 body params × 650,000,000,000 tokens ≈ 1.099 × 10²² FLOPs`.

| GPUs (H100-class) | Effective cluster throughput | Estimated days | Tokens/s (cluster) |
|---:|---:|---:|---:|
| 8 | 3.16 PFLOP/s | **40.2** | 187,140 |
| 16 | 6.33 PFLOP/s | 20.1 | 374,279 |
| 32 | 12.66 PFLOP/s | 10.1 | 748,559 |
| 64 | 25.32 PFLOP/s | 5.0 | 1,497,117 |
| 128 | 50.65 PFLOP/s | 2.5 | 2,994,235 |
| 256 | 101.3 PFLOP/s | 1.3 | 5,988,469 |

These are **idealised**. Real wall-clock should be budgeted at 1.5–3× these
figures once communication overhead, data-loader stalls, checkpoint I/O,
preemption/restart, loss-spike retries and lower-than-assumed MFU are included.
`estimate_training()` takes `--gpu-count`, `tflops_per_gpu` and `mfu` so the user
can substitute their own hardware.

## 5. Dataset storage (650B presented tokens)

| Item | Size |
|---|---:|
| Raw UTF-8 source (3.5 B/token) | 2.275 TB |
| Tokenised uint16 IDs (2 B/token) | 1.300 TB |
| Full resumable checkpoint | 46.5 GiB each |

The budget requires at least **162.5B unique deduplicated tokens**
(`min_unique_fraction = 0.25`, `max_epochs = 4`). That implies roughly **569 GB
raw / 325 GB tokenised** of deduplicated source material before repetition.
Shard manifests, dedup indexes and QC reports add overhead that Stage 2 will
measure.

## 6. Pre-flight checks before committing to training

1. Run `python -m rootism.cli.doctor` on the target node — it detects GPUs, VRAM,
   BF16 support and emits fit warnings.
2. Confirm `peak_per_shard_gb` < usable VRAM with ≥20% headroom.
3. If BF16 is unsupported (compute capability < 8.0), use the FP16 fallback path
   that Stage 3 must provide; do not silently assume BF16.
4. Verify storage throughput can sustain the estimated tokens/s (data-loader
   profiling is Stage 3).
5. Verify the deduplicated corpus really contains ≥162.5B tokens via
   `preflight_budget()`; otherwise the 650B budget would be inflated by repeats.
6. Re-run the vocabulary sweep on the licensed production corpus before freezing
   the tokenizer.

## 7. Known limitations of these estimates

* `6ND` ignores embedding/head FLOPs and attention's `O(T²)` term, so 32K-context
  training costs more than the estimate.
* MFU 40% is an assumption, not a measured value for this architecture.
* Activation formulas are analytical, not profiled.
* No communication-overhead model for FSDP / tensor parallelism.
* No cost model for preemption/restart or checkpoint frequency.
