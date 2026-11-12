# Rootism Operator, Training, Deployment & Troubleshooting Guide

This guide provides step-by-step operational procedures for training the custom `RootismTokenizer`, auditing commercial dataset provenance, launching and resuming distributed `Rootism-3B` 650B-token pretraining runs, evaluating checkpoints, quantising weights, and deploying the FastAPI + Root HTML Developer Platform.

---

## 1. Environment Installation & Reproducibility Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Verify installation and run the complete 42-test automated suite:
```bash
pytest -v
```

---

## 2. Architecture & Cluster Sizing Verification

Inspect the exact parameter breakdown (`3,009,630,720` trainable parameters) and compare candidate architectures:
```bash
rootism inspect-model --compare-candidates
```

Generate hardware memory, throughput, NVMe shard storage, and wall-clock duration estimates for your target GPU cluster:
```bash
# Example: 64x H100 SXM 80GB (~5.32 days for 650B tokens)
rootism plan-hardware --gpu h100_80gb_sxm --num-gpus 64 --seq-len 4096 --micro-batch 4 --tokens 650000000000 --ckpt selective
```

---

## 3. Commercial Licensing Audit, Tokenizer Training & Data Preparation

### 3.1 Verify Commercial Dataset Provenance & Export Attribution Notices
```bash
rootism verify-licenses \
  --registry configs/datasets/provenance_registry.yaml \
  --write-notices THIRD_PARTY_DATA_NOTICES.md
```
> Any dataset lacking an approved commercial SPDX expression (`MIT`, `Apache-2.0`, `BSD-2-Clause`, `BSD-3-Clause`, `ISC`, `CC0-1.0`, `CC-BY-4.0`, `Unlicense`, `0BSD`, `Python-2.0`, or `Rootism-Proprietary-Original`) or carrying `Non-Commercial`, `Research-Only`, `GPL`/`AGPL`, or model-ToS-restricted provenance is automatically rejected.

### 3.2 Train & Cryptographically Lock the 65,536-Vocab Byte-Level BPE Tokenizer
```bash
rootism train-tokenizer \
  --input /mnt/nvme/clean_commercial_corpus \
  --output artifacts/production_tokenizer/tokenizer.json \
  --vocab-size 65536
```
Bind `version_lock.fingerprint_sha256` from `tokenizer.json` into `configs/training/pretrain_650b.yaml` (`model.tokenizer_version_fingerprint`).

### 3.3 Generate Subprocess-Verified Original Synthetic Tasks & Inspect Binary `.rshard` Data
```bash
rootism generate-synthetic --count-per-type 100 --seed 20261112 --output artifacts/synthetic/verified_tasks.json
rootism inspect-data \
  --manifest data_shards/shard_manifest.json \
  --tokenizer artifacts/production_tokenizer/tokenizer.json
```

---

## 4. Distributed Pretraining Launch & Fault-Tolerant Resumption

### 4.1 Local End-to-End Pipeline Verification
Before launching a multi-node cluster job, verify the `RootismPretrainer`, `SequencePacker`, `TrainingStabilityMonitor`, and `AtomicCheckpointManager` locally:
```bash
rootism train-verify --steps 4 --checkpoint-dir artifacts/verify_train_ckpts
```

### 4.2 Multi-GPU / Multi-Node 650B Production Launch (`torchrun`)
`RootismPretrainer` automatically initializes PyTorch composable `FSDP2` (`fully_shard` + `DeviceMesh` hybrid sharding across nodes) and enforces the `650,000,000,000` (`>600B`) processed-token budget:
```bash
torchrun \
  --nnodes=8 \
  --nproc_per_node=8 \
  --rdzv_id=rootism-650b-run1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  -m rootism.cli train-verify --config configs/training/pretrain_650b.yaml
```
*(For custom streaming `.rshard` cluster scripts, instantiate `RootismPretrainer(RootismConfig.from_yaml("configs/training/pretrain_650b.yaml"))` and pass `RootismStreamingShardDataset` to `trainer.train(dataset, auto_resume=True)`.)*

### 4.3 Preemption & Corrupted Checkpoint Recovery
- **Graceful Preemption (`SIGTERM` / `SIGINT`)**: `PreemptionSignalCatcher` traps SLURM/Kubernetes preemption signals, finishes the in-flight optimizer step cleanly, commits an atomic `step_XXXXXXXX` checkpoint with `fsync` + SHA-256 manifest, and exits cleanly.
- **Automatic Fallback Recovery**: On restart (`auto_resume=True`), `AtomicCheckpointManager.find_latest_valid_checkpoint()` validates SHA-256 digests of `model.safetensors`, `training_state.pt`, and `config.yaml`. Any incomplete `.tmp` directory or corrupted checkpoint is logged and skipped automatically in favor of the latest intact recovery point.

---

## 5. Non-Destructive Quantisation & Reproducible Evaluation

### 5.1 Evaluate `INT8` and `INT4` Memory vs Quality Trade-Offs
```bash
rootism quantize-eval --config configs/model/rootism_tiny_verify.yaml
```
`quantize_model_non_destructive` never overwrites master `.safetensors` checkpoints. Always inspect `logit_cosine_similarity` and `perplexity_delta` before deploying `int4` in latency/VRAM-constrained environments.

### 5.2 Run Executable Multi-Language Coding, Reasoning & Generalisation Benchmarks
```bash
rootism evaluate \
  --artifact-dir artifacts/local_runtime \
  --output-dir artifacts/eval_reports
```
Outputs `evaluation_report.json` and `evaluation_report.md` with honest training-token disclosures, syntax validity rates, functional `pass@1`, reasoning quality scores, and 13-gram contamination audit status.

---

## 6. Deploying the Serving API & Root HTML Developer Platform

### 6.1 Start Production Server
Bind to `0.0.0.0` so both local and reverse-proxied clients can connect:
```bash
# Optional authentication key:
export ROOTISM_API_KEY="replace-with-strong-commercial-api-token"

rootism serve \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization none
```
When `--checkpoint-dir` and `--tokenizer-dir` are omitted on a fresh developer machine, `bootstrap_local_serving_artifacts` initializes a cryptographically verified local `RootismForCausalLM` checkpoint and `RootismTokenizer` in `artifacts/local_runtime` and **explicitly labels** the UI and `/api/v1/model/info` banner with its exact step and processed training tokens (`0` until trained).

### 6.2 Key Endpoints
| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/` | `GET` | Serves Root-Directory HTML Developer Platform (`index.html`, `styles.css`, `app.js`) |
| `/api/v1/health` | `GET` | Liveness, readiness, device/dtype/quantization, concurrency slots |
| `/api/v1/model/info` | `GET` | Honest training status, verified `Rootism-3B` breakdown, 19 languages, 5 reasoning modes |
| `/api/v1/chat/completions` | `POST` | Synchronous JSON or SSE `text/event-stream` chat & structured reasoning generation |
| `/api/v1/code/complete` | `POST` | Fill-In-The-Middle (`<|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>`) completion |
| `/api/v1/requests/{id}/cancel` | `POST` | Immediate cooperative cancellation of an in-flight generation |
| `/api/v1/workspace/*` | `GET/POST` | Confined workspace file tree, file read/save with unified diff, and 1-click rollback |
| `/api/v1/agent/{tool,run,confirm}` | `POST` | Multi-step `CodingAgentOrchestrator` execution & human safety confirmation gate |

---

## 7. Troubleshooting Guide

1. **`TokenizerCompatibilityError: Tokenizer fingerprint mismatch`**:
   The checkpoint's `model.tokenizer_version_fingerprint` does not match the loaded `tokenizer.json`. Always serve and resume checkpoints with the exact locked tokenizer artifact used during training.
2. **`CheckpointIntegrityError: Checksum mismatch`**:
   A disk or transfer fault corrupted a checkpoint file. Keep `auto_resume=True` enabled so `RootismPretrainer` automatically falls back to the preceding verified `step_XXXXXXXX` directory.
3. **`HTTP 401 Missing or invalid Bearer API key`**:
   `ROOTISM_API_KEY` is set on the server. Supply `Authorization: Bearer <key>` in API clients or enter the key in the **Sampling & Context** drawer of the Developer Platform.
4. **Agent Run Paused in `awaiting_confirmation` State**:
   `SafetyPolicyGuard` intercepted a destructive operation (`rm`, `delete_file`, `git reset --hard`, `DROP TABLE`). Review the reason and confirmation token in the amber UI banner and click **Approve & Execute** or **Reject Safely**.
