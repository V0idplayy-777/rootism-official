# Training Rootism & Loading Your Weights Into The Platform

This is the practical, end-to-end path from a fresh clone to a Rootism model that answers
prompts in the studio UI. Everything here uses commands that exist in `rootism/cli.py`.

> **Right now the platform runs in honest standby mode.** A step-0 checkpoint is initialized
> automatically, `processed_training_tokens` is `0`, and `/api/v1/health` reports
> `"trained_weights": false`. In standby: the workspace, editor, diffs, undo, symbol index,
> chat history export and agent inspection/sandbox tools all work; chat and FIM completion
> return a clean explanatory message instead of random-weight gibberish. **Nothing errors.**

---

## 0. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -v        # optional sanity check
```

---

## 1. Train the tokenizer (once per corpus)

```bash
rootism train-tokenizer \
  --input /path/to/clean_corpus_dir \
  --output artifacts/prod/tokenizer/tokenizer.json \
  --vocab-size 65536
```

* `--input` accepts a directory or a single text file.
* The tokenizer records a SHA-256 `fingerprint`; the model later locks to it, so **never
  change the tokenizer after training starts** (checkpoint loading rejects mismatches).

---

## 2. Audit dataset licensing, then shard the corpus

```bash
rootism verify-licenses --registry configs/datasets/provenance_registry.yaml
```

Only commercially approved licences pass the audit. Prepare `.rshard` / `.ridx` binary
shards using `rootism.data.sharding.build_shards_parallel` / `RootismShardWriter`
(see `docs/OPERATOR_AND_DEPLOYMENT_GUIDE.md` §3.3). Inspect the result:

```bash
rootism inspect-data \
  --manifest data_shards/shard_manifest.json \
  --tokenizer artifacts/prod/tokenizer/tokenizer.json
```

---

## 3. Train the model

### 3a. Quick local smoke run (CPU / single GPU, proves the pipeline)

```bash
rootism train-verify \
  --config configs/model/rootism_tiny_verify.yaml \
  --steps 4 \
  --checkpoint-dir artifacts/verify_train_ckpts
```

This writes real `step_XXXXXXXX/` checkpoints. Because they report
`processed_training_tokens > 0`, serving them flips the studio out of standby immediately.

### 3b. Production 3B run

```bash
# single node, 8 GPUs
torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --rdzv_id=rootism-3b-run1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="localhost:29500" \
  -m rootism.cli train-verify \
  --config configs/training/pretrain_650b.yaml \
  --steps 100000 \
  --checkpoint-dir checkpoints/rootism-3b
```

Notes:

* `configs/training/pretrain_650b.yaml` enforces the `>600B` production token budget
  (`650,000,000,000` target). Use `configs/model/rootism_tiny_verify.yaml` only for
  verification runs — it is explicitly non-production.
* For real streaming-shard training, instantiate
  `RootismPretrainer(RootismConfig.from_yaml(...)).train(RootismStreamingShardDataset(...), auto_resume=True)`.
* Checkpoints are atomic, `fsync`ed, SHA-256 manifest-verified, and resumable after
  preemption (`AtomicCheckpointManager`).

Each checkpoint directory looks like:

```
checkpoints/rootism-3b/step_00001234/
├── model.safetensors          # the weights
├── training_state.pt          # optimizer + scheduler + RNG state
├── config.yaml                # exact architecture + tokenizer fingerprint lock
└── checkpoint_manifest.json   # SHA-256 digests, provenance, processed_training_tokens
```

---

## 4. Serve the trained weights

### Option A — explicit flags (recommended)

```bash
rootism serve \
  --host 0.0.0.0 \
  --port 8000 \
  --checkpoint-dir checkpoints/rootism-3b/step_00001234 \
  --tokenizer-dir artifacts/prod/tokenizer \
  --quantization none        # or int8 / int4
```

### Option B — drop-in default path (no flags)

Place the artifacts where the platform looks by default, then just `rootism serve`:

```
artifacts/local_runtime/
├── tokenizer/tokenizer.json
└── checkpoints/step_00001234/   # the checkpoint directory above
```

Point `--checkpoint-dir` at the specific `step_XXXXXXXX` directory, **not** its parent, so
`checkpoint_manifest.json` is found.

### Option C — remote / hosted server

Run the server anywhere reachable, then in the studio open **Settings → Backend URL**, enter
`https://your-host:8000`, and (if you set `ROOTISM_API_KEY`) paste the key into the
**Bearer API key** field.

---

## 5. How you know it worked

```bash
curl -s http://localhost:8000/api/v1/health | python -m json.tool
```

* `"trained_weights": true`
* `"in_standby_mode": false`
* `"processed_training_tokens": <your real token count>`

`/api/v1/model/info` then shows the real checkpoint stage/step in `honest_training_status`,
chat completions stream real sampled tokens, and FIM completion fills code at the cursor.
No restart of the UI, no code edits — the same page picks it up on the next request.

---

## 6. Evaluations & quantisation (optional)

```bash
rootism evaluate --artifact-dir artifacts/local_runtime --output-dir artifacts/eval_reports
rootism quantize-eval --config configs/model/rootism_tiny_verify.yaml
```

`int8`/`int4` quantisation is non-destructive: master `.safetensors` files are never
overwritten, so you can always fall back to `--quantization none`.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
| :--- | :--- |
| `Missing checkpoint_manifest.json` | `--checkpoint-dir` must be the `step_XXXXXXXX` folder. |
| `Tokenizer version mismatch` | Tokenizer fingerprint in `config.yaml` differs from the `--tokenizer-dir` file. Retrain with the original tokenizer. |
| `Invalid or missing Rootism origin_signature` | The checkpoint was not produced by a Rootism from-scratch run. External/pretrained weights are intentionally rejected. |
| UI says "Standby mode" | The loaded checkpoint has `processed_training_tokens == 0`. Train, then serve the new `step_*` directory. |
| UI says "No backend" | Static preview only — set a **Backend URL** in Settings or run `rootism serve`. Editing still works locally in the browser. |
