# Reproducibility (directive G)

## Identities recorded for every run

| Item | Where | How |
|---|---|---|
| Model configuration | checkpoint manifest `config`, `config_id` | sha256 of canonical `ModelConfig.to_dict()` |
| Architecture | `architecture_id` | `rootism-decoder-v1` |
| Checkpoint format | `format_id`, `schema_version` | `rootism-ckpt-v1`, schema `1` |
| Tokenizer version | `tokenizer_id`, `fingerprint`, `corpus_sha256` | sha256 over vocab + merges + specials + config + corpus digest |
| Dataset manifest | `manifest_id` (Stage 2) | sha256 of canonical registry/manifest |
| Training config | Stage 3 | canonical JSON in checkpoint manifest |
| Software versions | checkpoint manifest `torch_version` | captured at save time |
| Random seed | Stage 3 | caller-supplied; `set_seed()` seeds Python/NumPy/Torch |
| Step / token counter | `TokenLedger`, manifest `step` | saved with every checkpoint |

## Determinism in Stage 1

* `initialize_weights(seed=...)` uses `torch.random.fork_rng()` — reproducible
  without clobbering the caller's RNG.
* BPE training is deterministic: pre-token counts are exact, pair tie-breaks use
  `(-count, token_string_a, token_string_b)`, and word iteration is sorted.
  Two runs on the same corpus produce identical merges and `tokenizer_id`
  (tested).
* `CorpusDigest` hashes length-prefixed document bytes in order, so corpus
  identity is order-sensitive and collision-resistant.
* `SamplingConfig(seed=...)` seeds a dedicated `torch.Generator` for multinomial
  sampling (tested).
* All JSON is canonical (sorted keys, no incidental whitespace).

## Deterministic execution

`set_seed(seed, deterministic=True)` calls `torch.use_deterministic_algorithms(True,
warn_only=True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`. This can
reduce throughput and is not available for every kernel.

## Known sources of nondeterminism

* Atomic floating-point reductions and some fused CUDA kernels (GPU-dependent).
* Nondeterministic algorithm selection (`warn_only=True` reports, does not abort).
* Distributed reduction order across workers (Stage 3 must record world size,
  rank layout and sharding strategy).
* Data-loader ordering, async prefetch and worker RNG seeding (Stage 2/3).
* Hardware/driver/compiler differences between machines; record torch version,
  CUDA version and GPU model in the run manifest.

## Verifying a checkpoint

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from rootism.io import verify_checkpoint, find_recovery_point, load_checkpoint
print(verify_checkpoint("path/to/ckpt.pt"))
print(find_recovery_point("path/to/checkpoints"))
state, manifest = load_checkpoint("path/to/ckpt.pt")
print(manifest["config_id"], manifest["tokenizer_id"], manifest["torch_version"])
PY
```

Loading with a mismatched config or tokenizer raises instead of silently
reinterpreting tensors.
