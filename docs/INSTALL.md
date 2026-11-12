# Installation and Environment Setup

## 1. Requirements

* Python ≥ 3.10 (Stage 1 developed and tested on Python 3.11.2)
* PyTorch ≥ 2.4 (Stage 1 tested on **2.15.0+cu132, CPU build**)
* NumPy ≥ 1.24 (tested on 2.4.6)
* pytest ≥ 8 for the test suite (tested on 9.2.0)

Full training additionally requires CUDA GPUs with BF16 (compute capability
≥ 8.0), NCCL, and storage/network capacity described in `HARDWARE.md`. The
sandbox used for Stage 1 had **no GPU**; GPU paths are marked unverified.

## 2. Create the environment

```bash
git clone <your-rootism-checkout>
cd rootism-official
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements-dev.txt
export PYTHONPATH=src
```

`requirements.txt` installs `torch` and `numpy` from PyPI. On a GPU machine,
install the build matching your CUDA runtime (the PyTorch installation selector
at pytorch.org gives the exact index URL); do not mix CUDA builds with an
incompatible driver. The PyPI `torch` wheel pulls NVIDIA runtime wheels — see
`LICENSING.md` before any commercial redistribution.

Optional extras in `pyproject.toml`:

```bash
.venv/bin/pip install -e ".[dev,serving,flash]"
```

`flash` (`flash-attn`) is optional; Rootism falls back to chunked SDPA masks when
it is absent. `serving` (FastAPI/uvicorn/pydantic) is for Stage 5 and is **not yet
used**.

## 3. Verify the installation

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
# 76 passed

PYTHONPATH=src .venv/bin/python -m rootism.cli.params --verify-meta
# ... TOTAL 3,120,737,280 (3.1207 B)  meta-device verification: MATCH

PYTHONPATH=src .venv/bin/python -m rootism.cli.doctor
PYTHONPATH=src .venv/bin/python scripts/rootism_smoke.py
```

`--verify-meta` instantiates the full 3B model on PyTorch's `meta` device, so it
needs only a few MB of RAM. **Do not** instantiate the production model in fp32
on CPU — that would require ~12 GiB before optimiser state.

## 4. Reproducing the Stage 1 analyses

```bash
PYTHONPATH=src .venv/bin/python scripts/rootism_param_search.py
PYTHONPATH=src .venv/bin/python scripts/rootism_vocab_sweep.py --documents 12000
PYTHONPATH=src .venv/bin/python -m rootism.cli.tokenizer pilot \
    --output artifacts/pilot_tokenizer.json
PYTHONPATH=src .venv/bin/python -m rootism.cli.tokenizer evaluate \
    --tokenizer artifacts/pilot_tokenizer.json --pilot
```

`artifacts/` is gitignored. The pilot tokenizer and sweep output are regenerable
test artifacts, not production assets.

## 5. Console scripts

When installed with `pip install -e .`, these are available:

| Command | Purpose |
|---|---|
| `rootism-params` | exact parameter breakdown + meta-device verification |
| `rootism-tokenizer` | train / evaluate / info / verify / pilot |
| `rootism-doctor` | hardware detection + resource estimates |

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: rootism` | `export PYTHONPATH=src` or `pip install -e .` |
| `externally-managed-environment` on pip | Use the venv; do not use `--break-system-packages` |
| `enable_gqa` TypeError | PyTorch < 2.5; `attention._sdpa` falls back to `repeat_kv` automatically |
| `KV cache overflow: need N slots` | Increase `max_seq_len` or shorten the prompt; generation bounds it by `max_position_embeddings` |
| Slow BPE training on a large corpus | Lower `max_unique_pretokens`, or use Stage 2's sharded counting when available |
| `checkpoint SHA-256 mismatch` | A torn/corrupted checkpoint; use `find_recovery_point()` to select the last valid one |

## 7. What this install does NOT give you

There is no trained model, no dataset, no API server, no coding agent and no web
UI yet. See `STATUS.md`. Running `rootism_smoke.py` trains a 106,944-parameter
random model for six steps; that is a pipeline test, not a trained Rootism.
