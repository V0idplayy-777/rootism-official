# Rootism — Realistic Hardware, Memory, Storage & Duration Estimates

All figures in this specification are computed programmatically via `rootism plan-hardware` (`rootism/training/hardware.py`) using explicit equations and non-inflated dense Tensor Core throughput assumptions (Directive E).

---

## 1. Fundamental Memory & Compute Equations (`P = 3,009,630,720` Parameters)

1. **Model Weights:**
   - `BF16 / FP16` (`2 bytes/param`): $2 \times 3,009,630,720 = 6,019,261,440\text{ bytes} = \mathbf{5.61\text{ GiB}}$ (`6.02 GB`)
   - `FP32` (`4 bytes/param`): $4 \times 3,009,630,720 = 12,038,522,880\text{ bytes} = \mathbf{11.21\text{ GiB}}$ (`12.04 GB`)
2. **Gradients (`BF16`):** $\mathbf{5.61\text{ GiB}}$ globally (sharded across $N_{\text{shard}}$ ranks under FSDP2).
3. **AdamW FP32 Master Weights + Moments (`m`, `v`):**
   - $(4 + 4 + 4) \times P = 12 P = 36,115,568,640\text{ bytes} = \mathbf{33.64\text{ GiB}}$ globally (`4.20 GiB/GPU` across an 8-GPU NVLink node).
4. **Inference KV Cache (`BF16`, batch = 1, 5:1 GQA `4 KV heads × 128 dim × 40 layers`):**
   - Per token: $2 \times 40 \times 4 \times 128 \times 2 = \mathbf{81,920\text{ bytes}} = \mathbf{80\text{ KiB/token}}$
   - At **4,096 tokens:** **0.3125 GiB**
   - At **32,768 tokens (Native):** **2.50 GiB** *(vs 12.50 GiB for standard MHA)*
   - At **65,536 tokens (Extended YaRN):** **5.00 GiB** *(vs 25.00 GiB for standard MHA)*
5. **Pretraining Compute Budget (`T = 650,000,000,000` net processed tokens):**
   - Forward + Backward FLOPs (excluding recomputation):
     $$C = 6 \cdot P \cdot T = 6 \times 3.00963072 \times 10^9 \times 6.5 \times 10^{11} = \mathbf{1.1738 \times 10^{22}\text{ FLOPs}}$$

---

## 2. Storage Requirements

| Storage Category | Size | Calculation & Engineering Basis |
|---|---:|---|
| **Single Full Resumable Checkpoint** | **42.13 GB** | `6.02 GB` BF16 weights + `36.11 GB` FP32 master & AdamW `(m, v)` states + RNG/step/shard-offset metadata |
| **Single Inference Checkpoint (BF16)** | **6.02 GB** | Safetensors model weights + locked `tokenizer.json` |
| **8-bit / 4-bit Quantised Inference Weights** | **~3.1 GB / ~1.7 GB** | INT8 / INT4 weight quantisation (preserving unquantised master checkpoint) |
| **Recommended Checkpoint Retention Pool** | **0.421 TB (~450 GB)** | 5 rolling fault-recovery checkpoints + 5 milestone checkpoints |
| **650B Binary Shards (`.rshard` + `.ridx`)** | **1.326 TB** | `650B × 2 bytes` (`uint16` vocab $\le 65,536$) + ~2% `.ridx` index headers |
| **Recommended Dataset Prep NVMe Workspace** | **2.92 TB (~3.0 TB)** | Raw source downloads + deduplication staging + final `.rshard` files |

---

## 3. Cluster Sizing & Training Duration Matrix (`650B` Tokens)

Computed using dense (non-sparse!) BF16 TFLOPS and realistic effective Model FLOPs Utilisation (MFU) after accounting for activation-checkpointing recomputation (`1.165×` selective, `1.330×` full):

| Cluster Configuration | Sharding Strategy | Activation Ckpt | Seq Len / Microbatch | Est. Peak VRAM / GPU | Cluster Throughput (tokens/s) | Est. Wall-Clock Duration (650B) |
|---|---|---|---|---:|---:|---:|
| **16× RTX 4090 (24 GB)** | `fsdp_zero3` | `full` | `2,048` / `mb=1` | `6.52 GiB / 24 GiB` | `41,821 tok/s` | **179.9 days** *(feasible for verification/annealing; slow for full 650B)* |
| **16× A100 SXM (80 GB)** | `fsdp_hybrid` | `selective` | `4,096` / `mb=4` | `22.92 GiB / 80 GiB` | `118,646 tok/s` | **63.4 days** |
| **64× H100 SXM (80 GB)** *(8 nodes)* | `fsdp_hybrid` | `selective` | `4,096` / `mb=4` | `22.92 GiB / 80 GiB` | `1,414,827 tok/s` | **5.32 days** *(Recommended production sweet spot)* |
| **256× H100 SXM (80 GB)** *(32 nodes)* | `fsdp_hybrid` | `selective` | `4,096` / `mb=4` | `22.92 GiB / 80 GiB` | `5,659,307 tok/s` | **1.33 days (~32 hours)** |

Run custom infrastructure sizing for any hardware configuration via:
```bash
rootism plan-hardware --gpu h100_80gb_sxm --num-gpus 64 --seq-len 4096 --micro-batch 4 --tokens 650000000000
```
