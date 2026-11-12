"""
Hardware Capability Discovery & Explicit Infrastructure Planner (Steps 47, 48, 59, Directive E).

Never assumes unlimited or uniform GPU hardware. Detects actual runtime capabilities
and provides explicit, formula-audited memory, storage, throughput, and duration estimates
so the user can plan their 650B-token training cluster accurately before launching.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import asdict, dataclass
from typing import Any, Literal

import psutil
import torch

from rootism.config import DEFAULT_PRODUCTION_TRAINING_TOKENS, RootismModelConfig


@dataclass(frozen=True)
class HardwareCapabilities:
    device_type: Literal["cuda", "cpu"]
    device_count: int
    device_names: tuple[str, ...]
    total_vram_per_gpu_gib: float
    cpu_logical_cores: int
    host_ram_gib: float
    free_disk_gib: float
    cuda_bf16_supported: bool
    cuda_fp16_supported: bool
    cuda_tf32_supported: bool
    nccl_available: bool
    gloo_available: bool
    env_world_size: int
    env_rank: int
    env_local_rank: int

    def recommended_precision(self) -> Literal["bf16", "fp16", "fp32"]:
        if self.device_type == "cuda":
            if self.cuda_bf16_supported:
                return "bf16"
            if self.cuda_fp16_supported:
                return "fp16"
        return "fp32"

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["device_names"] = list(self.device_names)
        out["recommended_precision"] = self.recommended_precision()
        return out


def detect_hardware_capabilities() -> HardwareCapabilities:
    """Inspect active host CPU, RAM, disk, CUDA GPUs, precision support, and distributed backends."""
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(".")
    gib = float(1024**3)

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        cnt = torch.cuda.device_count()
        names = tuple(torch.cuda.get_device_name(i) for i in range(cnt))
        props0 = torch.cuda.get_device_properties(0)
        vram_gib = round(props0.total_memory / gib, 2)
        bf16_ok = bool(torch.cuda.is_bf16_supported())
        fp16_ok = props0.major >= 6
        tf32_ok = props0.major >= 8
        dev_type: Literal["cuda", "cpu"] = "cuda"
    else:
        cnt = 1
        names = (platform.processor() or platform.machine() or "CPU",)
        vram_gib = 0.0
        bf16_ok = False
        fp16_ok = False
        tf32_ok = False
        dev_type = "cpu"

    nccl_ok = bool(torch.distributed.is_available() and torch.distributed.is_nccl_available())
    gloo_ok = bool(torch.distributed.is_available() and torch.distributed.is_gloo_available())

    return HardwareCapabilities(
        device_type=dev_type,
        device_count=cnt,
        device_names=names,
        total_vram_per_gpu_gib=vram_gib,
        cpu_logical_cores=os.cpu_count() or 1,
        host_ram_gib=round(vm.total / gib, 2),
        free_disk_gib=round(disk.free / gib, 2),
        cuda_bf16_supported=bf16_ok,
        cuda_fp16_supported=fp16_ok,
        cuda_tf32_supported=tf32_ok,
        nccl_available=nccl_ok,
        gloo_available=gloo_ok,
        env_world_size=int(os.environ.get("WORLD_SIZE", "1")),
        env_rank=int(os.environ.get("RANK", "0")),
        env_local_rank=int(os.environ.get("LOCAL_RANK", "0")),
    )


GPU_SPEC_PRESETS: dict[str, dict[str, float]] = {
    # Peak dense BF16/FP16 Tensor Core TFLOPS (without 2:4 sparsity inflation!) + VRAM GiB
    "rtx_4090_24gb": {"vram_gib": 24.0, "bf16_tflops": 165.2, "typical_mfu": 0.38},
    "l40s_48gb": {"vram_gib": 48.0, "bf16_tflops": 362.0, "typical_mfu": 0.42},
    "a100_40gb_sxm": {"vram_gib": 40.0, "bf16_tflops": 312.0, "typical_mfu": 0.48},
    "a100_80gb_sxm": {"vram_gib": 80.0, "bf16_tflops": 312.0, "typical_mfu": 0.50},
    "h100_80gb_sxm": {"vram_gib": 80.0, "bf16_tflops": 989.5, "typical_mfu": 0.47},
    "h200_141gb_sxm": {"vram_gib": 141.0, "bf16_tflops": 989.5, "typical_mfu": 0.48},
    "b200_192gb_sxm": {"vram_gib": 192.0, "bf16_tflops": 2250.0, "typical_mfu": 0.46},
}


@dataclass(frozen=True)
class InfrastructureEstimate:
    """Explicit, assumption-transparent hardware, memory, storage & duration estimate."""

    gpu_preset: str
    num_gpus: int
    vram_per_gpu_gib: float
    recommended_strategy: str
    activation_checkpointing: str
    sequence_length: int
    micro_batch_size: int

    # Per-GPU VRAM breakdown (GiB)
    model_weights_per_gpu_gib: float
    gradients_per_gpu_gib: float
    optimizer_states_per_gpu_gib: float
    activations_per_gpu_gib: float
    cuda_allocator_and_comm_buffer_gib: float
    estimated_peak_vram_per_gpu_gib: float
    fits_in_gpu_vram: bool

    # Inference KV-cache reference (GiB, BF16, batch=1)
    kv_cache_32k_gib: float
    kv_cache_64k_gib: float

    # Storage estimates (TB / GB)
    single_full_checkpoint_gb: float
    single_inference_weights_gb: float
    recommended_checkpoint_retention_tb: float
    binary_shards_650b_tb: float
    recommended_dataset_workspace_tb: float

    # Compute & duration for target_tokens
    target_training_tokens: int
    total_training_flops: float
    assumed_mfu: float
    estimated_cluster_tokens_per_sec: float
    estimated_training_days: float
    explicit_assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["explicit_assumptions"] = list(self.explicit_assumptions)
        return d


def estimate_training_infrastructure(
    model_config: RootismModelConfig | None = None,
    *,
    gpu_preset: str = "h100_80gb_sxm",
    num_gpus: int = 64,
    sequence_length: int = 4_096,
    micro_batch_size: int = 4,
    target_training_tokens: int = DEFAULT_PRODUCTION_TRAINING_TOKENS,
    activation_checkpointing: Literal["none", "selective", "full"] = "selective",
) -> InfrastructureEstimate:
    """
    Calculate realistic memory, storage, throughput, and duration figures for Rootism
    from explicit first-principles equations.
    """
    if gpu_preset not in GPU_SPEC_PRESETS:
        raise ValueError(f"Unknown gpu_preset {gpu_preset!r}; choose from {sorted(GPU_SPEC_PRESETS)}")
    if num_gpus < 1:
        raise ValueError("num_gpus must be >= 1")

    cfg = model_config or RootismModelConfig()
    spec = GPU_SPEC_PRESETS[gpu_preset]
    vram_gib = spec["vram_gib"]
    gib = float(1024**3)

    p = float(cfg.analytical_parameter_count())

    # Mixed-precision memory equations:
    # - BF16 weights: 2 * P bytes
    # - BF16 gradients: 2 * P bytes
    # - FP32 master weights + FP32 AdamW m + v moments: (4 + 4 + 4) * P = 12 * P bytes
    full_weights_gib = (2.0 * p) / gib
    full_grads_gib = (2.0 * p) / gib
    full_opt_gib = (12.0 * p) / gib

    # Select sharding strategy appropriate for GPU VRAM and cluster size
    intra_node_shard = min(num_gpus, 8)
    if num_gpus == 1:
        strategy = "single"
        w_div, g_div, o_div = 1.0, 1.0, 1.0
    elif vram_gib >= 80.0 and num_gpus >= 8:
        # Hybrid FSDP (ZeRO-3 within 8-GPU NVLink node, replicated across nodes)
        strategy = "fsdp_hybrid" if num_gpus > 8 else "fsdp_zero2"
        w_div = float(intra_node_shard) if strategy == "fsdp_hybrid" else 1.0
        g_div = float(intra_node_shard)
        o_div = float(intra_node_shard)
    else:
        # Full FSDP ZeRO-3 across ranks (+ full activation checkpointing recommended on <=24GB)
        strategy = "fsdp_zero3"
        effective_shard = float(min(num_gpus, 32))
        w_div, g_div, o_div = effective_shard, effective_shard, effective_shard

    weights_gpu = full_weights_gib / w_div + (0.25 if strategy in ("fsdp_zero3", "fsdp_hybrid") else 0.0)
    grads_gpu = full_grads_gib / g_div
    opt_gpu = full_opt_gib / o_div

    # Activation memory with FlashAttention (no S^2 score matrix) + chunked CE (0.55 GiB logit buffer):
    # Uncheckpointed per-layer BF16 stored activations ~ 16 * B * S * d_model bytes
    # Full checkpointing stores only 2 * B * S * d_model per layer + 1 active recompute layer
    # Selective checkpointing (every 2nd layer) halves stored layer activations
    b_s_d = float(micro_batch_size) * float(sequence_length) * float(cfg.d_model)
    per_layer_unckpt_bytes = 16.0 * b_s_d
    per_layer_ckpt_input_bytes = 2.0 * b_s_d
    if activation_checkpointing == "full":
        act_bytes = cfg.n_layers * per_layer_ckpt_input_bytes + per_layer_unckpt_bytes
    elif activation_checkpointing == "selective":
        half_layers = cfg.n_layers / 2.0
        act_bytes = half_layers * per_layer_unckpt_bytes + half_layers * per_layer_ckpt_input_bytes
    else:
        act_bytes = cfg.n_layers * per_layer_unckpt_bytes

    chunked_ce_buffer_bytes = float(2048 * cfg.vocab_size * 4)  # 2048-token FP32 logit chunk
    act_gpu = (act_bytes + chunked_ce_buffer_bytes) / gib

    comm_and_frag_gib = 2.5  # NCCL all-gather/reduce-scatter buckets + CUDA caching allocator headroom
    peak_vram = weights_gpu + grads_gpu + opt_gpu + act_gpu + comm_and_frag_gib

    kv_per_tok = cfg.kv_cache_bytes_per_token(bytes_per_element=2)
    kv_32k = (kv_per_tok * 32_768) / gib
    kv_64k = (kv_per_tok * 65_536) / gib

    # Storage (decimal GB / TB as reported by cloud block/object stores)
    full_ckpt_gb = (2.0 * p + 12.0 * p) / 1e9  # BF16 weights + FP32 master & AdamW moments
    inf_weights_gb = (2.0 * p) / 1e9
    ckpt_retention_tb = (full_ckpt_gb * 10.0) / 1000.0  # Rolling 5 recovery + 5 milestone checkpoints
    shards_tb = (float(target_training_tokens) * 2.04) / 1e12  # 2B uint16 + ~2% .ridx overhead
    workspace_tb = round(shards_tb * 2.2, 2)  # Raw downloads + dedup staging + final binary shards

    # Compute FLOPs: 6 * N_non_embed_or_total * T (with activation checkpointing factor: +16.5% selective, +33% full)
    recompute_mult = {"none": 1.0, "selective": 1.165, "full": 1.33}[activation_checkpointing]
    base_flops = 6.0 * p * float(target_training_tokens)
    effective_mfu = spec["typical_mfu"] / recompute_mult
    cluster_flops_per_sec = float(num_gpus) * spec["bf16_tflops"] * 1e12 * effective_mfu
    toks_per_sec = cluster_flops_per_sec / (6.0 * p)
    duration_days = (float(target_training_tokens) / toks_per_sec) / 86_400.0

    assumptions = (
        f"Peak dense BF16 GPU throughput = {spec['bf16_tflops']} TFLOPS/GPU (non-sparse).",
        f"Base hardware MFU = {spec['typical_mfu']:.2f}, adjusted by activation checkpointing factor {recompute_mult:.3f} -> net MFU {effective_mfu:.3f}.",
        "FlashAttention/SDPA enabled (O(S) attention memory) and Rootism chunked cross-entropy enabled (2048-token logit chunks).",
        "Checkpoint storage assumes 10 retained full resumable checkpoints (5 rolling recovery + 5 milestone evaluations).",
        "Dataset storage assumes uint16 .rshard binary format (2 bytes/token) plus 1.2x raw/staging overhead.",
    )

    return InfrastructureEstimate(
        gpu_preset=gpu_preset,
        num_gpus=num_gpus,
        vram_per_gpu_gib=vram_gib,
        recommended_strategy=strategy,
        activation_checkpointing=activation_checkpointing,
        sequence_length=sequence_length,
        micro_batch_size=micro_batch_size,
        model_weights_per_gpu_gib=round(weights_gpu, 2),
        gradients_per_gpu_gib=round(grads_gpu, 2),
        optimizer_states_per_gpu_gib=round(opt_gpu, 2),
        activations_per_gpu_gib=round(act_gpu, 2),
        cuda_allocator_and_comm_buffer_gib=comm_and_frag_gib,
        estimated_peak_vram_per_gpu_gib=round(peak_vram, 2),
        fits_in_gpu_vram=peak_vram <= (vram_gib * 0.94),
        kv_cache_32k_gib=round(kv_32k, 2),
        kv_cache_64k_gib=round(kv_64k, 2),
        single_full_checkpoint_gb=round(full_ckpt_gb, 2),
        single_inference_weights_gb=round(inf_weights_gb, 2),
        recommended_checkpoint_retention_tb=round(ckpt_retention_tb, 3),
        binary_shards_650b_tb=round(shards_tb, 3),
        recommended_dataset_workspace_tb=workspace_tb,
        target_training_tokens=target_training_tokens,
        total_training_flops=base_flops,
        assumed_mfu=round(effective_mfu, 4),
        estimated_cluster_tokens_per_sec=round(toks_per_sec, 1),
        estimated_training_days=round(duration_days, 2),
        explicit_assumptions=assumptions,
    )
