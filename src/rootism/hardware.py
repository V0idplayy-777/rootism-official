"""Analytical hardware/resource estimates (requirement E).

Every value here is derived from explicit assumptions in `ASSUMPTIONS`; these
are ESTIMATES, not measured throughput. The Stage 4 profiler replaces them with
measurements on the user's actual hardware.
"""

from __future__ import annotations

import math
import os
import platform
from typing import Any

import torch

from .config import ModelConfig
from .model.parameters import analytic_breakdown, kv_cache_bytes_per_token

ASSUMPTIONS: dict[str, Any] = {
    "bf16_bytes_per_param": 2,
    "grad_bytes_per_param": 2,
    "adamw_fp32_state_bytes_per_param": 12,  # fp32 master weight + first + second moment
    "source_bytes_per_token": 3.5,  # measured pilot: UTF-8 code is ~3.5 B/token
    "tokenized_bytes_per_token": 2,  # uint16 ids for a 49K vocabulary
    "flops_per_param_per_token": 6,  # 6ND approximation (fwd + bwd)
    "reference_gpu_tflops_bf16": 989.0,  # one H100 SXM BF16 dense peak
    "reference_mfu": 0.40,  # assumed achievable model flops utilisation
    "reference_gpu_vram_gb": 80.0,
}


def detect_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [],
        "gpu_vram_gb": [],
        "bf16_supported": False,
        "fp16_supported": torch.cuda.is_available(),
        "total_ram_gb": None,
        "bf16_cpu_emulation": not torch.cuda.is_available(),
    }
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        info["total_ram_gb"] = round(pages * page_size / (1024**3), 2)
    except (ValueError, OSError, AttributeError):
        pass
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpu_names"].append(props.name)
            info["gpu_vram_gb"].append(round(props.total_memory / (1024**3), 2))
        major, _ = torch.cuda.get_device_capability(0)
        info["bf16_supported"] = major >= 8
    return info


def activation_bytes_per_token(config: ModelConfig, *, checkpointing: bool) -> int:
    """Analytical activation estimate per token, bf16, flash attention.

    Without checkpointing we retain, per layer: residual input, two normalised
    inputs, q/k/v, attention output, gate/up/activated gate, mlp output.
    With checkpointing only the layer input (and a small recompute buffer) is
    retained; the rest is recomputed in the backward pass.
    """
    d = config.hidden_size
    h = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    ff = config.intermediate_size
    if checkpointing:
        elements_per_layer = 2 * d + d  # layer input + recompute staging
    else:
        elements_per_layer = 4 * d + h + 2 * kv + h + 3 * ff + d
    return config.num_hidden_layers * elements_per_layer * ASSUMPTIONS["bf16_bytes_per_param"]


def estimate_memory(
    config: ModelConfig,
    *,
    microbatch_tokens: int = 32_768,
    seq_len: int = 32_768,
    shards: int = 8,
    activation_checkpointing: bool = True,
) -> dict[str, float]:
    counts = analytic_breakdown(config)
    n = counts["total"]
    a = ASSUMPTIONS
    static = n * (a["bf16_bytes_per_param"] + a["grad_bytes_per_param"] + a["adamw_fp32_state_bytes_per_param"])
    activations = activation_bytes_per_token(config, checkpointing=activation_checkpointing) * microbatch_tokens
    kv_inference = kv_cache_bytes_per_token(config, torch.bfloat16) * seq_len
    return {
        "trainable_parameters": n,
        "bf16_weights_gb": n * a["bf16_bytes_per_param"] / 1024**3,
        "gradients_gb": n * a["grad_bytes_per_param"] / 1024**3,
        "adamw_states_gb": n * a["adamw_fp32_state_bytes_per_param"] / 1024**3,
        "static_total_gb": static / 1024**3,
        "static_per_shard_gb": static / shards / 1024**3,
        "activations_per_shard_gb": activations / shards / 1024**3,
        "peak_per_shard_gb": (static + activations) / shards / 1024**3,
        "kv_cache_gb_at_seq_len": kv_inference / 1024**3,
        "kv_cache_kib_per_token": kv_inference / seq_len / 1024,
        "full_checkpoint_gb": static / 1024**3,
        "bf16_model_only_checkpoint_gb": n * a["bf16_bytes_per_param"] / 1024**3,
    }


def estimate_training(
    config: ModelConfig,
    *,
    target_tokens: int = 650_000_000_000,
    gpu_count: int = 8,
    tflops_per_gpu: float = ASSUMPTIONS["reference_gpu_tflops_bf16"],
    mfu: float = ASSUMPTIONS["reference_mfu"],
) -> dict[str, float]:
    counts = analytic_breakdown(config)
    # Embedding/output matrices do not scale like dense layers in 6ND; use the
    # transformer-body parameter count for the estimate (documented approximation).
    n_body = counts["attention"] + counts["feed_forward"]
    flops = ASSUMPTIONS["flops_per_param_per_token"] * n_body * target_tokens
    effective_flops_per_second = gpu_count * tflops_per_gpu * 1e12 * mfu
    seconds = flops / effective_flops_per_second
    return {
        "body_parameters": n_body,
        "total_training_flops": flops,
        "effective_flops_per_second": effective_flops_per_second,
        "seconds": seconds,
        "hours": seconds / 3600,
        "days": seconds / 86400,
        "tokens_per_second_cluster": target_tokens / seconds,
        "gpu_count": gpu_count,
        "assumed_tflops_per_gpu": tflops_per_gpu,
        "assumed_mfu": mfu,
    }


def estimate_storage(*, target_tokens: int = 650_000_000_000) -> dict[str, float]:
    a = ASSUMPTIONS
    return {
        "raw_source_tb": target_tokens * a["source_bytes_per_token"] / 1e12,
        "tokenized_uint16_tb": target_tokens * a["tokenized_bytes_per_token"] / 1e12,
        "assumption_source_bytes_per_token": a["source_bytes_per_token"],
        "assumption_tokenized_bytes_per_token": a["tokenized_bytes_per_token"],
    }


def validate_hardware_for_training(config: ModelConfig, *, shards: int, microbatch_tokens: int) -> list[str]:
    """Return warnings for configurations that cannot fit the assumed devices."""
    hw = detect_hardware()
    mem = estimate_memory(config, microbatch_tokens=microbatch_tokens, shards=shards)
    warnings: list[str] = []
    if not hw["cuda_available"]:
        warnings.append("No CUDA device detected: full training requires GPUs; CPU is for tests only.")
        return warnings
    if not hw["bf16_supported"]:
        warnings.append("GPU compute capability < 8.0: BF16 unsupported, use the FP16 fallback path.")
    vram = min(hw["gpu_vram_gb"]) if hw["gpu_vram_gb"] else ASSUMPTIONS["reference_gpu_vram_gb"]
    if mem["peak_per_shard_gb"] > vram:
        warnings.append(
            f"Estimated peak {mem['peak_per_shard_gb']:.1f} GiB/shard exceeds "
            f"detected {vram:.1f} GiB/GPU; reduce microbatch tokens or increase shards."
        )
    if math.isnan(mem["peak_per_shard_gb"]):  # pragma: no cover
        warnings.append("memory estimate is not finite")
    return warnings
