"""
Distributed Training Strategy Orchestrator (Step 48).

Uses modern PyTorch 2.x DTensor-native FSDP2 (`fully_shard`) and DDP:
- Single-device execution (`single`)
- Distributed Data Parallel (`ddp`)
- Composable FSDP2 ZeRO-2 (`fsdp_zero2`: `reshard_after_forward=False` per `RootismDecoderLayer`)
- Composable FSDP2 ZeRO-3 (`fsdp_zero3`: `reshard_after_forward=True` per `RootismDecoderLayer`)
- Multi-node Hybrid FSDP2 (`fsdp_hybrid`: 2D `(replicate, shard)` `DeviceMesh`)
- Tensor Parallelism shard plan generation (`build_tensor_parallel_plan`)
- Pipeline Parallelism balanced layer stage partitioning (`partition_layers_for_pipeline`)
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from rootism.config import RootismDistributedConfig, RootismModelConfig
from rootism.model.transformer import RootismDecoderLayer
from rootism.training.hardware import HardwareCapabilities, detect_hardware_capabilities


@dataclass(frozen=True)
class DistributedRuntimeContext:
    initialized: bool
    backend: str
    strategy: str
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def resolve_distributed_strategy(
    dist_cfg: RootismDistributedConfig,
    hw: HardwareCapabilities | None = None,
) -> Literal["single", "ddp", "fsdp_zero2", "fsdp_zero3", "fsdp_hybrid"]:
    caps = hw or detect_hardware_capabilities()
    world_size = max(caps.env_world_size, 1)
    if dist.is_initialized():
        world_size = max(world_size, dist.get_world_size())

    if dist_cfg.strategy != "auto":
        if dist_cfg.strategy != "single" and world_size == 1 and not dist.is_initialized():
            return "single"
        return dist_cfg.strategy

    if world_size <= 1:
        return "single"
    if caps.device_type == "cpu":
        return "ddp"
    if caps.total_vram_per_gpu_gib >= 80.0:
        return "fsdp_hybrid" if (world_size > 8 and world_size % 8 == 0) else "fsdp_zero2"
    return "fsdp_zero3"


def initialize_distributed_context(
    dist_cfg: RootismDistributedConfig,
    *,
    init_method: str | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    local_rank: int | None = None,
) -> DistributedRuntimeContext:
    """Initialize (or attach to an existing) `torch.distributed` process group cleanly."""
    caps = detect_hardware_capabilities()
    eff_world = world_size if world_size is not None else caps.env_world_size
    eff_rank = rank if rank is not None else caps.env_rank
    eff_local = local_rank if local_rank is not None else caps.env_local_rank

    if dist_cfg.backend == "auto":
        backend = "nccl" if (caps.device_type == "cuda" and caps.nccl_available) else "gloo"
    else:
        backend = dist_cfg.backend

    if eff_world > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method or "env://",
            rank=eff_rank,
            world_size=eff_world,
        )

    is_init = dist.is_initialized()
    if is_init:
        eff_rank = dist.get_rank()
        eff_world = dist.get_world_size()

    if caps.device_type == "cuda":
        dev_idx = eff_local % max(1, caps.device_count)
        torch.cuda.set_device(dev_idx)
        device = torch.device("cuda", dev_idx)
    else:
        device = torch.device("cpu")

    strategy = resolve_distributed_strategy(dist_cfg, caps)
    if not is_init:
        strategy = "single"

    return DistributedRuntimeContext(
        initialized=is_init,
        backend=backend if is_init else "none",
        strategy=strategy,
        rank=eff_rank,
        local_rank=eff_local,
        world_size=eff_world,
        device=device,
    )


def wrap_model_for_distributed(
    model: nn.Module,
    ctx: DistributedRuntimeContext,
    dist_cfg: RootismDistributedConfig,
    param_dtype: torch.dtype | None = None,
) -> nn.Module:
    """Wrap `RootismForCausalLM` in DDP or composable DTensor FSDP2 (`fully_shard`)."""
    model = model.to(ctx.device)
    if not ctx.initialized or ctx.strategy == "single" or ctx.world_size == 1:
        return model

    if ctx.strategy == "ddp":
        device_ids = [ctx.device.index] if ctx.device.type == "cuda" else None
        return DDP(
            model,
            device_ids=device_ids,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
        )

    # Composable FSDP2 (`fully_shard`) across CUDA (`nccl`) and CPU (`gloo`)
    reshard_after_fwd = ctx.strategy in ("fsdp_zero3", "fsdp_hybrid")
    mp_policy = (
        MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
        )
        if param_dtype in (torch.bfloat16, torch.float16)
        else MixedPrecisionPolicy()
    )
    offload_policy: OffloadPolicy = (
        CPUOffloadPolicy() if dist_cfg.cpu_offload_params else OffloadPolicy()
    )

    dev_mesh_type = "cuda" if ctx.device.type == "cuda" else "cpu"
    if ctx.strategy == "fsdp_hybrid" and ctx.world_size >= 4 and ctx.world_size % 2 == 0:
        intra = min(8, ctx.world_size // 2)
        while ctx.world_size % intra != 0 and intra > 1:
            intra -= 1
        inter = ctx.world_size // intra
        mesh = init_device_mesh(dev_mesh_type, (inter, intra), mesh_dim_names=("replicate", "shard"))
    else:
        mesh = init_device_mesh(dev_mesh_type, (ctx.world_size,))

    for submodule in model.modules():
        if isinstance(submodule, RootismDecoderLayer):
            fully_shard(
                submodule,
                mesh=mesh,
                reshard_after_forward=reshard_after_fwd,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
            )

    fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=reshard_after_fwd,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
    )
    return model


@contextlib.contextmanager
def gradient_accumulation_sync_context(model: nn.Module, should_sync: bool) -> Iterator[None]:
    """
    Suppress redundant cross-worker gradient reductions on intermediate accumulation microbatches
    for both DDP (`no_sync()`) and FSDP2 (`set_requires_gradient_sync(False)`).
    """
    if should_sync:
        yield
        return

    if hasattr(model, "set_requires_gradient_sync"):
        model.set_requires_gradient_sync(False)  # type: ignore[operator]
        try:
            yield
        finally:
            model.set_requires_gradient_sync(True)  # type: ignore[operator]
    elif hasattr(model, "no_sync"):
        with model.no_sync():  # type: ignore[operator]
            yield
    else:
        yield


def all_reduce_token_counts(
    ctx: DistributedRuntimeContext,
    *,
    presented_tokens: int,
    active_loss_tokens: int,
    repeated_epoch_tokens: int,
    step_locally_valid: bool,
) -> tuple[int, int, int, bool]:
    """
    Synchronize exact global token counters and step validity across all ranks
    in a single 4-element `int64` collective.
    """
    if not ctx.initialized or ctx.world_size == 1:
        return presented_tokens, active_loss_tokens, repeated_epoch_tokens, step_locally_valid

    vec = torch.tensor(
        [
            int(presented_tokens),
            int(active_loss_tokens),
            int(repeated_epoch_tokens),
            0 if step_locally_valid else 1,
        ],
        dtype=torch.int64,
        device=ctx.device,
    )
    dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    vals = vec.cpu().tolist()
    globally_valid = int(vals[3]) == 0
    return int(vals[0]), int(vals[1]), int(vals[2]), globally_valid


def build_tensor_parallel_plan(
    model_config: RootismModelConfig,
    tp_size: int,
) -> dict[str, str]:
    if tp_size < 1:
        raise ValueError("tp_size must be >= 1")
    if tp_size == 1:
        return {}
    if model_config.n_heads % tp_size != 0 or model_config.n_kv_heads % tp_size != 0:
        raise ValueError(
            f"tp_size={tp_size} must divide both n_heads ({model_config.n_heads}) "
            f"and n_kv_heads ({model_config.n_kv_heads}) in Rootism GQA."
        )
    if model_config.intermediate_size % tp_size != 0:
        raise ValueError(
            f"tp_size={tp_size} must divide intermediate_size ({model_config.intermediate_size})."
        )
    plan: dict[str, str] = {}
    for layer_idx in range(model_config.n_layers):
        prefix = f"model.layers.{layer_idx}"
        plan[f"{prefix}.self_attn.q_proj"] = "colwise"
        plan[f"{prefix}.self_attn.k_proj"] = "colwise"
        plan[f"{prefix}.self_attn.v_proj"] = "colwise"
        plan[f"{prefix}.self_attn.o_proj"] = "rowwise"
        plan[f"{prefix}.mlp.gate_proj"] = "colwise"
        plan[f"{prefix}.mlp.up_proj"] = "colwise"
        plan[f"{prefix}.mlp.down_proj"] = "rowwise"
    return plan


def partition_layers_for_pipeline(
    model_config: RootismModelConfig,
    pp_size: int,
) -> list[dict[str, Any]]:
    if pp_size < 1 or pp_size > model_config.n_layers:
        raise ValueError(
            f"pp_size ({pp_size}) must be in [1, {model_config.n_layers}]"
        )
    base = model_config.n_layers // pp_size
    rem = model_config.n_layers % pp_size
    stages: list[dict[str, Any]] = []
    cursor = 0
    for stage_id in range(pp_size):
        count = base + (1 if stage_id < rem else 0)
        layer_range = list(range(cursor, cursor + count))
        cursor += count
        stages.append(
            {
                "stage_id": stage_id,
                "has_embeddings": stage_id == 0,
                "layer_indices": layer_range,
                "has_final_norm_and_lm_head": stage_id == pp_size - 1,
            }
        )
    return stages
