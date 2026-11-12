"""
Memory & Mixed-Precision Optimisation Controller (Step 49).

Implements:
- Hardware-verified BF16 mixed precision (where supported)
- FP16 dynamic loss-scaling fallback via `torch.amp.GradScaler` on GPUs without BF16
- Clean FP32 execution on generic CPUs
- Selective or full activation checkpointing
- Distributed-safe global gradient clipping (`FSDP.clip_grad_norm_` vs `nn.utils.clip_grad_norm_`)
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from typing import Literal

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from rootism.config import RootismMemoryConfig
from rootism.model.transformer import RootismForCausalLM
from rootism.training.hardware import HardwareCapabilities, detect_hardware_capabilities


class PrecisionAndMemoryController:
    """
    Coordinates mixed-precision `autocast`, `GradScaler`, activation checkpointing,
    and safe gradient clipping without assuming homogeneous GPU capabilities.
    """

    def __init__(
        self,
        memory_cfg: RootismMemoryConfig,
        hw: HardwareCapabilities | None = None,
    ) -> None:
        self.memory_cfg = memory_cfg
        self.hw = hw or detect_hardware_capabilities()
        self.resolved_precision: Literal["bf16", "fp16", "fp32"] = self._resolve_precision()
        self.device_type: Literal["cuda", "cpu"] = self.hw.device_type

        if self.device_type == "cuda" and self.hw.cuda_tf32_supported:
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = True

        self.use_grad_scaler: bool = self.resolved_precision == "fp16" and self.device_type == "cuda"
        self.scaler = torch.amp.GradScaler(
            device="cuda",
            enabled=self.use_grad_scaler,
        )

    def _resolve_precision(self) -> Literal["bf16", "fp16", "fp32"]:
        req = self.memory_cfg.precision
        if req == "auto":
            return self.hw.recommended_precision()
        if req == "bf16":
            if self.hw.device_type == "cuda" and not self.hw.cuda_bf16_supported:
                # Graceful hardware fallback to FP16 + GradScaler on pre-Ampere GPUs
                return "fp16" if self.hw.cuda_fp16_supported else "fp32"
            return "bf16"
        if req == "fp16":
            if self.hw.device_type == "cpu":
                # CPU FP16 matmuls lack dynamic-range safety; fall back to BF16 or FP32
                return "fp32"
            return "fp16"
        return "fp32"

    @property
    def autocast_dtype(self) -> torch.dtype:
        if self.resolved_precision == "bf16":
            return torch.bfloat16
        if self.resolved_precision == "fp16":
            return torch.float16
        return torch.float32

    def apply_model_memory_optimizations(self, model: RootismForCausalLM) -> None:
        """Configure selective or full activation checkpointing on `RootismForCausalLM`."""
        model.configure_activation_checkpointing(
            mode=self.memory_cfg.activation_checkpointing,
            every_n_layers=self.memory_cfg.selective_checkpoint_every_n_layers,
        )

    @contextlib.contextmanager
    def autocast_context(self) -> Iterator[None]:
        if self.resolved_precision == "fp32":
            yield
            return
        with torch.amp.autocast(
            device_type=self.device_type,
            dtype=self.autocast_dtype,
            enabled=True,
        ):
            yield

    def backward_scaled(self, loss: torch.Tensor) -> None:
        if self.use_grad_scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def unscale_clip_and_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        max_grad_norm: float,
    ) -> tuple[bool, float]:
        """
        Unscale gradients (if FP16 GradScaler active), compute & clip global gradient norm,
        check finite-ness, and execute `optimizer.step()` only when numerically valid.

        Returns `(step_executed, grad_norm_float)`.
        """
        if self.use_grad_scaler:
            self.scaler.unscale_(optimizer)

        if isinstance(model, FSDP):
            grad_norm_t = model.clip_grad_norm_(max_grad_norm)
        else:
            params = [p for p in model.parameters() if p.grad is not None]
            if params:
                grad_norm_t = nn.utils.clip_grad_norm_(params, max_grad_norm)
            else:
                grad_norm_t = torch.tensor(0.0)

        if hasattr(grad_norm_t, "full_tensor"):
            grad_norm = float(grad_norm_t.full_tensor().item())  # type: ignore[union-attr]
        elif torch.is_tensor(grad_norm_t):
            grad_norm = float(grad_norm_t.item())
        else:
            grad_norm = float(grad_norm_t)
        if not math.isfinite(grad_norm):
            if self.use_grad_scaler:
                self.scaler.update()
            optimizer.zero_grad(set_to_none=True)
            return False, grad_norm

        if self.use_grad_scaler:
            prev_scale = self.scaler.get_scale()
            self.scaler.step(optimizer)
            self.scaler.update()
            new_scale = self.scaler.get_scale()
            step_ok = new_scale >= prev_scale
        else:
            optimizer.step()
            step_ok = True

        optimizer.zero_grad(set_to_none=True)
        return step_ok, grad_norm
