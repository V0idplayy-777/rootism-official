"""
Rootism Optimiser Engineering (Step 50).

Implements:
1. Strict parameter-group partitioning:
   - 2D+ matrices (embeddings, Q/K/V/O projections, SwiGLU gate/up/down, lm_head) -> decayed (`weight_decay = 0.1`)
   - 1D scale/norm/bias vectors (`RMSNorm`, `QK-RMSNorm`, any biases) -> non-decayed (`weight_decay = 0.0`)
2. Hardware-adaptive PyTorch `AdamW` (`fused=True` on CUDA when compatible, `foreach=True` fallback)
3. `RootismFP32MasterAdamW`: independently implemented decoupled AdamW keeping FP32 master weights
   and FP32 moments `(exp_avg, exp_avg_sq)` when model parameters themselves reside in BF16/FP16,
   preventing low-precision mantissa update underflow!
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from rootism.config import RootismOptimizerConfig


@dataclass(frozen=True)
class OptimizerGroupSummary:
    decayed_tensor_count: int
    decayed_parameter_count: int
    non_decayed_tensor_count: int
    non_decayed_parameter_count: int
    optimizer_class: str
    used_fused_kernel: bool
    used_foreach_kernel: bool


def partition_parameter_groups(
    model: nn.Module,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], OptimizerGroupSummary]:
    """
    Separate model trainable parameters into weight-decayed (ndim >= 2) and
    non-decayed (ndim < 2, e.g. RMSNorm / QK-Norm / biases), deduplicating tied weights.
    """
    seen_ids: set[int] = set()
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen_ids:
            continue
        seen_ids.add(pid)

        is_norm_or_1d = (
            param.ndim < 2
            or name.endswith(".bias")
            or "norm" in name.lower()
        )
        if is_norm_or_1d:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups: list[dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": float(weight_decay), "group_name": "decayed"})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0, "group_name": "non_decayed"})

    summary = OptimizerGroupSummary(
        decayed_tensor_count=len(decay_params),
        decayed_parameter_count=sum(int(p.numel()) for p in decay_params),
        non_decayed_tensor_count=len(no_decay_params),
        non_decayed_parameter_count=sum(int(p.numel()) for p in no_decay_params),
        optimizer_class="",
        used_fused_kernel=False,
        used_foreach_kernel=False,
    )
    return groups, summary


class RootismFP32MasterAdamW(torch.optim.Optimizer):
    """
    Decoupled AdamW optimizer maintaining exact FP32 master copies and FP32 first/second
    moments even when `param.data` is stored in `torch.bfloat16` or `torch.float16`.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps: float = float(group["eps"])
            wd: float = float(group["weight_decay"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad_fp32 = p.grad.detach().to(torch.float32)
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["master_param"] = p.detach().to(torch.float32).clone()
                    state["exp_avg"] = torch.zeros_like(state["master_param"], dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(state["master_param"], dtype=torch.float32)

                state["step"] += 1
                step_t: int = state["step"]
                master: torch.Tensor = state["master_param"]
                exp_avg: torch.Tensor = state["exp_avg"]
                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]

                # Decoupled weight decay in FP32 master space
                if wd != 0.0:
                    master.mul_(1.0 - lr * wd)

                exp_avg.lerp_(grad_fp32, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

                bias_correction1 = 1.0 - (beta1**step_t)
                bias_correction2 = 1.0 - (beta2**step_t)
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                master.addcdiv_(exp_avg, denom, value=-step_size)
                p.copy_(master.to(p.dtype))

        return loss


def build_optimizer(
    model: nn.Module,
    opt_cfg: RootismOptimizerConfig,
) -> tuple[torch.optim.Optimizer, OptimizerGroupSummary]:
    """Build configured Rootism optimizer with hardware-aware fused/foreach dispatch."""
    groups, base_summary = partition_parameter_groups(model, weight_decay=opt_cfg.weight_decay)
    all_params = [p for g in groups for p in g["params"]]
    all_on_cuda = bool(all_params) and all(p.is_cuda for p in all_params)

    if opt_cfg.optimizer_type == "rootism_fp32_master_adamw":
        opt: torch.optim.Optimizer = RootismFP32MasterAdamW(
            groups,
            lr=opt_cfg.learning_rate,
            betas=(opt_cfg.beta1, opt_cfg.beta2),
            eps=opt_cfg.eps,
            weight_decay=opt_cfg.weight_decay,
        )
        return opt, OptimizerGroupSummary(
            decayed_tensor_count=base_summary.decayed_tensor_count,
            decayed_parameter_count=base_summary.decayed_parameter_count,
            non_decayed_tensor_count=base_summary.non_decayed_tensor_count,
            non_decayed_parameter_count=base_summary.non_decayed_parameter_count,
            optimizer_class="RootismFP32MasterAdamW",
            used_fused_kernel=False,
            used_foreach_kernel=False,
        )

    adamw_sig = inspect.signature(torch.optim.AdamW.__init__).parameters
    use_fused = bool(
        opt_cfg.prefer_fused
        and all_on_cuda
        and "fused" in adamw_sig
        and all(p.dtype in (torch.float32, torch.bfloat16, torch.float16) for p in all_params)
    )
    use_foreach = bool(
        not use_fused and opt_cfg.prefer_foreach and "foreach" in adamw_sig
    )

    kwargs: dict[str, Any] = {
        "lr": opt_cfg.learning_rate,
        "betas": (opt_cfg.beta1, opt_cfg.beta2),
        "eps": opt_cfg.eps,
        "weight_decay": opt_cfg.weight_decay,
    }
    if use_fused:
        kwargs["fused"] = True
    elif use_foreach:
        kwargs["foreach"] = True

    opt = torch.optim.AdamW(groups, **kwargs)
    return opt, OptimizerGroupSummary(
        decayed_tensor_count=base_summary.decayed_tensor_count,
        decayed_parameter_count=base_summary.decayed_parameter_count,
        non_decayed_tensor_count=base_summary.non_decayed_tensor_count,
        non_decayed_parameter_count=base_summary.non_decayed_parameter_count,
        optimizer_class="torch.optim.AdamW",
        used_fused_kernel=use_fused,
        used_foreach_kernel=use_foreach,
    )
