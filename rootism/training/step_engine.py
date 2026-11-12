"""
Training Step Execution Engine (Steps 47–50 Cohesive Integration).

Executes genuine forward + backward + gradient accumulation (`no_sync`) +
mixed-precision unscaling/clipping + decoupled AdamW updates + distributed
token accounting (`TokenAccountingLedger`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from rootism.config import RootismTrainingConfig
from rootism.core.token_accounting import TokenAccountingLedger, TokenSnapshot
from rootism.model.transformer import IGNORE_INDEX
from rootism.training.distributed import (
    DistributedRuntimeContext,
    all_reduce_token_counts,
    gradient_accumulation_sync_context,
)
from rootism.training.memory import PrecisionAndMemoryController


@dataclass(frozen=True)
class StepExecutionResult:
    step_valid: bool
    mean_loss: float
    grad_norm: float
    token_snapshot: TokenSnapshot


class TrainingStepExecutor:
    """
    Executes one complete global optimizer step across `gradient_accumulation_steps`
    microbatches with exact loss normalization and distributed token accounting.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        memory_ctrl: PrecisionAndMemoryController,
        dist_ctx: DistributedRuntimeContext,
        token_ledger: TokenAccountingLedger,
        train_cfg: RootismTrainingConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.memory_ctrl = memory_ctrl
        self.dist_ctx = dist_ctx
        self.token_ledger = token_ledger
        self.train_cfg = train_cfg

    def execute_global_step(
        self,
        microbatches: list[dict[str, torch.Tensor | int | bool]],
    ) -> StepExecutionResult:
        """
        Run forward & backward over `microbatches`, synchronize gradients only on
        the final microbatch, clip & step optimizer, and commit token ledger.
        """
        if not microbatches:
            raise ValueError("microbatches list cannot be empty")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        num_micro = len(microbatches)
        local_presented = 0
        local_active = 0
        local_repeated = 0
        weighted_loss_sum = 0.0
        any_nan_loss = False

        for m_idx, mb in enumerate(microbatches):
            is_last = m_idx == (num_micro - 1)
            input_ids = mb["input_ids"]
            assert isinstance(input_ids, torch.Tensor)
            input_ids = input_ids.to(self.dist_ctx.device, non_blocking=True)

            labels_raw = mb.get("labels")
            if isinstance(labels_raw, torch.Tensor):
                labels = labels_raw.to(self.dist_ctx.device, non_blocking=True)
            else:
                labels = input_ids.clone()
                pad_id = getattr(getattr(self.model, "config", None), "pad_token_id", 0)
                labels[labels == pad_id] = IGNORE_INDEX

            position_ids_raw = mb.get("position_ids")
            position_ids = (
                position_ids_raw.to(self.dist_ctx.device, non_blocking=True)
                if isinstance(position_ids_raw, torch.Tensor)
                else None
            )
            doc_ids_raw = mb.get("document_ids")
            document_ids = (
                doc_ids_raw.to(self.dist_ctx.device, non_blocking=True)
                if isinstance(doc_ids_raw, torch.Tensor)
                else None
            )

            # Causal next-token prediction predicts S - 1 targets per sequence
            presented_slots = int(input_ids.shape[0] * max(0, input_ids.shape[1] - 1))

            with gradient_accumulation_sync_context(self.model, should_sync=is_last):
                with self.memory_ctrl.autocast_context():
                    out = self.model(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        document_ids=document_ids,
                        labels=labels,
                        return_logits=False,
                    )
                    assert out.loss is not None
                    scaled_micro_loss = out.loss / float(num_micro)

                loss_val = float(out.loss.detach().item())
                if not math.isfinite(loss_val):
                    any_nan_loss = True
                else:
                    weighted_loss_sum += loss_val * max(1, out.active_loss_tokens)

                self.memory_ctrl.backward_scaled(scaled_micro_loss)

            active_toks = int(out.active_loss_tokens)
            if "repeated_epoch_tokens" in mb and isinstance(mb["repeated_epoch_tokens"], int):
                rep_toks = min(active_toks, max(0, int(mb["repeated_epoch_tokens"])))
            else:
                rep_toks = active_toks if bool(mb.get("is_repeated_epoch", False)) else 0
            local_presented += presented_slots
            local_active += active_toks
            local_repeated += rep_toks
            self.token_ledger.record_microbatch(
                total_sequence_slots=presented_slots,
                active_loss_tokens=active_toks,
                repeated_epoch_tokens=rep_toks,
            )

        if any_nan_loss:
            self.optimizer.zero_grad(set_to_none=True)
            step_locally_ok = False
            grad_norm = float("nan")
        else:
            step_locally_ok, grad_norm = self.memory_ctrl.unscale_clip_and_step(
                model=self.model,
                optimizer=self.optimizer,
                max_grad_norm=self.train_cfg.optimizer.max_grad_norm,
            )

        g_pres, g_act, g_rep, g_valid = all_reduce_token_counts(
            self.dist_ctx,
            presented_tokens=local_presented,
            active_loss_tokens=local_active,
            repeated_epoch_tokens=local_repeated,
            step_locally_valid=step_locally_ok,
        )

        snap = self.token_ledger.commit_optimizer_step(
            step_valid=g_valid,
            world_size_all_reduced_presented_tokens=g_pres,
            world_size_all_reduced_active_tokens=g_act,
            world_size_all_reduced_repeated_tokens=g_rep,
        )
        mean_loss = weighted_loss_sum / float(max(1, local_active)) if not any_nan_loss else float("nan")
        return StepExecutionResult(
            step_valid=g_valid,
            mean_loss=mean_loss,
            grad_norm=grad_norm,
            token_snapshot=snap,
        )
