"""
Complete Hardware-Adaptive, Resumable Rootism Pretraining Engine (Steps 51–60).

Integrates:
- `RootismForCausalLM` (independent scratch model)
- `SequencePacker` + `BatchScheduleCoordinator`
- `DistributedRuntimeContext` (Single / DDP / FSDP2)
- `PrecisionAndMemoryController` (BF16 / FP16 GradScaler / FP32 + activation checkpointing)
- `RootismLRScheduler` (WSD / Cosine)
- `TrainingStabilityMonitor`
- `AtomicCheckpointManager` + `PreemptionSignalCatcher`
- `TrainingThroughputProfiler`
- `TokenAccountingLedger` (strictly stopping at `target_training_tokens`, default 650B)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rootism.config import RootismConfig
from rootism.core.provenance import assert_independent_environment, seed_everything
from rootism.core.token_accounting import TokenAccountingLedger, TokenSnapshot
from rootism.model.transformer import RootismForCausalLM
from rootism.training.batching import BatchScheduleCoordinator
from rootism.training.checkpointing import AtomicCheckpointManager, PreemptionSignalCatcher
from rootism.training.distributed import (
    DistributedRuntimeContext,
    initialize_distributed_context,
    wrap_model_for_distributed,
)
from rootism.training.hardware import detect_hardware_capabilities
from rootism.training.memory import PrecisionAndMemoryController
from rootism.training.optimizer import build_optimizer
from rootism.training.packing import SequencePacker
from rootism.training.profiler import ThroughputSummaryReport, TrainingThroughputProfiler
from rootism.training.scheduler import RootismLRScheduler
from rootism.training.stability import StabilityDiagnostics, TrainingStabilityMonitor
from rootism.training.step_engine import TrainingStepExecutor


@dataclass(frozen=True)
class PretrainingRunSummary:
    resumed_from_checkpoint: str | None
    recovery_warnings: tuple[str, ...]
    final_step: int
    final_loss: float
    final_lr: float
    budget_reached: bool
    preempted_early: bool
    latest_checkpoint_path: str | None
    token_snapshot: TokenSnapshot
    throughput_summary: ThroughputSummaryReport
    last_stability_diagnostics: StabilityDiagnostics | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "resumed_from_checkpoint": self.resumed_from_checkpoint,
            "recovery_warnings": list(self.recovery_warnings),
            "final_step": self.final_step,
            "final_loss": round(self.final_loss, 6),
            "final_lr": self.final_lr,
            "budget_reached": self.budget_reached,
            "preempted_early": self.preempted_early,
            "latest_checkpoint_path": self.latest_checkpoint_path,
            "token_snapshot": self.token_snapshot.to_dict(),
            "throughput_summary": self.throughput_summary.to_dict(),
            "last_stability_diagnostics": (
                self.last_stability_diagnostics.to_dict()
                if self.last_stability_diagnostics is not None
                else None
            ),
        }


class RootismPretrainer:
    """
    Production pretraining orchestrator ready for single-GPU, multi-GPU, and multi-node clusters.
    """

    def __init__(
        self,
        config: RootismConfig,
        *,
        dist_ctx: DistributedRuntimeContext | None = None,
        max_retained_checkpoints: int = 5,
    ) -> None:
        assert_independent_environment()
        self.config = config
        self.hw = detect_hardware_capabilities()
        seed_everything(config.training.seed, deterministic=config.training.deterministic_ops)

        self.dist_ctx = dist_ctx or initialize_distributed_context(config.training.distributed)
        self.memory_ctrl = PrecisionAndMemoryController(config.training.memory, hw=self.hw)

        raw_model = RootismForCausalLM(config.model)
        self.memory_ctrl.apply_model_memory_optimizations(raw_model)
        self.model = wrap_model_for_distributed(
            raw_model,
            self.dist_ctx,
            config.training.distributed,
            param_dtype=self.memory_ctrl.autocast_dtype,
        )

        self.optimizer, self.optimizer_summary = build_optimizer(
            self.model, config.training.optimizer
        )
        self.scheduler = RootismLRScheduler(
            self.optimizer,
            peak_lr=config.training.optimizer.learning_rate,
            min_lr=config.training.optimizer.min_learning_rate,
            warmup_tokens=config.training.warmup_tokens,
            total_tokens=config.training.target_training_tokens,
            schedule_type="wsd",
        )
        self.token_ledger = TokenAccountingLedger(
            target_training_tokens=config.training.target_training_tokens,
            verification_mode=config.training.verification_mode,
        )
        self.batch_coord = BatchScheduleCoordinator(
            sequence_length=config.training.sequence_length,
            micro_batch_size=config.training.micro_batch_size,
            world_size=self.dist_ctx.world_size,
            gradient_accumulation_steps=config.training.memory.gradient_accumulation_steps,
        )
        self.packer = SequencePacker(
            sequence_length=config.training.sequence_length,
            pad_token_id=config.model.pad_token_id,
        )
        self.stability = TrainingStabilityMonitor()
        self.stability.attach_activation_hooks(self.model)

        self.ckpt_manager = AtomicCheckpointManager(
            config.training.checkpoint_dir,
            max_retained_checkpoints=max_retained_checkpoints,
        )
        self.profiler = TrainingThroughputProfiler(
            model_param_count=config.model.analytical_parameter_count(),
            world_size=self.dist_ctx.world_size,
        )
        self.step_executor = TrainingStepExecutor(
            model=self.model,
            optimizer=self.optimizer,
            memory_ctrl=self.memory_ctrl,
            dist_ctx=self.dist_ctx,
            token_ledger=self.token_ledger,
            train_cfg=config.training,
        )

        self.current_step: int = 0
        self.documents_consumed_cursor: int = 0

    def auto_resume_if_available(self) -> tuple[Path | None, list[str]]:
        """Automatically detect and restore the latest verified checkpoint, skipping corrupted states."""
        latest_dir, warnings = self.ckpt_manager.find_latest_valid_checkpoint(
            expected_config=self.config
        )
        if latest_dir is None:
            return None, warnings

        extra_state: dict[str, Any] = {}
        loaded = self.ckpt_manager.load_checkpoint(
            latest_dir,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            token_ledger=self.token_ledger,
            expected_config=self.config,
            restore_rng=True,
            extra_state_out=extra_state,
        )
        self.current_step = loaded.step
        self.documents_consumed_cursor = int(loaded.dataset_cursor.get("documents_consumed", 0))
        if "packer_state" in loaded.dataset_cursor:
            self.packer.load_state_dict(loaded.dataset_cursor["packer_state"])
        if "stability_state" in extra_state:
            self.stability.load_state_dict(extra_state["stability_state"])
        return latest_dir, warnings

    def train(
        self,
        document_stream: Iterable[dict[str, Any] | list[int]],
        *,
        auto_resume: bool = True,
        max_steps: int | None = None,
        save_final_checkpoint: bool = True,
    ) -> PretrainingRunSummary:
        """
        Run resumable training over `document_stream` until `token_ledger.should_stop`
        (`target_training_tokens` reached), `max_steps` is hit, or preemption is signaled.
        """
        resumed_dir: Path | None = None
        recovery_warnings: list[str] = []
        if auto_resume:
            resumed_dir, recovery_warnings = self.auto_resume_if_available()

        catcher = PreemptionSignalCatcher()
        catcher.install()

        latest_ckpt_saved: Path | None = resumed_dir
        last_loss = 0.0
        last_diag: StabilityDiagnostics | None = None
        preempted = False

        try:
            # Fast-forward stream past documents already consumed prior to resume!
            skipped = 0
            def _resumed_doc_iter():
                nonlocal skipped
                for item in document_stream:
                    if skipped < self.documents_consumed_cursor:
                        skipped += 1
                        continue
                    self.documents_consumed_cursor += 1
                    yield item

            mb_iter = self.packer.build_microbatches(
                _resumed_doc_iter(),
                micro_batch_size=self.config.training.micro_batch_size,
                flush_tail=True,
            )

            grad_accum = self.batch_coord.gradient_accumulation_steps

            while not self.token_ledger.should_stop:
                if max_steps is not None and self.current_step >= max_steps:
                    break
                if catcher.preemption_requested:
                    preempted = True
                    break

                t_data_start = self.profiler.now()
                microbatches: list[dict[str, Any]] = []
                for _ in range(grad_accum):
                    try:
                        microbatches.append(next(mb_iter))
                    except StopIteration:
                        break
                t_data_sec = self.profiler.now() - t_data_start

                if not microbatches:
                    break

                # Update LR smoothly according to net processed tokens so far
                self.scheduler.step_to_tokens(self.token_ledger.processed_training_tokens)

                prev_tokens = self.token_ledger.processed_training_tokens
                t_comp_start = self.profiler.now()
                step_res = self.step_executor.execute_global_step(microbatches)
                t_comp_sec = self.profiler.now() - t_comp_start

                if step_res.step_valid:
                    self.current_step += 1
                    last_loss = step_res.mean_loss
                    self.scheduler.step_to_tokens(self.token_ledger.processed_training_tokens)

                last_diag = self.stability.observe_step(
                    step=self.current_step,
                    loss=step_res.mean_loss,
                    grad_norm=step_res.grad_norm,
                    optimizer=self.optimizer,
                )

                step_tokens = self.token_ledger.processed_training_tokens - prev_tokens
                t_ckpt_sec = 0.0
                ckpt_bytes = 0

                should_ckpt = (
                    self.dist_ctx.is_primary
                    and step_res.step_valid
                    and (
                        self.current_step % self.config.training.checkpoint_interval_steps == 0
                        or catcher.preemption_requested
                    )
                )
                if should_ckpt:
                    t_c0 = self.profiler.now()
                    latest_ckpt_saved = self._save_current_checkpoint()
                    t_ckpt_sec = self.profiler.now() - t_c0
                    ckpt_bytes = sum(
                        f.stat().st_size for f in latest_ckpt_saved.iterdir() if f.is_file()
                    )

                self.profiler.record_step(
                    step=self.current_step,
                    tokens_processed=step_tokens,
                    data_wait_sec=t_data_sec,
                    compute_sec=t_comp_sec,
                    checkpoint_sec=t_ckpt_sec,
                    checkpoint_bytes_written=ckpt_bytes,
                )

                if catcher.preemption_requested:
                    preempted = True
                    break

            if save_final_checkpoint and self.dist_ctx.is_primary and self.current_step > 0:
                if latest_ckpt_saved is None or latest_ckpt_saved.name != f"step_{self.current_step:08d}":
                    latest_ckpt_saved = self._save_current_checkpoint()
        finally:
            catcher.uninstall()

        return PretrainingRunSummary(
            resumed_from_checkpoint=str(resumed_dir) if resumed_dir else None,
            recovery_warnings=tuple(recovery_warnings),
            final_step=self.current_step,
            final_loss=last_loss,
            final_lr=self.scheduler.current_lr,
            budget_reached=self.token_ledger.should_stop,
            preempted_early=preempted,
            latest_checkpoint_path=str(latest_ckpt_saved) if latest_ckpt_saved else None,
            token_snapshot=self.token_ledger.snapshot(),
            throughput_summary=self.profiler.summarize(),
            last_stability_diagnostics=last_diag,
        )

    def _save_current_checkpoint(self) -> Path:
        return self.ckpt_manager.save_checkpoint(
            step=self.current_step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            token_ledger=self.token_ledger,
            config=self.config,
            dataset_cursor={
                "documents_consumed": self.documents_consumed_cursor,
                "packer_state": self.packer.state_dict(),
            },
            distributed_metadata={
                "world_size": self.dist_ctx.world_size,
                "rank": self.dist_ctx.rank,
                "strategy": self.dist_ctx.strategy,
            },
            extra_state={"stability_state": self.stability.state_dict()},
        )
