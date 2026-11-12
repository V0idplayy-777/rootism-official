"""
Strict 6-Category Training Token Accounting Ledger (Steps 05 & 60).

Explicitly distinguishes and tracks:
1. unique_source_tokens: Raw tokens observed in ingested source corpora before deduplication.
2. deduplicated_tokens: Net unique corpus tokens surviving quality filtering and deduplication.
3. presented_tokens: Total sequence slots fed into model forward passes (including padding/masked/repeats).
4. repeated_epoch_tokens: Non-padding tokens drawn from documents on epoch > 1.
5. padding_and_masked_tokens: Pad tokens + loss-masked tokens (e.g., ignore_index = -100) excluded from loss.
6. processed_training_tokens: Actual non-padding, loss-active tokens processed in completed, valid optimizer steps.

`processed_training_tokens` is the ONLY counter credited toward the >=600B (default 650B) target.
Aborted/NaN steps, padding tokens, and masked prompt tokens NEVER inflate `processed_training_tokens`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rootism.config import DEFAULT_PRODUCTION_TRAINING_TOKENS, MINIMUM_PRODUCTION_TRAINING_TOKENS


@dataclass(frozen=True)
class TokenSnapshot:
    """Immutable snapshot of all 6 token accounting metrics and budget progress."""

    target_training_tokens: int
    verification_mode: bool
    unique_source_tokens: int
    deduplicated_tokens: int
    presented_tokens: int
    repeated_epoch_tokens: int
    padding_and_masked_tokens: int
    processed_training_tokens: int
    skipped_unstable_step_tokens: int
    completed_steps: int

    @property
    def remaining_training_tokens(self) -> int:
        return max(0, self.target_training_tokens - self.processed_training_tokens)

    @property
    def progress_fraction(self) -> float:
        if self.target_training_tokens <= 0:
            return 1.0
        return min(1.0, self.processed_training_tokens / self.target_training_tokens)

    @property
    def budget_reached(self) -> bool:
        return self.processed_training_tokens >= self.target_training_tokens

    @property
    def effective_packing_efficiency(self) -> float:
        if self.presented_tokens == 0:
            return 0.0
        active = self.presented_tokens - self.padding_and_masked_tokens
        return max(0.0, active / self.presented_tokens)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["remaining_training_tokens"] = self.remaining_training_tokens
        out["progress_fraction"] = round(self.progress_fraction, 8)
        out["budget_reached"] = self.budget_reached
        out["effective_packing_efficiency"] = round(self.effective_packing_efficiency, 6)
        return out


class TokenAccountingLedger:
    """
    Stateful, checkpoint-serializable ledger tracking corpus & training token metrics.
    """

    def __init__(
        self,
        target_training_tokens: int = DEFAULT_PRODUCTION_TRAINING_TOKENS,
        verification_mode: bool = False,
    ) -> None:
        if not verification_mode and target_training_tokens <= MINIMUM_PRODUCTION_TRAINING_TOKENS:
            raise ValueError(
                f"Production TokenAccountingLedger requires target_training_tokens > "
                f"{MINIMUM_PRODUCTION_TRAINING_TOKENS:,} (got {target_training_tokens:,})."
            )
        if target_training_tokens <= 0:
            raise ValueError("target_training_tokens must be positive")

        self.target_training_tokens: int = int(target_training_tokens)
        self.verification_mode: bool = bool(verification_mode)

        self.unique_source_tokens: int = 0
        self.deduplicated_tokens: int = 0
        self.presented_tokens: int = 0
        self.repeated_epoch_tokens: int = 0
        self.padding_and_masked_tokens: int = 0
        self.processed_training_tokens: int = 0
        self.skipped_unstable_step_tokens: int = 0
        self.completed_steps: int = 0

        # Pending accumulators across gradient-accumulation micro-steps prior to optimizer.step()
        self._pending_presented: int = 0
        self._pending_padding_masked: int = 0
        self._pending_repeated_epoch: int = 0
        self._pending_active_loss_tokens: int = 0

    def register_corpus_statistics(
        self,
        *,
        unique_source_tokens: int,
        deduplicated_tokens: int,
    ) -> None:
        """Record pre-training corpus preparation token totals."""
        if unique_source_tokens < 0 or deduplicated_tokens < 0:
            raise ValueError("Corpus token counts must be non-negative")
        if deduplicated_tokens > unique_source_tokens:
            raise ValueError(
                f"deduplicated_tokens ({deduplicated_tokens}) cannot exceed "
                f"unique_source_tokens ({unique_source_tokens})"
            )
        self.unique_source_tokens = int(unique_source_tokens)
        self.deduplicated_tokens = int(deduplicated_tokens)

    def record_microbatch(
        self,
        *,
        total_sequence_slots: int,
        active_loss_tokens: int,
        repeated_epoch_tokens: int = 0,
    ) -> None:
        """
        Stage token counts for a single microbatch forward/backward pass.
        Tokens are NOT credited to `processed_training_tokens` until `commit_optimizer_step(step_valid=True)`.
        """
        if total_sequence_slots < 0 or active_loss_tokens < 0 or repeated_epoch_tokens < 0:
            raise ValueError("Microbatch token counts cannot be negative")
        if active_loss_tokens > total_sequence_slots:
            raise ValueError(
                f"active_loss_tokens ({active_loss_tokens}) cannot exceed "
                f"total_sequence_slots ({total_sequence_slots})"
            )
        if repeated_epoch_tokens > active_loss_tokens:
            raise ValueError(
                f"repeated_epoch_tokens ({repeated_epoch_tokens}) cannot exceed "
                f"active_loss_tokens ({active_loss_tokens})"
            )

        padding_and_masked = total_sequence_slots - active_loss_tokens
        self._pending_presented += int(total_sequence_slots)
        self._pending_padding_masked += int(padding_and_masked)
        self._pending_repeated_epoch += int(repeated_epoch_tokens)
        self._pending_active_loss_tokens += int(active_loss_tokens)

    def commit_optimizer_step(
        self,
        *,
        step_valid: bool = True,
        world_size_all_reduced_active_tokens: int | None = None,
        world_size_all_reduced_presented_tokens: int | None = None,
        world_size_all_reduced_repeated_tokens: int | None = None,
    ) -> TokenSnapshot:
        """
        Commit (or roll back on NaN/Inf overflow) the accumulated microbatches for one global step.

        Supports passing exact distributed all-reduced global counts across ranks.
        """
        presented = (
            int(world_size_all_reduced_presented_tokens)
            if world_size_all_reduced_presented_tokens is not None
            else self._pending_presented
        )
        active = (
            int(world_size_all_reduced_active_tokens)
            if world_size_all_reduced_active_tokens is not None
            else self._pending_active_loss_tokens
        )
        repeated = (
            int(world_size_all_reduced_repeated_tokens)
            if world_size_all_reduced_repeated_tokens is not None
            else self._pending_repeated_epoch
        )
        if active > presented:
            raise ValueError("All-reduced active tokens cannot exceed all-reduced presented tokens")
        padding_masked = presented - active

        self._pending_presented = 0
        self._pending_padding_masked = 0
        self._pending_repeated_epoch = 0
        self._pending_active_loss_tokens = 0

        self.presented_tokens += presented
        self.padding_and_masked_tokens += padding_masked

        if step_valid:
            self.processed_training_tokens += active
            self.repeated_epoch_tokens += repeated
            self.completed_steps += 1
        else:
            # NaN/Inf/skipped step: never inflate processed_training_tokens!
            self.skipped_unstable_step_tokens += active

        return self.snapshot()

    @property
    def should_stop(self) -> bool:
        return self.processed_training_tokens >= self.target_training_tokens

    def snapshot(self) -> TokenSnapshot:
        return TokenSnapshot(
            target_training_tokens=self.target_training_tokens,
            verification_mode=self.verification_mode,
            unique_source_tokens=self.unique_source_tokens,
            deduplicated_tokens=self.deduplicated_tokens,
            presented_tokens=self.presented_tokens,
            repeated_epoch_tokens=self.repeated_epoch_tokens,
            padding_and_masked_tokens=self.padding_and_masked_tokens,
            processed_training_tokens=self.processed_training_tokens,
            skipped_unstable_step_tokens=self.skipped_unstable_step_tokens,
            completed_steps=self.completed_steps,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "target_training_tokens": self.target_training_tokens,
            "verification_mode": self.verification_mode,
            "unique_source_tokens": self.unique_source_tokens,
            "deduplicated_tokens": self.deduplicated_tokens,
            "presented_tokens": self.presented_tokens,
            "repeated_epoch_tokens": self.repeated_epoch_tokens,
            "padding_and_masked_tokens": self.padding_and_masked_tokens,
            "processed_training_tokens": self.processed_training_tokens,
            "skipped_unstable_step_tokens": self.skipped_unstable_step_tokens,
            "completed_steps": self.completed_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        target = int(state["target_training_tokens"])
        ver_mode = bool(state.get("verification_mode", False))
        if not ver_mode and target <= MINIMUM_PRODUCTION_TRAINING_TOKENS:
            raise ValueError(
                f"Refusing to restore non-verification ledger with target <= {MINIMUM_PRODUCTION_TRAINING_TOKENS:,}"
            )
        self.target_training_tokens = target
        self.verification_mode = ver_mode
        self.unique_source_tokens = int(state["unique_source_tokens"])
        self.deduplicated_tokens = int(state["deduplicated_tokens"])
        self.presented_tokens = int(state["presented_tokens"])
        self.repeated_epoch_tokens = int(state["repeated_epoch_tokens"])
        self.padding_and_masked_tokens = int(state["padding_and_masked_tokens"])
        self.processed_training_tokens = int(state["processed_training_tokens"])
        self.skipped_unstable_step_tokens = int(state.get("skipped_unstable_step_tokens", 0))
        self.completed_steps = int(state["completed_steps"])
