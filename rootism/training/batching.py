"""
Global-Batch & Microbatch Management System (Step 52).

Reconciles `micro_batch_size`, `sequence_length`, `gradient_accumulation_steps`,
and distributed `world_size` to guarantee exact global batch accounting across
arbitrary GPU cluster topologies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GlobalBatchPlan:
    sequence_length: int
    causal_targets_per_sequence: int
    micro_batch_size: int
    world_size: int
    gradient_accumulation_steps: int
    sequences_per_microbatch_per_rank: int
    sequences_per_global_step: int
    presented_tokens_per_microbatch_per_rank: int
    presented_tokens_per_global_step: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BatchScheduleCoordinator:
    """
    Computes and enforces consistent microbatch and gradient-accumulation schedules
    across single-node and multi-node distributed runs.
    """

    def __init__(
        self,
        *,
        sequence_length: int,
        micro_batch_size: int,
        world_size: int = 1,
        gradient_accumulation_steps: int | None = None,
        target_global_batch_tokens: int | None = None,
    ) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be >= 2 for causal next-token training")
        if micro_batch_size < 1 or world_size < 1:
            raise ValueError("micro_batch_size and world_size must be >= 1")

        self.sequence_length = int(sequence_length)
        self.micro_batch_size = int(micro_batch_size)
        self.world_size = int(world_size)

        targets_per_seq = self.sequence_length - 1
        tokens_per_step_unit = self.micro_batch_size * targets_per_seq * self.world_size

        if target_global_batch_tokens is not None:
            if target_global_batch_tokens < tokens_per_step_unit:
                grad_accum = 1
            else:
                grad_accum = max(1, round(target_global_batch_tokens / float(tokens_per_step_unit)))
        elif gradient_accumulation_steps is not None:
            if gradient_accumulation_steps < 1:
                raise ValueError("gradient_accumulation_steps must be >= 1")
            grad_accum = int(gradient_accumulation_steps)
        else:
            grad_accum = 1

        self.gradient_accumulation_steps: int = grad_accum
        self.plan = GlobalBatchPlan(
            sequence_length=self.sequence_length,
            causal_targets_per_sequence=targets_per_seq,
            micro_batch_size=self.micro_batch_size,
            world_size=self.world_size,
            gradient_accumulation_steps=grad_accum,
            sequences_per_microbatch_per_rank=self.micro_batch_size,
            sequences_per_global_step=self.micro_batch_size * self.world_size * grad_accum,
            presented_tokens_per_microbatch_per_rank=self.micro_batch_size * targets_per_seq,
            presented_tokens_per_global_step=tokens_per_step_unit * grad_accum,
        )
