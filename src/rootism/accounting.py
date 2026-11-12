"""Training-token accounting (requirements 05, 60).

Definitions are deliberately explicit so no number can be inflated:

* unique_source_tokens     raw tokens in collected source documents (pre-dedup)
* deduplicated_tokens      tokens remaining after deduplication (one clean epoch)
* presented_tokens         every token slot fed to the model, including padding
* repeated_tokens          processed tokens coming from epoch >= 1 (repeats)
* padded_tokens            presented slots that were padding
* masked_tokens            presented, non-padding slots excluded from the loss
* processed_training_tokens presented - padded - masked  ("actual" tokens)
* committed_tokens         processed tokens in steps that completed and were
                           checkpointed ("successfully processed")
* failed_tokens            processed tokens from steps that failed and were
                           retried; never counted toward the target

Only `committed_tokens` counts toward the 650B training budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .config import TrainingTokenBudget
from .utils.hashing import json_load, json_save
from .version import SCHEMA_VERSION


@dataclass
class TokenLedger:
    unique_source_tokens: int = 0
    deduplicated_tokens: int = 0
    presented_tokens: int = 0
    repeated_tokens: int = 0
    padded_tokens: int = 0
    masked_tokens: int = 0
    processed_training_tokens: int = 0
    committed_tokens: int = 0
    failed_tokens: int = 0
    epochs: dict[str, int] = field(default_factory=dict)
    steps: int = 0
    schema_version: str = SCHEMA_VERSION

    def set_corpus_counts(self, *, unique_source_tokens: int, deduplicated_tokens: int) -> None:
        if unique_source_tokens < deduplicated_tokens:
            raise ValueError("deduplicated_tokens cannot exceed unique_source_tokens")
        self.unique_source_tokens = unique_source_tokens
        self.deduplicated_tokens = deduplicated_tokens

    def record_step(
        self,
        *,
        presented_tokens: int,
        padded_tokens: int = 0,
        masked_tokens: int = 0,
        epoch: int = 0,
        committed: bool = True,
    ) -> int:
        """Record one completed optimizer step; return actual tokens added."""
        if padded_tokens + masked_tokens > presented_tokens:
            raise ValueError("padding + masked tokens exceed presented tokens")
        actual = presented_tokens - padded_tokens - masked_tokens
        if actual < 0:
            raise ValueError("negative token count")
        self.presented_tokens += presented_tokens
        self.padded_tokens += padded_tokens
        self.masked_tokens += masked_tokens
        self.repeated_tokens += actual if epoch >= 1 else 0
        self.processed_training_tokens += actual
        self.steps += 1
        key = str(epoch)
        self.epochs[key] = self.epochs.get(key, 0) + actual
        if committed:
            self.committed_tokens += actual
        else:
            self.failed_tokens += actual
        return actual

    @property
    def unique_fraction(self) -> float:
        return (
            (self.committed_tokens - self.repeated_tokens) / self.committed_tokens
            if self.committed_tokens
            else 0.0
        )

    def invariant_errors(self) -> list[str]:
        errors = []
        if self.presented_tokens - self.padded_tokens - self.masked_tokens != self.processed_training_tokens:
            errors.append("presented - padded - masked != processed_training_tokens")
        if self.committed_tokens + self.failed_tokens != self.processed_training_tokens:
            errors.append("committed + failed != processed_training_tokens")
        if sum(self.epochs.values()) != self.processed_training_tokens:
            errors.append("per-epoch totals do not sum to processed_training_tokens")
        if self.committed_tokens > self.processed_training_tokens:
            errors.append("committed exceeds processed")
        return errors

    def progress(self, budget: TrainingTokenBudget) -> dict[str, float | int]:
        return {
            "target_tokens": budget.target_tokens,
            "committed_tokens": self.committed_tokens,
            "fraction_complete": self.committed_tokens / budget.target_tokens,
            "remaining_tokens": max(0, budget.target_tokens - self.committed_tokens),
            "unique_fraction": self.unique_fraction,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "unique_source_tokens": self.unique_source_tokens,
            "deduplicated_tokens": self.deduplicated_tokens,
            "presented_tokens": self.presented_tokens,
            "repeated_tokens": self.repeated_tokens,
            "padded_tokens": self.padded_tokens,
            "masked_tokens": self.masked_tokens,
            "processed_training_tokens": self.processed_training_tokens,
            "committed_tokens": self.committed_tokens,
            "failed_tokens": self.failed_tokens,
            "epochs": self.epochs,
            "steps": self.steps,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TokenLedger":
        known = set(cls().__dict__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> None:
        json_save(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "TokenLedger":
        return cls.from_dict(json_load(path))


def preflight_budget(
    budget: TrainingTokenBudget,
    *,
    unique_source_tokens: int,
    deduplicated_tokens: int,
) -> dict[str, int]:
    """Validate the whole training configuration BEFORE training starts."""
    budget.validate(deduplicated_tokens=deduplicated_tokens)
    if deduplicated_tokens > unique_source_tokens:
        raise ValueError("deduplicated_tokens > unique_source_tokens is impossible")
    return {
        "target_tokens": budget.target_tokens,
        "hard_minimum_tokens": budget.hard_minimum_tokens,
        "minimum_unique_tokens": budget.minimum_unique_tokens,
        "required_epochs_ceiling": (
            budget.target_tokens // max(deduplicated_tokens, 1) + 1
        ),
        "max_epochs": budget.max_epochs,
    }


def assert_budget_complete(budget: TrainingTokenBudget, ledger: TokenLedger) -> None:
    """Refuse to declare a run complete unless the ACTUAL token floor is met."""
    if ledger.invariant_errors():
        raise ValueError("token ledger invariants broken: " + "; ".join(ledger.invariant_errors()))
    if ledger.committed_tokens <= budget.hard_minimum_tokens:
        raise ValueError(
            f"training has {ledger.committed_tokens:,} committed tokens; "
            f"MORE THAN {budget.hard_minimum_tokens:,} are required"
        )
    if ledger.committed_tokens < budget.target_tokens:
        raise ValueError(
            f"training stopped early: {ledger.committed_tokens:,} < {budget.target_tokens:,}"
        )
    if ledger.repeated_tokens > ledger.committed_tokens * (1.0 - budget.min_unique_fraction):
        raise ValueError("too much of the budget came from repeated epochs")
