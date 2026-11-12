"""
Configurable Learning-Rate Scheduling System (Step 51).

Supports both:
- `cosine`: Linear warmup from 0 -> `peak_lr` over `warmup_tokens`, followed by smooth
  cosine decay to `min_lr` at `total_tokens`.
- `wsd` (Warmup-Stable-Decay): Linear warmup -> constant `peak_lr` plateau -> smooth
  cosine annealing to `min_lr` over the final `decay_fraction` of `total_tokens`
  (preventing abrupt LR cliffs across multi-stage curriculum transitions).

Driven by exact `processed_training_tokens` (or step fallback) so gradient-accumulation
or sequence-length curriculum changes never introduce discontinuous LR jumps!
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch


class RootismLRScheduler:
    """
    Token-progress-indexed smooth learning-rate scheduler with full checkpoint state serialization.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        peak_lr: float,
        min_lr: float,
        warmup_tokens: int,
        total_tokens: int,
        schedule_type: Literal["cosine", "wsd"] = "wsd",
        wsd_decay_fraction: float = 0.15,
    ) -> None:
        if peak_lr <= 0.0:
            raise ValueError(f"peak_lr must be > 0, got {peak_lr}")
        if min_lr < 0.0 or min_lr > peak_lr:
            raise ValueError(f"min_lr ({min_lr}) must be in [0, peak_lr ({peak_lr})]")
        if total_tokens <= 0:
            raise ValueError(f"total_tokens must be > 0, got {total_tokens}")
        if warmup_tokens < 0:
            raise ValueError("warmup_tokens must be >= 0")
        if not (0.0 < wsd_decay_fraction < 1.0):
            raise ValueError("wsd_decay_fraction must be in (0, 1)")

        self.optimizer = optimizer
        self.peak_lr = float(peak_lr)
        self.min_lr = float(min_lr)
        self.warmup_tokens = min(int(warmup_tokens), int(total_tokens))
        self.total_tokens = int(total_tokens)
        self.schedule_type = schedule_type
        self.wsd_decay_fraction = float(wsd_decay_fraction)

        self.current_tokens: int = 0
        self.current_lr: float = self.compute_lr_at_tokens(0)
        self._apply_lr(self.current_lr)

    def compute_lr_at_tokens(self, processed_tokens: int) -> float:
        t = max(0, int(processed_tokens))

        if self.warmup_tokens > 0 and t < self.warmup_tokens:
            # Smooth positive floor on step 0 (1 / (warmup + 1)) so step 0 actually moves weights cleanly
            alpha = float(t + 1) / float(self.warmup_tokens + 1)
            return self.peak_lr * alpha

        if t >= self.total_tokens:
            return self.min_lr

        post_warmup_span = max(1, self.total_tokens - self.warmup_tokens)

        if self.schedule_type == "cosine":
            progress = float(t - self.warmup_tokens) / float(post_warmup_span)
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.peak_lr - self.min_lr) * cosine

        # Warmup-Stable-Decay (WSD)
        decay_tokens = max(1, int(round(self.total_tokens * self.wsd_decay_fraction)))
        decay_start = max(self.warmup_tokens, self.total_tokens - decay_tokens)
        if t <= decay_start:
            return self.peak_lr

        decay_span = max(1, self.total_tokens - decay_start)
        progress = min(1.0, max(0.0, float(t - decay_start) / float(decay_span)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    def compute_lr(self, processed_tokens: int) -> float:
        return self.compute_lr_at_tokens(processed_tokens)
    def _apply_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            lr_mult = float(group.get("lr_scale", 1.0))
            group["lr"] = lr * lr_mult

    def step_to_tokens(self, processed_tokens: int) -> float:
        """Advance schedule smoothly to `processed_tokens` and apply updated LR to optimizer."""
        self.current_tokens = max(0, int(processed_tokens))
        self.current_lr = self.compute_lr_at_tokens(self.current_tokens)
        self._apply_lr(self.current_lr)
        return self.current_lr

    def state_dict(self) -> dict[str, Any]:
        return {
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr,
            "warmup_tokens": self.warmup_tokens,
            "total_tokens": self.total_tokens,
            "schedule_type": self.schedule_type,
            "wsd_decay_fraction": self.wsd_decay_fraction,
            "current_tokens": self.current_tokens,
            "current_lr": self.current_lr,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.peak_lr = float(state["peak_lr"])
        self.min_lr = float(state["min_lr"])
        self.warmup_tokens = int(state["warmup_tokens"])
        self.total_tokens = int(state["total_tokens"])
        self.schedule_type = state["schedule_type"]
        self.wsd_decay_fraction = float(state.get("wsd_decay_fraction", 0.15))
        self.step_to_tokens(int(state["current_tokens"]))
