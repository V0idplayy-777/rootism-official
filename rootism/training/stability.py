"""
Training Stability & Numerical Anomaly Monitor (Step 54).

Monitors and diagnoses:
1. NaN / Inf values in loss, gradients, or optimizer state moments
2. Statistical loss spikes (`loss > ema_loss * spike_ratio` and `z_score > spike_z`)
3. Gradient explosions (`grad_norm > explosion_threshold`)
4. Vanishing gradients (`grad_norm < vanishing_threshold` sustained across `vanishing_patience` steps)
5. Abnormal hidden-state activation magnitudes via lightweight forward hooks (`max_abs` & RMS)
6. Optimiser moment instability (`exp_avg` / `exp_avg_sq` non-finiteness or extreme variance)
7. Sustained training divergence (monotonically worsening EMA loss after warmup)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class StabilityDiagnostics:
    step: int
    healthy: bool
    anomalies: tuple[str, ...]
    loss: float
    loss_ema: float
    grad_norm: float
    max_activation_abs: float
    mean_activation_rms: float
    optimizer_moments_finite: bool
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["anomalies"] = list(self.anomalies)
        return d


class TrainingStabilityMonitor:
    """
    Stateful numerical watchdog attached to `RootismForCausalLM` and its optimizer.
    """

    def __init__(
        self,
        *,
        ema_decay: float = 0.95,
        loss_spike_ratio: float = 2.25,
        loss_spike_z: float = 4.0,
        grad_explosion_threshold: float = 25.0,
        grad_vanishing_threshold: float = 1e-8,
        vanishing_patience: int = 5,
        activation_abs_ceiling: float = 65_000.0,
        divergence_window: int = 20,
    ) -> None:
        self.ema_decay = ema_decay
        self.loss_spike_ratio = loss_spike_ratio
        self.loss_spike_z = loss_spike_z
        self.grad_explosion_threshold = grad_explosion_threshold
        self.grad_vanishing_threshold = grad_vanishing_threshold
        self.vanishing_patience = vanishing_patience
        self.activation_abs_ceiling = activation_abs_ceiling
        self.divergence_window = divergence_window

        self._loss_history: deque[float] = deque(maxlen=128)
        self._ema_history: deque[float] = deque(maxlen=divergence_window)
        self.loss_ema: float | None = None
        self.consecutive_vanishing_steps: int = 0
        self.total_anomalies_recorded: int = 0

        self._hook_handles: list[Any] = []
        self._latest_act_max_abs: float = 0.0
        self._latest_act_rms_values: list[float] = []

    def attach_activation_hooks(self, model: nn.Module) -> None:
        """Attach zero-grad-overhead activation telemetry hooks to decoder norms/blocks."""
        self.remove_hooks()

        def _make_hook():
            def _hook(_mod: nn.Module, _inp: Any, out: Any) -> None:
                tensor = out[0] if isinstance(out, tuple) else out
                if not isinstance(tensor, torch.Tensor):
                    return
                with torch.no_grad():
                    det = tensor.detach()
                    if det.numel() == 0:
                        return
                    # Sample first row to avoid expensive full-batch sync
                    sample = det[0].to(torch.float32)
                    m_abs = float(sample.abs().max().item())
                    rms = float(sample.pow(2).mean().sqrt().item())
                    if not math.isfinite(m_abs) or m_abs > self._latest_act_max_abs:
                        self._latest_act_max_abs = m_abs
                    self._latest_act_rms_values.append(rms)

            return _hook

        raw = model.module if hasattr(model, "module") else model
        layers = getattr(getattr(raw, "model", None), "layers", None)
        if layers is not None and len(layers) > 0:
            # Instrument first, middle, and final transformer blocks
            indices = sorted({0, len(layers) // 2, len(layers) - 1})
            for i in indices:
                self._hook_handles.append(layers[i].register_forward_hook(_make_hook()))
        else:
            self._hook_handles.append(raw.register_forward_hook(_make_hook()))

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    @staticmethod
    def check_optimizer_moments_finite(optimizer: torch.optim.Optimizer) -> bool:
        for state in optimizer.state.values():
            if not isinstance(state, dict):
                continue
            for key in ("exp_avg", "exp_avg_sq", "master_param"):
                t = state.get(key)
                if isinstance(t, torch.Tensor):
                    local_t = t.to_local() if hasattr(t, "to_local") else t
                    if not bool(torch.isfinite(local_t).all().item()):
                        return False
        return True

    def observe_step(
        self,
        *,
        step: int,
        loss: float,
        grad_norm: float,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> StabilityDiagnostics:
        anomalies: list[str] = []

        act_max = self._latest_act_max_abs
        act_rms = (
            sum(self._latest_act_rms_values) / len(self._latest_act_rms_values)
            if self._latest_act_rms_values
            else 0.0
        )
        self._latest_act_max_abs = 0.0
        self._latest_act_rms_values.clear()

        if not math.isfinite(loss):
            anomalies.append("NON_FINITE_LOSS_NAN_OR_INF")
        if not math.isfinite(grad_norm):
            anomalies.append("NON_FINITE_GRADIENT_NORM")

        if not math.isfinite(act_max) or not math.isfinite(act_rms):
            anomalies.append("NON_FINITE_ACTIVATION")
        elif act_max > self.activation_abs_ceiling:
            anomalies.append("EXTREME_ACTIVATION_MAGNITUDE")

        if math.isfinite(grad_norm):
            if grad_norm > self.grad_explosion_threshold:
                anomalies.append("GRADIENT_EXPLOSION")
            if grad_norm < self.grad_vanishing_threshold:
                self.consecutive_vanishing_steps += 1
                if self.consecutive_vanishing_steps >= self.vanishing_patience:
                    anomalies.append("VANISHING_GRADIENTS")
            else:
                self.consecutive_vanishing_steps = 0

        if math.isfinite(loss):
            if len(self._loss_history) >= 8 and self.loss_ema is not None:
                mean_h = sum(self._loss_history) / len(self._loss_history)
                std_h = math.sqrt(
                    sum((x - mean_h) ** 2 for x in self._loss_history) / len(self._loss_history)
                )
                z = (loss - mean_h) / max(std_h, 1e-4)
                if loss > self.loss_ema * self.loss_spike_ratio and z > self.loss_spike_z:
                    anomalies.append("LOSS_SPIKE")

            if self.loss_ema is None:
                self.loss_ema = loss
            else:
                self.loss_ema = self.ema_decay * self.loss_ema + (1.0 - self.ema_decay) * loss
            self._loss_history.append(loss)
            self._ema_history.append(self.loss_ema)

            if len(self._ema_history) == self.divergence_window:
                first_q = list(self._ema_history)[: self.divergence_window // 4]
                last_q = list(self._ema_history)[-self.divergence_window // 4 :]
                if min(last_q) > max(first_q) * 1.35:
                    anomalies.append("TRAINING_DIVERGENCE")

        opt_ok = True
        if optimizer is not None:
            opt_ok = self.check_optimizer_moments_finite(optimizer)
            if not opt_ok:
                anomalies.append("OPTIMIZER_MOMENT_INSTABILITY")

        healthy = len(anomalies) == 0
        if not healthy:
            self.total_anomalies_recorded += len(anomalies)

        if "NON_FINITE_LOSS_NAN_OR_INF" in anomalies or "OPTIMIZER_MOMENT_INSTABILITY" in anomalies:
            action = "rollback_to_latest_valid_checkpoint_and_downscale_lr"
        elif "TRAINING_DIVERGENCE" in anomalies or "LOSS_SPIKE" in anomalies:
            action = "skip_or_dampen_step_and_inspect_data_shard"
        elif "GRADIENT_EXPLOSION" in anomalies:
            action = "rely_on_grad_clip_and_monitor"
        elif "VANISHING_GRADIENTS" in anomalies:
            action = "inspect_residual_scaling_and_learning_rate"
        else:
            action = "continue"

        return StabilityDiagnostics(
            step=step,
            healthy=healthy,
            anomalies=tuple(anomalies),
            loss=loss,
            loss_ema=self.loss_ema if self.loss_ema is not None else loss,
            grad_norm=grad_norm,
            max_activation_abs=round(act_max, 4),
            mean_activation_rms=round(act_rms, 4),
            optimizer_moments_finite=opt_ok,
            recommended_action=action,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "loss_ema": self.loss_ema,
            "loss_history": list(self._loss_history),
            "ema_history": list(self._ema_history),
            "consecutive_vanishing_steps": self.consecutive_vanishing_steps,
            "total_anomalies_recorded": self.total_anomalies_recorded,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.loss_ema = state.get("loss_ema")
        self._loss_history = deque(state.get("loss_history", []), maxlen=128)
        self._ema_history = deque(state.get("ema_history", []), maxlen=self.divergence_window)
        self.consecutive_vanishing_steps = int(state.get("consecutive_vanishing_steps", 0))
        self.total_anomalies_recorded = int(state.get("total_anomalies_recorded", 0))
