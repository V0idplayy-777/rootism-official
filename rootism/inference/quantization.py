"""
Non-Destructive INT8 & INT4 Weight Quantisation & Quality Evaluator (Step 78).

Implements:
1. `QuantizedLinearInt8`: Symmetric per-out-channel INT8 weight quantisation (`1 byte/weight`).
2. `QuantizedLinearInt4`: Group-wise symmetric INT4 weight quantisation packed 2 values per `uint8`
   byte (`0.5 bytes/weight` + per-group FP16/FP32 scales).
3. `quantize_model_non_destructive(model, mode="int8"|"int4")`: Clones and quantises `nn.Linear`
   blocks in memory (or exports to a separate `.quant.safetensors` artifact) WITHOUT modifying
   or overwriting the full-precision checkpoint!
4. `evaluate_quantization_quality(...)`: Measures actual byte footprint reduction, logit cosine
   similarity, mean absolute logit drift, and cross-entropy / perplexity delta on held-out sequences.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

QuantMode = Literal["none", "int8", "int4"]


class QuantizedLinearInt8(nn.Module):
    """Per-output-channel symmetric INT8 weight-only linear layer."""

    def __init__(self, in_features: int, out_features: int, has_bias: bool) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "qweight", torch.empty((out_features, in_features), dtype=torch.int8)
        )
        self.register_buffer(
            "scales", torch.empty((out_features, 1), dtype=torch.float32)
        )
        if has_bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_buffer("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> QuantizedLinearInt8:
        mod = cls(linear.in_features, linear.out_features, linear.bias is not None)
        with torch.no_grad():
            w = linear.weight.detach().to(torch.float32)
            max_abs = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            scales = max_abs / 127.0
            qw = torch.clamp(torch.round(w / scales), -127, 127).to(torch.int8)
            mod.qweight.copy_(qw)
            mod.scales.copy_(scales)
            if linear.bias is not None and mod.bias is not None:
                mod.bias.copy_(linear.bias.detach().to(torch.float32))
        return mod

    def dequantize_weight(self, dtype: torch.dtype) -> torch.Tensor:
        return (self.qweight.to(torch.float32) * self.scales).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize_weight(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)

    def storage_bytes(self) -> int:
        total = self.qweight.numel() * self.qweight.element_size()
        total += self.scales.numel() * self.scales.element_size()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return total


class QuantizedLinearInt4(nn.Module):
    """
    Group-wise symmetric signed INT4 (`[-7, +7]`) weight-only linear layer
    with two 4-bit weights packed per `uint8` byte (`0.5 B/param` + group scale).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool,
        group_size: int = 64,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = max(8, min(group_size, in_features))
        # Pad in_features to even multiple of group_size if needed
        self.padded_in = math.ceil(in_features / (2 * self.group_size)) * (2 * self.group_size)
        self.num_groups = self.padded_in // self.group_size

        self.register_buffer(
            "packed_qweight",
            torch.empty((out_features, self.padded_in // 2), dtype=torch.uint8),
        )
        self.register_buffer(
            "scales",
            torch.empty((out_features, self.num_groups), dtype=torch.float16),
        )
        if has_bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_buffer("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 64) -> QuantizedLinearInt4:
        mod = cls(linear.in_features, linear.out_features, linear.bias is not None, group_size=group_size)
        with torch.no_grad():
            w = linear.weight.detach().to(torch.float32)
            if mod.padded_in > mod.in_features:
                w = F.pad(w, (0, mod.padded_in - mod.in_features), value=0.0)
            grouped = w.view(mod.out_features, mod.num_groups, mod.group_size)
            max_abs = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            scales = max_abs / 7.0
            # Map signed [-7..+7] to unsigned nibble [1..15] via +8 offset
            q_signed = torch.clamp(torch.round(grouped / scales), -7, 7).to(torch.int16)
            q_unsigned = (q_signed + 8).to(torch.uint8).view(mod.out_features, mod.padded_in)

            low_nibble = q_unsigned[:, 0::2] & 0x0F
            high_nibble = (q_unsigned[:, 1::2] & 0x0F) << 4
            packed = low_nibble | high_nibble

            mod.packed_qweight.copy_(packed)
            mod.scales.copy_(scales.squeeze(-1).to(torch.float16))
            if linear.bias is not None and mod.bias is not None:
                mod.bias.copy_(linear.bias.detach().to(torch.float32))
        return mod

    def dequantize_weight(self, dtype: torch.dtype) -> torch.Tensor:
        low = (self.packed_qweight & 0x0F).to(torch.int16) - 8
        high = ((self.packed_qweight >> 4) & 0x0F).to(torch.int16) - 8
        unpacked = torch.stack((low, high), dim=-1).view(self.out_features, self.padded_in)
        grouped = unpacked.view(self.out_features, self.num_groups, self.group_size).to(torch.float32)
        deq = (grouped * self.scales.to(torch.float32).unsqueeze(-1)).view(self.out_features, self.padded_in)
        return deq[:, : self.in_features].to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize_weight(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)

    def storage_bytes(self) -> int:
        total = self.packed_qweight.numel() * self.packed_qweight.element_size()
        total += self.scales.numel() * self.scales.element_size()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return total


def compute_module_storage_bytes(model: nn.Module) -> int:
    total = 0
    seen: set[int] = set()
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            total += p.numel() * p.element_size()
    for b in model.buffers():
        if b.data_ptr() not in seen:
            seen.add(b.data_ptr())
            total += b.numel() * b.element_size()
    return total


def quantize_model_non_destructive(
    model: nn.Module,
    mode: QuantMode,
    *,
    int4_group_size: int = 64,
    keep_lm_head_fp: bool = True,
) -> nn.Module:
    """
    Create a deep-copied model with internal `nn.Linear` blocks converted to
    `QuantizedLinearInt8` or `QuantizedLinearInt4`, leaving the original `model` untouched!
    """
    if mode == "none":
        return model

    cloned = copy.deepcopy(model)
    cloned.eval()

    def _replace(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                if keep_lm_head_fp and full_name == "lm_head":
                    continue
                if mode == "int8":
                    setattr(parent, name, QuantizedLinearInt8.from_linear(child))
                elif mode == "int4":
                    setattr(parent, name, QuantizedLinearInt4.from_linear(child, group_size=int4_group_size))
            else:
                _replace(child, full_name)

    _replace(cloned)
    return cloned


@dataclass(frozen=True)
class QuantizationTradeoffReport:
    mode: QuantMode
    baseline_bytes: int
    quantized_bytes: int
    compression_ratio: float
    logit_cosine_similarity: float
    mean_abs_logit_error: float
    baseline_loss: float
    quantized_loss: float
    perplexity_delta: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@torch.no_grad()
def evaluate_quantization_quality(
    baseline_model: nn.Module,
    quantized_model: nn.Module,
    eval_input_ids: torch.Tensor,
    *,
    mode: QuantMode,
) -> QuantizationTradeoffReport:
    """
    Compare full-precision vs quantized model memory footprint, logit fidelity,
    and cross-entropy / perplexity drift on `eval_input_ids` (`[B, S]`).
    """
    baseline_model.eval()
    quantized_model.eval()

    labels = eval_input_ids.clone()
    base_out = baseline_model(input_ids=eval_input_ids, labels=labels)
    quant_out = quantized_model(input_ids=eval_input_ids, labels=labels)

    b_logits = base_out.logits.to(torch.float32).reshape(-1, base_out.logits.size(-1))
    q_logits = quant_out.logits.to(torch.float32).reshape(-1, quant_out.logits.size(-1))

    cos_sim = float(F.cosine_similarity(b_logits, q_logits, dim=-1).mean().item())
    mae = float((b_logits - q_logits).abs().mean().item())

    b_loss = float(base_out.loss.item()) if base_out.loss is not None else 0.0
    q_loss = float(quant_out.loss.item()) if quant_out.loss is not None else 0.0
    b_ppl = math.exp(min(20.0, b_loss))
    q_ppl = math.exp(min(20.0, q_loss))

    b_bytes = compute_module_storage_bytes(baseline_model)
    q_bytes = compute_module_storage_bytes(quantized_model)

    return QuantizationTradeoffReport(
        mode=mode,
        baseline_bytes=b_bytes,
        quantized_bytes=q_bytes,
        compression_ratio=round(float(b_bytes) / max(1.0, float(q_bytes)), 3),
        logit_cosine_similarity=round(cos_sim, 6),
        mean_abs_logit_error=round(mae, 6),
        baseline_loss=round(b_loss, 6),
        quantized_loss=round(q_loss, 6),
        perplexity_delta=round(q_ppl - b_ppl, 6),
    )
