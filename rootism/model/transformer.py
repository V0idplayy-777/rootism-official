"""
Rootism Core Autoregressive Decoder-Only Transformer (Steps 02, 03, 12, 18, 19).

Independently engineered neural architecture (`RootismForCausalLM`):
- Zero external pretrained dependencies; strict random initialization from scratch.
- Pre-RMSNorm residual highways with 1 / sqrt(2 * n_layers) output projection scaling.
- 5:1 Grouped-Query Attention with QK-RMSNorm, RoPE + YaRN scaling, and KV-cache support.
- SwiGLU gated feed-forward layers.
- Memory-exact chunked cross-entropy loss (`checkpoint` per vocabulary projection chunk)
  eliminating multi-gigabyte `[B, S, V]` logit spikes on 32K–64K sequence training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from rootism import __architecture_family__, __version__
from rootism.config import RootismModelConfig
from rootism.core.provenance import ROOTISM_ORIGIN_SIGNATURE, assert_independent_environment
from rootism.model.attention import KVCacheLayerState, RootismAttention
from rootism.model.ffn import RootismSwiGLU
from rootism.model.norms import RootismRMSNorm
from rootism.model.rope import RootismRotaryEmbedding

IGNORE_INDEX: int = -100


@dataclass
class CausalLMOutput:
    """Structured output of `RootismForCausalLM`."""

    loss: torch.Tensor | None
    logits: torch.Tensor | None
    past_key_values: list[KVCacheLayerState] | None
    active_loss_tokens: int


class RootismDecoderLayer(nn.Module):
    """Single pre-norm Transformer block: Attention + Residual -> SwiGLU + Residual."""

    def __init__(self, config: RootismModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RootismRMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.self_attn = RootismAttention(config=config, layer_idx=layer_idx)
        self.ffn_norm = RootismRMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mlp = RootismSwiGLU(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        past_key_value: KVCacheLayerState | None = None,
        use_cache: bool = False,
        force_math_reference: bool = False,
    ) -> tuple[torch.Tensor, KVCacheLayerState | None]:
        residual = hidden_states
        normed_attn_in = self.attn_norm(hidden_states)
        attn_out, next_kv = self.self_attn(
            normed_attn_in,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            document_ids=document_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            force_math_reference=force_math_reference,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        normed_ffn_in = self.ffn_norm(hidden_states)
        ffn_out = self.mlp(normed_ffn_in)
        hidden_states = residual + ffn_out

        return hidden_states, next_kv


class RootismModel(nn.Module):
    """Backbone stack of embeddings, RoPE, 40 RootismDecoderLayers, and final RMSNorm."""

    def __init__(self, config: RootismModelConfig) -> None:
        super().__init__()
        assert_independent_environment()
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model

        self.tok_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=None,  # Explicit loss masking handles pad tokens without zeroing grad norms silently
        )
        self.rotary_emb = RootismRotaryEmbedding(
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            scaling=config.rope_scaling,
        )
        self.layers = nn.ModuleList(
            [RootismDecoderLayer(config, layer_idx=i) for i in range(config.n_layers)]
        )
        self.norm = RootismRMSNorm(config.d_model, eps=config.rms_norm_eps)

        self._checkpoint_layer_flags: list[bool] = [False] * config.n_layers

    def configure_activation_checkpointing(
        self,
        mode: Literal["none", "selective", "full"] = "none",
        every_n_layers: int = 2,
    ) -> None:
        """Enable full or selective layer-wise activation checkpointing."""
        if mode == "none":
            self._checkpoint_layer_flags = [False] * len(self.layers)
        elif mode == "full":
            self._checkpoint_layer_flags = [True] * len(self.layers)
        elif mode == "selective":
            if every_n_layers < 1:
                raise ValueError("every_n_layers must be >= 1")
            self._checkpoint_layer_flags = [
                (idx % every_n_layers == 0) for idx in range(len(self.layers))
            ]
        else:
            raise ValueError(f"Unknown activation checkpointing mode: {mode!r}")

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        past_key_values: list[KVCacheLayerState] | None = None,
        use_cache: bool = False,
        force_math_reference: bool = False,
    ) -> tuple[torch.Tensor, list[KVCacheLayerState] | None]:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [batch, seq_len], got {tuple(input_ids.shape)}")
        bsz, seq_len = input_ids.shape

        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            if len(past_key_values) != len(self.layers):
                raise ValueError(
                    f"past_key_values length ({len(past_key_values)}) must match n_layers ({len(self.layers)})"
                )
            past_len = past_key_values[0].seq_length

        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + seq_len, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(bsz, seq_len)

        hidden_states = self.tok_embeddings(input_ids)
        cos, sin = self.rotary_emb(position_ids)

        next_cache_list: list[KVCacheLayerState] | None = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            use_ckpt = (
                self.training
                and not use_cache
                and layer_past is None
                and self._checkpoint_layer_flags[idx]
                and hidden_states.requires_grad
            )
            if use_ckpt:

                def _ckpt_forward(
                    h: torch.Tensor,
                    c: torch.Tensor,
                    s: torch.Tensor,
                    am: torch.Tensor | None,
                    dids: torch.Tensor | None,
                    _layer: RootismDecoderLayer = layer,
                ) -> torch.Tensor:
                    out_h, _ = _layer(
                        h,
                        cos=c,
                        sin=s,
                        attention_mask=am,
                        document_ids=dids,
                        past_key_value=None,
                        use_cache=False,
                        force_math_reference=force_math_reference,
                    )
                    return out_h

                hidden_states = torch_checkpoint(
                    _ckpt_forward,
                    hidden_states,
                    cos,
                    sin,
                    attention_mask,
                    document_ids,
                    use_reentrant=False,
                )
                next_kv = None
            else:
                hidden_states, next_kv = layer(
                    hidden_states,
                    cos=cos,
                    sin=sin,
                    attention_mask=attention_mask,
                    document_ids=document_ids,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                    force_math_reference=force_math_reference,
                )

            if use_cache and next_cache_list is not None and next_kv is not None:
                next_cache_list.append(next_kv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, next_cache_list


def _chunk_cross_entropy_sum(
    hidden_chunk: torch.Tensor,
    label_chunk: torch.Tensor,
    lm_weight: torch.Tensor,
    lm_bias: torch.Tensor | None,
) -> torch.Tensor:
    logits_chunk = F.linear(hidden_chunk, lm_weight, lm_bias).to(torch.float32)
    return F.cross_entropy(logits_chunk, label_chunk, ignore_index=IGNORE_INDEX, reduction="sum")


class RootismForCausalLM(nn.Module):
    """
    Complete Rootism Causal Language Model with independent scratch initialization,
    depth-scaled residual projections, and memory-efficient chunked cross-entropy.
    """

    def __init__(self, config: RootismModelConfig) -> None:
        super().__init__()
        assert_independent_environment()
        self.config = config
        self.model = RootismModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.tok_embeddings.weight

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize all parameters strictly from random distributions:
        - Standard projections & embeddings: N(0, init_std)
        - Residual branch output projections (`o_proj`, `down_proj`):
          N(0, init_std / sqrt(2 * n_layers)) to bound deep residual stream variance across 40 layers.
        - RMSNorm scales: 1.0
        """
        base_std = self.config.init_std
        residual_std = (
            base_std / math.sqrt(2.0 * float(self.config.n_layers))
            if self.config.residual_depth_scaling
            else base_std
        )

        for name, module in self.named_modules():
            if isinstance(module, RootismRMSNorm):
                module.reset_parameters()
            elif isinstance(module, nn.Embedding):
                if not module.weight.is_meta:
                    nn.init.normal_(module.weight, mean=0.0, std=base_std)
            elif isinstance(module, nn.Linear):
                is_residual_exit = name.endswith("self_attn.o_proj") or name.endswith("mlp.down_proj")
                std = residual_std if is_residual_exit else base_std
                if not module.weight.is_meta:
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None and not module.bias.is_meta:
                    nn.init.zeros_(module.bias)

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.tok_embeddings.weight

    def configure_activation_checkpointing(
        self,
        mode: Literal["none", "selective", "full"] = "none",
        every_n_layers: int = 2,
    ) -> None:
        self.model.configure_activation_checkpointing(mode=mode, every_n_layers=every_n_layers)

    def provenance_metadata(self) -> dict[str, str]:
        """Return checkpoint header metadata attesting to Rootism scratch provenance."""
        return {
            "origin_signature": ROOTISM_ORIGIN_SIGNATURE,
            "architecture_family": __architecture_family__,
            "software_version": __version__,
            "initialization_origin": self.config.initialization_origin,
            "model_config_sha256": self.config.config_hash(),
            "tokenizer_fingerprint_sha256": self.config.tokenizer_version_fingerprint or "",
        }

    def compute_chunked_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        loss_chunk_size: int = 2_048,
    ) -> tuple[torch.Tensor, int]:
        """
        Compute exact causal next-token cross-entropy in sequence chunks so peak VRAM
        is O(loss_chunk_size * vocab_size) rather than O(batch * seq_len * vocab_size).
        """
        shift_hidden = hidden_states[:, :-1, :].contiguous().view(-1, self.config.d_model)
        shift_labels = labels[:, 1:].contiguous().view(-1)

        valid_mask = shift_labels != IGNORE_INDEX
        valid_count_tensor = valid_mask.sum()
        active_tokens = int(valid_count_tensor.item())

        if active_tokens == 0:
            # Zero-loss connected to graph so distributed backward does not hang
            zero_loss = (shift_hidden[:1].sum() * 0.0) + (self.lm_head.weight[:1, :1].sum() * 0.0)
            return zero_loss.to(torch.float32), 0

        total_n = shift_hidden.shape[0]
        losses: list[torch.Tensor] = []
        use_ckpt = self.training and (shift_hidden.requires_grad or self.lm_head.weight.requires_grad)

        for start in range(0, total_n, loss_chunk_size):
            end = min(start + loss_chunk_size, total_n)
            lbl_chunk = shift_labels[start:end]
            if not bool((lbl_chunk != IGNORE_INDEX).any()):
                continue
            h_chunk = shift_hidden[start:end]
            if use_ckpt:
                chunk_sum = torch_checkpoint(
                    _chunk_cross_entropy_sum,
                    h_chunk,
                    lbl_chunk,
                    self.lm_head.weight,
                    self.lm_head.bias,
                    use_reentrant=False,
                )
            else:
                chunk_sum = _chunk_cross_entropy_sum(
                    h_chunk, lbl_chunk, self.lm_head.weight, self.lm_head.bias
                )
            losses.append(chunk_sum)

        total_loss_sum = torch.stack(losses).sum()
        mean_loss = total_loss_sum / valid_count_tensor.to(torch.float32)
        return mean_loss, active_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: list[KVCacheLayerState] | None = None,
        use_cache: bool = False,
        return_logits: bool = True,
        loss_chunk_size: int = 2_048,
        force_math_reference: bool = False,
    ) -> CausalLMOutput:
        hidden_states, next_kv = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            document_ids=document_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            force_math_reference=force_math_reference,
        )

        loss: torch.Tensor | None = None
        active_tokens = 0
        logits: torch.Tensor | None = None

        if labels is not None and not return_logits:
            loss, active_tokens = self.compute_chunked_loss(
                hidden_states=hidden_states,
                labels=labels,
                loss_chunk_size=loss_chunk_size,
            )
        else:
            logits = self.lm_head(hidden_states)
            if labels is not None:
                shift_logits = logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size).to(torch.float32)
                shift_labels = labels[:, 1:].contiguous().view(-1)
                active_tokens = int((shift_labels != IGNORE_INDEX).sum().item())
                if active_tokens > 0:
                    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX)
                else:
                    loss = shift_logits[:1, :1].sum() * 0.0

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=next_kv,
            active_loss_tokens=active_tokens,
        )
