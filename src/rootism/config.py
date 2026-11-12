"""Rootism configuration objects.

All persisted artifacts are built from these dataclasses. `config_id` is a
content hash of the canonical serialized config, so changing any architectural
value changes the identity of the model and invalidates old checkpoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

from .utils.hashing import json_load, json_save
from .version import (
    MODEL_ARCHITECTURE_ID,
    SCHEMA_VERSION,
    canonical_json,
    content_hash,
    short_id,
)

# Requirement 05: the training budget must be MORE THAN 600 billion actual tokens.
HARD_MINIMUM_TRAINING_TOKENS = 600_000_000_000
DEFAULT_TRAINING_TOKENS = 650_000_000_000


@dataclass(frozen=True)
class RopeScalingConfig:
    """RoPE context-extension configuration.

    type:
      "none"        plain RoPE, no scaling
      "linear"      position interpolation (positions / factor)
      "dynamic_ntk" dynamic NTK base adjustment beyond the original context
      "yarn"        NTK-by-parts with linear ramp + attention temperature
    """

    type: str = "none"
    factor: float = 1.0
    original_max_position_embeddings: int = 32_768
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    attention_factor: float | None = None

    def validate(self) -> None:
        if self.type not in {"none", "linear", "dynamic_ntk", "yarn"}:
            raise ValueError(f"unsupported RoPE scaling type {self.type!r}")
        if self.factor < 1.0:
            raise ValueError("RoPE scaling factor must be >= 1.0")
        if self.original_max_position_embeddings < 1:
            raise ValueError("original_max_position_embeddings must be positive")
        if self.type == "yarn" and self.beta_fast <= self.beta_slow:
            raise ValueError("yarn beta_fast must be greater than beta_slow")


@dataclass(frozen=True)
class ModelConfig:
    """Decoder-only Rootism architecture.

    The production preset is chosen by the parameter-budget search in
    scripts/rootism_param_search.py and documented in docs/MODEL.md.
    """

    vocab_size: int = 49_152
    hidden_size: int = 3_072
    num_hidden_layers: int = 28
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 8_192
    max_position_embeddings: int = 32_768
    rope_theta: float = 100_000.0
    rope_scaling: RopeScalingConfig = RopeScalingConfig()
    rms_norm_eps: float = 1e-5
    norm_type: str = "rmsnorm"
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02
    depth_scaled_init: bool = True
    qk_norm: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    attention_dropout: float = 0.0
    loss_chunk_size: int = 512
    attention_query_chunk_size: int = 2_048
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    use_cache: bool = True
    # Only for small unit-test configs; production builds MUST NOT set this.
    allow_short_context: bool = False
    model_type: str = "rootism"
    architecture: str = "RootismForCausalLM"
    schema_version: str = SCHEMA_VERSION

    # ---- validation -----------------------------------------------------
    def validate(self) -> None:
        if self.vocab_size < 256:
            raise ValueError("vocab_size must be >= 256")
        if self.hidden_size <= 0 or self.num_hidden_layers <= 0:
            raise ValueError("hidden_size and num_hidden_layers must be positive")
        if self.num_attention_heads <= 0 or self.num_key_value_heads <= 0:
            raise ValueError("head counts must be positive")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads (GQA)")
        if self.head_dim <= 0 or self.head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer")
        if self.num_attention_heads * self.head_dim != self.hidden_size:
            raise ValueError(
                "num_attention_heads * head_dim must equal hidden_size "
                "(keeps o_proj square and the parameter budget exact)"
            )
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.max_position_embeddings < 32_768 and not self.allow_short_context:
            raise ValueError(
                "production Rootism requires an initial context of at least 32768 tokens; "
                "set allow_short_context=True only for small unit-test configs"
            )
        if self.norm_type != "rmsnorm":
            raise ValueError(f"unsupported norm_type {self.norm_type!r}")
        if self.hidden_act != "silu":
            raise ValueError(f"unsupported hidden_act {self.hidden_act!r}")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if self.loss_chunk_size <= 0 or self.attention_query_chunk_size <= 0:
            raise ValueError("chunk sizes must be positive")
        self.rope_scaling.validate()
        if self.rope_scaling.type != "none":
            extended = int(self.rope_scaling.factor * self.rope_scaling.original_max_position_embeddings)
            if extended < self.max_position_embeddings:
                raise ValueError(
                    "RoPE scaling factor does not cover max_position_embeddings: "
                    f"{extended} < {self.max_position_embeddings}"
                )

    # ---- serialization --------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, RopeScalingConfig):
                out[f.name] = {sf.name: getattr(value, sf.name) for sf in fields(value)}
            else:
                out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown ModelConfig fields: {sorted(unknown)}")
        values: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            raw = data[f.name]
            if f.name == "rope_scaling":
                raw = RopeScalingConfig(**dict(raw))
            values[f.name] = raw
        cfg = cls(**values)
        cfg.validate()
        return cfg

    @property
    def config_id(self) -> str:
        return short_id("cfg", content_hash(self.to_dict()), length=16)

    @property
    def architecture_id(self) -> str:
        return MODEL_ARCHITECTURE_ID

    def save(self, path: str | Path) -> None:
        json_save(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        data = json_load(path)
        from .version import assert_schema

        assert_schema(data, kind="model config")
        return cls.from_dict(data)

    def with_overrides(self, **kwargs: Any) -> "ModelConfig":
        cfg = replace(self, **kwargs)
        cfg.validate()
        return cfg


@dataclass(frozen=True)
class TokenizerConfig:
    """Independent byte-level BPE tokenizer configuration."""

    vocab_size: int = 49_152  # includes special tokens and 256 byte symbols
    min_pair_frequency: int = 2
    max_pretoken_bytes: int = 128
    max_unique_pretokens: int = 2_000_000  # RAM bound for merge learning
    special_tokens_version: int = 1
    normalizer: str = "identity"  # no NFKC: code is byte-sensitive
    add_bos: bool = True
    add_eos: bool = True
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.vocab_size < 256 + 64:
            raise ValueError("vocab_size must leave room for byte alphabet and special tokens")
        if self.min_pair_frequency < 1:
            raise ValueError("min_pair_frequency must be >= 1")
        if self.max_pretoken_bytes < 8:
            raise ValueError("max_pretoken_bytes must be >= 8")
        if self.normalizer != "identity":
            raise ValueError("only identity normalization is supported (code must be byte-exact)")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TokenizerConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown TokenizerConfig fields: {sorted(unknown)}")
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(frozen=True)
class TrainingTokenBudget:
    """Enforces the 650B actual-token training target (requirements 05, 60).

    "Actual tokens" means non-padding, non-masked tokens that contributed to a
    real gradient-based update. Padding and masked tokens never count.
    """

    target_tokens: int = DEFAULT_TRAINING_TOKENS
    hard_minimum_tokens: int = HARD_MINIMUM_TRAINING_TOKENS
    max_epochs: int = 4
    # Fraction of the target that must come from deduplicated, non-repeated
    # tokens. With max_epochs=4 this is 1/4; prevents inflating the counter by
    # looping a tiny corpus.
    min_unique_fraction: float = 0.25

    def validate(self, *, deduplicated_tokens: int | None = None) -> None:
        if self.target_tokens <= self.hard_minimum_tokens:
            raise ValueError(
                f"target_tokens={self.target_tokens:,} is not MORE THAN the "
                f"{self.hard_minimum_tokens:,} token hard minimum (requirement 05)"
            )
        if self.target_tokens < DEFAULT_TRAINING_TOKENS:
            raise ValueError(
                f"target_tokens={self.target_tokens:,} is below the initial 650B target; "
                "lower budgets are only allowed by an explicit, documented override"
            )
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be >= 1")
        if not 0.0 < self.min_unique_fraction <= 1.0:
            raise ValueError("min_unique_fraction must be in (0, 1]")
        if deduplicated_tokens is not None:
            needed = math.ceil(self.target_tokens * self.min_unique_fraction)
            if deduplicated_tokens < needed:
                raise ValueError(
                    f"deduplicated corpus has {deduplicated_tokens:,} tokens but the budget needs "
                    f"at least {needed:,} unique deduplicated tokens "
                    f"(min_unique_fraction={self.min_unique_fraction})"
                )

    @property
    def minimum_unique_tokens(self) -> int:
        return math.ceil(self.target_tokens * self.min_unique_fraction)


def tiny_test_config(**kwargs: Any) -> ModelConfig:
    """Small config used by unit tests. Intentionally NOT production-valid."""
    base = dict(
        vocab_size=257,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        max_position_embeddings=128,
        rope_theta=10_000.0,
        rope_scaling=RopeScalingConfig(original_max_position_embeddings=128),
        loss_chunk_size=32,
        attention_query_chunk_size=64,
        allow_short_context=True,
    )
    base.update(kwargs)
    cfg = ModelConfig(**base)
    cfg.validate()
    return cfg


def canonical_config_json(cfg: ModelConfig) -> str:
    return canonical_json(cfg.to_dict())
