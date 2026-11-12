"""
Centralised, strongly validated configuration schemas for all Rootism subsystems.

Designed so model architecture, tokenizer, data pipeline, distributed training,
inference engine, coding agent, and serving layer share one canonical contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

MINIMUM_PRODUCTION_TRAINING_TOKENS: int = 600_000_000_000  # >600B floor (strictly >600B checked below)
DEFAULT_PRODUCTION_TRAINING_TOKENS: int = 650_000_000_000  # 650B initial production target
TARGET_PARAMETER_COUNT: int = 3_000_000_000
PRODUCTION_PARAMETER_TOLERANCE_RATIO: float = 0.05  # +/- 5% around 3.00B for production configs


PROHIBITED_PRETRAINED_KEYS: frozenset[str] = frozenset(
    {
        "pretrained_model_name_or_path",
        "hf_model_id",
        "huggingface_repo",
        "base_model",
        "teacher_model",
        "distillation_teacher",
        "lora_base_checkpoint",
        "finetune_from_external",
        "pretrained_weights_url",
    }
)


class RopeScalingConfig(BaseModel):
    """Configuration for Rotary Position Embedding context scaling (YaRN / NTK-by-parts)."""

    model_config = ConfigDict(extra="forbid")

    scaling_type: Literal["none", "yarn"] = "none"
    factor: float = Field(default=1.0, ge=1.0, le=16.0)
    original_max_position_embeddings: int = Field(default=32_768, ge=256)
    low_freq_factor: float = Field(default=1.0, gt=0.0)
    high_freq_factor: float = Field(default=4.0, gt=0.0)
    attention_factor: float | None = Field(
        default=None,
        description="Optional override for YaRN attention softmax scale m = 0.1 * ln(s) + 1.0.",
    )

    @model_validator(mode="after")
    def _validate_factors(self) -> "RopeScalingConfig":
        if self.high_freq_factor <= self.low_freq_factor:
            raise ValueError("high_freq_factor must be strictly greater than low_freq_factor")
        if self.scaling_type == "none" and self.factor != 1.0:
            raise ValueError("scaling_type='none' requires factor == 1.0; use scaling_type='yarn' for >1.0")
        return self


class RootismModelConfig(BaseModel):
    """
    Canonical Rootism Decoder-Only Transformer Architecture Configuration.

    Default values define the production `Rootism-3B` model:
    - Exact trainable parameters: 3,009,630,720 (~3.0096B)
    - Native context window: 32,768 tokens (extensible to 65,536+ via YaRN)
    - 5:1 Grouped-Query Attention (20 query heads, 4 KV heads, 128 head_dim)
    - 40 pre-norm transformer layers with QK-RMSNorm and bias-free SwiGLU (6,656 intermediate)
    """

    model_config = ConfigDict(extra="forbid")

    architecture_id: str = "rootism-decoder-v1"
    model_name: str = "Rootism-3B"
    verification_mode: bool = Field(
        default=False,
        description="Explicitly marks non-production tiny test models; never allowed in production runs.",
    )

    vocab_size: int = Field(default=65_536, ge=512, le=262_144)
    d_model: int = Field(default=2_560, ge=64)
    n_layers: int = Field(default=40, ge=1)
    n_heads: int = Field(default=20, ge=1)
    n_kv_heads: int = Field(default=4, ge=1)
    head_dim: int = Field(default=128, ge=16)
    intermediate_size: int = Field(default=6_656, ge=64)

    rms_norm_eps: float = Field(default=1e-6, gt=0.0)
    qk_norm: bool = Field(default=True, description="Per-head RMSNorm on Q and K for BF16/FP16 stability.")
    attention_bias: bool = Field(default=False)
    mlp_bias: bool = Field(default=False)
    tie_word_embeddings: bool = Field(default=False)
    attention_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    max_position_embeddings: int = Field(default=32_768, ge=256)
    extended_context_length: int = Field(default=65_536, ge=256)
    rope_theta: float = Field(default=500_000.0, gt=1_000.0)
    rope_scaling: RopeScalingConfig = Field(default_factory=RopeScalingConfig)

    init_std: float = Field(default=0.02, gt=0.0)
    residual_depth_scaling: bool = Field(
        default=True,
        description="Scale output projections (o_proj, down_proj) by 1 / sqrt(2 * n_layers).",
    )

    pad_token_id: int = Field(default=0, ge=0)
    bos_token_id: int = Field(default=1, ge=0)
    eos_token_id: int = Field(default=2, ge=0)
    unk_token_id: int = Field(default=3, ge=0)

    initialization_origin: Literal["random_scratch"] = "random_scratch"
    tokenizer_version_fingerprint: str | None = Field(
        default=None,
        description="SHA-256 fingerprint locking checkpoint weights to an exact RootismTokenizer version.",
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_pretrained_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            forbidden = PROHIBITED_PRETRAINED_KEYS.intersection(data.keys())
            if forbidden:
                raise ValueError(
                    f"Prohibited pretrained/distillation configuration keys detected: {sorted(forbidden)}. "
                    "Rootism must be initialized and trained strictly from random parameters."
                )
        return data

    @model_validator(mode="after")
    def _validate_architecture_invariants(self) -> "RootismModelConfig":
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError(
                f"n_heads ({self.n_heads}) * head_dim ({self.head_dim}) must equal d_model ({self.d_model})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads}) for GQA"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE pair rotation")
        for tok_name in ("pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id"):
            tok_id = getattr(self, tok_name)
            if tok_id >= self.vocab_size:
                raise ValueError(f"{tok_name} ({tok_id}) must be < vocab_size ({self.vocab_size})")

        if not self.verification_mode:
            if self.max_position_embeddings < 32_768:
                raise ValueError(
                    f"Production Rootism requires max_position_embeddings >= 32,768 (got {self.max_position_embeddings})"
                )
            analytical_params = self.analytical_parameter_count()
            lower = int(TARGET_PARAMETER_COUNT * (1.0 - PRODUCTION_PARAMETER_TOLERANCE_RATIO))
            upper = int(TARGET_PARAMETER_COUNT * (1.0 + PRODUCTION_PARAMETER_TOLERANCE_RATIO))
            if not (lower <= analytical_params <= upper):
                raise ValueError(
                    f"Production RootismModelConfig must stay within +/-5% of 3.00B parameters "
                    f"([{lower:,}, {upper:,}]), got {analytical_params:,}. "
                    "Set verification_mode=True only for explicitly non-production unit test fixtures."
                )
        return self

    @property
    def num_key_value_groups(self) -> int:
        return self.n_heads // self.n_kv_heads

    def analytical_parameter_breakdown(self) -> dict[str, int]:
        """Deterministic closed-form parameter count breakdown matching RootismForCausalLM."""
        embed_params = self.vocab_size * self.d_model

        q_params = self.d_model * (self.n_heads * self.head_dim)
        k_params = self.d_model * (self.n_kv_heads * self.head_dim)
        v_params = self.d_model * (self.n_kv_heads * self.head_dim)
        o_params = (self.n_heads * self.head_dim) * self.d_model
        if self.attention_bias:
            q_params += self.n_heads * self.head_dim
            k_params += self.n_kv_heads * self.head_dim
            v_params += self.n_kv_heads * self.head_dim
            o_params += self.d_model
        qk_norm_params = (2 * self.head_dim) if self.qk_norm else 0
        attn_per_layer = q_params + k_params + v_params + o_params + qk_norm_params

        ffn_per_layer = 3 * self.d_model * self.intermediate_size
        if self.mlp_bias:
            ffn_per_layer += 2 * self.intermediate_size + self.d_model

        norm_per_layer = 2 * self.d_model
        final_norm_params = self.d_model

        total_attn = self.n_layers * attn_per_layer
        total_ffn = self.n_layers * ffn_per_layer
        total_norm = self.n_layers * norm_per_layer + final_norm_params
        output_proj = 0 if self.tie_word_embeddings else (self.vocab_size * self.d_model)

        total = embed_params + total_attn + total_ffn + total_norm + output_proj
        return {
            "embedding_parameters": embed_params,
            "attention_parameters": total_attn,
            "feed_forward_parameters": total_ffn,
            "normalisation_parameters": total_norm,
            "output_projection_parameters": output_proj,
            "per_layer_parameters": attn_per_layer + ffn_per_layer + norm_per_layer,
            "total_trainable_parameters": total,
        }

    def analytical_parameter_count(self) -> int:
        return self.analytical_parameter_breakdown()["total_trainable_parameters"]

    def kv_cache_bytes_per_token(self, bytes_per_element: int = 2) -> int:
        """KV cache footprint per sequence token (default BF16/FP16 = 2 bytes)."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * bytes_per_element

    def config_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def tiny_verification(
        cls,
        vocab_size: int = 1024,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        n_kv_heads: int = 2,
        head_dim: int = 32,
        intermediate_size: int = 384,
        max_position_embeddings: int = 2048,
    ) -> "RootismModelConfig":
        """Explicitly tagged non-production tiny config for fast CPU unit/integration verification."""
        return cls(
            model_name="Rootism-Tiny-VerifyOnly",
            verification_mode=True,
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            extended_context_length=max_position_embeddings * 2,
        )


class RootismTokenizerConfig(BaseModel):
    """Configuration for Rootism's independent code-aware byte-level BPE tokenizer."""

    model_config = ConfigDict(extra="forbid")

    tokenizer_name: str = "rootism-bpe-v1"
    semver: str = "1.0.0"
    vocab_size: int = Field(default=65_536, ge=384, le=65_536)
    reserved_special_slots: int = Field(default=64, ge=32, le=256)
    digit_split_mode: Literal["single_digit", "three_digit"] = "single_digit"
    max_indentation_spaces: int = Field(default=16, ge=2, le=32)
    min_pair_frequency: int = Field(default=2, ge=1)
    pretokenizer_version: str = "code-byte-v1"


class MixtureDomainWeight(BaseModel):
    """Weight and metadata for a single training data domain in a curriculum stage."""

    model_config = ConfigDict(extra="forbid")

    domain_id: str
    weight: float = Field(gt=0.0)
    max_epochs: float = Field(default=4.0, gt=0.0)
    description: str = ""


class CurriculumStageConfig(BaseModel):
    """A curriculum stage within the >=600B token training schedule."""

    model_config = ConfigDict(extra="forbid")

    stage_name: str
    token_budget: int = Field(gt=0)
    sequence_length: int = Field(default=4_096, ge=128)
    rope_scaling_factor: float = Field(default=1.0, ge=1.0)
    domains: list[MixtureDomainWeight] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_weights(self) -> "CurriculumStageConfig":
        total_w = sum(d.weight for d in self.domains)
        if abs(total_w - 1.0) > 1e-5:
            raise ValueError(f"Domain weights in stage '{self.stage_name}' must sum to 1.0 (got {total_w:.6f})")
        return self


class RootismDataConfig(BaseModel):
    """Configuration for commercial licensing, quality filtering, deduplication, and sharding."""

    model_config = ConfigDict(extra="forbid")

    provenance_registry_path: str = "configs/datasets/provenance_registry.yaml"
    require_verified_commercial_license: bool = True
    scan_file_headers_for_copyleft: bool = True
    enable_exact_dedup: bool = True
    enable_minhash_dedup: bool = True
    minhash_num_perm: int = Field(default=128, ge=16)
    minhash_jaccard_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    enable_benchmark_decontamination: bool = True
    decontamination_ngram_size: int = Field(default=13, ge=5)
    shard_max_tokens: int = Field(default=100_000_000, ge=1_024)
    curriculum_stages: list[CurriculumStageConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _enforce_commercial_license_flag(self) -> "RootismDataConfig":
        if not self.require_verified_commercial_license:
            raise ValueError(
                "Commercial-use license verification is an absolute Rootism requirement and cannot be disabled."
            )
        return self


class RootismOptimizerConfig(BaseModel):
    """Configuration for decoupled AdamW optimizer and gradient clipping."""

    model_config = ConfigDict(extra="forbid")

    optimizer_type: Literal["adamw", "rootism_fp32_master_adamw"] = "adamw"
    learning_rate: float = Field(default=3.0e-4, gt=0.0)
    min_learning_rate: float = Field(default=3.0e-5, ge=0.0)
    beta1: float = Field(default=0.9, gt=0.0, lt=1.0)
    beta2: float = Field(default=0.95, gt=0.0, lt=1.0)
    eps: float = Field(default=1.0e-8, gt=0.0)
    weight_decay: float = Field(default=0.1, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    prefer_fused: bool = True
    prefer_foreach: bool = True


class RootismMemoryConfig(BaseModel):
    """Precision, activation checkpointing, and memory optimization policy."""

    model_config = ConfigDict(extra="forbid")

    precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    activation_checkpointing: Literal["none", "selective", "full"] = "selective"
    selective_checkpoint_every_n_layers: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    empty_cache_interval_steps: int = Field(default=0, ge=0)


class RootismDistributedConfig(BaseModel):
    """Distributed execution strategy (DDP, FSDP/ZeRO, Tensor & Pipeline Parallelism)."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["auto", "single", "ddp", "fsdp_zero2", "fsdp_zero3", "fsdp_hybrid"] = "auto"
    backend: Literal["auto", "nccl", "gloo"] = "auto"
    tensor_parallel_size: int = Field(default=1, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)
    cpu_offload_params: bool = False
    sync_module_states: bool = True


class RootismTrainingConfig(BaseModel):
    """Top-level training configuration enforcing the >600B token production requirement."""

    model_config = ConfigDict(extra="forbid")

    run_name: str = "rootism-3b-pretrain-650b"
    seed: int = Field(default=20261112)
    deterministic_ops: bool = False
    verification_mode: bool = Field(
        default=False,
        description="Must be False for production training; only True for small-scale pipeline tests.",
    )

    target_training_tokens: int = Field(default=DEFAULT_PRODUCTION_TRAINING_TOKENS, gt=0)
    sequence_length: int = Field(default=4_096, ge=64)
    micro_batch_size: int = Field(default=4, ge=1)
    warmup_tokens: int = Field(default=2_000_000_000, ge=0)

    checkpoint_dir: str = "checkpoints/rootism-3b"
    checkpoint_interval_steps: int = Field(default=1_000, ge=1)
    eval_interval_steps: int = Field(default=1_000, ge=1)
    log_interval_steps: int = Field(default=10, ge=1)

    optimizer: RootismOptimizerConfig = Field(default_factory=RootismOptimizerConfig)
    memory: RootismMemoryConfig = Field(default_factory=RootismMemoryConfig)
    distributed: RootismDistributedConfig = Field(default_factory=RootismDistributedConfig)

    @model_validator(mode="before")
    @classmethod
    def _reject_pretrained_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            forbidden = PROHIBITED_PRETRAINED_KEYS.intersection(data.keys())
            if forbidden:
                raise ValueError(
                    f"Prohibited pretrained/distillation training keys detected: {sorted(forbidden)}"
                )
        return data

    @model_validator(mode="after")
    def _enforce_token_budget(self) -> "RootismTrainingConfig":
        if not self.verification_mode:
            if self.target_training_tokens <= MINIMUM_PRODUCTION_TRAINING_TOKENS:
                raise ValueError(
                    f"Rootism production training requires MORE THAN {MINIMUM_PRODUCTION_TRAINING_TOKENS:,} "
                    f"actual training tokens (configured: {self.target_training_tokens:,}). "
                    f"Default target is {DEFAULT_PRODUCTION_TRAINING_TOKENS:,} tokens."
                )
        return self


class RootismConfig(BaseModel):
    """Unified master configuration combining model, tokenizer, data, and training."""

    model_config = ConfigDict(extra="forbid")

    model: RootismModelConfig = Field(default_factory=RootismModelConfig)
    tokenizer: RootismTokenizerConfig = Field(default_factory=RootismTokenizerConfig)
    data: RootismDataConfig = Field(default_factory=RootismDataConfig)
    training: RootismTrainingConfig = Field(default_factory=RootismTrainingConfig)

    @model_validator(mode="after")
    def _validate_cross_component_coherence(self) -> "RootismConfig":
        if self.model.vocab_size != self.tokenizer.vocab_size:
            raise ValueError(
                f"Model vocab_size ({self.model.vocab_size}) must match tokenizer vocab_size "
                f"({self.tokenizer.vocab_size})"
            )
        if self.model.verification_mode != self.training.verification_mode:
            raise ValueError(
                "model.verification_mode and training.verification_mode must agree to prevent "
                "accidentally running tiny verification models under production configurations or vice versa."
            )
        if self.data.curriculum_stages and not self.training.verification_mode:
            stage_sum = sum(s.token_budget for s in self.data.curriculum_stages)
            if stage_sum != self.training.target_training_tokens:
                raise ValueError(
                    f"Sum of curriculum stage token budgets ({stage_sum:,}) must equal "
                    f"training.target_training_tokens ({self.training.target_training_tokens:,})"
                )
        return self

    def to_yaml(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.model_dump(mode="json"), fh, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RootismConfig":
        with Path(path).open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if isinstance(raw, dict) and "model" not in raw and ("d_model" in raw or "architecture_id" in raw):
            is_verify = bool(raw.get("verification_mode", False))
            vocab_sz = int(raw.get("vocab_size", 65_536))
            train_dict: dict[str, Any] = {"verification_mode": is_verify}
            if is_verify:
                train_dict.update(
                    {
                        "target_training_tokens": 4_096,
                        "warmup_tokens": 512,
                        "sequence_length": 128,
                        "micro_batch_size": 2,
                        "memory": {"gradient_accumulation_steps": 2, "activation_checkpointing": "none"},
                    }
                )
            raw = {
                "model": raw,
                "tokenizer": {"vocab_size": vocab_sz},
                "training": train_dict,
            }
        return cls.model_validate(raw)
