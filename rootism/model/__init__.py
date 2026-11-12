"""Rootism Neural Network Architecture Package."""

from rootism.model.attention import KVCacheLayerState, RootismAttention, build_packed_causal_mask
from rootism.model.ffn import RootismSwiGLU
from rootism.model.inspector import (
    ParameterAuditReport,
    evaluate_architecture_candidates,
    inspect_model_parameters,
)
from rootism.model.norms import RootismRMSNorm
from rootism.model.rope import RootismRotaryEmbedding, apply_rotary_pos_emb, rotate_half
from rootism.model.transformer import (
    IGNORE_INDEX,
    CausalLMOutput,
    RootismDecoderLayer,
    RootismForCausalLM,
    RootismModel,
)

__all__ = [
    "IGNORE_INDEX",
    "KVCacheLayerState",
    "RootismAttention",
    "build_packed_causal_mask",
    "RootismSwiGLU",
    "RootismRMSNorm",
    "RootismRotaryEmbedding",
    "apply_rotary_pos_emb",
    "rotate_half",
    "CausalLMOutput",
    "RootismDecoderLayer",
    "RootismModel",
    "RootismForCausalLM",
    "ParameterAuditReport",
    "inspect_model_parameters",
    "evaluate_architecture_candidates",
]
