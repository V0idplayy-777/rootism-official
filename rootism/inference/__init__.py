"""Rootism Production Inference & Quantisation Package (Phase IX)."""

from rootism.inference.engine import (
    ContextWindowReport,
    GenerationSamplingParams,
    InferenceGenerationResult,
    InferenceStreamChunk,
    RootismInferenceEngine,
    bootstrap_local_serving_artifacts,
)
from rootism.inference.quantization import (
    QuantizedLinearInt4,
    QuantizedLinearInt8,
    QuantizationTradeoffReport,
    QuantMode,
    compute_module_storage_bytes,
    evaluate_quantization_quality,
    quantize_model_non_destructive,
)

__all__ = [
    "ContextWindowReport",
    "GenerationSamplingParams",
    "InferenceGenerationResult",
    "InferenceStreamChunk",
    "RootismInferenceEngine",
    "bootstrap_local_serving_artifacts",
    "QuantMode",
    "QuantizedLinearInt8",
    "QuantizedLinearInt4",
    "QuantizationTradeoffReport",
    "compute_module_storage_bytes",
    "quantize_model_non_destructive",
    "evaluate_quantization_quality",
]
