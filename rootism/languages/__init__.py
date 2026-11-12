"""Rootism Programming Language & Ecosystem Engineering Package."""

from rootism.languages.allocation import (
    CODE_LANGUAGE_SHARES,
    MODALITY_SHARES,
    LanguageCodeAllocation,
    ModalityAllocation,
    compute_training_token_allocation,
)
from rootism.languages.cross_language import (
    SEMANTIC_DIVERGENCE_MATRIX,
    CrossLanguageSemanticDivergence,
    query_semantic_divergences,
)
from rootism.languages.frameworks import (
    FRAMEWORKS_BY_SLUG,
    SUPPORTED_FRAMEWORKS,
    FrameworkSpec,
    detect_repository_frameworks,
)
from rootism.languages.registry import (
    ALL_SUPPORTED_LANGUAGES,
    EXTENSION_TO_LANGUAGE,
    PRIMARY_LANGUAGES,
    SECONDARY_LANGUAGES,
    LanguageScopeAssessment,
    LanguageSpec,
    SyntaxCheckResult,
    check_balanced_delimiters,
    classify_language_scope,
    validate_code_syntax,
)

__all__ = [
    "PRIMARY_LANGUAGES",
    "SECONDARY_LANGUAGES",
    "ALL_SUPPORTED_LANGUAGES",
    "EXTENSION_TO_LANGUAGE",
    "LanguageSpec",
    "LanguageScopeAssessment",
    "SyntaxCheckResult",
    "classify_language_scope",
    "check_balanced_delimiters",
    "validate_code_syntax",
    "SUPPORTED_FRAMEWORKS",
    "FRAMEWORKS_BY_SLUG",
    "FrameworkSpec",
    "detect_repository_frameworks",
    "SEMANTIC_DIVERGENCE_MATRIX",
    "CrossLanguageSemanticDivergence",
    "query_semantic_divergences",
    "MODALITY_SHARES",
    "CODE_LANGUAGE_SHARES",
    "ModalityAllocation",
    "LanguageCodeAllocation",
    "compute_training_token_allocation",
]
