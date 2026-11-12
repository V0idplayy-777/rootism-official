"""Rootism Independent Tokenizer Package."""

from rootism.tokenizer.bpe import (
    RootismTokenizer,
    TokenizerCompatibilityError,
    TokenizerVersionLock,
)
from rootism.tokenizer.evaluator import (
    TokenizerEvaluationReport,
    TokenizerEvaluator,
    compare_vocabulary_sizes,
)
from rootism.tokenizer.pretokenizer import (
    BYTES_TO_UNICODE,
    MULTI_CHAR_OPERATORS,
    UNICODE_TO_BYTES,
    CodeAwarePretokenizer,
)
from rootism.tokenizer.special_tokens import (
    CORE_SPECIAL_TOKEN_SPECS,
    RESERVED_SPECIAL_TOKEN_SLOTS,
    SPECIAL_TOKENS_BY_ID,
    SPECIAL_TOKENS_BY_SURFACE,
    SpecialTokenSpec,
    build_special_token_table,
)
from rootism.tokenizer.trainer import RootismTokenizerTrainer, TokenizerTrainingReport

__all__ = [
    "RootismTokenizer",
    "TokenizerCompatibilityError",
    "TokenizerVersionLock",
    "TokenizerEvaluationReport",
    "TokenizerEvaluator",
    "compare_vocabulary_sizes",
    "BYTES_TO_UNICODE",
    "UNICODE_TO_BYTES",
    "MULTI_CHAR_OPERATORS",
    "CodeAwarePretokenizer",
    "CORE_SPECIAL_TOKEN_SPECS",
    "RESERVED_SPECIAL_TOKEN_SLOTS",
    "SPECIAL_TOKENS_BY_ID",
    "SPECIAL_TOKENS_BY_SURFACE",
    "SpecialTokenSpec",
    "build_special_token_table",
    "RootismTokenizerTrainer",
    "TokenizerTrainingReport",
]
