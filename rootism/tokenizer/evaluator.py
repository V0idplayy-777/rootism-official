"""
Rootism Tokenizer Empirical Evaluation Suite (Steps 23 & 27).

Measures:
- Compression efficiency (UTF-8 bytes per token & characters per token)
- Average tokens per source-code file and per line
- Identifier fragmentation across naming conventions (snake_case, camelCase, PascalCase, SCREAMING_SNAKE)
- Indentation efficiency (tokens per 4-space / 8-space indented block)
- Unicode handling (lossless roundtrip across scripts, math symbols, emoji, box-drawing)
- Natural-language tokenization efficiency
- Cross-language consistency (per-language bytes/token + coefficient of variation)
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from rootism.config import RootismTokenizerConfig
from rootism.tokenizer.bpe import RootismTokenizer
from rootism.tokenizer.trainer import RootismTokenizerTrainer

_IDENTIFIER_REGEX = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")


@dataclass(frozen=True)
class TokenizerEvaluationReport:
    """Comprehensive empirical evaluation results for a `RootismTokenizer`."""

    tokenizer_fingerprint: str
    vocab_size: int
    overall_bytes_per_token: float
    overall_chars_per_token: float
    avg_tokens_per_code_file: float
    avg_tokens_per_code_line: float
    avg_tokens_per_identifier: float
    indentation_tokens_for_4_spaces: int
    indentation_tokens_for_newline_8_spaces: int
    natural_language_bytes_per_token: float
    unicode_roundtrip_pass_rate: float
    unicode_bytes_per_token: float
    per_language_bytes_per_token: dict[str, float]
    cross_language_consistency_cv: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_UNICODE_PROBES: tuple[str, ...] = (
    "∀x ∈ ℝ, ∃ε > 0 : ‖∇f(x)‖₂ ≤ ε — λ-calculus & ∑ᵢ₌₁ⁿ xᵢ²",
    "日本語のコメント: 配列の境界チェックを実行する",
    "한국어 주석: 비동기 작업 큐 초기화",
    "Ελληνικά μαθηματικά σύμβολα: α, β, γ, Δ, Ω",
    "Path: C:\\Users\\dev\\проект\\src\\🚀_deploy.py\n\treturn '✓ ok'",
    "Box tree:\n├── rootism/\n│   ├── model/\n│   └── tokenizer/\n└── tests/",
)


class TokenizerEvaluator:
    """Evaluates a `RootismTokenizer` across code, prose, identifiers, and Unicode."""

    def __init__(self, tokenizer: RootismTokenizer) -> None:
        self.tokenizer = tokenizer

    def evaluate(
        self,
        code_samples_by_language: dict[str, list[str]],
        natural_language_samples: list[str],
        unicode_probes: tuple[str, ...] = DEFAULT_UNICODE_PROBES,
    ) -> TokenizerEvaluationReport:
        total_bytes = 0
        total_chars = 0
        total_tokens = 0

        code_file_token_counts: list[int] = []
        code_line_token_counts: list[int] = []
        identifier_token_counts: list[int] = []
        per_lang_bpt: dict[str, float] = {}

        for lang, files in sorted(code_samples_by_language.items()):
            lang_bytes = 0
            lang_tokens = 0
            for content in files:
                if not content:
                    continue
                b_len = len(content.encode("utf-8"))
                toks = self.tokenizer.encode(content, allow_special=False)
                n_toks = max(1, len(toks))

                total_bytes += b_len
                total_chars += len(content)
                total_tokens += n_toks
                lang_bytes += b_len
                lang_tokens += n_toks

                code_file_token_counts.append(len(toks))
                n_lines = max(1, content.count("\n") + 1)
                code_line_token_counts.append(len(toks) / n_lines)

                for match in _IDENTIFIER_REGEX.findall(content)[:64]:
                    id_toks = self.tokenizer.encode(match, allow_special=False)
                    identifier_token_counts.append(len(id_toks))

            if lang_tokens > 0:
                per_lang_bpt[lang] = round(lang_bytes / lang_tokens, 4)

        nl_bytes = 0
        nl_tokens = 0
        for prose in natural_language_samples:
            if not prose:
                continue
            b_len = len(prose.encode("utf-8"))
            toks = self.tokenizer.encode(prose, allow_special=False)
            n_toks = max(1, len(toks))
            nl_bytes += b_len
            nl_tokens += n_toks
            total_bytes += b_len
            total_chars += len(prose)
            total_tokens += n_toks

        uni_passes = 0
        uni_bytes = 0
        uni_tokens = 0
        for probe in unicode_probes:
            toks = self.tokenizer.encode(probe, allow_special=False)
            dec = self.tokenizer.decode(toks, errors="strict")
            if dec == probe:
                uni_passes += 1
            uni_bytes += len(probe.encode("utf-8"))
            uni_tokens += max(1, len(toks))

        indent_4 = len(self.tokenizer.encode("    ", allow_special=False))
        indent_nl8 = len(self.tokenizer.encode("\n        ", allow_special=False))

        lang_values = list(per_lang_bpt.values())
        if len(lang_values) >= 2:
            mean_v = sum(lang_values) / len(lang_values)
            var_v = sum((v - mean_v) ** 2 for v in lang_values) / len(lang_values)
            cv = math.sqrt(var_v) / max(mean_v, 1e-9)
        else:
            cv = 0.0

        return TokenizerEvaluationReport(
            tokenizer_fingerprint=self.tokenizer.fingerprint,
            vocab_size=self.tokenizer.vocab_size,
            overall_bytes_per_token=round(total_bytes / max(1, total_tokens), 4),
            overall_chars_per_token=round(total_chars / max(1, total_tokens), 4),
            avg_tokens_per_code_file=round(
                sum(code_file_token_counts) / max(1, len(code_file_token_counts)), 2
            ),
            avg_tokens_per_code_line=round(
                sum(code_line_token_counts) / max(1, len(code_line_token_counts)), 2
            ),
            avg_tokens_per_identifier=round(
                sum(identifier_token_counts) / max(1, len(identifier_token_counts)), 4
            ),
            indentation_tokens_for_4_spaces=indent_4,
            indentation_tokens_for_newline_8_spaces=indent_nl8,
            natural_language_bytes_per_token=round(nl_bytes / max(1, nl_tokens), 4),
            unicode_roundtrip_pass_rate=round(uni_passes / max(1, len(unicode_probes)), 4),
            unicode_bytes_per_token=round(uni_bytes / max(1, uni_tokens), 4),
            per_language_bytes_per_token=per_lang_bpt,
            cross_language_consistency_cv=round(cv, 4),
        )


def compare_vocabulary_sizes(
    training_corpus: list[str],
    code_eval: dict[str, list[str]],
    nl_eval: list[str],
    candidate_vocab_sizes: tuple[int, ...] = (512, 1024, 2048),
) -> list[dict[str, Any]]:
    """
    Train and benchmark multiple vocabulary budgets on the same corpus to quantify
    compression vs embedding parameter trade-offs empirically (Step 23).
    """
    results: list[dict[str, Any]] = []
    for vs in candidate_vocab_sizes:
        cfg = RootismTokenizerConfig(vocab_size=vs)
        tok, train_rep = RootismTokenizerTrainer(cfg).train_from_iterator(training_corpus)
        eval_rep = TokenizerEvaluator(tok).evaluate(code_eval, nl_eval)
        results.append(
            {
                "vocab_size": vs,
                "learned_merges": train_rep.learned_merges,
                "overall_bytes_per_token": eval_rep.overall_bytes_per_token,
                "avg_tokens_per_identifier": eval_rep.avg_tokens_per_identifier,
                "cross_language_consistency_cv": eval_rep.cross_language_consistency_cv,
            }
        )
    return results
