"""
Benchmark Decontamination Engine (Step 43).

Prevents evaluation benchmark leakage into training shards by maintaining an isolated
index of held-out benchmark items (canonical SHA-256, normalized word n-grams, and
normalized 50-character code spans) and quarantining any overlapping training sample.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9_]+")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class BenchmarkItem:
    benchmark_name: str
    item_id: str
    prompt_text: str
    canonical_solution: str = ""


@dataclass(frozen=True)
class ContaminationCheckResult:
    contaminated: bool
    matched_benchmark: str | None
    matched_item_id: str | None
    match_type: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DecontaminationEngine:
    """
    Indexes evaluation benchmarks and enforces strict train/eval separation.
    """

    def __init__(self, *, word_ngram_size: int = 13, code_char_span: int = 50) -> None:
        self.word_ngram_size = word_ngram_size
        self.code_char_span = code_char_span

        self._exact_hashes: dict[str, tuple[str, str]] = {}
        self._word_ngrams: dict[bytes, tuple[str, str]] = {}
        self._code_spans: dict[bytes, tuple[str, str]] = {}
        self.total_checked: int = 0
        self.total_quarantined: int = 0

    @staticmethod
    def _normalize_words(text: str) -> list[str]:
        return _WORD_RE.findall(text.lower())

    @staticmethod
    def _normalize_code_chars(text: str) -> str:
        return _WS_RE.sub("", text).lower()

    def register_benchmark_items(self, items: list[BenchmarkItem]) -> None:
        for item in items:
            tag = (item.benchmark_name, item.item_id)
            for segment in (item.prompt_text, item.canonical_solution, f"{item.prompt_text}\n{item.canonical_solution}"):
                clean = segment.strip()
                if not clean:
                    continue
                sha = hashlib.sha256(clean.encode("utf-8")).hexdigest()
                self._exact_hashes[sha] = tag

                words = self._normalize_words(clean)
                if len(words) >= self.word_ngram_size:
                    for i in range(len(words) - self.word_ngram_size + 1):
                        gram = " ".join(words[i : i + self.word_ngram_size]).encode("utf-8")
                        digest = hashlib.blake2b(gram, digest_size=12).digest()
                        self._word_ngrams.setdefault(digest, tag)

                norm_chars = self._normalize_code_chars(clean)
                if len(norm_chars) >= self.code_char_span:
                    # Index strided 50-char spans covering canonical prompts/solutions
                    stride = max(1, self.code_char_span // 4)
                    for i in range(0, len(norm_chars) - self.code_char_span + 1, stride):
                        span = norm_chars[i : i + self.code_char_span].encode("utf-8")
                        digest = hashlib.blake2b(span, digest_size=12).digest()
                        self._code_spans.setdefault(digest, tag)

    def check_document(self, text: str) -> ContaminationCheckResult:
        self.total_checked += 1
        clean = text.strip()
        if not clean:
            return ContaminationCheckResult(False, None, None, None)

        sha = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        if sha in self._exact_hashes:
            bm, iid = self._exact_hashes[sha]
            self.total_quarantined += 1
            return ContaminationCheckResult(True, bm, iid, "EXACT_BENCHMARK_HASH")

        words = self._normalize_words(clean)
        if len(words) >= self.word_ngram_size and self._word_ngrams:
            for i in range(len(words) - self.word_ngram_size + 1):
                gram = " ".join(words[i : i + self.word_ngram_size]).encode("utf-8")
                digest = hashlib.blake2b(gram, digest_size=12).digest()
                hit = self._word_ngrams.get(digest)
                if hit is not None:
                    self.total_quarantined += 1
                    return ContaminationCheckResult(True, hit[0], hit[1], f"{self.word_ngram_size}_WORD_NGRAM_LEAK")

        norm_chars = self._normalize_code_chars(clean)
        if len(norm_chars) >= self.code_char_span and self._code_spans:
            for i in range(len(norm_chars) - self.code_char_span + 1):
                span = norm_chars[i : i + self.code_char_span].encode("utf-8")
                digest = hashlib.blake2b(span, digest_size=12).digest()
                hit = self._code_spans.get(digest)
                if hit is not None:
                    self.total_quarantined += 1
                    return ContaminationCheckResult(True, hit[0], hit[1], f"{self.code_char_span}_CHAR_CODE_SPAN_LEAK")

        return ContaminationCheckResult(False, None, None, None)

    def audit_summary(self) -> dict[str, Any]:
        return {
            "word_ngram_size": self.word_ngram_size,
            "code_char_span": self.code_char_span,
            "indexed_exact_hashes": len(self._exact_hashes),
            "indexed_word_ngrams": len(self._word_ngrams),
            "indexed_code_spans": len(self._code_spans),
            "total_documents_checked": self.total_checked,
            "total_documents_quarantined": self.total_quarantined,
        }
