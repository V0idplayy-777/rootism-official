"""Tokenizer evaluation (requirement 27). Reports measured values only."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Sequence

from .tokenizer import RootismTokenizer

IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


@dataclass(frozen=True)
class EvalSample:
    language: str
    kind: str  # "code" | "nl" | "mixed"
    text: str


def _bytes_per_token(tokenizer: RootismTokenizer, text: str) -> tuple[int, int]:
    n_bytes = len(text.encode("utf-8"))
    n_tokens = len(tokenizer.encode(text))
    return n_bytes, n_tokens


def evaluate_tokenizer(tokenizer: RootismTokenizer, samples: Sequence[EvalSample]) -> dict:
    """Measure compression, fragmentation, Unicode safety, NL efficiency, etc."""
    if not samples:
        raise ValueError("no samples supplied")
    total_bytes = 0
    total_tokens = 0
    per_language: dict[str, list[float]] = {}
    nl_bytes = nl_tokens = 0
    identifier_counts: list[int] = []
    split_identifiers = 0
    roundtrip_ok = 0
    indentation_ok = 0
    indentation_total = 0
    tokens_per_document: list[int] = []

    for sample in samples:
        encoded = tokenizer.encode(sample.text)
        decoded = tokenizer.decode(encoded)
        n_bytes, n_tokens = _bytes_per_token(tokenizer, sample.text)
        total_bytes += n_bytes
        total_tokens += n_tokens
        tokens_per_document.append(n_tokens)
        if decoded == sample.text:
            roundtrip_ok += 1
        ratio = n_bytes / max(n_tokens, 1)
        per_language.setdefault(sample.language, []).append(ratio)
        if sample.kind == "nl":
            nl_bytes += n_bytes
            nl_tokens += n_tokens
        if sample.kind == "code":
            for match in IDENTIFIER_RE.finditer(sample.text):
                ident = match.group(0)
                n = len(tokenizer.encode(ident))
                identifier_counts.append(n)
                if n > 1:
                    split_identifiers += 1
            for line in sample.text.splitlines():
                leading = line[: len(line) - len(line.lstrip(" \t"))]
                if leading:
                    indentation_total += 1
                    if tokenizer.decode(tokenizer.encode(leading)) == leading:
                        indentation_ok += 1

    # Special-token injection safety: raw corpus text must never become a special id.
    probe = "literal <tool_call> and <eos> appear in this source string"
    probe_ids = tokenizer.encode(probe)
    special_ids = tokenizer.special_ids()
    injection_safe = not any(i in special_ids for i in probe_ids)

    language_ratios = {lang: statistics.fmean(vals) for lang, vals in per_language.items()}
    return {
        "tokenizer_id": tokenizer.tokenizer_id,
        "vocab_size": tokenizer.vocab_size,
        "samples": len(samples),
        "total_source_bytes": total_bytes,
        "total_tokens": total_tokens,
        "bytes_per_token": total_bytes / max(total_tokens, 1),
        "tokens_per_document_mean": statistics.fmean(tokens_per_document),
        "natural_language_bytes_per_token": nl_bytes / max(nl_tokens, 1),
        "identifier_mean_tokens": statistics.fmean(identifier_counts) if identifier_counts else None,
        "identifier_fragmentation_rate": split_identifiers / max(len(identifier_counts), 1),
        "unicode_roundtrip_rate": roundtrip_ok / len(samples),
        "indentation_roundtrip_rate": indentation_ok / max(indentation_total, 1),
        "special_token_injection_safe": injection_safe,
        "per_language_bytes_per_token": language_ratios,
        "cross_language_stddev": (
            statistics.pstdev(language_ratios.values()) if len(language_ratios) > 1 else 0.0
        ),
        "cross_language_max_minus_min": (
            max(language_ratios.values()) - min(language_ratios.values()) if language_ratios else 0.0
        ),
    }


def validate_roundtrip(tokenizer: RootismTokenizer, documents: Sequence[str]) -> dict:
    """Strict byte-exact round-trip used as a training-time gate."""
    failures = []
    for i, doc in enumerate(documents):
        ids = tokenizer.encode(doc)
        if tokenizer.decode(ids) != doc:
            failures.append(i)
    return {
        "documents": len(documents),
        "failures": failures,
        "roundtrip_rate": (len(documents) - len(failures)) / max(len(documents), 1),
    }
