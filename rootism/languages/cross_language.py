"""
Cross-Language Semantic Reasoning & Equivalence Engine (Step 32).

Captures subtle semantic divergences across programming languages so Rootism
can translate code faithfully, explain semantic mismatches, and avoid naive
line-by-line transliteration bugs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CrossLanguageSemanticDivergence:
    """Documented semantic hazard when translating or interfacing between two languages."""

    concept: str
    source_language: str
    target_language: str
    source_semantics: str
    target_semantics: str
    translation_hazard: str
    idiomatic_resolution: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SEMANTIC_DIVERGENCE_MATRIX: tuple[CrossLanguageSemanticDivergence, ...] = (
    CrossLanguageSemanticDivergence(
        concept="integer_overflow",
        source_language="python",
        target_language="rust",
        source_semantics="Arbitrary-precision `int` never overflows.",
        target_semantics="Fixed-width `i64`/`u64` panics on overflow in debug builds and wraps in release unless explicit.",
        translation_hazard="Direct translation of large combinatorial or hash arithmetic panics or silently wraps.",
        idiomatic_resolution="Use `checked_add`/`saturating_mul`/`wrapping_mul` or `i128`/big-int crate explicitly.",
    ),
    CrossLanguageSemanticDivergence(
        concept="string_indexing_and_slicing",
        source_language="python",
        target_language="rust",
        source_semantics="`s[i]` indexes Unicode codepoints.",
        target_semantics="`&str` is UTF-8 byte-indexed; slicing mid-codepoint panics at runtime.",
        translation_hazard="Byte-offset vs char-index mismatch on multibyte UTF-8 characters.",
        idiomatic_resolution="Use `s.chars().nth(i)` or `s.char_indices()` when character semantics are required.",
    ),
    CrossLanguageSemanticDivergence(
        concept="utf16_vs_codepoint_length",
        source_language="javascript",
        target_language="go",
        source_semantics="`str.length` counts UTF-16 code units (surrogate pairs count as 2).",
        target_semantics="`len(s)` counts UTF-8 bytes; `utf8.RuneCountInString(s)` counts Unicode scalar values.",
        translation_hazard="Validation lengths and cursor offsets diverge on non-BMP characters (e.g. emoji).",
        idiomatic_resolution="Specify byte vs rune vs UTF-16 offset explicitly across API boundaries.",
    ),
    CrossLanguageSemanticDivergence(
        concept="modulo_negative_numbers",
        source_language="python",
        target_language="c",
        source_semantics="`-7 % 3 == 2` (floored division; remainder takes sign of divisor).",
        target_semantics="`-7 % 3 == -1` (truncated division; remainder takes sign of dividend).",
        translation_hazard="Negative cyclic array index wrapping (`(idx - 1) % n`) produces negative out-of-bounds index in C/C++/Go/Java/JS.",
        idiomatic_resolution="Use `((a % n) + n) % n` or `rem_euclid` in Rust.",
    ),
    CrossLanguageSemanticDivergence(
        concept="resource_cleanup_determinism",
        source_language="cpp",
        target_language="python",
        source_semantics="RAII destructors run deterministically on scope exit.",
        target_semantics="`__del__` finalizers have non-deterministic timing and can be delayed by reference cycles.",
        translation_hazard="File descriptors, locks, or sockets leak under load if ported to `__del__`.",
        idiomatic_resolution="Translate C++ RAII scopes into Python `with` context managers (`__enter__`/`__exit__`).",
    ),
    CrossLanguageSemanticDivergence(
        concept="null_three_valued_logic",
        source_language="python",
        target_language="sql",
        source_semantics="`None == None` is `True`; `x not in [1, None]` evaluates boolean membership.",
        target_semantics="`NULL = NULL` is `UNKNOWN`; `x NOT IN (1, NULL)` never evaluates to TRUE for any row.",
        translation_hazard="Filtering queries silently return 0 rows when nullable columns appear in `NOT IN` subqueries.",
        idiomatic_resolution="Use `IS NULL` / `IS DISTINCT FROM` and `NOT EXISTS` anti-joins instead of `NOT IN`.",
    ),
)


def query_semantic_divergences(lang_a: str, lang_b: str) -> list[CrossLanguageSemanticDivergence]:
    """Return known semantic divergences in either direction between `lang_a` and `lang_b`."""
    pair = {lang_a.lower(), lang_b.lower()}
    return [
        d
        for d in SEMANTIC_DIVERGENCE_MATRIX
        if {d.source_language, d.target_language} == pair
    ]
