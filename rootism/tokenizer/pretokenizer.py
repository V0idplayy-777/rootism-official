"""
Code-Aware Pre-Tokenizer & Reversible Byte-to-Unicode Bijection (Steps 22 & 24).

Specifically designed so programming syntax is never mangled prior to BPE:
- Bijective 256-byte printable mapping ensures 100% lossless UTF-8 and arbitrary binary coverage.
- Preserves newline + indentation runs (`\n    `, `    `, `\t\t`) without merging spaces into identifiers.
- Preserves multi-character operators (`===`, `!=`, `->`, `=>`, `::`, `&&`, `||`, `<<`, `>>`, `**`, `//`, `?.`).
- Splits digits deterministically (`single_digit` or `three_digit`) for consistent arithmetic/line-number tokens.
- Keeps file path separators and punctuation cleanly segmented.
"""

from __future__ import annotations

import re
from typing import Literal


def _build_byte_unicode_bijection() -> tuple[dict[int, str], dict[str, int]]:
    """
    Construct a deterministic 1-to-1 mapping from every byte 0..255 to a distinct
    printable Unicode codepoint so BPE tokens are human-inspectable JSON strings
    with zero control-character corruption.
    """
    printable: list[int] = (
        list(range(33, 127))      # '!' .. '~'
        + list(range(161, 173))   # '¡' .. '¬'
        + list(range(174, 256))   # '®' .. 'ÿ'
    )
    mapped: list[int] = list(printable)
    offset = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            mapped.append(256 + offset)
            offset += 1
    b2u = {b: chr(u) for b, u in zip(printable, mapped, strict=True)}
    u2b = {u_ch: b for b, u_ch in b2u.items()}
    return b2u, u2b


BYTES_TO_UNICODE: dict[int, str] = _build_byte_unicode_bijection()[0]
UNICODE_TO_BYTES: dict[str, int] = _build_byte_unicode_bijection()[1]


# Ordered longest-first so compound operators match atomically before single-char punctuation
MULTI_CHAR_OPERATORS: tuple[str, ...] = (
    "<!--",
    "-->",
    "===",
    "!==",
    ">>>",
    "<<=",
    ">>=",
    "**=",
    "//=",
    "<=>",
    "...",
    "..=",
    "==",
    "!=",
    "<=",
    ">=",
    "->",
    "=>",
    "::",
    "&&",
    "||",
    "<<",
    ">>",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "&=",
    "|=",
    "^=",
    "**",
    "//",
    "|>",
    "??",
    "?.",
    "++",
    "--",
    "/*",
    "*/",
)


class CodeAwarePretokenizer:
    """
    Deterministic regex-driven pre-tokenizer engineered jointly for source code and prose.
    """

    def __init__(
        self,
        digit_split_mode: Literal["single_digit", "three_digit"] = "single_digit",
        max_indentation_spaces: int = 16,
    ) -> None:
        self.digit_split_mode = digit_split_mode
        self.max_indentation_spaces = max_indentation_spaces

        escaped_ops = "|".join(re.escape(op) for op in MULTI_CHAR_OPERATORS)
        digit_pattern = r"[0-9]" if digit_split_mode == "single_digit" else r"[0-9]{1,3}"

        # Pattern order is critical:
        # 1. CRLF / LF followed by indentation spaces (up to max_indentation_spaces) or tabs
        # 2. Runs of 2..max_indentation_spaces spaces NOT followed by non-space word merge
        #    (or tab runs)
        # 3. English contractions ('s, 't, 're, 've, 'm, 'll, 'd)
        # 4. Multi-character programming operators (===, ->, ::, etc.)
        # 5. Optional single space + identifier / prose word (letters + underscores, keeping camel/snake segments clean)
        # 6. Digits (single or 1-3 group)
        # 7. Single space + single punctuation/operator (e.g. ` +`, ` =`, ` (`)
        # 8. Standalone newlines, punctuation, path slashes, Unicode symbols
        # 9. Any trailing horizontal whitespace
        pattern = (
            rf"\r?\n[ ]{{1,{max_indentation_spaces}}}|\r?\n\t+"
            rf"|[ ]{{2,{max_indentation_spaces}}}"
            rf"|\t+"
            rf"|'(?:[sSmMdDtT]|[rR][eE]|[vV][eE]|[lL][lL])"
            rf"|{escaped_ops}"
            rf"| ?[^\W\d_]+(?:_[^\W\d_]+)*"
            rf"|_+[^\W\d_]*"
            rf"|{digit_pattern}"
            rf"| ?[^\s\w]"
            rf"|\r?\n+"
            rf"|\s+"
        )
        self._regex = re.compile(pattern, flags=re.UNICODE)

    def split_text(self, text: str) -> list[str]:
        """Segment raw UTF-8 text into code-structure-preserving pre-tokens."""
        if not text:
            return []
        return self._regex.findall(text)

    def pretokenize_to_byte_strings(self, text: str) -> list[str]:
        """Segment text and map each UTF-8 byte of each chunk into the printable byte alphabet."""
        chunks = self.split_text(text)
        out: list[str] = []
        b2u = BYTES_TO_UNICODE
        for chunk in chunks:
            raw = chunk.encode("utf-8")
            out.append("".join(b2u[b] for b in raw))
        return out

    @staticmethod
    def byte_string_to_bytes(byte_str: str) -> bytes:
        u2b = UNICODE_TO_BYTES
        return bytes(u2b[ch] for ch in byte_str)
