"""Code-aware, byte-exact pre-tokenizer (requirements 22, 24).

Design: instead of a language-agnostic regex that folds whitespace into words
(which destroys Python indentation), whitespace runs are emitted as their own
pre-tokens. BPE then learns dedicated tokens such as four spaces, a tab, or a
newline-plus-indentation sequence. Operators and punctuation are emitted as
runs so multi-character operators (`=>`, `===`, `::`, `->`, `<<=`) can become
single learned tokens. Identifiers, numbers and Unicode strings are kept whole.

No Unicode normalisation is applied: code is byte-sensitive.
"""

from __future__ import annotations

import unicodedata
from typing import Iterator


def _is_ident_start(ch: str) -> bool:
    return ch == "_" or ch.isalpha() or (unicodedata.category(ch) in {"Lm", "Lt", "Nl"})


def _is_ident_part(ch: str) -> bool:
    return _is_ident_start(ch) or ch.isdigit() or ch == "\u200c" or ch == "\u200d"


def _is_hspace(ch: str) -> bool:
    return ch in " \t\f\v\u00a0" or (ch.isspace() and ch not in "\r\n")


def _byte_len(s: str) -> int:
    return len(s.encode("utf-8"))


def _split_long(chunk: str, max_bytes: int) -> Iterator[str]:
    """Split an oversized chunk at character boundaries (never inside a codepoint)."""
    if _byte_len(chunk) <= max_bytes:
        yield chunk
        return
    current: list[str] = []
    size = 0
    for ch in chunk:
        csize = len(ch.encode("utf-8"))
        if current and size + csize > max_bytes:
            yield "".join(current)
            current, size = [], 0
        current.append(ch)
        size += csize
    if current:
        yield "".join(current)


def pretokenize(text: str, *, max_bytes: int = 128) -> list[str]:
    """Return pre-tokens whose exact concatenation equals `text`."""
    if not text:
        return []
    chunks: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\r" and i + 1 < n and text[i + 1] == "\n":
            chunks.append("\r\n")
            i += 2
            continue
        if ch in "\r\n":
            j = i + 1
            while j < n and text[j] == ch:
                j += 1
            chunks.append(text[i:j])
            i = j
            continue
        if _is_hspace(ch):
            j = i + 1
            while j < n and _is_hspace(text[j]):
                j += 1
            chunks.append(text[i:j])
            i = j
            continue
        if _is_ident_start(ch):
            j = i + 1
            while j < n and _is_ident_part(text[j]):
                j += 1
            chunks.append(text[i:j])
            i = j
            continue
        if ch.isdigit():
            j = i + 1
            while j < n:
                c = text[j]
                if c.isalnum() or c in "._":
                    j += 1
                elif c in "+-" and text[j - 1] in "eE":
                    j += 1  # sign inside an exponent, e.g. 1e-5
                else:
                    break
            chunks.append(text[i:j])
            i = j
            continue
        # punctuation / operators / quotes / arbitrary Unicode symbols
        j = i + 1
        while j < n:
            c = text[j]
            if c in "\r\n" or _is_hspace(c) or _is_ident_start(c) or c.isdigit():
                break
            j += 1
        chunks.append(text[i:j])
        i = j

    out: list[str] = []
    for chunk in chunks:
        out.extend(_split_long(chunk, max_bytes))
    assert "".join(out) == text  # structural invariant, cheap relative to BPE
    return out


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2-style reversible byte<->Unicode alphabet (all 256 bytes printable)."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = printable[:]
    extra = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            cs.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(printable, cs)}


BYTE_TO_UNICODE = bytes_to_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}
