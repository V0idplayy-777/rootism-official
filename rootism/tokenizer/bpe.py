"""
Rootism Independent Byte-Level BPE Tokenizer & Cryptographic Version Lock (Steps 21–25, 28).

Token ID Layout (for default `reserved_special_slots = 64`, `vocab_size = 65,536`):
- IDs [0 .. 63]        : Permanent Special & Protocol Control Tokens
- IDs [64 .. 319]      : 256 Bijective Base UTF-8 Byte Tokens (0x00 .. 0xFF)
- IDs [320 .. V - 1]   : Ordered Learned BPE Merge Tokens (+ deterministic padding slots if corpus < V)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rootism.config import RootismModelConfig, RootismTokenizerConfig
from rootism.tokenizer.pretokenizer import (
    BYTES_TO_UNICODE,
    UNICODE_TO_BYTES,
    CodeAwarePretokenizer,
)
from rootism.tokenizer.special_tokens import (
    SpecialTokenSpec,
    build_special_token_table,
)


class TokenizerCompatibilityError(RuntimeError):
    """Raised when a tokenizer file or model checkpoint fails cryptographic version verification."""


@dataclass(frozen=True)
class TokenizerVersionLock:
    """Permanent cryptographic binding between a RootismTokenizer and model checkpoints."""

    tokenizer_name: str
    semver: str
    vocab_size: int
    reserved_special_slots: int
    pretokenizer_version: str
    digit_split_mode: str
    max_indentation_spaces: int
    special_tokens_sha256: str
    merges_sha256: str
    vocab_sha256: str
    fingerprint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RootismTokenizer:
    """
    Production byte-level BPE tokenizer built independently for Rootism.
    """

    def __init__(
        self,
        config: RootismTokenizerConfig,
        merges: list[tuple[str, str]],
        extra_padded_tokens: list[str] | None = None,
        expected_fingerprint: str | None = None,
    ) -> None:
        self.config = config
        self.special_specs: list[SpecialTokenSpec] = build_special_token_table(
            config.reserved_special_slots
        )
        self.pretokenizer = CodeAwarePretokenizer(
            digit_split_mode=config.digit_split_mode,
            max_indentation_spaces=config.max_indentation_spaces,
        )

        self.id_to_token: dict[int, str] = {}
        self.token_to_id: dict[str, int] = {}
        self.special_to_id: dict[str, int] = {}
        self.id_to_special: dict[int, SpecialTokenSpec] = {}

        # 1. Register special tokens [0 .. reserved_special_slots - 1]
        for spec in self.special_specs:
            self.id_to_token[spec.token_id] = spec.surface
            self.token_to_id[spec.surface] = spec.token_id
            self.special_to_id[spec.surface] = spec.token_id
            self.id_to_special[spec.token_id] = spec

        # 2. Register 256 raw byte tokens [reserved_special_slots .. reserved_special_slots + 255]
        next_id = config.reserved_special_slots
        for byte_val in range(256):
            surf = BYTES_TO_UNICODE[byte_val]
            self.id_to_token[next_id] = surf
            self.token_to_id[surf] = next_id
            next_id += 1

        # 3. Register ordered BPE merges
        self.merges: list[tuple[str, str]] = []
        self.bpe_ranks: dict[tuple[str, str], int] = {}
        for rank, (left, right) in enumerate(merges):
            merged = left + right
            if merged in self.token_to_id:
                continue
            if next_id >= config.vocab_size:
                break
            self.merges.append((left, right))
            self.bpe_ranks[(left, right)] = rank
            self.id_to_token[next_id] = merged
            self.token_to_id[merged] = next_id
            next_id += 1

        # 4. Register deterministic unused capacity slots if merges < target vocab_size
        self.extra_padded_tokens: list[str] = []
        if extra_padded_tokens:
            for tok in extra_padded_tokens:
                if next_id >= config.vocab_size:
                    break
                if tok not in self.token_to_id:
                    self.id_to_token[next_id] = tok
                    self.token_to_id[tok] = next_id
                    self.extra_padded_tokens.append(tok)
                    next_id += 1

        while next_id < config.vocab_size:
            pad_tok = f"<|unused_vocab_{next_id:05d}|>"
            self.id_to_token[next_id] = pad_tok
            self.token_to_id[pad_tok] = next_id
            self.extra_padded_tokens.append(pad_tok)
            next_id += 1

        # Compile special-token splitter (longest first) ONLY used when allow_special=True
        escaped_specials = [
            re.escape(s) for s in sorted(self.special_to_id.keys(), key=len, reverse=True)
        ]
        self._special_regex = re.compile(f"({'|'.join(escaped_specials)})")

        # Word-level BPE memoization cache for O(1) lookup on recurring keywords/indents/identifiers
        self._bpe_cache: dict[str, tuple[int, ...]] = {}
        self._max_cache_entries: int = 131_072

        self.version_lock: TokenizerVersionLock = self._compute_version_lock()
        if expected_fingerprint is not None and self.version_lock.fingerprint_sha256 != expected_fingerprint:
            raise TokenizerCompatibilityError(
                f"Tokenizer fingerprint mismatch: computed {self.version_lock.fingerprint_sha256} "
                f"!= expected {expected_fingerprint}"
            )

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_token_id(self) -> int:
        return self.special_to_id["<|pad|>"]

    @property
    def bos_token_id(self) -> int:
        return self.special_to_id["<|bos|>"]

    @property
    def eos_token_id(self) -> int:
        return self.special_to_id["<|eos|>"]

    @property
    def unk_token_id(self) -> int:
        return self.special_to_id["<|unk|>"]

    @property
    def fingerprint(self) -> str:
        return self.version_lock.fingerprint_sha256

    def _compute_version_lock(self) -> TokenizerVersionLock:
        special_payload = json.dumps(
            [(s.token_id, s.surface, s.category) for s in self.special_specs],
            separators=(",", ":"),
        )
        special_sha = hashlib.sha256(special_payload.encode("utf-8")).hexdigest()

        merges_payload = "\n".join(f"{a} {b}" for a, b in self.merges)
        merges_sha = hashlib.sha256(merges_payload.encode("utf-8")).hexdigest()

        vocab_items = [self.id_to_token[i] for i in range(len(self.id_to_token))]
        vocab_sha = hashlib.sha256(
            json.dumps(vocab_items, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        master_obj = {
            "tokenizer_name": self.config.tokenizer_name,
            "semver": self.config.semver,
            "vocab_size": len(self.id_to_token),
            "reserved_special_slots": self.config.reserved_special_slots,
            "pretokenizer_version": self.config.pretokenizer_version,
            "digit_split_mode": self.config.digit_split_mode,
            "max_indentation_spaces": self.config.max_indentation_spaces,
            "special_tokens_sha256": special_sha,
            "merges_sha256": merges_sha,
            "vocab_sha256": vocab_sha,
        }
        fingerprint = hashlib.sha256(
            json.dumps(master_obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        return TokenizerVersionLock(
            **master_obj,
            fingerprint_sha256=fingerprint,
        )

    def _bpe_word_to_ids(self, byte_word: str) -> tuple[int, ...]:
        cached = self._bpe_cache.get(byte_word)
        if cached is not None:
            return cached

        if len(byte_word) == 1:
            res = (self.token_to_id[byte_word],)
            self._bpe_cache[byte_word] = res
            return res

        symbols: list[str] = list(byte_word)
        ranks = self.bpe_ranks

        while len(symbols) > 1:
            best_rank = 1_000_000_000
            best_pair: tuple[str, str] | None = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                r = ranks.get(pair)
                if r is not None and r < best_rank:
                    best_rank = r
                    best_pair = pair

            if best_pair is None:
                break

            first, second = best_pair
            merged_sym = first + second
            new_symbols: list[str] = []
            i = 0
            n = len(symbols)
            while i < n:
                if i < n - 1 and symbols[i] == first and symbols[i + 1] == second:
                    new_symbols.append(merged_sym)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        ids = tuple(self.token_to_id[sym] for sym in symbols)
        if len(self._bpe_cache) >= self._max_cache_entries:
            self._bpe_cache.clear()
        self._bpe_cache[byte_word] = ids
        return ids

    def _encode_plain_span(self, text: str) -> list[int]:
        out: list[int] = []
        for byte_word in self.pretokenizer.pretokenize_to_byte_strings(text):
            out.extend(self._bpe_word_to_ids(byte_word))
        return out

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        allow_special: bool = False,
    ) -> list[int]:
        """
        Encode UTF-8 text into Rootism token IDs.

        Security invariant: `allow_special` defaults to `False` so untrusted user prompts,
        source files, and scraped documents can NEVER inject control tokens like
        `<|tool_call_start|>` or `<|system_start|>`.
        """
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_token_id)

        if text:
            if allow_special:
                parts = self._special_regex.split(text)
                for part in parts:
                    if not part:
                        continue
                    sp_id = self.special_to_id.get(part)
                    if sp_id is not None:
                        ids.append(sp_id)
                    else:
                        ids.extend(self._encode_plain_span(part))
            else:
                ids.extend(self._encode_plain_span(text))

        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def encode_structured(self, segments: list[tuple[str, bool]]) -> list[int]:
        """
        Safely encode a sequence of `(content, is_trusted_special_token)` segments.
        When `is_trusted_special_token=True`, `content` must be an exact registered special token.
        When `is_trusted_special_token=False`, `content` is encoded strictly with `allow_special=False`.
        """
        ids: list[int] = []
        for content, is_special in segments:
            if is_special:
                if content not in self.special_to_id:
                    raise ValueError(f"Unknown special token in structured encoding: {content!r}")
                ids.append(self.special_to_id[content])
            else:
                ids.extend(self._encode_plain_span(content))
        return ids

    def decode_bytes(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> bytes:
        """Reconstruct exact raw byte stream from token IDs."""
        buf = bytearray()
        u2b = UNICODE_TO_BYTES
        special_limit = self.config.reserved_special_slots

        for tid in token_ids:
            if tid < 0 or tid >= len(self.id_to_token):
                raise ValueError(f"Token ID {tid} out of vocabulary range [0, {len(self.id_to_token) - 1}]")
            if tid < special_limit:
                if not skip_special_tokens:
                    buf.extend(self.id_to_token[tid].encode("utf-8"))
                continue
            surf = self.id_to_token[tid]
            if surf.startswith("<|unused_vocab_") and surf.endswith("|>"):
                if not skip_special_tokens:
                    buf.extend(surf.encode("utf-8"))
                continue
            for ch in surf:
                buf.append(u2b[ch])
        return bytes(buf)

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        errors: str = "replace",
    ) -> str:
        """Decode token IDs back into a UTF-8 string."""
        raw = self.decode_bytes(token_ids, skip_special_tokens=skip_special_tokens)
        return raw.decode("utf-8", errors=errors)

    def bind_to_model_config(self, model_config: RootismModelConfig) -> RootismModelConfig:
        """
        Permanently bind this tokenizer's cryptographic version fingerprint to a `RootismModelConfig`,
        verifying vocabulary size and special token ID alignment.
        """
        if model_config.vocab_size != self.vocab_size:
            raise TokenizerCompatibilityError(
                f"Cannot bind tokenizer (vocab_size={self.vocab_size}) to model "
                f"(vocab_size={model_config.vocab_size})."
            )
        if (
            model_config.pad_token_id != self.pad_token_id
            or model_config.bos_token_id != self.bos_token_id
            or model_config.eos_token_id != self.eos_token_id
            or model_config.unk_token_id != self.unk_token_id
        ):
            raise TokenizerCompatibilityError(
                "Core special token IDs (pad/bos/eos/unk) disagree between model and tokenizer."
            )
        return model_config.model_copy(
            update={"tokenizer_version_fingerprint": self.version_lock.fingerprint_sha256}
        )

    def save(self, path: str | Path) -> Path:
        """Serialize tokenizer artifact with embedded SHA-256 version lock."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "rootism-tokenizer-artifact-v1",
            "config": self.config.model_dump(mode="json"),
            "version_lock": self.version_lock.to_dict(),
            "special_tokens": [asdict(s) for s in self.special_specs],
            "merges": [[a, b] for a, b in self.merges],
        }
        tmp = out.with_suffix(out.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        tmp.replace(out)
        return out

    @classmethod
    def load(cls, path: str | Path, expected_fingerprint: str | None = None) -> "RootismTokenizer":
        """Load tokenizer artifact and strictly verify its embedded SHA-256 version lock."""
        src = Path(path)
        with src.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("format") != "rootism-tokenizer-artifact-v1":
            raise TokenizerCompatibilityError(f"Unrecognized tokenizer artifact format in {src}")
        cfg = RootismTokenizerConfig.model_validate(payload["config"])
        merges = [(pair[0], pair[1]) for pair in payload["merges"]]
        stored_fp = payload["version_lock"]["fingerprint_sha256"]
        target_fp = expected_fingerprint or stored_fp
        tok = cls(config=cfg, merges=merges, expected_fingerprint=target_fp)
        if tok.version_lock.fingerprint_sha256 != stored_fp:
            raise TokenizerCompatibilityError(
                f"Corrupted tokenizer artifact {src}: stored fingerprint {stored_fp} "
                f"!= reconstructed {tok.version_lock.fingerprint_sha256}"
            )
        return tok
