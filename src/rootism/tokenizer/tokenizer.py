"""RootismTokenizer: encode/decode, special-token policy, versioning (req. 25-28)."""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from ..config import TokenizerConfig
from ..utils.hashing import json_load, json_save
from ..version import (
    SCHEMA_VERSION,
    assert_schema,
    content_hash,
    short_id,
)
from .bpe import BPEModel
from .pretokenizer import UNICODE_TO_BYTE, pretokenize
from .special_tokens import (
    NUM_SPECIAL_TOKENS,
    SPECIAL_BY_ID,
    SPECIAL_BY_TOKEN,
    SPECIAL_TOKENS,
    special_id,
)


class RootismTokenizer:
    def __init__(self, model: BPEModel) -> None:
        self.model = model
        self.config = TokenizerConfig.from_dict(model.config)
        self.config.validate()
        self.offset = NUM_SPECIAL_TOKENS
        self._id_to_str: list[str] = list(model.token_strings)
        self._merge_ranks = dict(model.merge_ranks)
        self._cache: dict[tuple[int, ...], tuple[int, ...]] = {}
        self._special_pattern: re.Pattern[str] | None = None
        self.validate()
        self.fingerprint = content_hash(self._fingerprint_payload())
        self.tokenizer_id = short_id("tok", self.fingerprint, length=16)

    # ---- construction ---------------------------------------------------
    @classmethod
    def from_bpe_model(cls, model: BPEModel) -> "RootismTokenizer":
        return cls(model)

    @classmethod
    def from_file(cls, path: str) -> "RootismTokenizer":
        data = json_load(path)
        assert_schema(data, kind="tokenizer")
        recorded = data.get("fingerprint")
        model = BPEModel(
            token_strings=data["token_strings"],
            merges=[tuple(pair) for pair in data["merges"]],
            config=data["config"],
            corpus_sha256=data["corpus_sha256"],
            corpus_documents=data["corpus_documents"],
            corpus_bytes=data["corpus_bytes"],
            pretoken_total=data["pretoken_total"],
            pretokens_used=data["pretokens_used"],
            algorithm_id=data["algorithm_id"],
            schema_version=data["schema_version"],
        )
        tok = cls(model)
        if recorded != tok.fingerprint:
            raise ValueError(
                "tokenizer integrity check failed: the serialized vocabulary/merges "
                "do not match the recorded fingerprint. Refusing to load."
            )
        return tok

    def _fingerprint_payload(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "algorithm_id": self.model.algorithm_id,
            "config": self.model.config,
            "special_tokens": [[s.id, s.token, s.kind, s.semantics] for s in SPECIAL_TOKENS],
            "token_strings": self._id_to_str,
            "merges": [[a, b] for a, b in self.model.merges],
            "corpus_sha256": self.model.corpus_sha256,
        }

    def to_dict(self) -> dict:
        payload = self._fingerprint_payload()
        payload.update(
            fingerprint=self.fingerprint,
            tokenizer_id=self.tokenizer_id,
            corpus_documents=self.model.corpus_documents,
            corpus_bytes=self.model.corpus_bytes,
            pretoken_total=self.model.pretoken_total,
            pretokens_used=self.model.pretokens_used,
            vocab_size=self.vocab_size,
        )
        return payload

    def save(self, path: str) -> None:
        json_save(path, self.to_dict())

    def validate(self) -> None:
        if self.model.algorithm_id != "rootism-byte-bpe-v1":
            raise ValueError("unknown tokenizer algorithm")
        if len(self._id_to_str) < self.offset + 256:
            raise ValueError("vocabulary is missing the 256 base byte symbols")
        if len(self._id_to_str) > self.config.vocab_size:
            raise ValueError("vocabulary exceeds configured vocab_size")
        for rank, (a, b) in enumerate(self.model.merges):
            if not (self.offset <= a < self.offset + 256 + rank):
                raise ValueError(f"merge {rank} references invalid left symbol {a}")
            if not (self.offset <= b < self.offset + 256 + rank):
                raise ValueError(f"merge {rank} references invalid right symbol {b}")
        for s in SPECIAL_TOKENS:
            if self._id_to_str[s.id] != s.token:
                raise ValueError(f"special token {s.token!r} is not at id {s.id}")

    # ---- identity -------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self._id_to_str)

    @property
    def pad_id(self) -> int:
        return special_id("<pad>")

    @property
    def bos_id(self) -> int:
        return special_id("<bos>")

    @property
    def eos_id(self) -> int:
        return special_id("<eos>")

    @property
    def unk_id(self) -> int:
        return special_id("<unk>")

    def special_ids(self) -> frozenset[int]:
        return frozenset(SPECIAL_BY_ID)

    # ---- encoding -------------------------------------------------------
    def _bpe(self, word: tuple[int, ...]) -> tuple[int, ...]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        symbols = list(word)
        while len(symbols) > 1:
            best_rank = None
            best_pair: tuple[int, int] | None = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self._merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, pair
            if best_pair is None:
                break
            a, b = best_pair
            merged: list[int] = []
            i = 0
            while i < len(symbols):
                if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
                    merged.append(self.offset + 256 + best_rank)  # type: ignore[operator]
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
        result = tuple(symbols)
        if len(self._cache) > 200_000:
            self._cache.clear()
        self._cache[word] = result
        return result

    def _split_special(self, text: str, allowed: Iterable[str] | str | None) -> list[tuple[str, str]]:
        if allowed is None:
            return [("text", text)]
        names = [s.token for s in SPECIAL_TOKENS] if allowed == "all" else list(allowed)  # type: ignore[arg-type]
        unknown = set(names) - set(SPECIAL_BY_TOKEN)
        if unknown:
            raise ValueError(f"unknown special tokens requested: {sorted(unknown)}")
        if not names:
            return [("text", text)]
        pattern = re.compile("(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + ")")
        segments: list[tuple[str, str]] = []
        pos = 0
        for match in pattern.finditer(text):
            if match.start() > pos:
                segments.append(("text", text[pos : match.start()]))
            segments.append(("special", match.group(0)))
            pos = match.end()
        if pos < len(text):
            segments.append(("text", text[pos:]))
        return segments

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        allowed_special: Iterable[str] | str | None = None,
    ) -> list[int]:
        """Encode text. Special token strings are NOT interpreted by default.

        This is a security decision: untrusted corpus/project text containing
        `<tool_call>` must not be able to inject a structured tool call. Input
        builders opt in explicitly via `allowed_special`.
        """
        ids: list[int] = []
        for kind, payload in self._split_special(text, allowed_special):
            if kind == "special":
                ids.append(special_id(payload))
                continue
            for piece in pretokenize(payload, max_bytes=self.config.max_pretoken_bytes):
                word = tuple(self.offset + b for b in piece.encode("utf-8"))
                ids.extend(self._bpe(word))
        if add_special_tokens:
            ids = [self.bos_id, *ids, self.eos_id]
        return ids

    def encode_batch(self, texts: Sequence[str], **kwargs: object) -> list[list[int]]:
        return [self.encode(t, **kwargs) for t in texts]  # type: ignore[arg-type]

    def encode_fim(self, prefix: str, suffix: str, *, add_special_tokens: bool = False) -> list[int]:
        """Prefix-Suffix-Middle ordering for code infilling."""
        ids = (
            [special_id("<fim_prefix>")]
            + self.encode(prefix)
            + [special_id("<fim_suffix>")]
            + self.encode(suffix)
            + [special_id("<fim_middle>")]
        )
        if add_special_tokens:
            ids = [self.bos_id, *ids]
        return ids

    # ---- decoding -------------------------------------------------------
    def token_bytes(self, token_id: int) -> bytes:
        if token_id in SPECIAL_BY_ID:
            return b""
        if not 0 <= token_id < len(self._id_to_str):
            return b""
        out = bytearray()
        for ch in self._id_to_str[token_id]:
            byte = UNICODE_TO_BYTE.get(ch)
            if byte is None:
                out.extend(ch.encode("utf-8"))
            else:
                out.append(byte)
        return bytes(out)

    def token_str(self, token_id: int) -> str:
        if 0 <= token_id < len(self._id_to_str):
            return self._id_to_str[token_id]
        return SPECIAL_BY_TOKEN["<unk>"].token

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special: bool = True,
        errors: str = "replace",
    ) -> str:
        buf = bytearray()
        for tid in token_ids:
            if tid in SPECIAL_BY_ID:
                if not skip_special:
                    buf.extend(SPECIAL_BY_ID[tid].token.encode("utf-8"))
                continue
            if not 0 <= tid < len(self._id_to_str):
                if not skip_special:
                    buf.extend(b"<unk>")
                continue
            for ch in self._id_to_str[tid]:
                byte = UNICODE_TO_BYTE.get(ch)
                if byte is None:
                    buf.extend(ch.encode("utf-8"))
                else:
                    buf.append(byte)
        return bytes(buf).decode("utf-8", errors=errors)

    def __len__(self) -> int:
        return self.vocab_size
