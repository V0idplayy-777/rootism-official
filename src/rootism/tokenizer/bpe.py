"""Independent byte-level BPE trainer (requirements 21, 22, 26).

Algorithm: frequency-weighted pair merging over pre-tokenised words, with an
inverted index (pair -> set of word ids) and a lazy max-heap for O(log P)
selection. Deterministic tie-breaking makes two runs on the same corpus produce
byte-identical vocabularies.

Scalability: pre-token frequencies are counted with a disk-backed SQLite spill
so the corpus does not need to fit in RAM. Merge learning is bounded by
`max_unique_pretokens`; rarer pre-tokens remain representable through the 256
base byte symbols, so no text becomes unencodable.

Known limitation (documented, not hidden): this is single-machine merge
learning. For the full 650B-token corpus, frequency counting must be sharded
across workers and reduced; that path is part of Stage 3 (data pipeline).
"""

from __future__ import annotations

import hashlib
import heapq
import os
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..config import TokenizerConfig
from ..version import SCHEMA_VERSION, TOKENIZER_ALGORITHM_ID
from .pretokenizer import BYTE_TO_UNICODE, pretokenize
from .special_tokens import NUM_SPECIAL_TOKENS


class CorpusDigest:
    """Order-sensitive, length-prefixed digest of the training corpus."""

    def __init__(self) -> None:
        self._h = hashlib.sha256()
        self.documents = 0
        self.bytes = 0

    def update(self, document: str) -> None:
        data = document.encode("utf-8")
        self._h.update(len(data).to_bytes(8, "big"))
        self._h.update(data)
        self.documents += 1
        self.bytes += len(data)

    def hexdigest(self) -> str:
        return self._h.hexdigest()


class PretokenCounter:
    """Streaming pre-token counter with a SQLite spill."""

    def __init__(self, *, max_in_memory: int = 250_000, spill_dir: str | None = None) -> None:
        self.memory: Counter[str] = Counter()
        self.max_in_memory = max_in_memory
        self.total = 0
        self.unique_seen_upper_bound = 0
        fd, self.db_path = tempfile.mkstemp(prefix="rootism-pretokens-", suffix=".sqlite3", dir=spill_dir)
        os.close(fd)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=OFF")
        self._conn.execute("PRAGMA synchronous=OFF")
        self._conn.execute("CREATE TABLE IF NOT EXISTS counts (token TEXT PRIMARY KEY, n INTEGER NOT NULL)")

    def add(self, token: str) -> None:
        self.memory[token] += 1
        self.total += 1
        if len(self.memory) >= self.max_in_memory:
            self._spill()

    def _spill(self) -> None:
        if not self.memory:
            return
        self._conn.executemany(
            "INSERT INTO counts(token,n) VALUES(?,?) "
            "ON CONFLICT(token) DO UPDATE SET n = n + excluded.n",
            self.memory.items(),
        )
        self._conn.commit()
        self.unique_seen_upper_bound += len(self.memory)
        self.memory.clear()

    def top_items(self, limit: int) -> list[tuple[str, int]]:
        self._spill()
        merged: dict[str, int] = {}
        for token, n in self._conn.execute("SELECT token, n FROM counts"):
            merged[token] = merged.get(token, 0) + n
        for token, n in self.memory.items():
            merged[token] = merged.get(token, 0) + n
        items = sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))
        return items[:limit]

    def close(self) -> None:
        self._conn.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def __enter__(self) -> "PretokenCounter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


@dataclass
class BPEModel:
    """Serialisable result of BPE training."""

    token_strings: list[str]
    merges: list[tuple[int, int]]
    config: dict
    corpus_sha256: str
    corpus_documents: int
    corpus_bytes: int
    pretoken_total: int
    pretokens_used: int
    algorithm_id: str = TOKENIZER_ALGORITHM_ID
    schema_version: str = SCHEMA_VERSION
    merge_ranks: dict[tuple[int, int], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.merge_ranks:
            self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    @property
    def vocab_size(self) -> int:
        return len(self.token_strings)

    def merges_as_strings(self) -> list[tuple[str, str]]:
        return [(self.token_strings[a], self.token_strings[b]) for a, b in self.merges]


class BPETrainer:
    def __init__(self, config: TokenizerConfig) -> None:
        config.validate()
        self.config = config
        self.offset = NUM_SPECIAL_TOKENS
        from .special_tokens import SPECIAL_TOKENS

        self.token_strings: list[str] = [s.token for s in SPECIAL_TOKENS] + [
            BYTE_TO_UNICODE[b] for b in range(256)
        ]
        self.merge_budget = config.vocab_size - len(self.token_strings)
        if self.merge_budget < 1:
            raise ValueError("vocab_size is too small to learn any merges")

    def train(self, documents: Iterable[str], *, verbose: bool = False) -> BPEModel:
        from .special_tokens import SPECIAL_TOKENS  # local import avoids cycles

        digest = CorpusDigest()
        with PretokenCounter(max_in_memory=min(250_000, self.config.max_unique_pretokens)) as counter:
            for doc in documents:
                digest.update(doc)
                for piece in pretokenize(doc, max_bytes=self.config.max_pretoken_bytes):
                    counter.add(piece)
            items = counter.top_items(self.config.max_unique_pretokens)
            pretoken_total = counter.total

        words: list[tuple[int, ...]] = [
            tuple(self.offset + b for b in token.encode("utf-8")) for token, _ in items
        ]
        freqs: list[int] = [n for _, n in items]
        if not words:
            raise ValueError("cannot train a tokenizer on an empty corpus")

        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_to_words: dict[tuple[int, int], set[int]] = {}
        heap: list[tuple[int, str, str, int, int]] = []

        def bump(pair: tuple[int, int], delta: int) -> None:
            new = pair_counts.get(pair, 0) + delta
            if new <= 0:
                pair_counts.pop(pair, None)
                return
            pair_counts[pair] = new
            if delta > 0:
                a, b = pair
                heapq.heappush(heap, (-new, self.token_strings[a], self.token_strings[b], a, b))

        def contribute(word: Sequence[int], freq: int, idx: int, sign: int) -> None:
            for pair in zip(word, word[1:]):
                bump(pair, sign * freq)
                bucket = pair_to_words.get(pair)
                if sign > 0:
                    if bucket is None:
                        pair_to_words[pair] = {idx}
                    else:
                        bucket.add(idx)
                elif bucket is not None:
                    bucket.discard(idx)
                    if not bucket:
                        pair_to_words.pop(pair, None)

        for idx, (word, freq) in enumerate(zip(words, freqs)):
            contribute(word, freq, idx, +1)
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        target = self.config.vocab_size
        while len(self.token_strings) < target and heap:
            # lazy heap: discard entries whose recorded count is stale
            best: tuple[int, str, str, int, int] | None = None
            while heap:
                candidate = heapq.heappop(heap)
                count, _, _, a, b = candidate
                if -count == pair_counts.get((a, b), 0):
                    best = candidate
                    break
            if best is None:
                break
            neg_count, _, _, a, b = best
            if -neg_count < self.config.min_pair_frequency:
                break
            pair = (a, b)
            new_id = len(self.token_strings)
            self.token_strings.append(self.token_strings[a] + self.token_strings[b])
            merges.append((a, b))
            affected = sorted(pair_to_words.pop(pair, set()))
            for idx in affected:
                old_word = words[idx]
                contribute(old_word, freqs[idx], idx, -1)
                new_word: list[int] = []
                i = 0
                while i < len(old_word):
                    if i + 1 < len(old_word) and old_word[i] == a and old_word[i + 1] == b:
                        new_word.append(new_id)
                        i += 2
                    else:
                        new_word.append(old_word[i])
                        i += 1
                words[idx] = tuple(new_word)
                contribute(words[idx], freqs[idx], idx, +1)
            pair_counts.pop(pair, None)
            if verbose and len(merges) % 5_000 == 0:
                print(f"  learned {len(merges):,} merges", flush=True)

        return BPEModel(
            token_strings=list(self.token_strings),
            merges=merges,
            config=self.config.to_dict(),
            corpus_sha256=digest.hexdigest(),
            corpus_documents=digest.documents,
            corpus_bytes=digest.bytes,
            pretoken_total=pretoken_total,
            pretokens_used=len(items),
        )
