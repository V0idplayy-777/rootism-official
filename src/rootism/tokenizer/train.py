"""Tokenizer training pipeline entry point (requirement 26)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from ..config import TokenizerConfig
from .bpe import BPETrainer
from .evaluation import validate_roundtrip
from .tokenizer import RootismTokenizer


def iter_text_files(paths: Sequence[str | os.PathLike[str]]) -> Iterator[str]:
    """Stream UTF-8 documents from files or directories (no full-corpus RAM use)."""
    for raw in paths:
        path = Path(raw)
        candidates = sorted(path.rglob("*.txt")) if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_dir():
                continue
            with open(candidate, "r", encoding="utf-8", errors="strict") as fh:
                yield fh.read()


def train_tokenizer(
    documents: Iterable[str],
    *,
    config: TokenizerConfig | None = None,
    validation_documents: Sequence[str] | None = None,
    verbose: bool = False,
) -> RootismTokenizer:
    cfg = config or TokenizerConfig()
    cfg.validate()
    model = BPETrainer(cfg).train(documents, verbose=verbose)
    tokenizer = RootismTokenizer(model)
    if validation_documents is not None:
        report = validate_roundtrip(tokenizer, validation_documents)
        if report["failures"]:
            raise RuntimeError(
                f"tokenizer round-trip validation failed for {len(report['failures'])} documents"
            )
    return tokenizer


def train_tokenizer_from_paths(
    paths: Sequence[str | os.PathLike[str]],
    *,
    config: TokenizerConfig | None = None,
    verbose: bool = False,
) -> RootismTokenizer:
    return train_tokenizer(iter_text_files(paths), config=config, verbose=verbose)
