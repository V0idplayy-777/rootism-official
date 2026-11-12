"""Deterministic JSON I/O and hashing helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..version import canonical_json, content_hash


def stable_hash(obj: Any, length: int = 64) -> str:
    return content_hash(obj, length=length)


def json_save(path: str | os.PathLike[str], obj: Any, *, atomic: bool = True) -> None:
    """Write canonical JSON, atomically by default (tmp file + fsync + rename)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(obj)
    if not atomic:
        target.write_text(payload, encoding="utf-8")
        return
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def json_load(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
