"""Central versioning and compatibility identities.

Every persisted Rootism artifact (model config, tokenizer, checkpoint, dataset
manifest, evaluation report) carries a schema version and a content hash so that
silent incompatibilities are detected rather than guessed at.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

ROOTISM_VERSION = "0.1.0"
__version__ = ROOTISM_VERSION

# Bump only on breaking changes to serialized formats.
SCHEMA_VERSION = "1"
MODEL_ARCHITECTURE_ID = "rootism-decoder-v1"
TOKENIZER_ALGORITHM_ID = "rootism-byte-bpe-v1"
CHECKPOINT_FORMAT_ID = "rootism-ckpt-v1"
DATASET_MANIFEST_FORMAT_ID = "rootism-dataset-manifest-v1"


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_hash(obj: Any, *, length: int = 64) -> str:
    """SHA-256 over canonical JSON. Used for config/tokenizer/manifest identity."""
    digest = hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
    return digest[:length]


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def short_id(prefix: str, digest: str, *, length: int = 12) -> str:
    """Human-readable, still unique-enough identity derived from a digest."""
    return f"{prefix}-{digest[:length]}"


def assert_schema(meta: Mapping[str, Any], *, expected: str = SCHEMA_VERSION, kind: str = "artifact") -> None:
    """Reject artifacts written by an incompatible schema version."""
    actual = str(meta.get("schema_version", ""))
    if actual != expected:
        raise ValueError(
            f"{kind} schema mismatch: artifact has {actual!r}, this build expects {expected!r}. "
            "Refusing to load: silently reinterpreting persisted state is not permitted."
        )


def compatibility_report(
    *,
    config_id: str,
    tokenizer_id: str,
    checkpoint_config_id: str | None,
    checkpoint_tokenizer_id: str | None,
    checkpoint_architecture_id: str | None,
) -> None:
    """Fail closed when a checkpoint does not match the model + tokenizer in use.

    Requirement 28 / 100: tokenizer versions are permanently bound to checkpoints.
    """
    if checkpoint_architecture_id is not None and checkpoint_architecture_id != MODEL_ARCHITECTURE_ID:
        raise ValueError(f"checkpoint architecture {checkpoint_architecture_id!r} != {MODEL_ARCHITECTURE_ID!r}")
    if checkpoint_config_id is not None and checkpoint_config_id != config_id:
        raise ValueError(
            "checkpoint was produced by a different model configuration "
            f"({checkpoint_config_id!r} != {config_id!r})"
        )
    if checkpoint_tokenizer_id is not None and checkpoint_tokenizer_id != tokenizer_id:
        raise ValueError(
            "checkpoint tokenizer mismatch: token mappings may not be changed after training begins "
            f"({checkpoint_tokenizer_id!r} != {tokenizer_id!r})"
        )
