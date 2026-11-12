"""Atomic, integrity-checked checkpoint I/O (requirements 55, 57).

Stage 1 implements model-state checkpointing and generic state checkpoints.
Stage 4 extends the same format with optimiser/scheduler/RNG/dataset state and
validates true training resume. The format version is fixed here.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from ..config import ModelConfig
from ..utils.hashing import json_save
from ..version import (
    CHECKPOINT_FORMAT_ID,
    SCHEMA_VERSION,
    assert_schema,
    compatibility_report,
    file_sha256,
)


def _manifest_path(ckpt_path: str | os.PathLike[str]) -> Path:
    return Path(str(Path(ckpt_path)) + ".manifest.json")


def checkpoint_path(directory: str | os.PathLike[str], step: int, *, prefix: str = "ckpt") -> Path:
    return Path(directory) / f"{prefix}-step-{step:09d}.pt"


def _atomic_bytes(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        writer(tmp)
        with open(tmp, "r+b") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_checkpoint(
    state: Mapping[str, Any],
    path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write `torch.save` state plus a versioned manifest with a SHA-256 digest."""
    target = Path(path)
    meta = dict(metadata or {})
    meta.setdefault("step", 0)
    meta.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    meta.setdefault("torch_version", torch.__version__)
    meta.setdefault("world_size", 1)

    def write(tmp: str) -> None:
        torch.save(dict(state), tmp)

    _atomic_bytes(target, write)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "format_id": CHECKPOINT_FORMAT_ID,
        "sha256": file_sha256(target),
        "size_bytes": target.stat().st_size,
        "filename": target.name,
        **meta,
    }
    json_save(_manifest_path(target), manifest)
    return manifest


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
    verify: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = Path(path)
    manifest = json.loads(_manifest_path(target).read_text(encoding="utf-8"))
    assert_schema(manifest, kind="checkpoint")
    if manifest.get("format_id") != CHECKPOINT_FORMAT_ID:
        raise ValueError("unsupported checkpoint format")
    if verify:
        if target.stat().st_size != manifest["size_bytes"]:
            raise ValueError("checkpoint size mismatch")
        if file_sha256(target) != manifest["sha256"]:
            raise ValueError("checkpoint SHA-256 mismatch: file is corrupted or incomplete")
    state = torch.load(target, map_location=map_location, weights_only=True)
    return state, manifest


def verify_checkpoint(path: str | os.PathLike[str]) -> bool:
    try:
        target = Path(path)
        manifest = json.loads(_manifest_path(target).read_text(encoding="utf-8"))
        assert_schema(manifest, kind="checkpoint")
        return (
            target.exists()
            and target.stat().st_size == manifest["size_bytes"]
            and file_sha256(target) == manifest["sha256"]
        )
    except Exception:
        return False


def find_recovery_point(directory: str | os.PathLike[str]) -> Path | None:
    """Return the highest-step checkpoint whose manifest and digest validate."""
    candidates: list[tuple[int, Path]] = []
    for manifest_file in Path(directory).glob("*.manifest.json"):
        # filename is ckpt-step-000000001.pt.manifest.json -> restore .pt
        ckpt = Path(str(manifest_file)[: -len(".manifest.json")])
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            step = int(data.get("step", -1))
        except Exception:
            continue
        if verify_checkpoint(ckpt):
            candidates.append((step, ckpt))
    if not candidates:
        return None
    return max(candidates, key=lambda pair: (pair[0], str(pair[1])))[1]


def save_model_checkpoint(
    model: nn.Module,
    path: str | os.PathLike[str],
    *,
    tokenizer_id: str | None = None,
    step: int = 0,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg: ModelConfig = model.config  # type: ignore[attr-defined]
    state = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    metadata = {
        "config": cfg.to_dict(),
        "config_id": cfg.config_id,
        "architecture_id": cfg.architecture_id,
        "tokenizer_id": tokenizer_id,
        "step": step,
        "kind": "model",
    }
    metadata.update(extra_metadata or {})
    return save_checkpoint(state, path, metadata=metadata)


def load_model_checkpoint(
    path: str | os.PathLike[str],
    *,
    tokenizer_id: str | None = None,
    expected_config: ModelConfig | None = None,
    map_location: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build Rootism from the stored config and load weights strictly."""
    from ..model.config_bridge import build_model

    state, manifest = load_checkpoint(path, map_location=map_location)
    cfg = ModelConfig.from_dict(manifest["config"])
    compatibility_report(
        config_id=(expected_config.config_id if expected_config is not None else cfg.config_id),
        tokenizer_id=tokenizer_id or manifest.get("tokenizer_id") or "",
        checkpoint_config_id=manifest.get("config_id"),
        checkpoint_tokenizer_id=manifest.get("tokenizer_id") if tokenizer_id else None,
        checkpoint_architecture_id=manifest.get("architecture_id"),
    )
    model = build_model(expected_config or cfg, device=map_location, init_weights=False)
    missing, unexpected = model.load_state_dict(state["model"], strict=True)
    if missing or unexpected:  # strict=True already raises, kept for clarity
        raise RuntimeError(f"state_dict mismatch: missing={missing} unexpected={unexpected}")
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model, manifest
