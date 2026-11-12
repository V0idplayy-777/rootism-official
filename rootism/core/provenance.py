"""
Provenance & Reproducibility Guardrails (Steps 02, 03, 09, Directive G).

Enforces Rootism's non-negotiable independence mandate:
- Zero external pretrained models, tokenizers, or checkpoints.
- Cryptographically hashed reproducibility manifests linking model config,
  tokenizer fingerprint, dataset provenance hash, software version, and seed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

from rootism import __architecture_family__, __project__, __version__
from rootism.config import RootismConfig, RootismModelConfig

FORBIDDEN_RUNTIME_MODULES: tuple[str, ...] = (
    "transformers",
    "peft",
    "trl",
    "sentence_transformers",
)

ROOTISM_ORIGIN_SIGNATURE = "ROOTISM_INDEPENDENT_SCRATCH_V1"


class PretrainedDependencyViolationError(RuntimeError):
    """Raised when forbidden pretrained-model packages or external checkpoints are detected."""


@dataclass(frozen=True)
class ReproducibilityManifest:
    """Identifiable lineage record bound to every Rootism training run and checkpoint."""

    project: str
    software_version: str
    architecture_family: str
    origin_signature: str
    created_utc: str
    model_config_sha256: str
    tokenizer_fingerprint_sha256: str
    dataset_manifest_sha256: str
    training_config_sha256: str
    random_seed: int
    deterministic_ops_enabled: bool
    python_version: str
    pytorch_version: str
    platform_info: str
    cuda_available: bool
    device_summary: str
    nondeterminism_notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["nondeterminism_notes"] = list(self.nondeterminism_notes)
        return data

    def manifest_sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assert_independent_environment() -> None:
    """Verify no external pretrained model frameworks have been imported into the runtime."""
    loaded_forbidden = [
        mod for mod in FORBIDDEN_RUNTIME_MODULES if mod in sys.modules
    ]
    if loaded_forbidden:
        raise PretrainedDependencyViolationError(
            f"Forbidden external pretrained-model modules loaded in sys.modules: {loaded_forbidden}. "
            "Rootism prohibits importing external pretrained model libraries."
        )


def verify_rootism_checkpoint_provenance(
    metadata: dict[str, Any],
    expected_model_config: RootismModelConfig | None = None,
    expected_tokenizer_fingerprint: str | None = None,
) -> None:
    """
    Verify that a loaded checkpoint was genuinely created by Rootism from scratch
    and matches the active architecture and tokenizer mapping.
    """
    if not isinstance(metadata, dict):
        raise PretrainedDependencyViolationError("Checkpoint metadata missing or malformed.")

    origin = metadata.get("origin_signature")
    if origin != ROOTISM_ORIGIN_SIGNATURE:
        raise PretrainedDependencyViolationError(
            f"Invalid or missing Rootism origin_signature ({origin!r}). "
            "External or pretrained checkpoints cannot be loaded into Rootism."
        )

    arch = metadata.get("architecture_family")
    if arch != __architecture_family__:
        raise PretrainedDependencyViolationError(
            f"Incompatible checkpoint architecture_family {arch!r} (expected {__architecture_family__!r})."
        )

    init_origin = metadata.get("initialization_origin")
    if init_origin != "random_scratch":
        raise PretrainedDependencyViolationError(
            f"Checkpoint initialization_origin must be 'random_scratch', got {init_origin!r}."
        )

    if expected_model_config is not None:
        ckpt_cfg_hash = metadata.get("model_config_sha256")
        if ckpt_cfg_hash and ckpt_cfg_hash != expected_model_config.config_hash():
            raise PretrainedDependencyViolationError(
                f"Model configuration SHA-256 mismatch: checkpoint={ckpt_cfg_hash} "
                f"vs active={expected_model_config.config_hash()}."
            )

    tok_fp = expected_tokenizer_fingerprint or (
        expected_model_config.tokenizer_version_fingerprint if expected_model_config else None
    )
    if tok_fp is not None:
        ckpt_tok_fp = metadata.get("tokenizer_fingerprint_sha256")
        if ckpt_tok_fp != tok_fp:
            raise PretrainedDependencyViolationError(
                f"Tokenizer version mismatch: checkpoint locked to {ckpt_tok_fp!r}, "
                f"active tokenizer is {tok_fp!r}. Token mappings must never silently change."
            )


def seed_everything(seed: int, deterministic: bool = False) -> tuple[str, ...]:
    """
    Seed Python, NumPy, and PyTorch RNGs across CPU and CUDA.
    Returns explicit notes documenting any remaining hardware-level nondeterminism.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    notes: list[str] = []
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        notes.append(
            "Deterministic mode enabled (warn_only=True); cross-GPU-architecture floating-point "
            "reductions and asynchronous NCCL all-reduce order can still differ across distinct cluster topologies."
        )
    else:
        notes.append(
            "Standard high-throughput mode: FlashAttention backward atomic accumulations and "
            "NCCL ring/tree all-reduce floating-point non-associativity introduce minor bit-level variance."
        )
    return tuple(notes)


def create_reproducibility_manifest(
    config: RootismConfig,
    tokenizer_fingerprint: str,
    dataset_manifest_sha256: str,
    nondeterminism_notes: tuple[str, ...] = (),
) -> ReproducibilityManifest:
    """Create a complete, immutable lineage manifest for a training run."""
    assert_independent_environment()

    training_json = json.dumps(
        config.training.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    training_hash = hashlib.sha256(training_json.encode("utf-8")).hexdigest()

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(count)]
        device_summary = f"{count}x CUDA ({', '.join(sorted(set(names)))})"
    else:
        device_summary = f"CPU ({platform.processor() or platform.machine()}, {os.cpu_count() or 1} cores)"

    return ReproducibilityManifest(
        project=__project__,
        software_version=__version__,
        architecture_family=__architecture_family__,
        origin_signature=ROOTISM_ORIGIN_SIGNATURE,
        created_utc=datetime.now(timezone.utc).isoformat(),
        model_config_sha256=config.model.config_hash(),
        tokenizer_fingerprint_sha256=tokenizer_fingerprint,
        dataset_manifest_sha256=dataset_manifest_sha256,
        training_config_sha256=training_hash,
        random_seed=config.training.seed,
        deterministic_ops_enabled=config.training.deterministic_ops,
        python_version=sys.version.split()[0],
        pytorch_version=torch.__version__,
        platform_info=platform.platform(),
        cuda_available=torch.cuda.is_available(),
        device_summary=device_summary,
        nondeterminism_notes=nondeterminism_notes,
    )
