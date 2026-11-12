"""
Atomic Resumable Checkpointing, Integrity Validation & Fault Recovery (Steps 55, 56, 57).

Saves and validates:
1. Model weights (`model.safetensors`) with embedded Rootism scratch provenance header
2. Optimiser states (`optimizer_and_rng.pt`)
3. Learning-rate scheduler state
4. Python, NumPy, PyTorch CPU & CUDA random-number generator states
5. Global training step & 6-category `TokenAccountingLedger` state
6. Complete `RootismConfig` & `TokenizerVersionLock` fingerprint
7. Dataset & `SequencePacker` position cursor (`dataset_cursor`)
8. Distributed training world metadata

Fault-tolerance guarantees:
- Writes inside a staging directory (`step_XXXXXXXX.tmp`), `fsync`s every file, records SHA-256
  digests in `checkpoint_manifest.json`, and atomically renames to `step_XXXXXXXX`.
- Incomplete `.tmp` directories or corrupted checkpoints (truncated files / hash mismatch)
  are automatically detected and skipped during recovery in favor of the latest verified
  recovery point!
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import signal
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load, save_file as safetensors_save

from rootism.config import RootismConfig
from rootism.core.provenance import verify_rootism_checkpoint_provenance
from rootism.core.token_accounting import TokenAccountingLedger
from rootism.training.scheduler import RootismLRScheduler


class CheckpointIntegrityError(RuntimeError):
    """Raised when a checkpoint fails SHA-256, manifest, or provenance verification."""


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _fsync_file_and_dir(path: Path) -> None:
    with path.open("rb") as fh:
        os.fsync(fh.fileno())
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _extract_clean_model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    raw = model.module if hasattr(model, "module") else model
    clean: dict[str, torch.Tensor] = {}
    seen_ptrs: dict[int, str] = {}
    for k, v in raw.state_dict().items():
        local_t = v.full_tensor() if hasattr(v, "full_tensor") else v
        cpu_t = local_t.detach().cpu().contiguous()
        ptr = cpu_t.data_ptr()
        if ptr in seen_ptrs:
            # Safetensors requires distinct storage for tied weights; clone on save & re-tie on load
            cpu_t = cpu_t.clone()
        else:
            seen_ptrs[ptr] = k
        clean[k] = cpu_t
    return clean


@dataclass(frozen=True)
class LoadedCheckpointState:
    checkpoint_dir: Path
    step: int
    dataset_cursor: dict[str, Any]
    distributed_metadata: dict[str, Any]
    manifest: dict[str, Any]


class PreemptionSignalCatcher:
    """
    Catches `SIGTERM` / `SIGINT` from cluster schedulers (SLURM / Kubernetes / spot preemption)
    so the active training step finishes cleanly and commits an atomic emergency checkpoint
    instead of corrupting state mid-write (Step 56).
    """

    def __init__(self) -> None:
        self.preemption_requested: bool = False
        self.signal_received: int | None = None
        self._prev_handlers: dict[int, Any] = {}

    def install(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._prev_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle)

    def uninstall(self) -> None:
        for sig, prev in self._prev_handlers.items():
            signal.signal(sig, prev)
        self._prev_handlers.clear()

    def _handle(self, signum: int, _frame: Any) -> None:
        self.preemption_requested = True
        self.signal_received = signum


class AtomicCheckpointManager:
    """
    Manages atomic checkpoint creation, rolling retention, cryptographic verification,
    and automatic fallback recovery.
    """

    def __init__(
        self,
        checkpoint_root: str | Path,
        *,
        max_retained_checkpoints: int = 5,
    ) -> None:
        self.checkpoint_root = Path(checkpoint_root)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.max_retained_checkpoints = max(1, int(max_retained_checkpoints))

    @staticmethod
    def capture_rng_states() -> dict[str, Any]:
        rng: dict[str, Any] = {
            "python_random": random.getstate(),
            "numpy_random": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng["torch_cuda_all"] = torch.cuda.get_rng_state_all()
        return rng

    @staticmethod
    def restore_rng_states(rng: dict[str, Any]) -> None:
        if "python_random" in rng:
            random.setstate(rng["python_random"])
        if "numpy_random" in rng:
            np.random.set_state(rng["numpy_random"])
        if "torch_cpu" in rng:
            torch.set_rng_state(rng["torch_cpu"])
        if "torch_cuda_all" in rng and torch.cuda.is_available():
            cuda_states = rng["torch_cuda_all"]
            if len(cuda_states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cuda_states)

    def save_checkpoint(
        self,
        *,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: RootismLRScheduler,
        token_ledger: TokenAccountingLedger,
        config: RootismConfig,
        dataset_cursor: dict[str, Any],
        distributed_metadata: dict[str, Any] | None = None,
        extra_state: dict[str, Any] | None = None,
        training_stage_tag: str = "pretraining",
    ) -> Path:
        raw_model = model.module if hasattr(model, "module") else model
        provenance_meta = raw_model.provenance_metadata()  # type: ignore[operator]
        provenance_meta["training_stage_tag"] = training_stage_tag
        provenance_meta["saved_step"] = str(step)
        provenance_meta["processed_training_tokens"] = str(token_ledger.processed_training_tokens)

        final_dir = self.checkpoint_root / f"step_{step:08d}"
        tmp_dir = self.checkpoint_root / f"step_{step:08d}.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save Safetensors weights + embedded provenance metadata
        weights_path = tmp_dir / "model.safetensors"
        state_tensors = _extract_clean_model_state_dict(model)
        safetensors_save(state_tensors, str(weights_path), metadata=provenance_meta)
        _fsync_file_and_dir(weights_path)

        # 2. Save training state (optimizer, scheduler, RNGs, ledger, dataset cursor)
        opt_state_raw = optimizer.state_dict()
        # Convert any DTensor optimizer state tensors cleanly to CPU tensors for topology-agnostic resume
        clean_opt_state = self._cpu_clean_optimizer_state(opt_state_raw)
        trainer_state_path = tmp_dir / "training_state.pt"
        torch.save(
            {
                "step": int(step),
                "optimizer": clean_opt_state,
                "scheduler": scheduler.state_dict(),
                "rng_states": self.capture_rng_states(),
                "token_ledger": token_ledger.state_dict(),
                "dataset_cursor": dataset_cursor,
                "distributed_metadata": distributed_metadata or {"world_size": 1, "rank": 0},
                "extra_state": extra_state or {},
            },
            trainer_state_path,
        )
        _fsync_file_and_dir(trainer_state_path)

        # 3. Save canonical config snapshot
        config_path = tmp_dir / "config.yaml"
        config.to_yaml(config_path)
        _fsync_file_and_dir(config_path)

        # 4. Write cryptographic checkpoint manifest
        manifest = {
            "format": "rootism-checkpoint-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "step": int(step),
            "training_stage_tag": training_stage_tag,
            "processed_training_tokens": int(token_ledger.processed_training_tokens),
            "target_training_tokens": int(token_ledger.target_training_tokens),
            "provenance": provenance_meta,
            "files": {
                "model.safetensors": _sha256_path(weights_path),
                "training_state.pt": _sha256_path(trainer_state_path),
                "config.yaml": _sha256_path(config_path),
            },
        }
        manifest_path = tmp_dir / "checkpoint_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _fsync_file_and_dir(manifest_path)

        # 5. Atomic directory swap
        if final_dir.exists():
            backup_dir = self.checkpoint_root / f"step_{step:08d}.old"
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            final_dir.replace(backup_dir)
            tmp_dir.replace(final_dir)
            shutil.rmtree(backup_dir)
        else:
            tmp_dir.replace(final_dir)

        # 6. Atomic pointer update
        pointer_tmp = self.checkpoint_root / "latest_valid.json.tmp"
        pointer_final = self.checkpoint_root / "latest_valid.json"
        pointer_tmp.write_text(
            json.dumps({"latest_checkpoint": final_dir.name, "step": int(step)}, indent=2),
            encoding="utf-8",
        )
        _fsync_file_and_dir(pointer_tmp)
        pointer_tmp.replace(pointer_final)

        self._prune_old_checkpoints()
        return final_dir

    def _cpu_clean_optimizer_state(self, obj: Any) -> Any:
        if hasattr(obj, "full_tensor"):
            return obj.full_tensor().detach().cpu()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: self._cpu_clean_optimizer_state(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._cpu_clean_optimizer_state(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._cpu_clean_optimizer_state(v) for v in obj)
        return obj

    def _prune_old_checkpoints(self) -> None:
        # Clean up any leftover aborted `.tmp` directories first
        for tmp_cand in self.checkpoint_root.glob("step_*.tmp"):
            if tmp_cand.is_dir():
                shutil.rmtree(tmp_cand, ignore_errors=True)

        valid_dirs = sorted(
            d for d in self.checkpoint_root.glob("step_[0-9]*")
            if d.is_dir() and not d.name.endswith(".tmp") and not d.name.endswith(".old")
        )
        if len(valid_dirs) > self.max_retained_checkpoints:
            for stale in valid_dirs[: len(valid_dirs) - self.max_retained_checkpoints]:
                shutil.rmtree(stale, ignore_errors=True)

    def validate_checkpoint_dir(
        self,
        ckpt_dir: str | Path,
        expected_config: RootismConfig | None = None,
    ) -> dict[str, Any]:
        """
        Strictly verify manifest existence, SHA-256 digests of all constituent files,
        and Rootism scratch provenance metadata.
        """
        cdir = Path(ckpt_dir)
        if not cdir.is_dir() or cdir.name.endswith(".tmp"):
            raise CheckpointIntegrityError(f"Checkpoint directory invalid or incomplete (.tmp): {cdir}")

        manifest_path = cdir / "checkpoint_manifest.json"
        if not manifest_path.is_file():
            raise CheckpointIntegrityError(f"Missing checkpoint_manifest.json in {cdir}")

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CheckpointIntegrityError(f"Malformed checkpoint_manifest.json in {cdir}: {exc}") from exc

        if manifest.get("format") != "rootism-checkpoint-v1":
            raise CheckpointIntegrityError(f"Unsupported checkpoint format in {cdir}")

        files_map = manifest.get("files", {})
        for req_file in ("model.safetensors", "training_state.pt", "config.yaml"):
            if req_file not in files_map:
                raise CheckpointIntegrityError(f"Manifest missing checksum entry for {req_file}")
            fpath = cdir / req_file
            if not fpath.is_file():
                raise CheckpointIntegrityError(f"Missing required checkpoint file {fpath}")
            actual_sha = _sha256_path(fpath)
            if actual_sha != files_map[req_file]:
                raise CheckpointIntegrityError(
                    f"Checksum mismatch for {fpath}: {actual_sha} != {files_map[req_file]}"
                )

        prov = manifest.get("provenance", {})
        verify_rootism_checkpoint_provenance(
            prov,
            expected_model_config=expected_config.model if expected_config else None,
        )
        return manifest

    def find_latest_valid_checkpoint(
        self,
        expected_config: RootismConfig | None = None,
    ) -> tuple[Path | None, list[str]]:
        """
        Locate the newest uncorrupted checkpoint in `checkpoint_root`.
        If the newest candidate was interrupted or corrupted, logs a recovery warning
        and automatically falls back to the preceding verified recovery point!
        """
        warnings: list[str] = []
        candidates = sorted(
            (
                d
                for d in self.checkpoint_root.glob("step_*")
                if d.is_dir()
            ),
            key=lambda p: p.name,
            reverse=True,
        )
        for cand in candidates:
            try:
                self.validate_checkpoint_dir(cand, expected_config=expected_config)
                return cand, warnings
            except Exception as exc:
                warnings.append(f"Skipped invalid/incomplete checkpoint '{cand.name}': {exc}")
        return None, warnings

    def load_checkpoint(
        self,
        ckpt_dir: str | Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: RootismLRScheduler | None = None,
        token_ledger: TokenAccountingLedger | None = None,
        expected_config: RootismConfig | None = None,
        restore_rng: bool = True,
        extra_state_out: dict[str, Any] | None = None,
    ) -> LoadedCheckpointState:
        """Validate and restore full training state from `ckpt_dir`."""
        cdir = Path(ckpt_dir)
        manifest = self.validate_checkpoint_dir(cdir, expected_config=expected_config)

        weights = safetensors_load(str(cdir / "model.safetensors"), device="cpu")
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(weights, strict=True)
        if getattr(getattr(raw_model, "config", None), "tie_word_embeddings", False):
            raw_model.lm_head.weight = raw_model.model.tok_embeddings.weight  # type: ignore[union-attr]

        payload = torch.load(cdir / "training_state.pt", map_location="cpu", weights_only=False)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        if token_ledger is not None and "token_ledger" in payload:
            token_ledger.load_state_dict(payload["token_ledger"])
        if restore_rng and "rng_states" in payload:
            self.restore_rng_states(payload["rng_states"])
        if extra_state_out is not None and "extra_state" in payload:
            extra_state_out.update(payload["extra_state"])

        return LoadedCheckpointState(
            checkpoint_dir=cdir,
            step=int(payload["step"]),
            dataset_cursor=dict(payload.get("dataset_cursor", {})),
            distributed_metadata=dict(payload.get("distributed_metadata", {})),
            manifest=manifest,
        )
