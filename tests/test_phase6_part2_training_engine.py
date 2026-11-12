"""
Automated Tests for Phase VI Part 2: Complete Resumable Training Engine (Steps 51–60).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rootism.config import RootismConfig
from rootism.model.transformer import RootismForCausalLM
from rootism.training.batching import BatchScheduleCoordinator
from rootism.training.checkpointing import AtomicCheckpointManager, CheckpointIntegrityError
from rootism.training.optimizer import build_optimizer
from rootism.training.packing import SequencePacker
from rootism.training.scheduler import RootismLRScheduler
from rootism.training.stability import TrainingStabilityMonitor
from rootism.training.trainer import RootismPretrainer


def _load_tiny_cfg(tmp_path: Path) -> RootismConfig:
    cfg = RootismConfig.from_yaml("configs/model/rootism_tiny_verify.yaml")
    t_cfg = cfg.training.model_copy(
        update={
            "checkpoint_dir": str(tmp_path / "ckpts"),
            "sequence_length": 64,
            "micro_batch_size": 2,
            "target_training_tokens": 900,
            "warmup_tokens": 128,
            "checkpoint_interval_steps": 2,
        }
    )
    return cfg.model_copy(update={"training": t_cfg})


def test_lr_scheduler_and_batch_coordinator(tmp_path: Path) -> None:
    cfg = _load_tiny_cfg(tmp_path)
    model = RootismForCausalLM(cfg.model)
    opt, _ = build_optimizer(model, cfg.training.optimizer)

    sched = RootismLRScheduler(
        opt,
        peak_lr=3e-4,
        min_lr=3e-5,
        warmup_tokens=100,
        total_tokens=1000,
        schedule_type="wsd",
    )
    assert sched.compute_lr(0) > 0.0
    assert abs(sched.compute_lr(50) - (3e-4 * 51.0 / 101.0)) < 1e-8
    assert abs(sched.compute_lr(200) - 3e-4) < 1e-8  # WSD stable plateau
    assert abs(sched.compute_lr(1000) - 3e-5) < 1e-8  # WSD final cosine decay

    coord = BatchScheduleCoordinator(
        sequence_length=4096,
        micro_batch_size=4,
        world_size=64,
        target_global_batch_tokens=4_194_304,
    )
    assert coord.gradient_accumulation_steps == 4
    assert coord.plan.sequences_per_global_step == 1024
    assert coord.plan.presented_tokens_per_global_step == 1024 * 4095


def test_sequence_packer_causal_document_isolation(tmp_path: Path) -> None:
    """
    Prove that packed documents have reset `position_ids`, `-100` boundary label guards,
    and zero cross-document attention leakage in `RootismForCausalLM`!
    """
    cfg = _load_tiny_cfg(tmp_path)
    model = RootismForCausalLM(cfg.model).eval()

    packer = SequencePacker(sequence_length=16, pad_token_id=cfg.model.pad_token_id)
    doc_a1 = [100, 101, 102, 103, 104, 105]
    doc_a2 = [250, 251, 252, 253, 254, 255]  # Altered Document 1
    doc_b = [150, 151, 152, 153, 154, 155]   # Identical Document 2

    mbs_1 = list(packer.build_microbatches([doc_a1, doc_b], micro_batch_size=1, flush_tail=True))
    mbs_2 = list(packer.build_microbatches([doc_a2, doc_b], micro_batch_size=1, flush_tail=True))

    mb1, mb2 = mbs_1[0], mbs_2[0]
    # Position IDs must reset to 0 at index 6 (start of doc_b)!
    assert mb1["position_ids"][0, :12].tolist() == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
    # Label at first token of doc_b (index 6, i.e. shift_labels[5]) must be -100 so doc_a never predicts doc_b!
    assert int(mb1["labels"][0, 6].item()) == -100

    with torch.no_grad():
        out1 = model(
            input_ids=mb1["input_ids"],
            position_ids=mb1["position_ids"],
            document_ids=mb1["document_ids"],
            labels=mb1["labels"],
        )
        out2 = model(
            input_ids=mb2["input_ids"],
            position_ids=mb2["position_ids"],
            document_ids=mb2["document_ids"],
            labels=mb2["labels"],
        )
    # Logits across Document 2 positions [6..11] must be IDENTICAL despite Document 1 changing completely!
    max_diff_doc2 = (out1.logits[0, 6:12] - out2.logits[0, 6:12]).abs().max().item()
    assert max_diff_doc2 < 1e-5


def test_stability_monitor_anomalies(tmp_path: Path) -> None:
    cfg = _load_tiny_cfg(tmp_path)
    model = RootismForCausalLM(cfg.model)
    monitor = TrainingStabilityMonitor(vanishing_patience=2, grad_explosion_threshold=10.0)
    monitor.attach_activation_hooks(model)

    inp = torch.randint(64, 300, (1, 16), dtype=torch.long)
    out = model(input_ids=inp, labels=inp)
    diag_ok = monitor.observe_step(step=1, loss=float(out.loss.item()), grad_norm=1.2)
    assert diag_ok.healthy is True
    assert diag_ok.max_activation_abs > 0.0

    # Loss spike & gradient explosion
    for s in range(2, 12):
        monitor.observe_step(step=s, loss=2.0, grad_norm=1.0)
    spike_diag = monitor.observe_step(step=12, loss=16.0, grad_norm=25.0)
    assert "LOSS_SPIKE" in spike_diag.anomalies
    assert "GRADIENT_EXPLOSION" in spike_diag.anomalies

    # Non-finite NaN detection
    nan_diag = monitor.observe_step(step=13, loss=float("nan"), grad_norm=1.0)
    assert "NON_FINITE_LOSS_NAN_OR_INF" in nan_diag.anomalies
    monitor.remove_hooks()


def test_atomic_checkpoint_resume_and_corrupted_fallback(tmp_path: Path) -> None:
    cfg = _load_tiny_cfg(tmp_path)
    docs = [
        {"token_ids": [((i * 13 + j) % 240) + 64 for j in range(40)]}
        for i in range(60)
    ]

    # Phase 1: Train for 2 steps and write checkpoint `step_00000002`
    trainer1 = RootismPretrainer(cfg)
    sum1 = trainer1.train(docs, max_steps=2)
    assert sum1.final_step == 2
    ckpt_root = Path(cfg.training.checkpoint_dir)
    assert (ckpt_root / "step_00000002").is_dir()

    # Simulate both an aborted `.tmp` checkpoint AND a corrupted newer checkpoint `step_00000003`!
    aborted_tmp = ckpt_root / "step_00000004.tmp"
    aborted_tmp.mkdir(parents=True)
    (aborted_tmp / "partial.bin").write_bytes(b"incomplete")

    corrupted_dir = ckpt_root / "step_00000003"
    corrupted_dir.mkdir(parents=True)
    for f in (ckpt_root / "step_00000002").iterdir():
        (corrupted_dir / f.name).write_bytes(f.read_bytes())
    # Flip bytes inside `step_00000003/model.safetensors` so SHA-256 verification catches corruption!
    bad_weights = bytearray((corrupted_dir / "model.safetensors").read_bytes())
    bad_weights[-1] ^= 0xFF
    (corrupted_dir / "model.safetensors").write_bytes(bytes(bad_weights))

    mgr = AtomicCheckpointManager(ckpt_root)
    with pytest.raises(CheckpointIntegrityError):
        mgr.validate_checkpoint_dir(corrupted_dir, expected_config=cfg)

    # Phase 2: Fresh `RootismPretrainer` must skip `.tmp` & corrupted `step_00000003`,
    # automatically recover `step_00000002`, and train until `target_training_tokens` (220) is reached!
    trainer2 = RootismPretrainer(cfg)
    sum2 = trainer2.train(docs, auto_resume=True)
    assert sum2.resumed_from_checkpoint is not None
    assert sum2.resumed_from_checkpoint.endswith("step_00000002")
    assert len(sum2.recovery_warnings) >= 2
    assert sum2.budget_reached is True
    assert sum2.token_snapshot.processed_training_tokens >= 900
    assert sum2.throughput_summary.measured_steps >= 1
