"""Automated Tests for Phase I: Project Identity & Non-Negotiable Requirements (Steps 01–10)."""

from __future__ import annotations

import pytest

from rootism.config import (
    DEFAULT_PRODUCTION_TRAINING_TOKENS,
    MINIMUM_PRODUCTION_TRAINING_TOKENS,
    RootismConfig,
    RootismModelConfig,
    RootismTrainingConfig,
)
from rootism.core.provenance import (
    PretrainedDependencyViolationError,
    assert_independent_environment,
    create_reproducibility_manifest,
    seed_everything,
    verify_rootism_checkpoint_provenance,
)
from rootism.core.token_accounting import TokenAccountingLedger
from rootism.model.transformer import RootismForCausalLM


def test_independent_environment_has_zero_pretrained_frameworks() -> None:
    assert_independent_environment()


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "pretrained_model_name_or_path",
        "hf_model_id",
        "teacher_model",
        "distillation_teacher",
        "lora_base_checkpoint",
    ],
)
def test_configs_reject_pretrained_and_distillation_fields(forbidden_key: str) -> None:
    with pytest.raises(ValueError, match="Prohibited pretrained/distillation"):
        RootismModelConfig.model_validate({forbidden_key: "external/model"})

    with pytest.raises(ValueError, match="Prohibited pretrained/distillation"):
        RootismTrainingConfig.model_validate({forbidden_key: "external/teacher"})


def test_production_training_token_budget_enforces_strictly_above_600b() -> None:
    default_cfg = RootismTrainingConfig()
    assert default_cfg.target_training_tokens == DEFAULT_PRODUCTION_TRAINING_TOKENS == 650_000_000_000

    # Strictly MORE THAN 600B required: exact 600B must fail!
    with pytest.raises(ValueError, match="MORE THAN 600,000,000,000"):
        RootismTrainingConfig(target_training_tokens=MINIMUM_PRODUCTION_TRAINING_TOKENS)

    with pytest.raises(ValueError, match="MORE THAN 600,000,000,000"):
        RootismTrainingConfig(target_training_tokens=100_000_000_000)

    # Larger budgets (e.g. 1 Trillion tokens) are supported
    larger = RootismTrainingConfig(target_training_tokens=1_000_000_000_000)
    assert larger.target_training_tokens == 1_000_000_000_000


def test_six_category_token_accounting_ledger_prevents_inflation() -> None:
    ledger = TokenAccountingLedger(target_training_tokens=650_000_000_000)
    ledger.register_corpus_statistics(
        unique_source_tokens=900_000_000_000,
        deduplicated_tokens=500_000_000_000,
    )

    # Step 1: 2 microbatches of 4096 slots each, 3800 active non-pad tokens, 500 from epoch 2
    ledger.record_microbatch(total_sequence_slots=4096, active_loss_tokens=3800, repeated_epoch_tokens=0)
    ledger.record_microbatch(total_sequence_slots=4096, active_loss_tokens=3800, repeated_epoch_tokens=500)
    snap1 = ledger.commit_optimizer_step(step_valid=True)

    assert snap1.unique_source_tokens == 900_000_000_000
    assert snap1.deduplicated_tokens == 500_000_000_000
    assert snap1.presented_tokens == 8192
    assert snap1.padding_and_masked_tokens == 592
    assert snap1.repeated_epoch_tokens == 500
    assert snap1.processed_training_tokens == 7600
    assert snap1.completed_steps == 1

    # Step 2: Unstable NaN step -> must NOT credit processed_training_tokens!
    ledger.record_microbatch(total_sequence_slots=4096, active_loss_tokens=4000, repeated_epoch_tokens=100)
    snap2 = ledger.commit_optimizer_step(step_valid=False)
    assert snap2.processed_training_tokens == 7600
    assert snap2.skipped_unstable_step_tokens == 4000
    assert snap2.completed_steps == 1

    # State serialization roundtrip
    restored = TokenAccountingLedger()
    restored.load_state_dict(ledger.state_dict())
    assert restored.snapshot() == snap2


def test_checkpoint_provenance_verification_and_reproducibility_manifest() -> None:
    tiny_cfg = RootismModelConfig.tiny_verification().model_copy(
        update={"tokenizer_version_fingerprint": "abc123fingerprint"}
    )
    model = RootismForCausalLM(tiny_cfg)
    meta = model.provenance_metadata()
    verify_rootism_checkpoint_provenance(
        meta,
        expected_model_config=tiny_cfg,
        expected_tokenizer_fingerprint="abc123fingerprint",
    )

    # Reject external checkpoint
    with pytest.raises(PretrainedDependencyViolationError):
        verify_rootism_checkpoint_provenance({"origin_signature": "huggingface-llama"})

    # Reject silent tokenizer version drift
    with pytest.raises(PretrainedDependencyViolationError, match="Tokenizer version mismatch"):
        verify_rootism_checkpoint_provenance(
            meta,
            expected_model_config=tiny_cfg,
            expected_tokenizer_fingerprint="different_tokenizer_sha256",
        )

    master_cfg = RootismConfig.from_yaml("configs/training/pretrain_650b.yaml")
    notes = seed_everything(master_cfg.training.seed, deterministic=False)
    manifest = create_reproducibility_manifest(
        config=master_cfg,
        tokenizer_fingerprint="tok_sha256",
        dataset_manifest_sha256="data_sha256",
        nondeterminism_notes=notes,
    )
    assert len(manifest.manifest_sha256()) == 64
    assert manifest.random_seed == 20261112
