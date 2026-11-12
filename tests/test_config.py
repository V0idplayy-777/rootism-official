from __future__ import annotations

import pytest

from rootism.config import (
    DEFAULT_TRAINING_TOKENS,
    HARD_MINIMUM_TRAINING_TOKENS,
    ModelConfig,
    RopeScalingConfig,
    TokenizerConfig,
    TrainingTokenBudget,
    tiny_test_config,
)
from rootism.version import SCHEMA_VERSION, assert_schema, canonical_json, content_hash


def test_production_config_is_valid_and_stable():
    cfg = ModelConfig()
    cfg.validate()
    assert cfg.max_position_embeddings >= 32_768
    assert cfg.config_id == ModelConfig().config_id
    assert cfg.schema_version == SCHEMA_VERSION
    assert content_hash(cfg.to_dict()) == content_hash(ModelConfig().to_dict())
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_config_serialization_roundtrip(tmp_path):
    cfg = ModelConfig()
    path = tmp_path / "config.json"
    cfg.save(path)
    loaded = ModelConfig.load(path)
    assert loaded == cfg
    assert loaded.config_id == cfg.config_id


def test_unknown_and_invalid_config_fields_rejected():
    with pytest.raises(ValueError, match="unknown ModelConfig fields"):
        ModelConfig.from_dict({**ModelConfig().to_dict(), "surprise": 1})
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(num_attention_heads=25, num_key_value_heads=8).validate()
    with pytest.raises(ValueError, match="at least 32768"):
        ModelConfig(max_position_embeddings=8_192).validate()
    with pytest.raises(ValueError, match="unknown TokenizerConfig fields"):
        TokenizerConfig.from_dict({**TokenizerConfig().to_dict(), "surprise": 1})


def test_short_context_only_allowed_explicitly():
    tiny = tiny_test_config()
    assert tiny.allow_short_context is True
    tiny.validate()
    with pytest.raises(ValueError):
        ModelConfig(max_position_embeddings=1024).validate()


def test_rope_scaling_validation():
    with pytest.raises(ValueError):
        RopeScalingConfig(type="magic").validate()
    with pytest.raises(ValueError, match="does not cover"):
        ModelConfig(
            max_position_embeddings=65_536,
            rope_scaling=RopeScalingConfig(type="yarn", factor=1.5, original_max_position_embeddings=32_768),
        ).validate()
    ModelConfig(
        max_position_embeddings=65_536,
        rope_scaling=RopeScalingConfig(type="yarn", factor=2.0, original_max_position_embeddings=32_768),
    ).validate()


def test_tokenizer_config_validation():
    TokenizerConfig().validate()
    with pytest.raises(ValueError):
        TokenizerConfig(vocab_size=128).validate()
    with pytest.raises(ValueError):
        TokenizerConfig(normalizer="nfkc").validate()


def test_training_budget_enforces_600b_minimum():
    budget = TrainingTokenBudget()
    budget.validate()
    assert budget.target_tokens == DEFAULT_TRAINING_TOKENS == 650_000_000_000
    assert budget.target_tokens > HARD_MINIMUM_TRAINING_TOKENS == 600_000_000_000
    assert budget.minimum_unique_tokens == 162_500_000_000
    with pytest.raises(ValueError, match="MORE THAN"):
        TrainingTokenBudget(target_tokens=600_000_000_000).validate()
    with pytest.raises(ValueError, match="MORE THAN"):
        TrainingTokenBudget(target_tokens=500_000_000_000).validate()
    with pytest.raises(ValueError, match="below the initial 650B"):
        TrainingTokenBudget(target_tokens=601_000_000_000).validate()
    # A larger budget is allowed.
    TrainingTokenBudget(target_tokens=1_000_000_000_000).validate()


def test_schema_mismatch_is_rejected():
    with pytest.raises(ValueError, match="schema mismatch"):
        assert_schema({"schema_version": "999"}, kind="artifact")
