from __future__ import annotations

import pytest
import torch

from rootism.config import ModelConfig, RopeScalingConfig, tiny_test_config
from rootism.model.config_bridge import build_model
from rootism.tokenizer import RootismTokenizer
from rootism.tokenizer.pilot_corpus import build_pilot_corpus
from rootism.tokenizer.train import train_tokenizer
from rootism.config import TokenizerConfig


@pytest.fixture(scope="session")
def pilot_tokenizer() -> RootismTokenizer:
    docs = build_pilot_corpus(generated_documents=60)
    return train_tokenizer(
        docs,
        config=TokenizerConfig(vocab_size=1024, min_pair_frequency=1),
        validation_documents=docs,
    )


@pytest.fixture
def tiny_config() -> ModelConfig:
    return tiny_test_config()


@pytest.fixture
def tiny_model(tiny_config: ModelConfig):
    torch.manual_seed(0)
    return build_model(tiny_config, seed=0)


@pytest.fixture
def long_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=257,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        max_position_embeddings=32_768,
        rope_theta=100_000.0,
        rope_scaling=RopeScalingConfig(original_max_position_embeddings=32_768),
        loss_chunk_size=32,
        attention_query_chunk_size=2_048,
    )


@pytest.fixture
def long_model(long_config: ModelConfig):
    return build_model(long_config, seed=0)
