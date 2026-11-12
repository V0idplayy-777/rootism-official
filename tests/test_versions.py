from __future__ import annotations

import json

import pytest

from rootism.config import ModelConfig
from rootism.version import (
    MODEL_ARCHITECTURE_ID,
    SCHEMA_VERSION,
    compatibility_report,
    content_hash,
    file_sha256,
    short_id,
)


def test_config_identity_changes_with_architecture():
    base = ModelConfig()
    deeper = base.with_overrides(num_hidden_layers=29)
    assert base.config_id != deeper.config_id
    assert base.config_id == base.with_overrides().config_id
    assert deeper.architecture_id == MODEL_ARCHITECTURE_ID


def test_compatibility_report_fails_closed():
    compatibility_report(
        config_id="cfg-1", tokenizer_id="tok-1",
        checkpoint_config_id="cfg-1", checkpoint_tokenizer_id="tok-1",
        checkpoint_architecture_id=MODEL_ARCHITECTURE_ID,
    )
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        compatibility_report(
            config_id="cfg-1", tokenizer_id="tok-1",
            checkpoint_config_id="cfg-1", checkpoint_tokenizer_id="tok-2",
            checkpoint_architecture_id=None,
        )
    with pytest.raises(ValueError, match="different model configuration"):
        compatibility_report(
            config_id="cfg-1", tokenizer_id="tok-1",
            checkpoint_config_id="cfg-9", checkpoint_tokenizer_id=None,
            checkpoint_architecture_id=None,
        )
    with pytest.raises(ValueError, match="architecture"):
        compatibility_report(
            config_id="cfg-1", tokenizer_id="tok-1",
            checkpoint_config_id=None, checkpoint_tokenizer_id=None,
            checkpoint_architecture_id="some-other-architecture",
        )


def test_hashing_helpers(tmp_path):
    assert content_hash({"a": 1}) == content_hash({"a": 1})
    assert content_hash({"a": 1}) != content_hash({"a": 2})
    assert short_id("x", "abcdef1234567890", length=4) == "x-abcd"
    path = tmp_path / "f.bin"
    path.write_bytes(b"rootism")
    assert len(file_sha256(str(path))) == 64
    assert file_sha256(str(path)) == file_sha256(str(path))


def test_config_load_rejects_future_schema(tmp_path):
    cfg = ModelConfig()
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({**cfg.to_dict(), "schema_version": "999"}))
    with pytest.raises(ValueError, match="schema mismatch"):
        ModelConfig.load(path)
    assert SCHEMA_VERSION == "1"
