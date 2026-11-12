from __future__ import annotations

import json

import pytest
import torch

from rootism.config import ModelConfig, tiny_test_config
from rootism.io import (
    checkpoint_path,
    find_recovery_point,
    load_checkpoint,
    load_model_checkpoint,
    save_checkpoint,
    save_model_checkpoint,
    verify_checkpoint,
)
from rootism.model.config_bridge import build_model


def test_generic_checkpoint_roundtrip(tmp_path):
    state = {
        "model": {"w": torch.arange(12).reshape(3, 4).float()},
        "optimizer": {"step": 42, "lr": 1e-4},
        "rng_state": torch.tensor([1, 2, 3], dtype=torch.uint8),
        "step": 42,
    }
    path = checkpoint_path(tmp_path, 42)
    manifest = save_checkpoint(state, path, metadata={"step": 42, "tokenizer_id": "tok-test"})
    loaded, loaded_manifest = load_checkpoint(path)
    assert torch.equal(loaded["model"]["w"], state["model"]["w"])
    assert loaded["optimizer"] == state["optimizer"]
    assert torch.equal(loaded["rng_state"], state["rng_state"])
    assert loaded_manifest["sha256"] == manifest["sha256"]
    assert verify_checkpoint(path)


def test_atomic_manifest_detects_corruption(tmp_path):
    path = checkpoint_path(tmp_path, 1)
    save_checkpoint({"x": torch.ones(4)}, path, metadata={"step": 1})
    raw = bytearray(path.read_bytes())
    raw[-2] ^= 0xFF
    path.write_bytes(bytes(raw))
    assert verify_checkpoint(path) is False
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_checkpoint(path)


def test_recovery_point_skips_incomplete_checkpoint(tmp_path):
    for step in (1, 2, 3):
        save_checkpoint({"x": torch.full((4,), float(step))}, checkpoint_path(tmp_path, step),
                        metadata={"step": step})
    corrupted = checkpoint_path(tmp_path, 3)
    data = json.loads((tmp_path / (corrupted.name + ".manifest.json")).read_text())
    data["step"] = 999  # valid JSON but wrong metadata is still digest-checked
    (tmp_path / (corrupted.name + ".manifest.json")).write_text(json.dumps(data))
    assert verify_checkpoint(corrupted) is True  # data bytes are intact
    raw = bytearray(corrupted.read_bytes())
    raw[-1] ^= 0xFF
    corrupted.write_bytes(bytes(raw))
    recovered = find_recovery_point(tmp_path)
    assert recovered == checkpoint_path(tmp_path, 2)


def test_model_checkpoint_restores_exact_outputs(tiny_model, tiny_config, pilot_tokenizer, tmp_path):
    path = checkpoint_path(tmp_path, 7)
    manifest = save_model_checkpoint(tiny_model, path, tokenizer_id=pilot_tokenizer.tokenizer_id, step=7)
    assert manifest["config_id"] == tiny_config.config_id
    assert manifest["architecture_id"] == tiny_config.architecture_id
    restored, loaded = load_model_checkpoint(path, tokenizer_id=pilot_tokenizer.tokenizer_id)
    x = torch.randint(4, tiny_config.vocab_size, (1, 6))
    assert torch.equal(tiny_model(x).logits, restored(x).logits)
    assert loaded["step"] == 7


def test_tokenizer_version_mismatch_is_rejected(tiny_model, pilot_tokenizer, tmp_path):
    path = checkpoint_path(tmp_path, 1)
    save_model_checkpoint(tiny_model, path, tokenizer_id=pilot_tokenizer.tokenizer_id)
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        load_model_checkpoint(path, tokenizer_id="tok-a-different-tokenizer")


def test_model_config_mismatch_is_rejected(tiny_model, tmp_path):
    path = checkpoint_path(tmp_path, 1)
    save_model_checkpoint(tiny_model, path)
    other = tiny_test_config(hidden_size=96, num_attention_heads=6, head_dim=16)
    with pytest.raises(ValueError, match="different model configuration"):
        load_model_checkpoint(path, expected_config=other)


def test_checkpoint_filename_and_manifest_are_adjacent(tmp_path):
    path = checkpoint_path(tmp_path, 123)
    save_checkpoint({"a": torch.zeros(1)}, path, metadata={"step": 123})
    assert (tmp_path / "ckpt-step-000000123.pt.manifest.json").exists()
    assert ModelConfig().config_id
