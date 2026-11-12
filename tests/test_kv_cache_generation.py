from __future__ import annotations

import threading

import pytest
import torch

from rootism.model.generation import ByteStreamDecoder, Generator, SamplingConfig, sample_next_token
from rootism.model.kv_cache import RootismKVCache


def test_kv_cache_matches_full_recompute(tiny_model, tiny_config):
    cfg = tiny_config
    prompt = [5, 9, 11, 4, 7]
    x = torch.tensor([prompt])
    cache = RootismKVCache(cfg, batch_size=1, max_seq_len=32, dtype=torch.float32)
    with torch.no_grad():
        full = tiny_model(x, use_cache=False, num_logits_to_keep=1).logits[0, -1]
        cached = tiny_model(x, past_key_values=cache, use_cache=True, num_logits_to_keep=1).logits[0, -1]
    assert torch.allclose(full, cached, atol=1e-6)
    assert cache.length == len(prompt)
    with torch.no_grad():
        nxt = torch.tensor([[12]])
        full2 = tiny_model(torch.cat([x, nxt], 1), use_cache=False, num_logits_to_keep=1).logits[0, -1]
        cached2 = tiny_model(nxt, past_key_values=cache, use_cache=True, num_logits_to_keep=1).logits[0, -1]
    assert torch.allclose(full2, cached2, atol=1e-5)
    assert cache.length == len(prompt) + 1


def test_cache_overflow_is_rejected(tiny_config):
    cache = RootismKVCache(tiny_config, batch_size=1, max_seq_len=4, dtype=torch.float32)
    with pytest.raises(IndexError, match="overflow"):
        cache.prepare(torch.tensor([[9]]))


def test_streaming_events_and_finish_reasons(tiny_model, tiny_config):
    g = Generator(tiny_model)
    cfg = SamplingConfig(max_new_tokens=4, do_sample=False)
    events = list(g.stream([5, 6, 7], cfg))
    assert [e.type for e in events] == ["token"] * 4 + ["done"]
    assert events[-1].finish_reason == "max_new_tokens"
    assert len(events[-1].token_ids) == 4
    assert events[-1].prompt_tokens == 3


def test_eos_and_stop_token_terminate_generation(tiny_model, tiny_config):
    g = Generator(tiny_model)
    # Greedy decoding on a random model picks a fixed first token; force it as EOS.
    cfg = SamplingConfig(max_new_tokens=10, do_sample=False)
    first = g.generate([5, 6], cfg).token_ids[0]
    cfg2 = SamplingConfig(max_new_tokens=10, do_sample=False, stop_token_ids=(first,))
    res = g.generate([5, 6], cfg2)
    assert res.finish_reason == "stop_token"
    assert len(res.token_ids) == 1
    cfg3 = SamplingConfig(max_new_tokens=10, do_sample=False, eos_token_id=first)
    assert g.generate([5, 6], cfg3).finish_reason == "eos"


def test_sampling_is_reproducible_with_seed(tiny_model):
    g = Generator(tiny_model)
    cfg = SamplingConfig(max_new_tokens=8, do_sample=True, temperature=1.0, top_k=10, top_p=0.9, seed=17)
    a = g.generate([5, 6, 7], cfg)
    b = g.generate([5, 6, 7], cfg)
    assert a.token_ids == b.token_ids


def test_cancel_event_stops_generation(tiny_model):
    g = Generator(tiny_model)
    cancel = threading.Event()
    cfg = SamplingConfig(max_new_tokens=50, do_sample=False)
    seen = 0
    for event in g.stream([5, 6], cfg, cancel=cancel):
        if event.type == "token":
            seen += 1
            if seen == 2:
                cancel.set()
    assert seen == 2


def test_sampling_config_validation():
    with pytest.raises(ValueError, match="temperature"):
        SamplingConfig(temperature=0.0).validate()
    with pytest.raises(ValueError, match="top_p"):
        SamplingConfig(top_p=1.5).validate()
    with pytest.raises(ValueError, match="max_new_tokens"):
        SamplingConfig(max_new_tokens=0).validate()


def test_byte_stream_decoder_handles_split_utf8():
    dec = ByteStreamDecoder()
    data = "héllo 🚀".encode("utf-8")
    assert dec.push(data[:2]) == "h"  # 'é' is buffered mid-codepoint
    assert dec.push(data[2:]) == "éllo 🚀"
    assert dec.finish() == ""
    assert "h" + "éllo 🚀" == "héllo 🚀"


def test_top_k_and_repetition_penalty_are_applied():
    logits = torch.tensor([1.0, 5.0, 3.0, 2.0])
    cfg = SamplingConfig(top_k=1, temperature=1.0, repetition_penalty=10.0, seed=0)
    gen = torch.Generator().manual_seed(0)
    for _ in range(5):
        assert sample_next_token(logits, cfg, generated=[], step=0, generator=gen) == 1
    # rep penalty pushes the previously generated token away from argmax
    cfg2 = SamplingConfig(top_k=0, top_p=1.0, temperature=1.0, repetition_penalty=10.0, seed=0)
    penalties = [sample_next_token(logits, cfg2, generated=[1] * 20, step=0, generator=gen) for _ in range(20)]
    assert any(t != 1 for t in penalties)


@pytest.mark.slow
def test_decode_at_position_32767_with_full_size_cache(long_model, long_config):
    """Long-context support: real forward at the END of a 32,768-slot KV cache.

    This verifies RoPE at position 32,767, cache capacity/indexing and a
    32,768-key decode step. A full 32K-token *prefill* requires GPU memory and
    is explicitly marked unverified in docs/STATUS.md.
    """
    cache = RootismKVCache(long_config, batch_size=1, max_seq_len=32_768, dtype=torch.float32)
    for buf in cache.k + cache.v:
        buf.zero_()
    with torch.no_grad():
        first = long_model(torch.tensor([[5]]), past_key_values=cache, use_cache=True)
        assert torch.isfinite(first.logits).all()
        seek = long_model(
            torch.tensor([[7]]),
            past_key_values=cache,
            use_cache=True,
            position_ids=torch.tensor([[32_767]]),
        )
    assert torch.isfinite(seek.logits).all()
    assert cache.length == 32_768
    assert cache.bytes_per_token == 2 * 2 * 2 * 16 * 4
    assert cache.memory_bytes(32_768) == cache.bytes_per_token * 32_768
