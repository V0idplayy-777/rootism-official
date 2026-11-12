from __future__ import annotations

import torch

from rootism.config import tiny_test_config
from rootism.model.config_bridge import build_model
from rootism.model.generation import Generator, SamplingConfig


def test_tokenizer_to_model_to_generation_roundtrip(pilot_tokenizer):
    """End-to-end: real token ids -> real forward -> real generation -> real text."""
    tok = pilot_tokenizer
    cfg = tiny_test_config(vocab_size=tok.vocab_size, max_position_embeddings=128)
    model = build_model(cfg, seed=0)
    prompt = tok.encode("def f():\n    return 1\n", add_special_tokens=True)
    assert prompt[0] == tok.bos_id

    x = torch.tensor([prompt])
    out = model(x, labels=x)
    assert torch.isfinite(out.loss)
    assert out.num_valid_tokens.item() == len(prompt) - 1

    generator = Generator(model, token_bytes=tok.token_bytes)
    events = list(generator.stream(prompt, SamplingConfig(max_new_tokens=5, do_sample=False)))
    tokens = [e for e in events if e.type == "token"]
    done = events[-1]
    assert done.type == "done"
    assert len(tokens) == 5
    assert done.text.startswith("".join(e.text_delta for e in tokens))
    assert isinstance(done.text, str)
    assert all(0 <= t < tok.vocab_size for t in done.token_ids)
    # Generated ids must all be decodable through the real tokenizer.
    assert isinstance(tok.decode(done.token_ids, errors="replace"), str)


def test_batch_with_padding_and_document_packing_are_both_trainable(pilot_tokenizer):
    tok = pilot_tokenizer
    cfg = tiny_test_config(vocab_size=tok.vocab_size)
    model = build_model(cfg, seed=1)
    a = tok.encode("alpha = 1", add_special_tokens=True)
    b = tok.encode("beta = 22", add_special_tokens=True)
    # right-padded batch
    max_len = max(len(a), len(b))
    padded = [a + [tok.pad_id] * (max_len - len(a)), b + [tok.pad_id] * (max_len - len(b))]
    mask = torch.tensor([[1] * len(a) + [0] * (max_len - len(a)),
                         [1] * len(b) + [0] * (max_len - len(b))])
    ids = torch.tensor(padded)
    labels = ids.clone()
    labels[mask == 0] = -100
    out = model(ids, attention_mask=mask, labels=labels)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    # packed single-row sequence with document boundaries
    packed = a + b
    doc = torch.tensor([[0] * len(a) + [1] * len(b)])
    out2 = model(torch.tensor([packed]), document_ids=doc, labels=torch.tensor([packed]))
    assert torch.isfinite(out2.loss)
