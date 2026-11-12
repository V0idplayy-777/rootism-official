"""Chunked cross-entropy: exact loss without materialising full-vocabulary logits.

At 32768 tokens x 49152 vocab, materialising logits in fp32 would require
~6.4 GiB for a single sequence (and would be duplicated by autograd). Computing
the loss in token chunks bounds that buffer to `chunk_size * vocab` while
producing the mathematically identical sum reduction.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch.nn import functional as F


def chunked_causal_lm_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    chunk_size: int = 512,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (sum_of_losses, number_of_valid_tokens).

    The sum reduction is intentional: the distributed trainer divides by the
    *global* count of non-padding tokens, not by a per-microbatch mean.
    """
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("hidden_states and labels must already be shift-aligned")
    bsz, seq_len, hidden_size = hidden_states.shape
    flat_hidden = hidden_states.reshape(-1, hidden_size)
    flat_labels = labels.reshape(-1)
    total = torch.zeros((), dtype=torch.float32, device=hidden_states.device)
    for start in range(0, flat_hidden.shape[0], chunk_size):
        stop = min(start + chunk_size, flat_hidden.shape[0])
        logits = logits_fn(flat_hidden[start:stop]).to(torch.float32)
        total = total + F.cross_entropy(
            logits, flat_labels[start:stop], ignore_index=ignore_index, reduction="sum"
        )
    valid = (flat_labels != ignore_index).sum()
    return total, valid
