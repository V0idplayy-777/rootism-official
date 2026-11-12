"""Static, preallocated KV cache for Rootism inference (requirements 14, 78)."""

from __future__ import annotations

import torch

from ..config import ModelConfig
from ..utils.seed import resolve_dtype


class RootismKVCache:
    """Per-layer (B, KV_heads, max_seq_len, head_dim) buffers.

    Preallocation avoids reallocation/copying during autoregressive decoding.
    `prepare()`/`commit()` bracket one forward pass so that every layer sees the
    same visible length, including when positions are non-contiguous (e.g. a
    long-context seek test).
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        batch_size: int = 1,
        max_seq_len: int | None = None,
        dtype: str | torch.dtype = "float32",
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len or config.max_position_embeddings
        self.dtype = resolve_dtype(dtype)
        self.device = torch.device(device)
        shape = (
            batch_size,
            config.num_key_value_heads,
            self.max_seq_len,
            config.head_dim,
        )
        empty = lambda: torch.empty(shape, dtype=self.dtype, device=self.device)
        self.k = [empty() for _ in range(config.num_hidden_layers)]
        self.v = [empty() for _ in range(config.num_hidden_layers)]
        self.length = 0
        self._next_length = 0

    def prepare(self, position_ids: torch.Tensor) -> None:
        self._next_length = max(self.length, int(position_ids.max().item()) + 1)
        if self._next_length > self.max_seq_len:
            raise IndexError(
                f"KV cache overflow: need {self._next_length} slots, allocated {self.max_seq_len}"
            )

    def commit(self) -> None:
        self.length = self._next_length

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new K/V at absolute positions and return the visible prefix."""
        bsz, kv_heads, q_len, head_dim = k.shape
        if bsz != self.batch_size:
            raise ValueError("KV cache batch size mismatch")
        positions = position_ids.to(torch.long)
        b_idx = torch.arange(bsz, device=k.device)[:, None].expand(bsz, q_len)
        self.k[layer_idx][b_idx, :, positions] = k.permute(0, 2, 1, 3)
        self.v[layer_idx][b_idx, :, positions] = v.permute(0, 2, 1, 3)
        length = self._next_length
        return self.k[layer_idx][:, :, :length, :], self.v[layer_idx][:, :, :length, :]

    @property
    def bytes_per_token(self) -> int:
        elements = 2 * self.config.num_hidden_layers * self.config.num_key_value_heads * self.config.head_dim
        return elements * torch.empty((), dtype=self.dtype).element_size()

    def memory_bytes(self, seq_len: int | None = None) -> int:
        return self.bytes_per_token * (seq_len or self.max_seq_len) * self.batch_size
