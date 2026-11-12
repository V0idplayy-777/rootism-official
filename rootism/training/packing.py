"""
Boundary-Safe Sequence Packing Engine (Step 53).

Packs multiple variable-length tokenized documents into dense `[micro_batch_size, sequence_length]`
training tensors to eliminate wasted padding while strictly preserving:
1. Sample Causal Isolation: emits `document_ids` (`0, 0, ..., 1, 1, ..., -1` for trailing pad)
   consumed directly by `RootismAttention`'s block-diagonal causal mask.
2. Position Reset: resets `position_ids` to `0, 1, 2, ...` at the start of every packed document
   so RoPE angles match standalone single-document positions!
3. Cross-Boundary Loss Masking: sets `labels[:, t] = IGNORE_INDEX (-100)` at the first token
   of every newly packed document (`t > 0` where `document_ids[t] != document_ids[t-1]`),
   preventing the final `<|eos|>` token of document k from being trained to predict the opening
   token of unrelated document k+1!
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import torch

from rootism.model.transformer import IGNORE_INDEX


class SequencePacker:
    """
    Stateful streaming sequence packer with checkpoint-serializable buffer state
    for exact mid-shard training resumption.
    """

    def __init__(
        self,
        *,
        sequence_length: int,
        pad_token_id: int = 0,
    ) -> None:
        if sequence_length < 4:
            raise ValueError("sequence_length must be >= 4")
        self.sequence_length = int(sequence_length)
        self.pad_token_id = int(pad_token_id)

        # Carry-over state when a long document spans across multiple packed sequences
        self._carry_tokens: list[int] = []
        self._carry_pos_start: int = 0
        self._carry_is_repeated: bool = False
        self.documents_ingested: int = 0
        self.packed_rows_emitted: int = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "pad_token_id": self.pad_token_id,
            "carry_tokens": list(self._carry_tokens),
            "carry_pos_start": self._carry_pos_start,
            "carry_is_repeated": self._carry_is_repeated,
            "documents_ingested": self.documents_ingested,
            "packed_rows_emitted": self.packed_rows_emitted,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.sequence_length = int(state["sequence_length"])
        self.pad_token_id = int(state["pad_token_id"])
        self._carry_tokens = [int(x) for x in state.get("carry_tokens", [])]
        self._carry_pos_start = int(state.get("carry_pos_start", 0))
        self._carry_is_repeated = bool(state.get("carry_is_repeated", False))
        self.documents_ingested = int(state.get("documents_ingested", 0))
        self.packed_rows_emitted = int(state.get("packed_rows_emitted", 0))

    def pack_rows(
        self,
        doc_stream: Iterable[dict[str, Any] | list[int] | np.ndarray],
        *,
        flush_tail_with_padding: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """
        Stream packed 1D sequence dictionaries of length `self.sequence_length`:
        `{input_ids, labels, position_ids, document_ids, repeated_active_tokens}`.
        """
        cur_tokens: list[int] = []
        cur_labels: list[int] = []
        cur_positions: list[int] = []
        cur_doc_ids: list[int] = []
        cur_doc_local_idx = 0
        cur_repeated_tokens = 0

        def _emit_current_row() -> dict[str, Any]:
            pad_needed = self.sequence_length - len(cur_tokens)
            if pad_needed > 0:
                cur_tokens.extend([self.pad_token_id] * pad_needed)
                cur_labels.extend([IGNORE_INDEX] * pad_needed)
                cur_positions.extend([0] * pad_needed)
                cur_doc_ids.extend([-1] * pad_needed)

            self.packed_rows_emitted += 1
            return {
                "input_ids": list(cur_tokens),
                "labels": list(cur_labels),
                "position_ids": list(cur_positions),
                "document_ids": list(cur_doc_ids),
                "repeated_active_tokens": cur_repeated_tokens,
            }

        def _append_span(tokens: list[int], pos_offset: int, is_repeated: bool) -> None:
            nonlocal cur_tokens, cur_labels, cur_positions, cur_doc_ids, cur_doc_local_idx, cur_repeated_tokens
            idx = 0
            n = len(tokens)
            while idx < n:
                rem_cap = self.sequence_length - len(cur_tokens)
                # Avoid stuffing a tiny 1-token orphan stub at the end of a row that has no next-token target
                if rem_cap == 1 and (n - idx) >= 2 and len(cur_tokens) > 0:
                    return_overflow = tokens[idx:]
                    self._carry_tokens = return_overflow
                    self._carry_pos_start = pos_offset + idx
                    self._carry_is_repeated = is_repeated
                    return

                take = min(rem_cap, n - idx)
                chunk = tokens[idx : idx + take]
                start_in_row = len(cur_tokens)

                cur_tokens.extend(chunk)
                chunk_labels = list(chunk)
                if start_in_row > 0 and pos_offset + idx == 0:
                    # First token of a newly packed document inside an existing row:
                    # NEVER let previous document's final token predict this boundary!
                    chunk_labels[0] = IGNORE_INDEX

                cur_labels.extend(chunk_labels)
                cur_positions.extend(range(pos_offset + idx, pos_offset + idx + take))
                cur_doc_ids.extend([cur_doc_local_idx] * take)

                if is_repeated:
                    # Count active target labels contributed by this repeated-epoch span (excluding slot 0 of row)
                    for local_k, lbl in enumerate(chunk_labels):
                        row_pos = start_in_row + local_k
                        if row_pos > 0 and lbl != IGNORE_INDEX:
                            cur_repeated_tokens += 1

                idx += take
                if len(cur_tokens) == self.sequence_length:
                    if idx < n:
                        self._carry_tokens = tokens[idx:]
                        self._carry_pos_start = pos_offset + idx
                        self._carry_is_repeated = is_repeated
                    return
            cur_doc_local_idx += 1

        # Drain any prior checkpoint carry-over first
        while self._carry_tokens:
            c_toks = self._carry_tokens
            c_pos = self._carry_pos_start
            c_rep = self._carry_is_repeated
            self._carry_tokens = []
            self._carry_pos_start = 0
            _append_span(c_toks, c_pos, c_rep)
            if len(cur_tokens) == self.sequence_length or self._carry_tokens:
                yield _emit_current_row()
                cur_tokens, cur_labels, cur_positions, cur_doc_ids = [], [], [], []
                cur_doc_local_idx = 0
                cur_repeated_tokens = 0

        for item in doc_stream:
            if isinstance(item, dict):
                raw_ids = item["token_ids"]
                tok_list = raw_ids.tolist() if isinstance(raw_ids, np.ndarray) else [int(x) for x in raw_ids]
                is_rep = bool(item.get("is_repeated_epoch", False))
            elif isinstance(item, np.ndarray):
                tok_list = item.tolist()
                is_rep = False
            else:
                tok_list = [int(x) for x in item]
                is_rep = False

            if len(tok_list) < 2:
                continue

            self.documents_ingested += 1
            _append_span(tok_list, 0, is_rep)

            while len(cur_tokens) == self.sequence_length or self._carry_tokens:
                if len(cur_tokens) > 0:
                    yield _emit_current_row()
                    cur_tokens, cur_labels, cur_positions, cur_doc_ids = [], [], [], []
                    cur_doc_local_idx = 0
                    cur_repeated_tokens = 0
                if self._carry_tokens:
                    c_toks = self._carry_tokens
                    c_pos = self._carry_pos_start
                    c_rep = self._carry_is_repeated
                    self._carry_tokens = []
                    self._carry_pos_start = 0
                    _append_span(c_toks, c_pos, c_rep)
                    if len(cur_tokens) < self.sequence_length and not self._carry_tokens:
                        break

        if flush_tail_with_padding and len(cur_tokens) >= 2:
            yield _emit_current_row()

    def build_microbatches(
        self,
        doc_stream: Iterable[dict[str, Any] | list[int] | np.ndarray],
        *,
        micro_batch_size: int,
        flush_tail: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Group packed rows into PyTorch tensor microbatches ready for `TrainingStepExecutor`."""
        if micro_batch_size < 1:
            raise ValueError("micro_batch_size must be >= 1")

        batch_rows: list[dict[str, Any]] = []
        for row in self.pack_rows(doc_stream, flush_tail_with_padding=flush_tail):
            batch_rows.append(row)
            if len(batch_rows) == micro_batch_size:
                yield self._collate_rows(batch_rows)
                batch_rows = []

        if flush_tail and batch_rows:
            yield self._collate_rows(batch_rows)

    @staticmethod
    def _collate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = torch.tensor([r["input_ids"] for r in rows], dtype=torch.long)
        labels = torch.tensor([r["labels"] for r in rows], dtype=torch.long)
        position_ids = torch.tensor([r["position_ids"] for r in rows], dtype=torch.long)
        document_ids = torch.tensor([r["document_ids"] for r in rows], dtype=torch.long)
        rep_tokens = sum(int(r["repeated_active_tokens"]) for r in rows)
        active_total = int((labels[:, 1:] != IGNORE_INDEX).sum().item())
        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "document_ids": document_ids,
            "repeated_epoch_tokens": rep_tokens,
            "is_repeated_epoch": rep_tokens > (active_total // 2),
        }
