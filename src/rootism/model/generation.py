"""Autoregressive inference with KV caching and configurable sampling (req. 78).

This module runs the real RootismForCausalLM forward pass. There is no canned
response, template answer or substitute model anywhere in this path.
"""

from __future__ import annotations

import codecs
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

import torch
from torch.nn import functional as F

from ..config import ModelConfig
from .kv_cache import RootismKVCache
from .language_model import RootismForCausalLM


@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int = 256
    min_new_tokens: int = 0
    do_sample: bool = True
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    seed: int | None = None
    eos_token_id: int = 2
    pad_token_id: int = 0
    stop_token_ids: tuple[int, ...] = ()
    stop_strings: tuple[str, ...] = ()
    stop_sequences: tuple[tuple[int, ...], ...] = ()

    def validate(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.min_new_tokens < 0 or self.min_new_tokens > self.max_new_tokens:
            raise ValueError("min_new_tokens out of range")
        if self.do_sample:
            if self.temperature <= 0:
                raise ValueError("temperature must be > 0")
            if not 0.0 < self.top_p <= 1.0:
                raise ValueError("top_p must be in (0, 1]")
            if self.top_k < 0:
                raise ValueError("top_k must be >= 0")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")


class ByteStreamDecoder:
    """Incremental UTF-8 decoding so multi-byte tokens stream correctly."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, data: bytes) -> str:
        return self._decoder.decode(data, final=False)

    def finish(self) -> str:
        return self._decoder.decode(b"", final=True)


@dataclass
class GenerationEvent:
    type: str  # "token" | "done"
    token_id: int | None = None
    text_delta: str = ""
    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int = 0
    generated_tokens: int = 0


@dataclass
class GenerationResult:
    token_ids: list[int]
    text: str
    finish_reason: str
    prompt_tokens: int
    generated_tokens: int


def _apply_repetition_penalty(logits: torch.Tensor, seen: Sequence[int], penalty: float) -> None:
    if penalty == 1.0 or not seen:
        return
    ids = torch.tensor(list(set(seen)), device=logits.device, dtype=torch.long)
    score = logits.gather(0, ids)
    score = torch.where(score > 0, score / penalty, score * penalty)
    logits.scatter_(0, ids, score)


def sample_next_token(
    logits: torch.Tensor,
    cfg: SamplingConfig,
    *,
    generated: Sequence[int],
    step: int,
    generator: torch.Generator | None,
) -> int:
    logits = logits.clone()
    _apply_repetition_penalty(logits, generated, cfg.repetition_penalty)
    if step < cfg.min_new_tokens:
        logits[cfg.eos_token_id] = -float("inf")
        for tid in cfg.stop_token_ids:
            logits[tid] = -float("inf")

    if not cfg.do_sample:
        return int(torch.argmax(logits).item())

    logits = logits / cfg.temperature
    if cfg.top_k > 0:
        k = min(cfg.top_k, logits.shape[-1])
        cutoff = torch.topk(logits, k).values[..., -1]
        logits = torch.where(logits < cutoff, torch.full_like(logits, -float("inf")), logits)
    if cfg.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative - probs > cfg.top_p
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all() or probs.sum() <= 0:
        return int(torch.argmax(logits).item())
    return int(torch.multinomial(probs, 1, generator=generator).item())


class Generator:
    """Streaming generator bound to a real Rootism checkpoint-backed model."""

    def __init__(
        self,
        model: RootismForCausalLM,
        *,
        token_bytes: Callable[[int], bytes] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.config: ModelConfig = model.config
        self.token_bytes = token_bytes
        if device is not None:
            self.model.to(torch.device(device))
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

    def stream(
        self,
        prompt_ids: Sequence[int],
        cfg: SamplingConfig,
        *,
        cancel: threading.Event | None = None,
    ) -> Iterator[GenerationEvent]:
        cfg.validate()
        prompt = list(prompt_ids)
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        max_len = min(
            self.config.max_position_embeddings,
            len(prompt) + cfg.max_new_tokens + 1,
        )
        if max_len <= len(prompt):
            raise ValueError("prompt leaves no room for generation within the context window")

        with torch.inference_mode():
            cache = RootismKVCache(
                self.config,
                batch_size=1,
                max_seq_len=max_len,
                dtype=self.dtype,
                device=self.device,
            )
            input_ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
            output = self.model(input_ids, past_key_values=cache, use_cache=True, num_logits_to_keep=1)
            logits = output.logits[0, -1, :]

            gen = None
            if cfg.seed is not None:
                gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
            decoder = ByteStreamDecoder()
            generated: list[int] = []
            history = list(prompt)
            text = ""
            finish_reason = "max_new_tokens"

            for step in range(cfg.max_new_tokens):
                if cancel is not None and cancel.is_set():
                    finish_reason = "cancelled"
                    break
                token_id = sample_next_token(
                    logits, cfg, generated=generated, step=step, generator=gen
                )
                generated.append(token_id)
                delta = ""
                if self.token_bytes is not None:
                    delta = decoder.push(self.token_bytes(token_id))
                    text += delta
                else:
                    text += f"<{token_id}>"
                yield GenerationEvent(
                    type="token",
                    token_id=token_id,
                    text_delta=delta,
                    text=text,
                    token_ids=list(generated),
                    prompt_tokens=len(prompt),
                    generated_tokens=len(generated),
                )

                terminal_ids = {cfg.eos_token_id, *cfg.stop_token_ids}
                if token_id in terminal_ids:
                    finish_reason = "eos" if token_id == cfg.eos_token_id else "stop_token"
                    break
                for seq in cfg.stop_sequences:
                    if tuple(generated[-len(seq):]) == tuple(seq):
                        finish_reason = "stop_sequence"
                        break
                if finish_reason == "stop_sequence":
                    break
                if cfg.stop_strings:
                    hit = min(
                        (text.find(s) for s in cfg.stop_strings if s in text),
                        default=-1,
                    )
                    if hit >= 0:
                        text = text[:hit]
                        finish_reason = "stop_string"
                        break
                if step + 1 >= cfg.max_new_tokens:
                    break
                if cache.length + 1 >= max_len:
                    finish_reason = "context_limit"
                    break
                nxt = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
                output = self.model(nxt, past_key_values=cache, use_cache=True, num_logits_to_keep=1)
                logits = output.logits[0, -1, :]

            if self.token_bytes is not None:
                tail = decoder.finish()
                text += tail
            yield GenerationEvent(
                type="done",
                text=text,
                token_ids=list(generated),
                finish_reason=finish_reason,
                prompt_tokens=len(prompt),
                generated_tokens=len(generated),
            )

    def generate(
        self,
        prompt_ids: Sequence[int],
        cfg: SamplingConfig,
        *,
        cancel: threading.Event | None = None,
    ) -> GenerationResult:
        last: GenerationEvent | None = None
        for event in self.stream(prompt_ids, cfg, cancel=cancel):
            last = event
        assert last is not None and last.type == "done"
        return GenerationResult(
            token_ids=last.token_ids,
            text=last.text,
            finish_reason=last.finish_reason or "unknown",
            prompt_tokens=last.prompt_tokens,
            generated_tokens=last.generated_tokens,
        )
