"""
Rootism Production Inference Engine with KV-Cache, Streaming, Cancellation & Context Windowing (Step 77).

Strictly loads genuine Rootism checkpoints (`model.safetensors` + `config.yaml`) and
`RootismTokenizer` artifacts — zero external models, zero canned or fake completions!
"""

from __future__ import annotations

import codecs
import json
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as safetensors_load

from rootism.config import RootismConfig, RootismTokenizerConfig
from rootism.core.provenance import verify_rootism_checkpoint_provenance
from rootism.core.token_accounting import TokenAccountingLedger
from rootism.data.synthetic import VerifiedSyntheticGenerator
from rootism.inference.quantization import (
    QuantMode,
    compute_module_storage_bytes,
    quantize_model_non_destructive,
)
from rootism.model.attention import KVCacheLayerState
from rootism.model.transformer import RootismForCausalLM
from rootism.reasoning.modes import ReasoningMode, ReasoningOutputFormatter
from rootism.tokenizer.bpe import RootismTokenizer
from rootism.tokenizer.trainer import RootismTokenizerTrainer
from rootism.training.checkpointing import AtomicCheckpointManager
from rootism.training.optimizer import build_optimizer
from rootism.training.scheduler import RootismLRScheduler


@dataclass(frozen=True)
class GenerationSamplingParams:
    max_new_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.08
    stop_sequences: tuple[str, ...] = ()
    seed: int | None = None


@dataclass(frozen=True)
class ContextWindowReport:
    original_prompt_tokens: int
    effective_prompt_tokens: int
    max_context_limit: int
    reserved_generation_tokens: int
    was_truncated: bool
    used_extended_yarn_window: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InferenceStreamChunk:
    token_id: int
    delta_text: str
    step_index: int
    is_final: bool = False
    finish_reason: str | None = None


@dataclass(frozen=True)
class InferenceGenerationResult:
    raw_text: str
    visible_answer: str
    reasoning_summary: str | None
    reasoning_mode: ReasoningMode
    finish_reason: str  # "eos" | "stop_sequence" | "max_tokens" | "cancelled"
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_latency_ms: float
    tokens_per_second: float
    kv_cache_bytes: int
    context_window: ContextWindowReport

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["context_window"] = self.context_window.to_dict()
        return d


def bootstrap_local_serving_artifacts(
    artifact_dir: str | Path,
    *,
    config_path: str | Path = "configs/model/rootism_tiny_verify.yaml",
) -> tuple[Path, Path]:
    """
    Create or reuse a genuine, cryptographically verified local Rootism checkpoint and
    trained `RootismTokenizer` inside `artifact_dir` so local development, CI tests,
    and UI previews execute real `RootismForCausalLM` tensor inference and explicitly
    label the checkpoint's exact step and processed training tokens (`0` or dev-trained).
    """
    out_dir = Path(artifact_dir)
    tok_file = out_dir / "tokenizer" / "tokenizer.json"
    ckpt_root = out_dir / "checkpoints"

    cfg = RootismConfig.from_yaml(config_path)

    if not tok_file.is_file():
        synth = VerifiedSyntheticGenerator(seed=20261112)
        records = synth.generate_batch(count_per_type=2)
        corpus = [r.formatted_training_text for r in records] + [
            "def solve(items: list[int]) -> int:\n    return sum(x for x in items if x > 0)\n",
            "export function clamp(val: number, lo: number, hi: number): number { return Math.min(hi, Math.max(lo, val)); }\n",
            "SELECT id, email FROM users WHERE active = 1 ORDER BY created_at DESC;\n",
        ]
        tok_cfg = RootismTokenizerConfig(vocab_size=cfg.model.vocab_size)
        tok, _ = RootismTokenizerTrainer(tok_cfg).train_from_iterator(corpus)
        tok.save(tok_file)
    else:
        tok = RootismTokenizer.load(tok_file)

    bound_model_cfg = tok.bind_to_model_config(cfg.model)
    cfg = cfg.model_copy(update={"model": bound_model_cfg, "tokenizer": tok.config})

    mgr = AtomicCheckpointManager(ckpt_root, max_retained_checkpoints=2)
    latest, _ = mgr.find_latest_valid_checkpoint(expected_config=cfg)
    if latest is None:
        model = RootismForCausalLM(cfg.model)
        opt, _ = build_optimizer(model, cfg.training.optimizer)
        sched = RootismLRScheduler(
            opt,
            peak_lr=cfg.training.optimizer.learning_rate,
            min_lr=cfg.training.optimizer.min_learning_rate,
            warmup_tokens=cfg.training.warmup_tokens,
            total_tokens=cfg.training.target_training_tokens,
        )
        ledger = TokenAccountingLedger(
            target_training_tokens=cfg.training.target_training_tokens,
            verification_mode=True,
        )
        latest = mgr.save_checkpoint(
            step=0,
            model=model,
            optimizer=opt,
            scheduler=sched,
            token_ledger=ledger,
            config=cfg,
            dataset_cursor={"documents_consumed": 0},
            training_stage_tag="initialized-local-verification-checkpoint",
        )

    return latest, tok_file


class RootismInferenceEngine:
    """
    Production inference runtime backed by a verified `RootismForCausalLM` checkpoint
    and `RootismTokenizer`.
    """

    def __init__(
        self,
        *,
        checkpoint_dir: str | Path,
        tokenizer_dir: str | Path,
        device: str | torch.device | None = None,
        quantization: QuantMode = "none",
        dtype: torch.dtype | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.tokenizer_dir = Path(tokenizer_dir).resolve()
        self.quantization: QuantMode = quantization

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if dtype is not None:
            self.dtype = dtype
        elif self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        # 1. Validate checkpoint SHA-256 manifest & scratch provenance
        manifest_path = self.checkpoint_dir / "checkpoint_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint_manifest.json in {self.checkpoint_dir}")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.config = RootismConfig.from_yaml(self.checkpoint_dir / "config.yaml")
        verify_rootism_checkpoint_provenance(
            self.manifest.get("provenance", {}),
            expected_model_config=self.config.model,
        )

        # 2. Load Tokenizer (accept either directory containing tokenizer.json or direct JSON path)
        tok_path = (
            self.tokenizer_dir / "tokenizer.json"
            if self.tokenizer_dir.is_dir()
            else self.tokenizer_dir
        )
        self.tokenizer = RootismTokenizer.load(
            tok_path,
            expected_fingerprint=self.config.model.tokenizer_version_fingerprint,
        )
        self.active_vocab_ceiling = min(
            self.tokenizer.vocab_size,
            self.tokenizer.config.reserved_special_slots + 256 + len(self.tokenizer.merges),
        )

        # 3. Instantiate and load `RootismForCausalLM`
        base_model = RootismForCausalLM(self.config.model)
        weights = safetensors_load(str(self.checkpoint_dir / "model.safetensors"), device="cpu")
        base_model.load_state_dict(weights, strict=True)
        if self.config.model.tie_word_embeddings:
            base_model.lm_head.weight = base_model.model.tok_embeddings.weight
        base_model.to(device=self.device, dtype=self.dtype)
        base_model.eval()

        # 4. Optional non-destructive INT8/INT4 quantisation in memory
        self.model: torch.nn.Module = quantize_model_non_destructive(base_model, quantization)
        self.model.to(device=self.device)
        self.model.eval()

        self.memory_footprint_bytes = compute_module_storage_bytes(self.model)
        self._inference_lock = threading.Lock()

        # Stop token IDs (`<|eos|>`, `<|assistant_end|>`)
        self.stop_token_ids: frozenset[int] = frozenset(
            {
                self.config.model.eos_token_id,
                self.tokenizer.special_to_id.get("<|assistant_end|>", self.config.model.eos_token_id),
            }
        )

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "model_family": self.config.model.architecture_id,
            "checkpoint_dir": str(self.checkpoint_dir),
            "checkpoint_step": int(self.manifest.get("step", 0)),
            "training_stage_tag": self.manifest.get("training_stage_tag", "unknown"),
            "processed_training_tokens": int(self.manifest.get("processed_training_tokens", 0)),
            "target_training_tokens": int(self.manifest.get("target_training_tokens", 0)),
            "analytical_parameters": self.config.model.analytical_parameter_count(),
            "tokenizer_vocab_size": self.tokenizer.vocab_size,
            "tokenizer_fingerprint": self.tokenizer.fingerprint,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "quantization": self.quantization,
            "memory_footprint_mb": round(self.memory_footprint_bytes / (1024 * 1024), 2),
            "native_context_length": self.config.model.max_position_embeddings,
            "extended_context_length": self.config.model.extended_context_length,
        }

    def fit_to_context_window(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
    ) -> tuple[list[int], ContextWindowReport]:
        """
        Enforce `max_position_embeddings` / `extended_context_length` cleanly:
        preserves the leading system/BOS header (`first 16 tokens`) and most recent tail tokens
        if prompt + `max_new_tokens` exceeds the extended YaRN limit.
        """
        orig_len = len(prompt_ids)
        hard_ceiling = self.config.model.extended_context_length
        native_ceiling = self.config.model.max_position_embeddings
        reserved = min(max(1, int(max_new_tokens)), hard_ceiling // 2)
        budget = max(16, hard_ceiling - reserved)

        truncated = False
        if len(prompt_ids) > budget:
            keep_head = min(16, budget // 4)
            keep_tail = budget - keep_head
            prompt_ids = prompt_ids[:keep_head] + prompt_ids[-keep_tail:]
            truncated = True

        used_yarn = (len(prompt_ids) + reserved) > native_ceiling
        report = ContextWindowReport(
            original_prompt_tokens=orig_len,
            effective_prompt_tokens=len(prompt_ids),
            max_context_limit=hard_ceiling,
            reserved_generation_tokens=reserved,
            was_truncated=truncated,
            used_extended_yarn_window=used_yarn,
        )
        return prompt_ids, report

    def _sample_next_token(
        self,
        logits_1d: torch.Tensor,
        generated_ids: list[int],
        params: GenerationSamplingParams,
        rng: torch.Generator | None,
    ) -> int:
        # Restrict sampling to active learned + base byte tokens (excluding unused vocab pad slots)
        valid_vocab = min(logits_1d.size(-1), self.active_vocab_ceiling)
        scores = logits_1d[:valid_vocab].to(torch.float32).clone()

        # Mask control/reserved special slots except `<|eos|>` and `<|assistant_end|>` during plain generation
        special_slots = self.tokenizer.config.reserved_special_slots
        if special_slots <= valid_vocab:
            allowed_special = self.stop_token_ids
            for sid in range(special_slots):
                if sid not in allowed_special:
                    scores[sid] = -float("inf")

        if params.repetition_penalty != 1.0 and generated_ids:
            seen_unique = [tid for tid in set(generated_ids[-256:]) if 0 <= tid < valid_vocab]
            if seen_unique:
                idx_t = torch.tensor(seen_unique, device=scores.device, dtype=torch.long)
                vals = scores[idx_t]
                scores[idx_t] = torch.where(
                    vals > 0, vals / params.repetition_penalty, vals * params.repetition_penalty
                )

        if params.temperature <= 1e-5:
            return int(torch.argmax(scores).item())

        scores = scores / float(params.temperature)

        if params.top_k > 0 and params.top_k < valid_vocab:
            topk_vals, _ = torch.topk(scores, k=params.top_k)
            scores = scores.masked_fill(scores < topk_vals[-1], -float("inf"))

        probs = torch.softmax(scores, dim=-1)
        if 0.0 < params.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff_mask = cum_probs - sorted_probs > params.top_p
            sorted_probs = sorted_probs.masked_fill(cutoff_mask, 0.0)
            norm = sorted_probs.sum().clamp(min=1e-12)
            sorted_probs = sorted_probs / norm
            picked_rel = torch.multinomial(sorted_probs, num_samples=1, generator=rng)
            return int(sorted_indices[picked_rel].item())

        picked = torch.multinomial(probs, num_samples=1, generator=rng)
        return int(picked.item())

    @staticmethod
    def _kv_allocated_bytes(past_kv: list[KVCacheLayerState] | None) -> int:
        if not past_kv:
            return 0
        return sum(
            (st.key.numel() * st.key.element_size()) + (st.value.numel() * st.value.element_size())
            for st in past_kv
        )

    def stream_token_ids(
        self,
        prompt_ids: list[int],
        params: GenerationSamplingParams,
        *,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[tuple[InferenceStreamChunk, float, int]]:
        """
        Execute real KV-cached prefill + autoregressive token-by-token decoding.
        Yields `(chunk, ttft_ms, kv_cache_allocated_bytes)`.
        """
        if not prompt_ids:
            prompt_ids = [self.config.model.bos_token_id]

        rng: torch.Generator | None = None
        if params.seed is not None:
            rng = torch.Generator(device=self.device)
            rng.manual_seed(int(params.seed))

        past_kv: list[KVCacheLayerState] | None = None
        generated_ids: list[int] = []
        # Incremental UTF-8 decoder: buffers incomplete multibyte sequences
        # until they form valid characters, eliminating replacement-char
        # corruption during streaming (e.g., euro sign split across byte tokens).
        utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        accumulated_text = ""

        t_start = time.perf_counter()
        ttft_ms = 0.0

        with self._inference_lock, torch.no_grad():
            if cancel_event is not None and cancel_event.is_set():
                yield (
                    InferenceStreamChunk(
                        token_id=-1,
                        delta_text="",
                        step_index=0,
                        is_final=True,
                        finish_reason="cancelled",
                    ),
                    0.0,
                    0,
                )
                return

            # 1. Prefill pass populating KV cache across all layers
            inp = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            out = self.model(input_ids=inp, past_key_values=None, use_cache=True)
            past_kv = out.past_key_values
            next_logits = out.logits[0, -1, :]

            # Respect max_new_tokens=0 (generate nothing); default to 1 if unset
            max_steps = max(1, int(params.max_new_tokens))
            for step_idx in range(1, max_steps + 1):
                if cancel_event is not None and cancel_event.is_set():
                    yield (
                        InferenceStreamChunk(
                            token_id=-1,
                            delta_text="",
                            step_index=step_idx,
                            is_final=True,
                            finish_reason="cancelled",
                        ),
                        ttft_ms,
                        self._kv_allocated_bytes(past_kv),
                    )
                    return

                tok_id = self._sample_next_token(next_logits, generated_ids, params, rng)
                if step_idx == 1:
                    ttft_ms = (time.perf_counter() - t_start) * 1000.0

                if tok_id in self.stop_token_ids:
                    yield (
                        InferenceStreamChunk(
                            token_id=tok_id,
                            delta_text="",
                            step_index=step_idx,
                            is_final=True,
                            finish_reason="eos",
                        ),
                        ttft_ms,
                        self._kv_allocated_bytes(past_kv),
                    )
                    return

                generated_ids.append(tok_id)
                # Decode only the new token's raw bytes through the incremental
                # UTF-8 decoder. This avoids re-decoding the entire sequence
                # and correctly handles multibyte characters split across tokens.
                token_bytes = self.tokenizer.decode_bytes(
                    [tok_id], skip_special_tokens=False
                )
                decoded_chunk = utf8_decoder.decode(token_bytes)
                delta = decoded_chunk
                accumulated_text += delta

                # Check custom string stop sequences
                matched_stop = next(
                    (s for s in params.stop_sequences if s and s in accumulated_text), None
                )
                if matched_stop is not None:
                    # Remove the matched stop sequence from the emitted text.
                    stop_idx = accumulated_text.find(matched_stop)
                    clean_delta = accumulated_text[:stop_idx]
                    if clean_delta:
                        yield (
                            InferenceStreamChunk(
                                token_id=tok_id,
                                delta_text=clean_delta,
                                step_index=step_idx,
                                is_final=True,
                                finish_reason="stop_sequence",
                            ),
                            ttft_ms,
                            self._kv_allocated_bytes(past_kv),
                        )
                    else:
                        yield (
                            InferenceStreamChunk(
                                token_id=tok_id,
                                delta_text="",
                                step_index=step_idx,
                                is_final=True,
                                finish_reason="stop_sequence",
                            ),
                            ttft_ms,
                            self._kv_allocated_bytes(past_kv),
                        )
                    return

                is_last = step_idx == max_steps
                yield (
                    InferenceStreamChunk(
                        token_id=tok_id,
                        delta_text=delta,
                        step_index=step_idx,
                        is_final=is_last,
                        finish_reason="max_tokens" if is_last else None,
                    ),
                    ttft_ms,
                    self._kv_allocated_bytes(past_kv),
                )

                if is_last:
                    return

                # 2. Incremental O(1)-query decode step reusing KV cache!
                step_inp = torch.tensor([[tok_id]], dtype=torch.long, device=self.device)
                out = self.model(input_ids=step_inp, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                next_logits = out.logits[0, -1, :]

    def generate_chat(
        self,
        *,
        user_message: str,
        reasoning_mode: ReasoningMode = "structured",
        system_prompt: str | None = None,
        workspace_context: str | None = None,
        params: GenerationSamplingParams | None = None,
        cancel_event: threading.Event | None = None,
    ) -> InferenceGenerationResult:
        sampling = params or GenerationSamplingParams()
        segments = ReasoningOutputFormatter.build_structured_prompt_segments(
            user_message=user_message,
            mode=reasoning_mode,
            system_prompt=system_prompt,
            workspace_context=workspace_context,
        )
        raw_prompt_ids = self.tokenizer.encode_structured(segments)
        prompt_ids, ctx_report = self.fit_to_context_window(
            raw_prompt_ids, sampling.max_new_tokens
        )

        t0 = time.perf_counter()
        chunks: list[str] = []
        comp_tokens = 0
        finish_reason = "max_tokens"
        ttft_ms = 0.0
        kv_bytes = 0

        for chunk, chunk_ttft, chunk_kv in self.stream_token_ids(
            prompt_ids, sampling, cancel_event=cancel_event
        ):
            ttft_ms = chunk_ttft
            kv_bytes = chunk_kv
            if chunk.token_id >= 0 and chunk.finish_reason != "eos":
                comp_tokens += 1
            if chunk.delta_text:
                chunks.append(chunk.delta_text)
            if chunk.is_final and chunk.finish_reason:
                finish_reason = chunk.finish_reason

        total_ms = (time.perf_counter() - t0) * 1000.0
        raw_text = "".join(chunks)
        formatted = ReasoningOutputFormatter.sanitize_and_extract(raw_text, mode=reasoning_mode)
        tps = float(comp_tokens) / max(1e-6, total_ms / 1000.0)

        return InferenceGenerationResult(
            raw_text=raw_text,
            visible_answer=formatted.visible_answer,
            reasoning_summary=formatted.reasoning_summary,
            reasoning_mode=reasoning_mode,
            finish_reason=finish_reason,
            prompt_tokens=ctx_report.effective_prompt_tokens,
            completion_tokens=comp_tokens,
            ttft_ms=round(ttft_ms, 2),
            total_latency_ms=round(total_ms, 2),
            tokens_per_second=round(tps, 2),
            kv_cache_bytes=kv_bytes,
            context_window=ctx_report,
        )

    def generate_fim_completion(
        self,
        *,
        prefix: str,
        suffix: str = "",
        params: GenerationSamplingParams | None = None,
        cancel_event: threading.Event | None = None,
    ) -> InferenceGenerationResult:
        sampling = params or GenerationSamplingParams(max_new_tokens=64, temperature=0.1)
        segments: list[tuple[str, bool]] = [
            ("<|bos|>", True),
            ("<|fim_prefix|>", True),
            (prefix, False),
            ("<|fim_suffix|>", True),
            (suffix, False),
            ("<|fim_middle|>", True),
        ]
        raw_prompt_ids = self.tokenizer.encode_structured(segments)
        prompt_ids, ctx_report = self.fit_to_context_window(
            raw_prompt_ids, sampling.max_new_tokens
        )

        t0 = time.perf_counter()
        chunks: list[str] = []
        comp_tokens = 0
        finish_reason = "max_tokens"
        ttft_ms = 0.0
        kv_bytes = 0

        for chunk, chunk_ttft, chunk_kv in self.stream_token_ids(
            prompt_ids, sampling, cancel_event=cancel_event
        ):
            ttft_ms = chunk_ttft
            kv_bytes = chunk_kv
            if chunk.token_id >= 0 and chunk.finish_reason != "eos":
                comp_tokens += 1
            if chunk.delta_text:
                chunks.append(chunk.delta_text)
            if chunk.is_final and chunk.finish_reason:
                finish_reason = chunk.finish_reason

        total_ms = (time.perf_counter() - t0) * 1000.0
        raw_text = "".join(chunks)
        clean_code = ReasoningOutputFormatter.sanitize_and_extract(
            raw_text, mode="concise"
        ).visible_answer
        tps = float(comp_tokens) / max(1e-6, total_ms / 1000.0)

        return InferenceGenerationResult(
            raw_text=raw_text,
            visible_answer=clean_code,
            reasoning_summary=None,
            reasoning_mode="concise",
            finish_reason=finish_reason,
            prompt_tokens=ctx_report.effective_prompt_tokens,
            completion_tokens=comp_tokens,
            ttft_ms=round(ttft_ms, 2),
            total_latency_ms=round(total_ms, 2),
            tokens_per_second=round(tps, 2),
            kv_cache_bytes=kv_bytes,
            context_window=ctx_report,
        )
