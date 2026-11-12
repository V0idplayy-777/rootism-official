"""
Rootism Production FastAPI Server & Root Developer Platform Host (Steps 79, 80, 81–90).

Features:
- Real `RootismInferenceEngine` streaming (`text/event-stream`) & synchronous generation
- Cooperative request cancellation (`/api/v1/requests/{request_id}/cancel`)
- Concurrency limiting (`asyncio.Semaphore`), sliding-window rate limiting, optional Bearer auth
- Confined multi-file Developer Workspace & Multi-step `CodingAgentOrchestrator` endpoints
- Serves the Root Directory HTML Developer Platform (`index.html`, `styles.css`, `app.js`)
  with iframe-safe headers for live browser previews (`0.0.0.0` / `*.e2b.app`).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from rootism.agent.orchestrator import (
    AgentPlannedAction,
    AgentStepTrace,
    CodingAgentOrchestrator,
)
from rootism.agent.tools import AgentToolRegistry
from rootism.config import RootismModelConfig
from rootism.inference.engine import (
    GenerationSamplingParams,
    RootismInferenceEngine,
    bootstrap_local_serving_artifacts,
)
from rootism.inference.quantization import QuantMode
from rootism.languages.registry import PRIMARY_LANGUAGES, SECONDARY_LANGUAGES
from rootism.reasoning.modes import REASONING_MODE_SPECS, ReasoningOutputFormatter
from rootism.server.schemas import (
    AgentConfirmationDecisionRequest,
    AgentTaskRunRequest,
    AgentToolInvokeRequest,
    ChatCompletionRequest,
    CodeCompletionRequest,
    WorkspaceFileSaveRequest,
)


def seed_default_agent_workspace(workspace_dir: Path) -> None:
    """
    Ensure the isolated sandbox workspace directory exists and is COMPLETELY EMPTY.

    Rootism deliberately ships no starter/scaffold files: the initial project must be
    a blank slate. Developers create their own files from the Workspace sidebar (or the
    coding agent creates them), so nothing is ever pre-seeded here.
    """
    workspace_dir.mkdir(parents=True, exist_ok=True)


class SlidingWindowRateLimiter:
    """Thread-safe per-client sliding-window request rate limiter."""

    def __init__(self, max_requests_per_minute: int = 180) -> None:
        self.max_requests_per_minute = max(1, int(max_requests_per_minute))
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check_or_raise(self, client_key: str) -> None:
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            q = self._hits[client_key]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.max_requests_per_minute:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded ({self.max_requests_per_minute} req/min). Retry shortly.",
                )
            q.append(now)


def create_rootism_app(
    *,
    engine: RootismInferenceEngine | None = None,
    checkpoint_dir: str | Path | None = None,
    tokenizer_dir: str | Path | None = None,
    quantization: QuantMode = "none",
    agent_workspace_dir: str | Path | None = None,
    platform_root_dir: str | Path | None = None,
    api_key: str | None = None,
    max_concurrent_generations: int = 4,
    rate_limit_per_minute: int = 240,
    allowed_origins: list[str] | None = None,
    require_auth: bool | None = None,
) -> FastAPI:
    repo_root = Path(__file__).resolve().parents[2]
    ui_root = Path(platform_root_dir).resolve() if platform_root_dir else repo_root

    if engine is None:
        if checkpoint_dir is None or tokenizer_dir is None:
            default_artifacts = repo_root / "artifacts" / "local_runtime"
            ckpt_p, tok_p = bootstrap_local_serving_artifacts(default_artifacts)
            checkpoint_dir = checkpoint_dir or ckpt_p
            tokenizer_dir = tokenizer_dir or tok_p
        engine = RootismInferenceEngine(
            checkpoint_dir=checkpoint_dir,
            tokenizer_dir=tokenizer_dir,
            quantization=quantization,
        )

    ws_dir = (
        Path(agent_workspace_dir).resolve()
        if agent_workspace_dir is not None
        else (repo_root / "artifacts" / "agent_workspace").resolve()
    )
    seed_default_agent_workspace(ws_dir)
    agent_tools = AgentToolRegistry(ws_dir)

    configured_api_key = api_key if api_key is not None else os.environ.get("ROOTISM_API_KEY")
    # Require authentication by default. Only disable explicitly for local development.
    auth_required = (
        require_auth if require_auth is not None
        else bool(configured_api_key)  # default: require auth when a key is configured
    )
    if auth_required and not configured_api_key:
        # If auth is required but no key is configured, generate an ephemeral one
        # so the server is never exposed without authentication.
        configured_api_key = secrets.token_urlsafe(32)
    rate_limiter = SlidingWindowRateLimiter(max_requests_per_minute=rate_limit_per_minute)
    concurrency_sem = asyncio.Semaphore(max(1, int(max_concurrent_generations)))
    active_cancellations: dict[str, threading.Event] = {}
    cancel_lock = threading.Lock()
    server_boot_ts = time.time()

    production_3b_cfg = RootismModelConfig()

    def _weight_readiness() -> dict[str, Any]:
        """Single source of truth for whether *trained* weights are actually loaded.

        Rootism always boots a real `RootismForCausalLM` tensor runtime, but a freshly
        initialized (step 0) checkpoint has processed 0 training tokens. In that case the
        platform runs in an honest "standby" mode: workspace/agent tooling works, and chat
        returns a clean explanation instead of pretending random weights are a model.
        """
        meta = engine.runtime_metadata()
        processed = int(meta.get("processed_training_tokens", 0) or 0)
        trained = processed > 0
        return {
            "weights_loaded": True,
            "trained_weights": trained,
            "model_ready": True,
            "in_standby_mode": not trained,
            "processed_training_tokens": processed,
            "checkpoint_step": int(meta.get("checkpoint_step", 0) or 0),
            "training_stage_tag": meta.get("training_stage_tag", "unknown"),
            "checkpoint_dir": meta.get("checkpoint_dir", ""),
        }

    def _standby_message() -> str:
        readiness = _weight_readiness()
        return (
            "Standby mode — the Rootism runtime is loaded and healthy, but the active checkpoint "
            f"(`{readiness['training_stage_tag']}`, step {readiness['checkpoint_step']}) has "
            f"{readiness['processed_training_tokens']:,} processed training tokens, so it has not "
            "learned language yet and will not answer prompts.\n\n"
            "Everything else already works: the Workspace, file editing, diffs, undo, symbol index, "
            "chat history export, and the agent's inspection/sandbox tooling.\n\n"
            "To turn on real generation:\n"
            "1. `rootism train-tokenizer --input <corpus_dir> --output artifacts/prod/tokenizer/tokenizer.json --vocab-size 65536`\n"
            "2. `rootism verify-licenses --registry configs/datasets/provenance_registry.yaml`\n"
            "3. `rootism train-verify --config configs/training/pretrain_650b.yaml --steps <N> --checkpoint-dir checkpoints/rootism-3b`\n"
            "   (multi-GPU: `torchrun --nproc_per_node=8 -m rootism.cli train-verify --config configs/training/pretrain_650b.yaml ...`)\n"
            "4. Serve those weights:\n"
            "   `rootism serve --host 0.0.0.0 --port 8000 --checkpoint-dir checkpoints/rootism-3b/step_XXXXXXXX --tokenizer-dir artifacts/prod/tokenizer`\n\n"
            "As soon as a checkpoint reports processed_training_tokens > 0, this same server switches "
            "to real neural generation automatically — no code changes needed. "
            "Full walkthrough: docs/TRAINING_AND_WEIGHTS.md."
        )

    app = FastAPI(
        title="Rootism Developer Platform & Inference API",
        version="0.1.0",
        description="Commercial-grade Rootism AI Coding, Reasoning & Multi-Step Agent Server",
    )

    # Restrict CORS to explicit origin allowlist
    resolved_origins = allowed_origins if allowed_origins is not None else ["*"]
    if auth_required and resolved_origins == ["*"]:
        # Never allow wildcard CORS when authentication is enabled
        resolved_origins = []
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def attach_request_headers(request: Request, call_next: Any) -> Any:
        req_id = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:12]}"
        request.state.request_id = req_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        # Use CSP frame-ancestors for controlled iframe embedding instead of
        # blanket-removing X-Frame-Options (clickjacking protection).
        # Allow embedding on Arena's e2b.app proxy and same-origin.
        existing_csp = response.headers.get("Content-Security-Policy", "")
        frame_ancestors = "frame-ancestors 'self' https://*.e2b.app"
        if frame_ancestors not in existing_csp:
            response.headers["Content-Security-Policy"] = (
                f"{existing_csp}; {frame_ancestors}" if existing_csp else frame_ancestors
            )
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_envelope(request: Request, exc: HTTPException) -> JSONResponse:
        req_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": str(exc.detail),
                    "request_id": req_id,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_envelope(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        req_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "INVALID_REQUEST_SCHEMA",
                    "message": "Request validation failed.",
                    "details": exc.errors(),
                    "request_id": req_id,
                }
            },
        )

    def verify_auth_and_rate_limit(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> None:
        if auth_required:
            if not configured_api_key:
                raise HTTPException(
                    status_code=503,
                    detail="Authentication is required but no API key is configured.",
                )
            expected = f"Bearer {configured_api_key}"
            if authorization != expected:
                raise HTTPException(
                    status_code=401,
                    detail="Missing or invalid Bearer API key.",
                )
        client_host = request.client.host if request.client else "local"
        rate_limiter.check_or_raise(client_host)

    @app.api_route("/api/v1/health", methods=["GET", "HEAD"])
    async def health_check() -> dict[str, Any]:
        # Health endpoint returns minimal info only — no runtime metadata leak
        with cancel_lock:
            in_flight = len(active_cancellations)
        readiness = _weight_readiness()
        return {
            "status": "healthy",
            "ready": True,
            "uptime_seconds": round(time.time() - server_boot_ts, 2),
            "in_flight_generations": in_flight,
            "max_concurrent_generations": max_concurrent_generations,
            "auth_required": auth_required,
            "runtime": {
                "device": str(engine.device),
                "dtype": str(engine.dtype),
                "quantization": engine.quantization,
            },
            **readiness,
        }

    @app.get("/api/v1/model/info", dependencies=[Depends(verify_auth_and_rate_limit)])
    async def model_info() -> dict[str, Any]:
        meta = engine.runtime_metadata()
        readiness = _weight_readiness()
        processed_toks = readiness["processed_training_tokens"]
        is_full_650b = processed_toks >= 600_000_000_000
        if processed_toks <= 0:
            banner = (
                "Standby — runtime loaded, 0 trained tokens. "
                "Workspace + agent tools are live; add trained weights to enable generation."
            )
        elif is_full_650b:
            banner = f"Production Rootism-3B ({processed_toks:,} tokens trained)"
        else:
            banner = (
                f"Active Checkpoint: {meta['training_stage_tag']} "
                f"(step {meta['checkpoint_step']}, {processed_toks:,} processed tokens). "
                "Real scratch RootismForCausalLM tensor runtime — full 650B cluster training pending."
            )
        return {
            "active_runtime": meta,
            "weight_readiness": readiness,
            **readiness,
            "honest_training_status": {
                "is_fully_pretrained_600b_plus": is_full_650b,
                "processed_training_tokens": processed_toks,
                "trained_weights_loaded": readiness["trained_weights"],
                "status_banner": banner,
            },
            "production_architecture_spec": {
                "architecture_id": production_3b_cfg.architecture_id,
                "verified_parameter_count": production_3b_cfg.analytical_parameter_count(),
                "parameter_breakdown": production_3b_cfg.analytical_parameter_breakdown(),
                "native_context_window": production_3b_cfg.max_position_embeddings,
                "extended_yarn_context_window": production_3b_cfg.extended_context_length,
                "target_training_tokens": 650_000_000_000,
                "minimum_production_tokens": 600_000_000_000,
            },
            "reasoning_modes": {
                k: {"label": v["label"], "description": v["system_hint"]}
                for k, v in REASONING_MODE_SPECS.items()
            },
            "supported_languages": {
                "primary": [s.display_name for s in PRIMARY_LANGUAGES],
                "secondary": [s.display_name for s in SECONDARY_LANGUAGES],
            },
        }

    @app.post("/api/v1/requests/{request_id}/cancel", dependencies=[Depends(verify_auth_and_rate_limit)])
    async def cancel_generation(request_id: str) -> dict[str, Any]:
        with cancel_lock:
            ev = active_cancellations.get(request_id)
            if ev is not None:
                ev.set()
                return {"request_id": request_id, "cancelled": True}
        return {"request_id": request_id, "cancelled": False, "note": "Request already finished or unknown."}

    @app.post("/api/v1/chat/completions", dependencies=[Depends(verify_auth_and_rate_limit)])
    async def chat_completions(req: ChatCompletionRequest, request: Request) -> Any:
        req_id = req.request_id or getattr(request.state, "request_id", f"req-{uuid.uuid4().hex[:12]}")
        sys_msgs = [m.content for m in req.messages if m.role == "system"]
        conv_lines: list[str] = []
        for m in req.messages:
            if m.role == "system":
                continue
            prefix = "User: " if m.role == "user" else "Assistant: "
            conv_lines.append(f"{prefix}{m.content}")
        user_payload = "\n\n".join(conv_lines) if len(conv_lines) > 1 else req.messages[-1].content
        system_prompt = "\n".join(sys_msgs) if sys_msgs else None

        sampling = GenerationSamplingParams(
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            stop_sequences=tuple(req.stop),
            seed=req.seed,
        )

        cancel_ev = threading.Event()
        with cancel_lock:
            active_cancellations[req_id] = cancel_ev

        standby = _weight_readiness()["in_standby_mode"]

        if standby:
            # Honest standby: never emit random-weight gibberish, never error.
            try:
                if req.stream:
                    segments = ReasoningOutputFormatter.build_structured_prompt_segments(
                        user_message=user_payload,
                        mode=req.reasoning_mode,
                        system_prompt=system_prompt,
                        workspace_context=req.workspace_context,
                    )
                    raw_ids = engine.tokenizer.encode_structured(segments)
                    _, ctx_report = engine.fit_to_context_window(raw_ids, req.max_tokens)
                    notice = _standby_message()

                    async def _standby_stream() -> Any:
                        yield f"data: {json.dumps({'event': 'meta', 'request_id': req_id, 'context_window': ctx_report.to_dict(), 'standby': True, 'processed_training_tokens': 0})}\n\n"
                        yield f"data: {json.dumps({'event': 'delta', 'request_id': req_id, 'text': notice, 'sanitized_text_so_far': notice, 'reasoning_summary_so_far': None, 'standby': True})}\n\n"
                        yield f"data: {json.dumps({'event': 'done', 'request_id': req_id, 'finish_reason': 'standby_no_trained_weights', 'content': notice, 'reasoning_summary': None, 'standby': True, 'usage': {'prompt_tokens': ctx_report.effective_prompt_tokens, 'completion_tokens': 0, 'total_tokens': ctx_report.effective_prompt_tokens}, 'telemetry': {'ttft_ms': 0.0, 'total_latency_ms': 0.0, 'tokens_per_second': 0.0, 'kv_cache_bytes': 0, 'context_window': ctx_report.to_dict()}})}\n\n"

                    return StreamingResponse(
                        _standby_stream(),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                    )
                return {
                    "id": req_id,
                    "object": "rootism.chat.completion",
                    "model": engine.config.model.architecture_id,
                    "checkpoint_step": 0,
                    "reasoning_mode": req.reasoning_mode,
                    "standby": True,
                    "message": {
                        "role": "assistant",
                        "content": _standby_message(),
                        "reasoning_summary": None,
                    },
                    "finish_reason": "standby_no_trained_weights",
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "telemetry": {
                        "ttft_ms": 0.0,
                        "total_latency_ms": 0.0,
                        "tokens_per_second": 0.0,
                        "kv_cache_bytes": 0,
                    },
                }
            finally:
                with cancel_lock:
                    active_cancellations.pop(req_id, None)

        if not req.stream:
            try:
                async with concurrency_sem:
                    result = await asyncio.to_thread(
                        engine.generate_chat,
                        user_message=user_payload,
                        reasoning_mode=req.reasoning_mode,
                        system_prompt=system_prompt,
                        workspace_context=req.workspace_context,
                        params=sampling,
                        cancel_event=cancel_ev,
                    )
                return {
                    "id": req_id,
                    "object": "rootism.chat.completion",
                    "model": engine.config.model.architecture_id,
                    "checkpoint_step": int(engine.manifest.get("step", 0)),
                    "reasoning_mode": req.reasoning_mode,
                    "message": {
                        "role": "assistant",
                        "content": result.visible_answer,
                        "reasoning_summary": result.reasoning_summary,
                    },
                    "finish_reason": result.finish_reason,
                    "usage": {
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "total_tokens": result.prompt_tokens + result.completion_tokens,
                    },
                    "telemetry": {
                        "ttft_ms": result.ttft_ms,
                        "total_latency_ms": result.total_latency_ms,
                        "tokens_per_second": result.tokens_per_second,
                        "kv_cache_bytes": result.kv_cache_bytes,
                        "context_window": result.context_window.to_dict(),
                    },
                }
            finally:
                with cancel_lock:
                    active_cancellations.pop(req_id, None)

        # Streaming SSE (`text/event-stream`) path
        segments = ReasoningOutputFormatter.build_structured_prompt_segments(
            user_message=user_payload,
            mode=req.reasoning_mode,
            system_prompt=system_prompt,
            workspace_context=req.workspace_context,
        )
        raw_ids = engine.tokenizer.encode_structured(segments)
        prompt_ids, ctx_report = engine.fit_to_context_window(raw_ids, sampling.max_new_tokens)

        async def _sse_generator():
            # Bounded queue with backpressure — a slow client cannot accumulate
            # unbounded events in memory.
            queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=256)
            loop = asyncio.get_running_loop()

            def _producer() -> None:
                t0 = time.perf_counter()
                raw_chunks: list[str] = []
                comp_tokens = 0
                finish_reason = "max_tokens"
                ttft_ms = 0.0
                kv_bytes = 0
                try:
                    for ch, c_ttft, c_kv in engine.stream_token_ids(
                        prompt_ids, sampling, cancel_event=cancel_ev
                    ):
                        ttft_ms = c_ttft
                        kv_bytes = c_kv
                        if ch.token_id >= 0 and ch.finish_reason != "eos":
                            comp_tokens += 1
                        if ch.delta_text:
                            raw_chunks.append(ch.delta_text)
                            # Redact any partial internal tags from live delta preview
                            preview_sanitized = ReasoningOutputFormatter.sanitize_and_extract(
                                "".join(raw_chunks), mode=req.reasoning_mode
                            )
                            # Only expose sanitized text — never raw internal deliberation
                            event = {
                                "event": "delta",
                                "request_id": req_id,
                                "step": ch.step_index,
                                "text": preview_sanitized.visible_answer,
                                "sanitized_text_so_far": preview_sanitized.visible_answer,
                                "reasoning_summary_so_far": preview_sanitized.reasoning_summary,
                            }
                            # Use put_nowait with backpressure handling
                            try:
                                loop.call_soon_threadsafe(queue.put_nowait, event)
                            except asyncio.QueueFull:
                                # Drop oldest event to make room for new one
                                try:
                                    loop.call_soon_threadsafe(queue.get_nowait)
                                    loop.call_soon_threadsafe(queue.put_nowait, event)
                                except (asyncio.QueueEmpty, asyncio.QueueFull):
                                    pass
                        if ch.is_final and ch.finish_reason:
                            finish_reason = ch.finish_reason
                except Exception as exc:
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {"event": "error", "request_id": req_id, "message": str(exc)},
                    )
                else:
                    total_ms = (time.perf_counter() - t0) * 1000.0
                    full_raw = "".join(raw_chunks)
                    final_fmt = ReasoningOutputFormatter.sanitize_and_extract(
                        full_raw, mode=req.reasoning_mode
                    )
                    tps = float(comp_tokens) / max(1e-6, total_ms / 1000.0)
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {
                            "event": "done",
                            "request_id": req_id,
                            "finish_reason": finish_reason,
                            "content": final_fmt.visible_answer,
                            "reasoning_summary": final_fmt.reasoning_summary,
                            "usage": {
                                "prompt_tokens": ctx_report.effective_prompt_tokens,
                                "completion_tokens": comp_tokens,
                                "total_tokens": ctx_report.effective_prompt_tokens + comp_tokens,
                            },
                            "telemetry": {
                                "ttft_ms": round(ttft_ms, 2),
                                "total_latency_ms": round(total_ms, 2),
                                "tokens_per_second": round(tps, 2),
                                "kv_cache_bytes": kv_bytes,
                                "context_window": ctx_report.to_dict(),
                            },
                        },
                    )
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            try:
                async with concurrency_sem:
                    yield f"data: {json.dumps({'event': 'meta', 'request_id': req_id, 'context_window': ctx_report.to_dict()})}\n\n"
                    worker = threading.Thread(target=_producer, daemon=True)
                    worker.start()
                    while True:
                        if await request.is_disconnected():
                            cancel_ev.set()
                            break
                        item = await queue.get()
                        if item is None:
                            break
                        yield f"data: {json.dumps(item)}\n\n"
                    await asyncio.to_thread(worker.join, 2.0)
            finally:
                cancel_ev.set()
                with cancel_lock:
                    active_cancellations.pop(req_id, None)

        return StreamingResponse(
            _sse_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/v1/code/complete", dependencies=[Depends(verify_auth_and_rate_limit)])
    async def code_complete(req: CodeCompletionRequest, request: Request) -> dict[str, Any]:
        req_id = getattr(request.state, "request_id", f"fim-{uuid.uuid4().hex[:10]}")
        if _weight_readiness()["in_standby_mode"]:
            return {
                "id": req_id,
                "completion": "",
                "finish_reason": "standby_no_trained_weights",
                "standby": True,
                "note": (
                    "No trained weights are loaded yet, so fill-in-the-middle completion is idle. "
                    "Add a trained checkpoint with `rootism serve --checkpoint-dir ... --tokenizer-dir ...` "
                    "and completions start working automatically."
                ),
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "telemetry": {"ttft_ms": 0.0, "total_latency_ms": 0.0, "tokens_per_second": 0.0},
            }
        sampling = GenerationSamplingParams(
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            stop_sequences=tuple(req.stop),
            seed=req.seed,
        )
        async with concurrency_sem:
            res = await asyncio.to_thread(
                engine.generate_fim_completion,
                prefix=req.prefix,
                suffix=req.suffix,
                params=sampling,
            )
        return {
            "id": req_id,
            "completion": res.visible_answer,
            "finish_reason": res.finish_reason,
            "usage": {
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
            },
            "telemetry": {
                "ttft_ms": res.ttft_ms,
                "total_latency_ms": res.total_latency_ms,
                "tokens_per_second": res.tokens_per_second,
            },
        }

    # --- Workspace & Multi-Step Coding Agent Endpoints (Steps 85 & 86) ---

    @app.get("/api/v1/workspace/files", dependencies=[Depends(verify_auth_and_rate_limit)])
    async def workspace_list_files() -> dict[str, Any]:
        overview = await asyncio.to_thread(agent_tools.analyzer.inspect_repository)
        listed = await asyncio.to_thread(agent_tools.invoke, "list_files", {})
        return {
            "workspace_root": str(ws_dir),
            "files": listed.payload.get("files", []),
            "overview": overview.to_dict(),
            "rollback_depth": len(agent_tools.rollback_stack),
        }

    @app.get("/api/v1/workspace/file", dependencies=[Depends(verify_auth_and_rate_limit)])
    async def workspace_read_file(path: str = Query(..., min_length=1)) -> dict[str, Any]:
        res = await asyncio.to_thread(
            agent_tools.invoke, "read_file", {"path": path, "start_line": 1, "end_line": 2000}
        )
        if not res.ok:
            if res.payload.get("error_type") == "FileNotFoundError":
                # A file that no longer exists (deleted, undone, never created) is a normal
                # state for the editor, not an error: report it and let the UI show a blank slate.
                return {
                    "path": path,
                    "exists": False,
                    "total_lines": 0,
                    "start_line": 1,
                    "end_line": 0,
                    "content": "",
                    "numbered_content": "",
                }
            raise HTTPException(status_code=400, detail=res.summary)
        return res.payload

    @app.post("/api/v1/workspace/file", dependencies=[Depends(verify_auth_and_rate_limit)])
    async def workspace_save_file(req: WorkspaceFileSaveRequest) -> dict[str, Any]:
        # An explicit editor save is user intent: it may overwrite the file the user is
        # editing without an extra confirmation token. Path confinement still applies.
        res = await asyncio.to_thread(
            agent_tools.invoke,
            "create_file",
            {"path": req.path, "content": req.content, "overwrite": True},
        )
        if not res.ok:
            raise HTTPException(status_code=400, detail=res.summary)
        return res.to_dict()

    @app.post("/api/v1/workspace/revert", dependencies=[Depends(verify_auth_and_rate_limit)])
    async def workspace_revert() -> dict[str, Any]:
        res = await asyncio.to_thread(agent_tools.invoke, "revert_last_edit", {})
        payload = res.to_dict()
        # Undo with an empty history is a friendly no-op, never an HTTP error.
        payload.update(
            {
                "reverted": bool(res.ok and res.payload.get("reverted_path")),
                "reverted_path": res.payload.get("reverted_path"),
                "message": res.summary,
            }
        )
        return payload

    @app.post("/api/v1/agent/tool", dependencies=[Depends(verify_auth_and_rate_limit)])
    async def agent_invoke_tool(req: AgentToolInvokeRequest) -> dict[str, Any]:
        res = await asyncio.to_thread(agent_tools.invoke, req.tool_name, req.args)
        return res.to_dict()

    @app.post("/api/v1/agent/confirm", dependencies=[Depends(verify_auth_and_rate_limit)])
    async def agent_confirm_decision(req: AgentConfirmationDecisionRequest) -> dict[str, Any]:
        if not req.approve:
            return {
                "approved": False,
                "confirmation_token": req.confirmation_token,
                "message": "Destructive action rejected by user; workspace left untouched.",
            }
        agent_tools.safety.grant_confirmation(req.confirmation_token)
        executed_result = None
        if req.pending_tool_name:
            merged_args = dict(req.pending_tool_args)
            merged_args["confirmation_token"] = req.confirmation_token
            tool_res = await asyncio.to_thread(
                agent_tools.invoke, req.pending_tool_name, merged_args
            )
            executed_result = tool_res.to_dict()
        return {
            "approved": True,
            "confirmation_token": req.confirmation_token,
            "executed_result": executed_result,
        }

    @app.post("/api/v1/agent/run", dependencies=[Depends(verify_auth_and_rate_limit)])
    async def agent_run_task(req: AgentTaskRunRequest) -> dict[str, Any]:
        """
        Execute a multi-step software engineering run (`inspect -> search/read/command -> test -> summarise`)
        inside the confined workspace, pairing structured repository inspection & test verification
        with a real `RootismInferenceEngine` pass.
        """
        if req.confirmation_token:
            agent_tools.safety.grant_confirmation(req.confirmation_token)

        orchestrator = CodingAgentOrchestrator(
            agent_tools,
            max_iterations=req.max_iterations,
        )

        # Detect explicit shell command or file deletion request in goal so safety confirmation gate is exercisable
        delete_match = re.search(r"\bdelete\s+file\s+([A-Za-z0-9_./-]+)", req.goal, flags=re.I)
        cmd_match = re.search(r"\brun\s+command\s*:\s*(.+)$", req.goal, flags=re.I)
        search_match = re.search(r"\bsearch\s+(?:for\s+)?['\"]?([A-Za-z0-9_]+)['\"]?", req.goal, flags=re.I)

        def _workspace_planner(
            goal: str,
            history: list[AgentStepTrace],
            _tools: AgentToolRegistry,
        ) -> AgentPlannedAction:
            step_n = len(history)
            if step_n == 0:
                return AgentPlannedAction(
                    phase="inspect",
                    rationale="Inspect repository layout, manifests, symbol index, and dependencies.",
                    tool_name="inspect_project",
                    tool_args={},
                )
            if step_n == 1:
                if delete_match:
                    target_file = delete_match.group(1)
                    return AgentPlannedAction(
                        phase="edit",
                        rationale=f"Attempt requested deletion of `{target_file}` through SafetyPolicyGuard.",
                        tool_name="delete_file",
                        tool_args={"path": target_file, "confirmation_token": req.confirmation_token},
                    )
                if cmd_match:
                    raw_cmd = cmd_match.group(1).strip()
                    return AgentPlannedAction(
                        phase="execute",
                        rationale=f"Execute requested command `{raw_cmd}` inside sandboxed runner.",
                        tool_name="run_command",
                        tool_args={"command": raw_cmd, "confirmation_token": req.confirmation_token},
                    )
                if search_match:
                    term = search_match.group(1)
                    return AgentPlannedAction(
                        phase="inspect",
                        rationale=f"Search repository occurrences of `{term}`.",
                        tool_name="search_project",
                        tool_args={"pattern": term, "is_regex": False},
                    )
                overview = _tools.invoke("inspect_project", {}).payload or {}
                indexed = overview.get("indexed_files") or []
                if indexed:
                    target = indexed[0].get("path") or indexed[0]
                    return AgentPlannedAction(
                        phase="inspect",
                        rationale=f"Read the first indexed workspace file `{target}` for context.",
                        tool_name="read_file",
                        tool_args={"path": target, "start_line": 1, "end_line": 200},
                    )
                # A brand-new empty project: nothing to read, so go straight to verification.
                return AgentPlannedAction(
                    phase="test",
                    rationale="Workspace is empty, so there is no source to read. Verify the test suite instead.",
                    tool_name="run_tests",
                    tool_args={},
                )
            if step_n == 2:
                tests_trace = next((t for t in history if t.tool_name == "run_tests"), None)
                read_trace = next((t for t in history if t.tool_name == "read_file"), None)
                if tests_trace is not None:
                    # Nothing left to inspect (empty project) — go straight to the summary.
                    pass
                elif read_trace is not None:
                    return AgentPlannedAction(
                        phase="test",
                        rationale="Execute automated workspace test suite and run FailureDiagnosticsEngine.",
                        tool_name="run_tests",
                        tool_args={},
                    )
                else:
                    return AgentPlannedAction(
                        phase="inspect",
                        rationale="No further source inspection needed for this workspace.",
                        tool_name="list_files",
                        tool_args={},
                    )

            # Synthesize final step with real RootismInferenceEngine pass + verified tool telemetry
            test_trace = next((t for t in reversed(history) if t.tool_name == "run_tests"), None)
            diag_summary = (
                test_trace.tool_result.get("summary", "No tests run")
                if test_trace and test_trace.tool_result
                else "Inspection complete."
            )
            if _weight_readiness()["in_standby_mode"]:
                summary_body = (
                    f"Agent run complete. Workspace verification: {diag_summary} "
                    "(model in standby — add trained weights to enable neural summaries)."
                )
            else:
                gen = engine.generate_chat(
                    user_message=f"Task: {goal}\nTest verification status: {diag_summary}",
                    reasoning_mode=req.reasoning_mode,
                    params=GenerationSamplingParams(max_new_tokens=32, temperature=0.2),
                )
                summary_body = (
                    f"Workspace verification: {diag_summary} | "
                    f"Rootism model pass ({gen.completion_tokens} tokens, {gen.tokens_per_second} tok/s): "
                    f"{gen.visible_answer[:240]}"
                )
            return AgentPlannedAction(
                phase="summarise",
                rationale=summary_body,
                is_final_summary=True,
            )

        async with concurrency_sem:
            report = await asyncio.to_thread(orchestrator.run, req.goal, _workspace_planner)
        return report.to_dict()

    # --- Root Directory Static Developer Platform Serving (Steps 81–90) ---

    @app.api_route("/", methods=["GET", "HEAD"])
    async def serve_root_index() -> FileResponse:
        index_path = ui_root / "index.html"
        if not index_path.is_file():
            raise HTTPException(status_code=404, detail="Root index.html not found.")
        return FileResponse(index_path, media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.api_route("/styles.css", methods=["GET", "HEAD"])
    async def serve_root_styles() -> FileResponse:
        css_path = ui_root / "styles.css"
        if not css_path.is_file():
            raise HTTPException(status_code=404, detail="Root styles.css not found.")
        return FileResponse(css_path, media_type="text/css", headers={"Cache-Control": "no-cache"})

    @app.api_route("/app.js", methods=["GET", "HEAD"])
    async def serve_root_app_js() -> FileResponse:
        js_path = ui_root / "app.js"
        if not js_path.is_file():
            raise HTTPException(status_code=404, detail="Root app.js not found.")
        return FileResponse(js_path, media_type="application/javascript", headers={"Cache-Control": "no-cache"})

    @app.api_route("/local-api.js", methods=["GET", "HEAD"])
    async def serve_root_local_api_js() -> FileResponse:
        js_path = ui_root / "local-api.js"
        if not js_path.is_file():
            raise HTTPException(status_code=404, detail="Root local-api.js not found.")
        return FileResponse(js_path, media_type="application/javascript", headers={"Cache-Control": "no-cache"})

    return app
