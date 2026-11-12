"""
Automated Tests for Phase IX (Inference & Quantisation), Phase X (Root Web Platform),
and Phase XI (Evaluation, Generalisation & Regression) (Steps 77–96).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from rootism.config import RootismConfig
from rootism.evaluation.coding_bench import CodingBenchmarkSuite
from rootism.evaluation.generalization import GeneralizationVerifier
from rootism.evaluation.reasoning_bench import ReasoningBenchmarkSuite
from rootism.evaluation.regression import (
    CheckpointRegressionTracker,
    EvaluationReportBuilder,
    EvaluationSnapshot,
)
from rootism.inference.engine import (
    GenerationSamplingParams,
    RootismInferenceEngine,
    bootstrap_local_serving_artifacts,
)
from rootism.inference.quantization import (
    compute_module_storage_bytes,
    evaluate_quantization_quality,
    quantize_model_non_destructive,
)
from rootism.model.transformer import RootismForCausalLM
from rootism.server.app import create_rootism_app


def test_non_destructive_int8_and_int4_quantization() -> None:
    cfg = RootismConfig.from_yaml("configs/model/rootism_tiny_verify.yaml")
    fp_model = RootismForCausalLM(cfg.model).eval()
    orig_weight_snapshot = fp_model.model.layers[0].mlp.gate_proj.weight.detach().clone()

    eval_ids = torch.randint(64, 300, (2, 32), dtype=torch.long)
    q8 = quantize_model_non_destructive(fp_model, "int8")
    q4 = quantize_model_non_destructive(fp_model, "int4")

    # Non-destructive guarantee: master FP32 model must be bitwise untouched!
    assert torch.equal(fp_model.model.layers[0].mlp.gate_proj.weight, orig_weight_snapshot)

    rep8 = evaluate_quantization_quality(fp_model, q8, eval_ids, mode="int8")
    rep4 = evaluate_quantization_quality(fp_model, q4, eval_ids, mode="int4")

    assert rep8.quantized_bytes < compute_module_storage_bytes(fp_model)
    assert rep4.quantized_bytes < rep8.quantized_bytes
    assert rep8.logit_cosine_similarity > 0.99
    assert rep4.logit_cosine_similarity > 0.92


def test_inference_engine_streaming_determinism_cancellation_and_context(tmp_path: Path) -> None:
    ckpt_dir, tok_file = bootstrap_local_serving_artifacts(tmp_path / "rt")
    engine = RootismInferenceEngine(checkpoint_dir=ckpt_dir, tokenizer_dir=tok_file)

    # 1. Deterministic seeded generation with KV cache
    params = GenerationSamplingParams(max_new_tokens=10, temperature=0.0, seed=1234)
    r1 = engine.generate_chat(user_message="def add(a, b):", reasoning_mode="concise", params=params)
    r2 = engine.generate_chat(user_message="def add(a, b):", reasoning_mode="concise", params=params)
    assert r1.raw_text == r2.raw_text
    assert r1.completion_tokens >= 1
    assert r1.kv_cache_bytes > 0

    # 2. Cooperative cancellation
    ev = threading.Event()
    ev.set()
    cancelled = engine.generate_chat(user_message="Hello", params=params, cancel_event=ev)
    assert cancelled.finish_reason == "cancelled"
    assert cancelled.completion_tokens == 0

    # 3. Context window overflow truncation
    huge_ids = [100] * (engine.config.model.extended_context_length + 200)
    fitted, ctx_rep = engine.fit_to_context_window(huge_ids, max_new_tokens=64)
    assert ctx_rep.was_truncated is True
    assert len(fitted) + 64 <= engine.config.model.extended_context_length


def test_fastapi_server_sse_auth_workspace_and_root_html_platform(tmp_path: Path) -> None:
    ckpt_dir, tok_file = bootstrap_local_serving_artifacts(tmp_path / "srv_rt")
    ws_dir = tmp_path / "srv_ws"

    engine = RootismInferenceEngine(checkpoint_dir=ckpt_dir, tokenizer_dir=tok_file)
    app = create_rootism_app(
        engine=engine,
        agent_workspace_dir=ws_dir,
        api_key="test-secret-key",
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer test-secret-key"}

    # 1. Health & Auth checks
    h_resp = client.get("/api/v1/health")
    assert h_resp.status_code == 200
    h_json = h_resp.json()
    assert h_json["ready"] is True
    # A fresh runtime is honestly reported as loaded-but-untrained (standby mode).
    assert h_json["weights_loaded"] is True
    assert h_json["trained_weights"] is False
    assert h_json["in_standby_mode"] is True
    assert h_json["processed_training_tokens"] == 0

    unauth = client.get("/api/v1/model/info")
    assert unauth.status_code == 401

    info_resp = client.get("/api/v1/model/info", headers=headers)
    assert info_resp.status_code == 200
    info_json = info_resp.json()
    assert info_json["production_architecture_spec"]["verified_parameter_count"] == 3_009_630_720
    assert info_json["honest_training_status"]["is_fully_pretrained_600b_plus"] is False
    assert info_json["trained_weights"] is False
    assert "Standby" in info_json["honest_training_status"]["status_banner"]

    # 2. Untrained/standby chat is clean and helpful — never an error, never gibberish
    sync_chat = client.post(
        "/api/v1/chat/completions",
        headers=headers,
        json={
            "messages": [{"role": "user", "content": "Write a tiny helper"}],
            "reasoning_mode": "structured",
            "stream": False,
            "max_tokens": 8,
            "temperature": 0.0,
        },
    )
    assert sync_chat.status_code == 200
    standby_json = sync_chat.json()
    assert standby_json["standby"] is True
    assert standby_json["finish_reason"] == "standby_no_trained_weights"
    assert "Standby mode" in standby_json["message"]["content"]
    assert "checkpoint-dir" in standby_json["message"]["content"]

    with client.stream(
        "POST",
        "/api/v1/chat/completions",
        headers=headers,
        json={
            "messages": [{"role": "user", "content": "Stream test"}],
            "stream": True,
            "max_tokens": 6,
            "temperature": 0.0,
        },
    ) as stream_resp:
        assert stream_resp.status_code == 200
        events = [
            json.loads(line[5:].strip())
            for line in stream_resp.iter_lines()
            if line.startswith("data:")
        ]
    event_types = [e["event"] for e in events]
    assert "meta" in event_types and "done" in event_types
    assert events[-1]["standby"] is True
    assert events[-1]["finish_reason"] == "standby_no_trained_weights"

    # 3. Seamless switch to real neural generation once trained weights are present.
    #    Serving a checkpoint whose manifest reports processed_training_tokens > 0 flips the
    #    exact same running process into real `RootismForCausalLM` tensor decoding.
    engine.manifest["processed_training_tokens"] = 12_000
    engine.manifest["training_stage_tag"] = "test-trained-checkpoint"
    trained_health = client.get("/api/v1/health").json()
    assert trained_health["trained_weights"] is True
    assert trained_health["in_standby_mode"] is False

    trained_chat = client.post(
        "/api/v1/chat/completions",
        headers=headers,
        json={
            "messages": [{"role": "user", "content": "Write a tiny helper"}],
            "stream": False,
            "max_tokens": 6,
            "temperature": 0.0,
        },
    ).json()
    assert "standby" not in trained_chat
    assert trained_chat["usage"]["completion_tokens"] >= 1
    assert trained_chat["checkpoint_step"] == 0

    trained_fim = client.post(
        "/api/v1/code/complete",
        headers=headers,
        json={"prefix": "def f(x):\n    ", "suffix": "\n    return y", "max_tokens": 4},
    ).json()
    assert "standby" not in trained_fim

    with client.stream(
        "POST",
        "/api/v1/chat/completions",
        headers=headers,
        json={"messages": [{"role": "user", "content": "Stream trained"}], "stream": True, "max_tokens": 4},
    ) as trained_stream:
        trained_events = [
            json.loads(line[5:].strip())
            for line in trained_stream.iter_lines()
            if line.startswith("data:")
        ]
    assert "standby" not in trained_events[-1]
    assert trained_events[-1]["event"] == "done"

    # Return to the honest untrained standby state for the remaining assertions.
    engine.manifest["processed_training_tokens"] = 0
    engine.manifest["training_stage_tag"] = "initialized-local-verification-checkpoint"

    # 4. Workspace starts COMPLETELY EMPTY (no default/scaffold files)
    ws_files = client.get("/api/v1/workspace/files", headers=headers).json()
    assert ws_files["files"] == []
    assert ws_files["rollback_depth"] == 0

    # Save a brand-new file, then overwrite it: both must succeed without a 400.
    created = client.post(
        "/api/v1/workspace/file",
        headers=headers,
        json={"path": "src/metrics.py", "content": "def p95(xs):\n    return 1.0\n"},
    )
    assert created.status_code == 200
    assert created.json()["ok"] is True

    overwritten = client.post(
        "/api/v1/workspace/file",
        headers=headers,
        json={"path": "src/metrics.py", "content": "def p95(xs):\n    return 2.0\n"},
    )
    assert overwritten.status_code == 200
    assert overwritten.json()["ok"] is True
    assert (ws_dir / "src" / "metrics.py").read_text(encoding="utf-8").endswith("return 2.0\n")

    read_back = client.get("/api/v1/workspace/file", headers=headers, params={"path": "src/metrics.py"})
    assert read_back.status_code == 200
    assert "return 2.0" in read_back.json()["content"]

    # Reading a file that does not exist is a normal empty-slate state, not an HTTP error.
    missing = client.get(
        "/api/v1/workspace/file", headers=headers, params={"path": "does/not/exist.py"}
    )
    assert missing.status_code == 200
    assert missing.json()["exists"] is False
    assert missing.json()["content"] == ""

    ws_files = client.get("/api/v1/workspace/files", headers=headers).json()
    assert "src/metrics.py" in ws_files["files"]
    assert ws_files["rollback_depth"] == 2

    # Undo restores the previous content; a second undo on an empty stack is a clean no-op.
    reverted = client.post("/api/v1/workspace/revert", headers=headers)
    assert reverted.status_code == 200
    assert reverted.json()["reverted"] is True
    assert reverted.json()["reverted_path"] == "src/metrics.py"
    assert "return 1.0" in (ws_dir / "src" / "metrics.py").read_text(encoding="utf-8")

    # Second undo rolls back the file creation itself, leaving the project empty again.
    revert_create = client.post("/api/v1/workspace/revert", headers=headers)
    assert revert_create.status_code == 200
    assert revert_create.json()["reverted"] is True
    assert not (ws_dir / "src" / "metrics.py").exists()

    # Undo with an empty history is a clean 200 no-op, never a 400.
    revert_empty = client.post("/api/v1/workspace/revert", headers=headers)
    assert revert_empty.status_code == 200
    assert revert_empty.json()["reverted"] is False
    assert revert_empty.json()["message"] == "Nothing to undo yet."

    # 5. FIM Code Completion is a clean standby no-op without trained weights
    fim_resp = client.post(
        "/api/v1/code/complete",
        headers=headers,
        json={"prefix": "def f(x):\n    ", "suffix": "\n    return y", "max_tokens": 6},
    )
    assert fim_resp.status_code == 200
    assert fim_resp.json()["standby"] is True
    assert fim_resp.json()["completion"] == ""

    # 6. Safety confirmation gate still protects destructive agent actions
    client.post(
        "/api/v1/workspace/file",
        headers=headers,
        json={"path": "src/ui_helpers.ts", "content": "export const x = 1;\n"},
    )
    gate_run = client.post(
        "/api/v1/agent/run",
        headers=headers,
        json={"goal": "delete file src/ui_helpers.ts", "max_iterations": 4},
    ).json()
    assert gate_run["status"] == "awaiting_confirmation"
    tok = gate_run["pending_confirmation_token"]
    assert tok is not None

    reject_res = client.post(
        "/api/v1/agent/confirm",
        headers=headers,
        json={"confirmation_token": tok, "approve": False},
    ).json()
    assert reject_res["approved"] is False
    assert (ws_dir / "src" / "ui_helpers.ts").is_file()

    # 7. Agent runs cleanly against a completely empty project
    empty_app = create_rootism_app(
        checkpoint_dir=ckpt_dir,
        tokenizer_dir=tok_file,
        agent_workspace_dir=tmp_path / "empty_ws",
        api_key="test-secret-key",
    )
    empty_client = TestClient(empty_app)
    empty_report = empty_client.post(
        "/api/v1/agent/run",
        headers=headers,
        json={"goal": "inspect this project and verify tests", "max_iterations": 6},
    ).json()
    assert empty_report["status"] == "completed"
    assert all("Error" not in json.dumps(t.get("tool_result", {})) for t in empty_report["traces"])
    assert empty_client.get("/api/v1/workspace/files", headers=headers).json()["files"] == []

    # 8. Verify Root Directory HTML Platform files & strictly relative URLs (Phase X Steps 81–90)
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "index.html").is_file()
    assert (repo_root / "styles.css").is_file()
    assert (repo_root / "app.js").is_file()
    assert (repo_root / "local-api.js").is_file()

    for route in ("/", "/styles.css", "/app.js", "/local-api.js"):
        r = client.get(route)
        assert r.status_code == 200
        assert "X-Frame-Options" not in r.headers

    assert "window.RootismLocal" in client.get("/local-api.js").text

    app_js_source = (repo_root / "app.js").read_text(encoding="utf-8")
    assert "localhost" not in app_js_source
    assert "127.0.0.1" not in app_js_source
    assert "./api/v1/chat/completions" in app_js_source
    # Standalone/static preview safety: local bridge is always guarded.
    assert "window.RootismLocal ||" in app_js_source
    # Enter sends, Shift+Enter inserts a newline.
    assert "!ev.shiftKey" in app_js_source

    index_source = (repo_root / "index.html").read_text(encoding="utf-8")
    assert "./local-api.js" in index_source
    assert "Enter to send" in index_source
    # Clean inline New-File flow (no modal browser prompts that iframes can suppress)
    assert 'id="new-file-form"' in index_source
    assert 'id="new-file-path-input"' in index_source
    assert "window.prompt" not in app_js_source

    styles_source = (repo_root / "styles.css").read_text(encoding="utf-8")
    # Neutral black dark theme — no bluish slate palette anywhere.
    for bluish in ("#0d1117", "#151b23", "#212830", "#010409", "#30363d"):
        assert bluish not in styles_source
    assert "--bg-app: #0a0a0a;" in styles_source
    assert "prefers-reduced-motion" in styles_source


def test_evaluation_benchmarks_generalization_and_regression(tmp_path: Path) -> None:
    # 1. Verify held-out multi-language coding suite executes real Python, Node JS, SQLite & GCC C99
    coding_suite = CodingBenchmarkSuite()
    ref_report = coding_suite.run_suite(lambda task: task.reference_solution)
    assert ref_report["syntax_valid_rate"] == 1.0
    assert ref_report["functional_pass_at_1"] == 1.0

    bad_report = coding_suite.run_suite(lambda _task: "def broken(:\n")
    assert bad_report["functional_pass_at_1"] == 0.0

    # 2. Verify reasoning benchmark suite
    reas_suite = ReasoningBenchmarkSuite(tasks_per_faculty=1)
    reas_rep = reas_suite.run_suite(lambda t: f"{t.faithful_user_summary}\n{t.final_answer}")
    assert reas_rep["conclusion_accuracy"] == 1.0
    assert reas_rep["mean_reasoning_quality_score"] > 0.85

    # 3. Verify contamination detection & generalisation audit
    verifier = GeneralizationVerifier()
    clean_audit = verifier.verify_generalization(
        training_corpus_texts=["def unrelated_fn(a, b): return a + b"],
        solver_fn=lambda _p: "FALSE",
    )
    assert clean_audit.decontamination_clean is True

    # 4. Verify regression gate & honest report builder
    tracker = CheckpointRegressionTracker()
    snap_a = EvaluationSnapshot("step_0010", 10, 1_000_000, 2.40, 0.90, 0.70, 0.75, 120.0)
    snap_b = EvaluationSnapshot("step_0020", 20, 2_000_000, 2.65, 0.82, 0.60, 0.68, 80.0)
    reg_fail = tracker.compare(snap_a, snap_b)
    assert reg_fail.passed_gate is False
    assert len(reg_fail.regressions_detected) >= 3

    json_p, md_p = EvaluationReportBuilder.write_reports(
        tmp_path / "reports",
        runtime_meta={"model_family": "rootism-verify", "processed_training_tokens": 0, "checkpoint_step": 0},
        coding_report=ref_report,
        reasoning_report=reas_rep,
        generalization_report=clean_audit.to_dict(),
        regression_report=reg_fail.to_dict(),
    )
    assert json_p.is_file() and md_p.is_file()
    assert "DEVELOPMENT / INITIALIZED CHECKPOINT" in md_p.read_text(encoding="utf-8")
