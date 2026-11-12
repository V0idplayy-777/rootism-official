"""
Automated Tests for Phase VII (Reasoning) & Phase VIII (Coding Agent & Sandbox) (Steps 61–76).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rootism.agent.orchestrator import (
    AgentPlannedAction,
    AgentStepTrace,
    CodingAgentOrchestrator,
)
from rootism.agent.safety import SafetyPolicyGuard, WorkspaceSecurityViolation
from rootism.agent.sandbox import SandboxedCommandRunner
from rootism.agent.testing import AgentTestVerifier, FailureDiagnosticsEngine
from rootism.agent.tools import AgentToolRegistry
from rootism.reasoning.modes import ReasoningOutputFormatter
from rootism.reasoning.supervision import (
    ReasoningQualityEvaluator,
    build_contrastive_flaw_repair_example,
    build_plan_implement_verify_example,
)
from rootism.reasoning.tasks import ProgressiveReasoningGenerator


def test_reasoning_curriculum_supervision_and_privacy_redaction() -> None:
    gen = ProgressiveReasoningGenerator(seed=42)
    suite = gen.generate_curriculum_suite(tasks_per_faculty=2)
    assert len(suite) == 8
    faculties = {t.faculty for t in suite}
    assert faculties == {"deduction", "induction", "constraint_satisfaction", "causal_error_diagnosis"}

    sup = build_plan_implement_verify_example(suite[0])
    assert sup.span_loss_weights["final_conclusion"] > sup.span_loss_weights["internal_plan"]

    contrastive = build_contrastive_flaw_repair_example(1, n_val=6)
    assert "15" in contrastive.formatted_sequence

    evaluator = ReasoningQualityEvaluator()
    good_score = evaluator.evaluate_response(
        "Chaining invariants proves the statement is FALSE.",
        expected_ground_truth_key="FALSE",
    )
    wrong_verbose = evaluator.evaluate_response(
        " ".join(["Let us ponder deeply about every invariant in detail."] * 25) + " Answer: TRUE.",
        expected_ground_truth_key="FALSE",
    )
    assert good_score.conclusion_valid is True
    assert good_score.overall_quality_score > 0.85
    assert wrong_verbose.conclusion_valid is False
    assert wrong_verbose.overall_quality_score < 0.15

    # Privacy boundary: internal `<|reason_plan_start|>` & `<|reason_verify_start|>` MUST never leak!
    raw_model_stream = (
        "<|reason_plan_start|>PRIVATE_SCRATCHPAD_SECRET_STEP<|reason_plan_end|>"
        "<|reason_verify_start|>PRIVATE_INTERNAL_CHECK<|reason_verify_end|>"
        "<|reason_summary_start|>Validated interval midpoint termination.<|reason_summary_end|>\n"
        "<|code_start|><|lang_tag|>python\ndef ok():\n    return 1\n<|code_end|><|assistant_end|>"
    )
    fmt_struct = ReasoningOutputFormatter.sanitize_and_extract(raw_model_stream, mode="structured")
    assert "PRIVATE_SCRATCHPAD_SECRET_STEP" not in fmt_struct.visible_answer
    assert "PRIVATE_INTERNAL_CHECK" not in fmt_struct.visible_answer
    assert fmt_struct.internal_deliberation_redacted is True
    assert fmt_struct.reasoning_summary == "Validated interval midpoint termination."
    assert "```python" in fmt_struct.visible_answer

    fmt_concise = ReasoningOutputFormatter.sanitize_and_extract(raw_model_stream, mode="concise")
    assert fmt_concise.reasoning_summary is None


def test_sandbox_confinement_timeouts_and_secret_stripping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    guard = SafetyPolicyGuard(ws)

    # 1. Block path traversal outside workspace
    with pytest.raises(WorkspaceSecurityViolation):
        guard.resolve_confined_path("../../etc/passwd")

    # 2. Unconditional block on sudo & root rm
    assert guard.evaluate_command("sudo ls").blocked_unconditionally is True
    assert guard.evaluate_command("rm -rf /").blocked_unconditionally is True

    # 3. Confirmation gate on `rm`
    rm_dec = guard.evaluate_command("rm temp.txt")
    assert rm_dec.requires_user_confirmation is True
    assert rm_dec.confirmation_token is not None

    # 4. Secret environment stripping & timeout enforcement
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_super_secret_should_not_leak")
    runner = SandboxedCommandRunner(guard, default_timeout_sec=1.0)
    env_res = runner.run("env")
    assert env_res.executed is True
    assert "ghp_super_secret_should_not_leak" not in env_res.stdout

    timeout_res = runner.run("python3 -c 'import time; time.sleep(5)'", timeout_sec=0.4)
    assert timeout_res.timed_out is True
    diag = FailureDiagnosticsEngine.diagnose(timeout_res)
    assert diag.category == "TIMEOUT_OR_RESOURCE_LIMIT"


def test_coding_agent_repair_loop_rollback_and_guardrails(tmp_path: Path) -> None:
    ws = tmp_path / "project"
    ws.mkdir()
    (ws / "calc.py").write_text(
        "def add_positive(a: int, b: int) -> int:\n    return a - b  # intentional bug\n",
        encoding="utf-8",
    )
    (ws / "test_calc.py").write_text(
        "from calc import add_positive\n\ndef test_add() -> None:\n    assert add_positive(4, 3) == 7\n",
        encoding="utf-8",
    )

    tools = AgentToolRegistry(ws)
    orchestrator = CodingAgentOrchestrator(tools, max_iterations=8)

    # Multi-step repair loop: inspect -> run_tests (fails assertion) -> edit_file -> run_tests (passes) -> summarise
    # Pre-compute confirmation token for the edit_file overwrite (safety gate)
    confirm_token = tools.safety.compute_confirmation_token("file_mutation", "edit:calc.py")

    def _repair_planner(
        _goal: str, history: list[AgentStepTrace], _reg: AgentToolRegistry
    ) -> AgentPlannedAction:
        step = len(history)
        if step == 0:
            return AgentPlannedAction("inspect", "Inspect repo", "inspect_project", {})
        if step == 1:
            return AgentPlannedAction("test", "Run initial pytest", "run_tests", {})
        if step == 2:
            # Verify Step 1 diagnosed FAILING_ASSERTION_OR_LOGIC_BUG
            prev_diag = history[-1].tool_result["payload"]["diagnosis"]  # type: ignore[index]
            assert prev_diag["category"] == "FAILING_ASSERTION_OR_LOGIC_BUG"
            return AgentPlannedAction(
                "repair",
                "Fix subtraction bug in calc.py",
                "edit_file",
                {"path": "calc.py", "old_text": "return a - b", "new_text": "return a + b", "confirmation_token": confirm_token},
            )
        if step == 3:
            return AgentPlannedAction("test", "Verify fix with pytest", "run_tests", {})
        return AgentPlannedAction("summarise", "Bug diagnosed, patched, and verified.", is_final_summary=True)

    report = orchestrator.run("Fix add_positive bug", _repair_planner)
    assert report.status == "completed"
    assert report.tests_verified is True
    assert "calc.py" in report.modified_files
    assert len(report.unified_diffs) == 1

    # Verify automated test synthesizer & reversible rollback
    verifier = AgentTestVerifier(tools.sandbox)
    rel_test, src_test = verifier.synthesize_python_smoke_test("calc.py")
    assert rel_test == "tests/test_auto_calc.py"
    assert "add_positive" in src_test

    rev = tools.invoke("revert_last_edit", {})
    assert rev.ok is True
    assert "a - b" in (ws / "calc.py").read_text(encoding="utf-8")

    # Verify anti-infinite-loop duplicate action guardrail!
    loop_report = orchestrator.run(
        "Stuck loop test",
        lambda _g, _h, _t: AgentPlannedAction("inspect", "Repeat same read", "list_files", {}),
    )
    assert loop_report.status == "stopped_guardrail"
    assert "repeated identical tool call" in loop_report.stop_reason

    # Verify destructive file deletion pauses for explicit confirmation token!
    confirm_report = orchestrator.run(
        "Delete calc.py",
        lambda _g, _h, _t: AgentPlannedAction("edit", "Delete file", "delete_file", {"path": "calc.py"}),
    )
    assert confirm_report.status == "awaiting_confirmation"
    assert confirm_report.pending_confirmation_token is not None
    assert (ws / "calc.py").exists()  # Must NOT have deleted without approval!
