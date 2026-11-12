"""
Multi-Step Coding Agent Orchestrator & Anti-Loop Guardrails (Steps 68, 72, 75, 76).

Executes the structured engineering loop:
  `inspect -> plan -> edit -> execute -> test -> diagnose -> repair -> summarise`

Anti-infinite-loop guardrails (Step 75):
1. Strict `max_iterations` hard ceiling.
2. Exact repeated tool-action fingerprint detector (`sha256(tool_name + canonical_args)`):
   halts immediately if an identical failing/no-op action is attempted more than `max_duplicate_actions` times.
3. Stagnation detector: tracks consecutive identical failure diagnosis signatures and stops cleanly
   when `max_stagnant_failures` is reached rather than thrashing indefinitely.
4. Safety confirmation pause: immediately suspends execution with status `"awaiting_confirmation"`
   and returns the confirmation token when a destructive operation is requested (Step 76).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from rootism.agent.tools import AgentToolRegistry, AgentToolResult


@dataclass(frozen=True)
class AgentPlannedAction:
    phase: str  # "inspect" | "plan" | "edit" | "execute" | "test" | "diagnose" | "repair" | "summarise"
    rationale: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    is_final_summary: bool = False


@dataclass(frozen=True)
class AgentStepTrace:
    step_index: int
    phase: str
    rationale: str
    tool_name: str | None
    tool_args: dict[str, Any] | None
    tool_result: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentRunReport:
    goal: str
    status: str  # "completed" | "awaiting_confirmation" | "stopped_guardrail" | "failed"
    stop_reason: str
    iterations_executed: int
    tests_verified: bool
    pending_confirmation_token: str | None
    pending_action: dict[str, Any] | None
    modified_files: tuple[str, ...]
    unified_diffs: tuple[str, ...]
    final_summary: str
    traces: tuple[AgentStepTrace, ...]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["modified_files"] = list(self.modified_files)
        d["unified_diffs"] = list(self.unified_diffs)
        d["traces"] = [t.to_dict() for t in self.traces]
        return d


AgentPlannerCallback = Callable[[str, list[AgentStepTrace], AgentToolRegistry], AgentPlannedAction]


class CodingAgentOrchestrator:
    """
    Coordinates multi-step repository inspection, planning, editing, verification,
    failure diagnosis, iterative repair, and final reporting.
    """

    def __init__(
        self,
        tools: AgentToolRegistry,
        *,
        max_iterations: int = 12,
        max_duplicate_actions: int = 1,
        max_stagnant_failures: int = 3,
    ) -> None:
        self.tools = tools
        self.max_iterations = max(1, int(max_iterations))
        self.max_duplicate_actions = max(1, int(max_duplicate_actions))
        self.max_stagnant_failures = max(1, int(max_stagnant_failures))

    @staticmethod
    def action_fingerprint(tool_name: str, args: dict[str, Any] | None) -> str:
        canonical = json.dumps({"tool": tool_name, "args": args or {}}, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def run(
        self,
        goal: str,
        planner: AgentPlannerCallback,
        *,
        on_step_event: Callable[[AgentStepTrace], None] | None = None,
    ) -> AgentRunReport:
        traces: list[AgentStepTrace] = []
        action_counts: dict[str, int] = {}
        last_failure_sig: str | None = None
        stagnant_count = 0
        tests_verified = False

        for step_idx in range(1, self.max_iterations + 1):
            planned = planner(goal, traces, self.tools)

            if planned.is_final_summary or planned.tool_name is None:
                trace = AgentStepTrace(
                    step_index=step_idx,
                    phase="summarise",
                    rationale=planned.rationale,
                    tool_name=None,
                    tool_args=None,
                    tool_result=None,
                )
                traces.append(trace)
                if on_step_event:
                    on_step_event(trace)
                return self._finalize_report(
                    goal=goal,
                    status="completed",
                    stop_reason="Agent completed task and produced final summary.",
                    tests_verified=tests_verified,
                    final_summary=planned.rationale,
                    traces=traces,
                )

            # Anti-loop Guardrail 1: Repeated identical action check
            # Read-only test runs after an edit are allowed, so reset duplicate map on successful file edits!
            fp = self.action_fingerprint(planned.tool_name, planned.tool_args)
            action_counts[fp] = action_counts.get(fp, 0) + 1
            if action_counts[fp] > self.max_duplicate_actions:
                return self._finalize_report(
                    goal=goal,
                    status="stopped_guardrail",
                    stop_reason=(
                        f"Guardrail halted execution: repeated identical tool call "
                        f"`{planned.tool_name}` ({action_counts[fp]} times) without intervening workspace change."
                    ),
                    tests_verified=tests_verified,
                    final_summary="Stopped early due to repeated identical action guardrail.",
                    traces=traces,
                )

            result: AgentToolResult = self.tools.invoke(planned.tool_name, planned.tool_args)
            trace = AgentStepTrace(
                step_index=step_idx,
                phase=planned.phase,
                rationale=planned.rationale,
                tool_name=planned.tool_name,
                tool_args=planned.tool_args,
                tool_result=result.to_dict(),
            )
            traces.append(trace)
            if on_step_event:
                on_step_event(trace)

            # Human-in-the-loop Safety Confirmation Pause (Step 76)
            if result.requires_confirmation:
                return self._finalize_report(
                    goal=goal,
                    status="awaiting_confirmation",
                    stop_reason=result.summary,
                    tests_verified=tests_verified,
                    final_summary=f"Paused awaiting explicit user confirmation: {result.summary}",
                    traces=traces,
                    pending_token=result.confirmation_token,
                    pending_action={"tool_name": planned.tool_name, "tool_args": planned.tool_args},
                )

            if planned.tool_name in ("create_file", "edit_file", "revert_last_edit") and result.ok:
                # Workspace state changed -> allow fresh test/command re-runs
                action_counts.clear()
                tests_verified = False

            if planned.tool_name == "run_tests":
                if result.ok:
                    tests_verified = True
                    stagnant_count = 0
                    last_failure_sig = None
                else:
                    diag = result.payload.get("diagnosis", {})
                    sig = f"{diag.get('category')}:{diag.get('file_location')}:{diag.get('line_number')}:{diag.get('error_type')}"
                    if sig == last_failure_sig:
                        stagnant_count += 1
                    else:
                        last_failure_sig = sig
                        stagnant_count = 1
                    if stagnant_count >= self.max_stagnant_failures:
                        return self._finalize_report(
                            goal=goal,
                            status="stopped_guardrail",
                            stop_reason=(
                                f"Guardrail halted execution: {stagnant_count} consecutive attempts "
                                f"produced unchanged failure signature ({sig})."
                            ),
                            tests_verified=False,
                            final_summary=f"Stopped due to stagnation on {sig}.",
                            traces=traces,
                        )

        return self._finalize_report(
            goal=goal,
            status="stopped_guardrail",
            stop_reason=f"Reached maximum iteration budget ({self.max_iterations} steps).",
            tests_verified=tests_verified,
            final_summary=f"Stopped cleanly at max_iterations={self.max_iterations}.",
            traces=traces,
        )

    def _finalize_report(
        self,
        *,
        goal: str,
        status: str,
        stop_reason: str,
        tests_verified: bool,
        final_summary: str,
        traces: list[AgentStepTrace],
        pending_token: str | None = None,
        pending_action: dict[str, Any] | None = None,
    ) -> AgentRunReport:
        mod_files = list(dict.fromkeys(e.relative_path for e in self.tools.rollback_stack))
        diffs = [e.unified_diff for e in self.tools.rollback_stack if e.unified_diff]
        return AgentRunReport(
            goal=goal,
            status=status,
            stop_reason=stop_reason,
            iterations_executed=len(traces),
            tests_verified=tests_verified,
            pending_confirmation_token=pending_token,
            pending_action=pending_action,
            modified_files=tuple(mod_files),
            unified_diffs=tuple(diffs),
            final_summary=final_summary,
            traces=tuple(traces),
        )
