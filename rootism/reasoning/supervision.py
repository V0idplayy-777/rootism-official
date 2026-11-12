"""
Multi-Objective Structured Reasoning Supervision & Quality Evaluation (Steps 64 & 65).

Avoids relying exclusively on chain-of-thought imitation by implementing three distinct
supervision formulations:
1. `plan_implement_verify`: Structured decomposition + self-verification + solution
   with differential per-span loss weights (prioritizing verified conclusions & working code!).
2. `contrastive_flaw_repair`: Teaches invalid-step detection and repair on plausible-looking
   flawed derivations so verbosity without validity is penalized.
3. `execution_state_trace`: Direct prediction of program state invariants without prose filler.

Also implements `ReasoningQualityEvaluator` judging correctness, consistency, and
conciseness efficiency (never equating longer output with stronger reasoning).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from rootism.reasoning.tasks import ReasoningCurriculumTask


@dataclass(frozen=True)
class SupervisedReasoningExample:
    task_id: str
    objective_type: str
    formatted_sequence: str
    span_loss_weights: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_plan_implement_verify_example(task: ReasoningCurriculumTask) -> SupervisedReasoningExample:
    """
    Format a task with explicit plan, verification, user-facing summary, and final answer spans,
    assigning highest supervision weight to verification & final answer accuracy.
    """
    seq = (
        f"<|user_start|><|reason_mode:structured|>\n{task.prompt}\n<|user_end|>"
        f"<|assistant_start|>"
        f"<|reason_plan_start|>{task.structured_plan}<|reason_plan_end|>"
        f"<|reason_verify_start|>{task.verification_check}<|reason_verify_end|>"
        f"<|reason_summary_start|>{task.faithful_user_summary}<|reason_summary_end|>\n"
        f"{task.final_answer}"
        f"<|assistant_end|>"
    )
    return SupervisedReasoningExample(
        task_id=task.task_id,
        objective_type="plan_implement_verify",
        formatted_sequence=seq,
        span_loss_weights={
            "prompt": 0.0,
            "internal_plan": 0.65,
            "self_verification": 1.15,
            "user_summary": 1.0,
            "final_conclusion": 1.5,
        },
    )


def build_contrastive_flaw_repair_example(idx: int, n_val: int = 6) -> SupervisedReasoningExample:
    """
    Supervise explicit identification of an invalid logical step in a flawed proof/algorithm
    followed by the corrected derivation (Step 65).
    """
    true_sum = n_val * (n_val - 1) // 2
    wrong_sum = n_val * (n_val + 1) // 2
    flawed_draft = (
        f"Step 1: Count unordered pairs from {n_val} distinct services.\n"
        f"Step 2: Apply sum 1..{n_val} = {n_val}*({n_val}+1)/2 = {wrong_sum}."
    )
    critique_and_fix = (
        f"Flaw at Step 2: unordered pairs of distinct elements equal C({n_val}, 2) = "
        f"sum_{{i=1}}^{{{n_val - 1}}} i = {n_val}*({n_val}-1)/2 = {true_sum}, not {wrong_sum} "
        f"(which erroneously includes {n_val} self-pairs)."
    )
    seq = (
        "<|user_start|><|reason_mode:explain|>\n"
        f"Audit the following derivation for logical soundness and correct any invalid step:\n{flawed_draft}\n"
        "<|user_end|><|assistant_start|>"
        f"<|reason_verify_start|>Test base case n=2: draft gives 3 pairs, actual is 1 pair -> Step 2 invalid.<|reason_verify_end|>"
        f"<|reason_summary_start|>{critique_and_fix}<|reason_summary_end|>\n"
        f"Corrected count: **{true_sum}**.<|assistant_end|>"
    )
    return SupervisedReasoningExample(
        task_id=f"contrastive-flaw-{idx:05d}",
        objective_type="contrastive_flaw_repair",
        formatted_sequence=seq,
        span_loss_weights={"prompt": 0.0, "self_verification": 1.2, "user_summary": 1.2, "final_conclusion": 1.5},
    )


@dataclass(frozen=True)
class ReasoningQualityScore:
    conclusion_valid: bool
    internally_consistent: bool
    repetition_penalty_factor: float
    conciseness_efficiency: float
    overall_quality_score: float
    failure_notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["failure_notes"] = list(self.failure_notes)
        return d


class ReasoningQualityEvaluator:
    """
    Evaluates reasoning quality by conclusion validity, logical non-contradiction,
    and conciseness — explicitly penalizing verbose circular filler (Step 64).
    """

    def evaluate_response(
        self,
        response_text: str,
        *,
        expected_ground_truth_key: str,
        reference_token_budget: int = 120,
    ) -> ReasoningQualityScore:
        notes: list[str] = []
        clean = response_text.strip()
        if not clean:
            return ReasoningQualityScore(False, False, 0.0, 0.0, 0.0, ("EMPTY_RESPONSE",))

        conclusion_ok = expected_ground_truth_key.lower() in clean.lower()
        if not conclusion_ok:
            notes.append(f"Expected conclusion marker {expected_ground_truth_key!r} missing.")

        # Detect direct self-contradictions (e.g. claiming both IS TRUE and IS FALSE for same conclusion)
        upper = clean.upper()
        contradictory = (
            ("MUST BE TRUE" in upper and "MUST BE FALSE" in upper)
            or ("ANSWER IS TRUE" in upper and "ANSWER IS FALSE" in upper)
        )
        consistent = not contradictory
        if contradictory:
            notes.append("Detected contradictory truth-value assertions in explanation.")

        # Detect circular sentence repetition
        sentences = [
            s.strip().lower()
            for s in re.split(r"[.!?\n]+", clean)
            if len(s.strip()) >= 16
        ]
        if sentences:
            unique_frac = len(set(sentences)) / float(len(sentences))
        else:
            unique_frac = 1.0
        if unique_frac < 0.75:
            notes.append(f"Repetitive/circular reasoning detected (unique sentence ratio={unique_frac:.2f}).")

        # Conciseness efficiency: 1.0 within reference budget, gently decaying if excessively bloated
        approx_words = max(1, len(clean.split()))
        if approx_words <= reference_token_budget:
            efficiency = 1.0
        else:
            over_ratio = float(reference_token_budget) / float(approx_words)
            efficiency = max(0.35, round(over_ratio**0.5, 4))

        if not conclusion_ok:
            # An eloquent 500-word explanation of a wrong conclusion scores near zero!
            overall = 0.10 * (1.0 if consistent else 0.0) * unique_frac
        else:
            base = 0.75 if consistent else 0.25
            overall = (base + 0.25 * efficiency) * unique_frac

        return ReasoningQualityScore(
            conclusion_valid=conclusion_ok,
            internally_consistent=consistent,
            repetition_penalty_factor=round(unique_frac, 4),
            conciseness_efficiency=round(efficiency, 4),
            overall_quality_score=round(min(1.0, overall), 4),
            failure_notes=tuple(notes),
        )
