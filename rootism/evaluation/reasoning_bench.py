"""
Multi-Faculty Reasoning Evaluation Suite (Step 92).

Evaluates multi-step deduction, mathematical/programmatic induction, DAG constraint
satisfaction, and causal error diagnosis using `ReasoningQualityEvaluator`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rootism.reasoning.supervision import ReasoningQualityEvaluator
from rootism.reasoning.tasks import ProgressiveReasoningGenerator, ReasoningCurriculumTask


class ReasoningBenchmarkSuite:
    """
    Runs held-out reasoning benchmarks generated with an isolated evaluation seed
    distinct from training synthetic seeds!
    """

    def __init__(self, eval_seed: int = 99_887_766, tasks_per_faculty: int = 2) -> None:
        gen = ProgressiveReasoningGenerator(seed=eval_seed)
        self.tasks: list[ReasoningCurriculumTask] = gen.generate_curriculum_suite(
            tasks_per_faculty=tasks_per_faculty
        )
        self.evaluator = ReasoningQualityEvaluator()

    def run_suite(self, responder_fn: Callable[[ReasoningCurriculumTask], str]) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        faculty_scores: dict[str, list[float]] = {}

        for task in self.tasks:
            resp = responder_fn(task)
            score = self.evaluator.evaluate_response(
                resp, expected_ground_truth_key=task.ground_truth_key
            )
            faculty_scores.setdefault(task.faculty, []).append(score.overall_quality_score)
            records.append(
                {
                    "task_id": task.task_id,
                    "faculty": task.faculty,
                    "difficulty": task.difficulty,
                    "score": score.to_dict(),
                }
            )

        n = max(1, len(records))
        accuracy = sum(1 for r in records if r["score"]["conclusion_valid"]) / float(n)
        mean_quality = sum(r["score"]["overall_quality_score"] for r in records) / float(n)
        per_faculty = {
            fac: round(sum(vals) / len(vals), 4) for fac, vals in faculty_scores.items()
        }
        return {
            "total_tasks": len(records),
            "conclusion_accuracy": round(accuracy, 4),
            "mean_reasoning_quality_score": round(mean_quality, 4),
            "per_faculty_quality": per_faculty,
            "results": records,
        }
