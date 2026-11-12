"""Rootism Evaluation, Benchmarking, Generalisation & Regression Package (Phase XI)."""

from rootism.evaluation.coding_bench import (
    CodingBenchmarkSuite,
    CodingBenchmarkTask,
    CodingTaskResult,
    build_heldout_coding_tasks,
    extract_code_from_response,
)
from rootism.evaluation.generalization import (
    GeneralizationAuditReport,
    GeneralizationVerifier,
)
from rootism.evaluation.reasoning_bench import ReasoningBenchmarkSuite
from rootism.evaluation.regression import (
    CheckpointRegressionTracker,
    EvaluationReportBuilder,
    EvaluationSnapshot,
    RegressionComparisonReport,
)

__all__ = [
    "CodingBenchmarkTask",
    "CodingTaskResult",
    "CodingBenchmarkSuite",
    "build_heldout_coding_tasks",
    "extract_code_from_response",
    "ReasoningBenchmarkSuite",
    "GeneralizationAuditReport",
    "GeneralizationVerifier",
    "EvaluationSnapshot",
    "RegressionComparisonReport",
    "CheckpointRegressionTracker",
    "EvaluationReportBuilder",
]
