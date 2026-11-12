"""Rootism Multi-Step Reasoning, Supervision & Presentation Package (Phase VII)."""

from rootism.reasoning.modes import (
    REASONING_MODE_SPECS,
    FormattedReasoningResponse,
    ReasoningMode,
    ReasoningOutputFormatter,
)
from rootism.reasoning.supervision import (
    ReasoningQualityEvaluator,
    ReasoningQualityScore,
    SupervisedReasoningExample,
    build_contrastive_flaw_repair_example,
    build_plan_implement_verify_example,
)
from rootism.reasoning.tasks import (
    ProgressiveReasoningGenerator,
    ReasoningCurriculumTask,
)

__all__ = [
    "REASONING_MODE_SPECS",
    "ReasoningMode",
    "FormattedReasoningResponse",
    "ReasoningOutputFormatter",
    "ReasoningQualityEvaluator",
    "ReasoningQualityScore",
    "SupervisedReasoningExample",
    "build_plan_implement_verify_example",
    "build_contrastive_flaw_repair_example",
    "ProgressiveReasoningGenerator",
    "ReasoningCurriculumTask",
]
