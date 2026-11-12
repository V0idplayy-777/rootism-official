"""
Checkpoint Regression Detection & Structured Evaluation Reporting (Steps 94 & 96).

Compares evaluation metrics across checkpoints (`baseline` vs `candidate`) and emits
reproducible JSON + Markdown reports with honest training-token disclosure.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationSnapshot:
    checkpoint_name: str
    step: int
    processed_training_tokens: int
    validation_loss: float
    coding_syntax_valid_rate: float
    coding_pass_at_1: float
    reasoning_quality_score: float
    tokens_per_second: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegressionComparisonReport:
    baseline_checkpoint: str
    candidate_checkpoint: str
    passed_gate: bool
    regressions_detected: tuple[str, ...]
    improvements_detected: tuple[str, ...]
    metric_deltas: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["regressions_detected"] = list(self.regressions_detected)
        d["improvements_detected"] = list(self.improvements_detected)
        return d


class CheckpointRegressionTracker:
    """
    Detects statistically meaningful quality or throughput regressions between
    model checkpoints before promoting a new checkpoint to serving (Step 94).
    """

    def __init__(
        self,
        *,
        max_allowed_loss_increase: float = 0.05,
        max_allowed_score_drop: float = 0.02,
        max_allowed_throughput_drop_ratio: float = 0.20,
    ) -> None:
        self.max_allowed_loss_increase = max_allowed_loss_increase
        self.max_allowed_score_drop = max_allowed_score_drop
        self.max_allowed_throughput_drop_ratio = max_allowed_throughput_drop_ratio

    def compare(
        self,
        baseline: EvaluationSnapshot,
        candidate: EvaluationSnapshot,
    ) -> RegressionComparisonReport:
        regressions: list[str] = []
        improvements: list[str] = []

        d_loss = candidate.validation_loss - baseline.validation_loss
        d_syn = candidate.coding_syntax_valid_rate - baseline.coding_syntax_valid_rate
        d_pass = candidate.coding_pass_at_1 - baseline.coding_pass_at_1
        d_reas = candidate.reasoning_quality_score - baseline.reasoning_quality_score
        d_tps = candidate.tokens_per_second - baseline.tokens_per_second

        if d_loss > self.max_allowed_loss_increase:
            regressions.append(f"Validation loss worsened by +{d_loss:.4f}")
        elif d_loss < -0.01:
            improvements.append(f"Validation loss improved by {d_loss:.4f}")

        if d_syn < -self.max_allowed_score_drop:
            regressions.append(f"Coding syntax validity dropped by {d_syn:.4f}")
        elif d_syn > 0.01:
            improvements.append(f"Coding syntax validity improved by +{d_syn:.4f}")

        if d_pass < -self.max_allowed_score_drop:
            regressions.append(f"Coding functional pass@1 dropped by {d_pass:.4f}")
        elif d_pass > 0.01:
            improvements.append(f"Coding functional pass@1 improved by +{d_pass:.4f}")

        if d_reas < -self.max_allowed_score_drop:
            regressions.append(f"Reasoning quality score dropped by {d_reas:.4f}")
        elif d_reas > 0.01:
            improvements.append(f"Reasoning quality score improved by +{d_reas:.4f}")

        if baseline.tokens_per_second > 0:
            tps_drop_ratio = (baseline.tokens_per_second - candidate.tokens_per_second) / baseline.tokens_per_second
            if tps_drop_ratio > self.max_allowed_throughput_drop_ratio:
                regressions.append(
                    f"Inference throughput regressed by {tps_drop_ratio * 100:.1f}% "
                    f"({baseline.tokens_per_second:.1f} -> {candidate.tokens_per_second:.1f} tok/s)"
                )

        return RegressionComparisonReport(
            baseline_checkpoint=baseline.checkpoint_name,
            candidate_checkpoint=candidate.checkpoint_name,
            passed_gate=(len(regressions) == 0),
            regressions_detected=tuple(regressions),
            improvements_detected=tuple(improvements),
            metric_deltas={
                "validation_loss_delta": round(d_loss, 6),
                "coding_syntax_valid_delta": round(d_syn, 4),
                "coding_pass_at_1_delta": round(d_pass, 4),
                "reasoning_quality_delta": round(d_reas, 4),
                "tokens_per_second_delta": round(d_tps, 2),
            },
        )


class EvaluationReportBuilder:
    """
    Produces reproducible JSON and Markdown evaluation reports with strict honesty
    disclosures whenever `processed_training_tokens < 600_000_000_000` (Step 96).
    """

    @staticmethod
    def write_reports(
        output_dir: str | Path,
        *,
        runtime_meta: dict[str, Any],
        coding_report: dict[str, Any],
        reasoning_report: dict[str, Any],
        generalization_report: dict[str, Any] | None = None,
        regression_report: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        processed = int(runtime_meta.get("processed_training_tokens", 0))
        production_complete = processed >= 600_000_000_000

        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "production_600b_budget_complete": production_complete,
            "honesty_disclosure": (
                "Full >600B commercial pretraining budget satisfied."
                if production_complete
                else (
                    f"DEVELOPMENT / INITIALIZED CHECKPOINT ({processed:,} processed training tokens). "
                    "Metrics verify end-to-end harness & model execution; full >600B cluster pretraining is required for production capability."
                )
            ),
            "runtime_metadata": runtime_meta,
            "coding_evaluation": coding_report,
            "reasoning_evaluation": reasoning_report,
            "generalization_audit": generalization_report,
            "regression_comparison": regression_report,
        }

        json_path = out / "evaluation_report.json"
        md_path = out / "evaluation_report.md"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        md_lines = [
            "# Rootism Reproducible Evaluation Report",
            "",
            f"- **Generated (UTC)**: `{payload['generated_utc']}`",
            f"- **Architecture**: `{runtime_meta.get('model_family')}` (`{runtime_meta.get('analytical_parameters', 0):,}` params)",
            f"- **Checkpoint Step**: `{runtime_meta.get('checkpoint_step')}` (`{runtime_meta.get('training_stage_tag')}`)",
            f"- **Processed Training Tokens**: `{processed:,}`",
            f"- **Disclosure**: {payload['honesty_disclosure']}",
            "",
            "## 1. Multi-Language Coding Suite",
            f"- Total Tasks: `{coding_report['total_tasks']}`",
            f"- Syntax Validity Rate: `{coding_report['syntax_valid_rate']:.2%}`",
            f"- Functional Pass@1: `{coding_report['functional_pass_at_1']:.2%}`",
            "",
            "## 2. Multi-Faculty Reasoning Suite",
            f"- Total Tasks: `{reasoning_report['total_tasks']}`",
            f"- Conclusion Accuracy: `{reasoning_report['conclusion_accuracy']:.2%}`",
            f"- Overall Quality Score: `{reasoning_report['mean_reasoning_quality_score']:.4f}`",
        ]
        if generalization_report:
            md_lines.extend(
                [
                    "",
                    "## 3. Contamination & Generalisation Audit",
                    f"- Decontamination Clean (0 13-gram leaks): `{generalization_report['decontamination_clean']}`",
                    f"- Base Accuracy vs Isomorphic Perturbation Accuracy: `{generalization_report['base_seed_accuracy']:.2%}` vs `{generalization_report['perturbed_isomorph_accuracy']:.2%}`",
                ]
            )
        if regression_report:
            md_lines.extend(
                [
                    "",
                    "## 4. Checkpoint Regression Gate",
                    f"- Passed Gate: `{regression_report['passed_gate']}`",
                    f"- Regressions: `{regression_report['regressions_detected']}`",
                ]
            )
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        return json_path, md_path
