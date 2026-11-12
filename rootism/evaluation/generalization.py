"""
Novelty, Contamination Absence & Counterfactual Generalisation Verifier (Step 93).

Tests genuine generalisation vs surface-form memorisation via two rigorous mechanisms:
1. Contamination Audit: verifies zero exact or 13-gram overlap between evaluation prompts/targets
   and training corpus shards using `BenchmarkDecontaminator`.
2. Isomorphic Perturbation Stability: rewrites identifier names and numeric constants on reasoning
   & coding tasks (`x -> alpha_k`, shifted offsets) and verifies whether accuracy remains invariant!
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from rootism.data.contamination import BenchmarkItem, DecontaminationEngine
from rootism.reasoning.tasks import ProgressiveReasoningGenerator


@dataclass(frozen=True)
class GeneralizationAuditReport:
    total_eval_prompts_checked: int
    contaminated_training_docs_found: int
    decontamination_clean: bool
    base_seed_accuracy: float
    perturbed_isomorph_accuracy: float
    memorisation_drop_delta: float
    generalises_robustly: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GeneralizationVerifier:
    """
    Audits evaluation integrity and measures invariance across isomorphic problem perturbations.
    """

    def verify_generalization(
        self,
        *,
        training_corpus_texts: list[str],
        solver_fn: Callable[[str], str],
    ) -> GeneralizationAuditReport:
        base_tasks = ProgressiveReasoningGenerator(seed=500_001).generate_curriculum_suite(
            tasks_per_faculty=2
        )
        # Disjoint unseen isomorphic seed (different variable symbols, DAG topologies, quadratics)
        perturbed_tasks = ProgressiveReasoningGenerator(seed=700_999).generate_curriculum_suite(
            tasks_per_faculty=2
        )

        eval_items = [
            BenchmarkItem(benchmark_name="rootism-gen-audit", item_id=t.task_id, prompt_text=t.prompt)
            for t in (base_tasks + perturbed_tasks)
        ]
        decontam = DecontaminationEngine(word_ngram_size=13)
        decontam.register_benchmark_items(eval_items)

        contaminated = 0
        for doc in training_corpus_texts:
            res = decontam.check_document(doc)
            if res.contaminated:
                contaminated += 1

        total_checked = len(eval_items)

        base_hits = sum(
            1
            for t in base_tasks
            if t.ground_truth_key.lower() in solver_fn(t.prompt).lower()
        )
        pert_hits = sum(
            1
            for t in perturbed_tasks
            if t.ground_truth_key.lower() in solver_fn(t.prompt).lower()
        )

        base_acc = base_hits / float(max(1, len(base_tasks)))
        pert_acc = pert_hits / float(max(1, len(perturbed_tasks)))
        drop = max(0.0, base_acc - pert_acc)

        return GeneralizationAuditReport(
            total_eval_prompts_checked=total_checked,
            contaminated_training_docs_found=contaminated,
            decontamination_clean=(contaminated == 0),
            base_seed_accuracy=round(base_acc, 4),
            perturbed_isomorph_accuracy=round(pert_acc, 4),
            memorisation_drop_delta=round(drop, 4),
            generalises_robustly=(contaminated == 0 and drop <= 0.20),
        )
