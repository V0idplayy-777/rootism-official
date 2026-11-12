"""
Principled 650B Training Token Budget Allocation (Step 33).

Allocates tokens across modalities and programming languages based on:
1. Practical developer utility and reasoning transfer
2. Permissive commercial-license corpus availability
3. Syntactic/semantic complexity (systems languages like Rust/C++ benefit from higher density per syntax construct)

Never uses a naive uniform split across languages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rootism.config import DEFAULT_PRODUCTION_TRAINING_TOKENS


@dataclass(frozen=True)
class ModalityAllocation:
    modality_id: str
    share_fraction: float
    token_budget: int
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LanguageCodeAllocation:
    language: str
    tier: str
    share_of_code_fraction: float
    token_budget: int
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# High-level modality breakdown across the 650B target (sums strictly to 1.0000)
MODALITY_SHARES: tuple[tuple[str, float, str], ...] = (
    (
        "source_code_curated",
        0.54,
        "High-quality permissively licensed single-file & multi-language source code.",
    ),
    (
        "repository_and_debugging_workflows",
        0.12,
        "Multi-file repository contexts, commit diffs, bug-fix pairs, code reviews, and test suites.",
    ),
    (
        "algorithmic_and_mathematical_reasoning",
        0.14,
        "Multi-step formal deduction, discrete math, algorithms, constraint satisfaction, and verified synthetic reasoning.",
    ),
    (
        "technical_docs_tutorials_and_api_refs",
        0.12,
        "Official framework/stdlib documentation, technical tutorials, architecture specs, and engineering discussions.",
    ),
    (
        "general_natural_language_foundation",
        0.08,
        "Curated non-code prose preserving strong instruction comprehension, nuance, and clear technical communication.",
    ),
)

# Relative shares within source-code budget across 10 Primary (82%), 9 Secondary (16%), and Config/Build DSLs (2%)
CODE_LANGUAGE_SHARES: tuple[tuple[str, str, float, str], ...] = (
    # Primary (0.820 total)
    ("python", "primary", 0.180, "Anchor language for AI/ML, backend APIs, scripting, and algorithmic reasoning transfer."),
    ("typescript", "primary", 0.130, "Typed full-stack web & application engineering; strong type-constraint supervision."),
    ("javascript", "primary", 0.090, "Ubiquitous web/Node.js runtime competence."),
    ("rust", "primary", 0.095, "Ownership/lifetime/type-system rigor sharply improves general invariant & memory reasoning."),
    ("cpp", "primary", 0.085, "High-performance systems, templates, RAII, and algorithmic data structures."),
    ("go", "primary", 0.070, "Cloud-native distributed systems, concurrency, and clean standard-library idioms."),
    ("c", "primary", 0.055, "Low-level memory model, OS interfaces, ABI/FFI foundations."),
    ("sql", "primary", 0.045, "Relational set logic, query planning, schema design, and analytical transformations."),
    ("html", "primary", 0.035, "Semantic accessible document structure and web platform integration."),
    ("css", "primary", 0.035, "Responsive layout geometry, Flexbox/Grid, and design systems."),
    # Secondary (0.160 total)
    ("java", "secondary", 0.035, "Enterprise JVM backend & concurrency ecosystems."),
    ("csharp", "secondary", 0.030, ".NET enterprise, async/LINQ, and systems tooling."),
    ("bash", "secondary", 0.022, "Essential DevOps, build automation, and coding-agent terminal execution."),
    ("php", "secondary", 0.016, "Web backend maintenance and modern PHP 8+ strict typing."),
    ("kotlin", "secondary", 0.015, "Modern JVM/Android & structured coroutines."),
    ("swift", "secondary", 0.013, "Apple/systems ARC & actor concurrency."),
    ("ruby", "secondary", 0.012, "Expressive web/scripting & test DSLs."),
    ("lua", "secondary", 0.009, "Embedded scripting, game/editor configuration, coroutines."),
    ("powershell", "secondary", 0.008, "Cross-platform/Windows automation & object pipelines."),
    # Build & structured workspace manifests (0.020 total)
    ("workspace_manifests", "auxiliary", 0.020, "Dockerfile, Makefile, CMake, TOML, YAML, JSON schemas supporting multi-file repos."),
)


def compute_training_token_allocation(
    total_training_tokens: int = DEFAULT_PRODUCTION_TRAINING_TOKENS,
) -> dict[str, Any]:
    """Compute exact integer token budgets per modality and per programming language."""
    modalities: list[ModalityAllocation] = []
    allocated_mod = 0
    for idx, (m_id, frac, reason) in enumerate(MODALITY_SHARES):
        if idx == len(MODALITY_SHARES) - 1:
            tok = total_training_tokens - allocated_mod
        else:
            tok = int(round(total_training_tokens * frac))
            allocated_mod += tok
        modalities.append(ModalityAllocation(m_id, frac, tok, reason))

    code_pool = sum(
        m.token_budget
        for m in modalities
        if m.modality_id in ("source_code_curated", "repository_and_debugging_workflows")
    )
    lang_allocs: list[LanguageCodeAllocation] = []
    allocated_lang = 0
    for idx, (lang, tier, frac, reason) in enumerate(CODE_LANGUAGE_SHARES):
        if idx == len(CODE_LANGUAGE_SHARES) - 1:
            tok = code_pool - allocated_lang
        else:
            tok = int(round(code_pool * frac))
            allocated_lang += tok
        lang_allocs.append(LanguageCodeAllocation(lang, tier, frac, tok, reason))

    return {
        "total_training_tokens": total_training_tokens,
        "code_and_repo_pool_tokens": code_pool,
        "modalities": [m.to_dict() for m in modalities],
        "languages": [la.to_dict() for la in lang_allocs],
    }
