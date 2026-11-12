"""Automated Tests for Phase IV: Programming Language Capabilities (Steps 29–34)."""

from __future__ import annotations

from rootism.languages import (
    ALL_SUPPORTED_LANGUAGES,
    PRIMARY_LANGUAGES,
    SECONDARY_LANGUAGES,
    classify_language_scope,
    compute_training_token_allocation,
    detect_repository_frameworks,
    query_semantic_divergences,
    validate_code_syntax,
)


def test_deliberate_19_language_scope_and_out_of_scope_calibration() -> None:
    primary_slugs = {s.slug for s in PRIMARY_LANGUAGES}
    secondary_slugs = {s.slug for s in SECONDARY_LANGUAGES}

    assert primary_slugs == {
        "python", "javascript", "typescript", "html", "css",
        "sql", "c", "cpp", "rust", "go",
    }
    assert secondary_slugs == {
        "java", "csharp", "php", "bash", "powershell",
        "lua", "kotlin", "swift", "ruby",
    }
    assert len(ALL_SUPPORTED_LANGUAGES) == 19

    assert classify_language_scope("src/main.rs").tier == "primary"
    assert classify_language_scope("C++").language == "cpp"
    assert classify_language_scope("scripts/deploy.sh").tier == "secondary"

    unsupported = classify_language_scope("solver.hs")
    assert unsupported.tier == "unsupported"
    assert unsupported.language == "haskell"
    assert unsupported.scope_token == "<|lang_scope:unsupported|>"
    assert "outside Rootism's 19" in unsupported.calibration_policy


def test_real_compiler_and_ast_syntax_validators() -> None:
    # Valid snippets across Python, C, C++, JS, Bash, Rust
    assert validate_code_syntax("python", "def add(a: int, b: int) -> int:\n    return a + b\n").valid
    assert not validate_code_syntax("python", "def broken(a:\n    return a\n").valid

    assert validate_code_syntax("c", "int square(int x) { return x * x; }\n").valid
    assert not validate_code_syntax("c", "int broken(int x) { return x * }\n").valid

    assert validate_code_syntax("cpp", "#include <vector>\nint f() { std::vector<int> v{1, 2}; return v[0]; }\n").valid
    assert not validate_code_syntax("cpp", "int f() { int x = ; return x; }\n").valid

    assert validate_code_syntax("javascript", "export function inc(x) { return x + 1; }\n").valid
    assert not validate_code_syntax("javascript", "export function inc(x { return x + 1; }\n").valid

    assert validate_code_syntax("bash", "#!/usr/bin/env bash\nset -euo pipefail\necho 'ok'\n").valid
    assert not validate_code_syntax("bash", "if [ -f foo ]; then echo hi\n").valid

    assert validate_code_syntax("rust", "fn main() { let v = vec![1, 2, 3]; println!(\"{:?}\", v); }\n").valid
    assert not validate_code_syntax("rust", "fn main() { let v = vec![1, 2, 3); }\n").valid


def test_framework_detection_cross_language_semantics_and_650b_allocation() -> None:
    repo = {
        "pyproject.toml": '[project]\ndependencies = ["fastapi>=0.111", "torch>=2.4"]\n',
        "frontend/package.json": '{"dependencies": {"react": "^19.0.0", "next": "15.0.0"}}',
        "services/gateway/Cargo.toml": '[dependencies]\naxum = "0.7"\ntokio = "1"\n',
    }
    detected = {f.slug for f in detect_repository_frameworks(repo)}
    assert {"fastapi", "pytorch", "react", "nextjs", "axum"}.issubset(detected)

    divergences = query_semantic_divergences("python", "rust")
    assert len(divergences) >= 2
    assert {d.concept for d in divergences} >= {"integer_overflow", "string_indexing_and_slicing"}

    alloc = compute_training_token_allocation(650_000_000_000)
    assert sum(m["token_budget"] for m in alloc["modalities"]) == 650_000_000_000
    lang_budgets = {item["language"]: item["token_budget"] for item in alloc["languages"]}
    assert sum(lang_budgets.values()) == alloc["code_and_repo_pool_tokens"]
    # Verify deliberate non-equal weighting: primary Python > secondary Lua
    assert lang_budgets["python"] > lang_budgets["lua"] * 10
