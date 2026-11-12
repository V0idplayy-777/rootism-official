"""
Rootism Programming Language Registry, Competence Profiles & Scope Classifier (Steps 29, 30, 34).

Defines the exact 10 Primary and 9 Secondary supported programming languages,
provides real compiler/AST/structural syntax validation, and implements honest
epistemic calibration for unsupported out-of-scope languages.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

LanguageTier = Literal["primary", "secondary", "unsupported"]


@dataclass(frozen=True)
class LanguageSpec:
    """Comprehensive specification of syntax, semantics, stdlib, errors, and tooling."""

    slug: str
    display_name: str
    tier: Literal["primary", "secondary"]
    extensions: tuple[str, ...]
    shebang_tokens: tuple[str, ...]
    line_comment: str | None
    block_comment: tuple[str, str] | None
    typing_discipline: str
    memory_model: str
    concurrency_model: str
    standard_library_pillars: tuple[str, ...]
    common_error_patterns: tuple[str, ...]
    workflow_tools: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRIMARY_LANGUAGES: tuple[LanguageSpec, ...] = (
    LanguageSpec(
        slug="python",
        display_name="Python",
        tier="primary",
        extensions=(".py", ".pyi"),
        shebang_tokens=("python", "python3"),
        line_comment="#",
        block_comment=('"""', '"""'),
        typing_discipline="dynamic with gradual PEP 484 static type annotations",
        memory_model="reference counting + cyclic generational garbage collector",
        concurrency_model="asyncio event loop, multiprocessing, threading (GIL / free-threaded)",
        standard_library_pillars=("collections", "itertools", "pathlib", "asyncio", "typing", "dataclasses", "subprocess", "json"),
        common_error_patterns=("mutable default arguments", "late-binding closure loop variables", "blocking I/O inside async coroutines", "NoneType attribute errors"),
        workflow_tools=("pytest", "pip", "venv", "mypy", "ruff"),
    ),
    LanguageSpec(
        slug="javascript",
        display_name="JavaScript",
        tier="primary",
        extensions=(".js", ".mjs", ".cjs", ".jsx"),
        shebang_tokens=("node", "deno", "bun"),
        line_comment="//",
        block_comment=("/*", "*/"),
        typing_discipline="dynamic prototype-based",
        memory_model="generational mark-and-sweep garbage collection",
        concurrency_model="single-threaded event loop, microtask queue, Promises, Worker threads",
        standard_library_pillars=("Promise", "Map/Set", "Array", "URL", "fetch", "Intl", "crypto", "fs/promises"),
        common_error_patterns=("unhandled Promise rejections", "== loose coercion bugs", "`this` binding loss", "floating-point IEEE-754 precision"),
        workflow_tools=("npm", "node", "vitest", "jest", "eslint"),
    ),
    LanguageSpec(
        slug="typescript",
        display_name="TypeScript",
        tier="primary",
        extensions=(".ts", ".tsx", ".mts", ".cts"),
        shebang_tokens=("ts-node", "tsx"),
        line_comment="//",
        block_comment=("/*", "*/"),
        typing_discipline="structural static type system compiled to JavaScript",
        memory_model="generational mark-and-sweep garbage collection (JS runtime)",
        concurrency_model="event loop, typed Promises, async/await, Web Workers",
        standard_library_pillars=("Utility types (Partial, Pick, Record, Awaited)", "discriminated unions", "generics", "DOM/Node typings"),
        common_error_patterns=("unsound `any` / `as` assertions", "runtime vs compile-time type erasure assumptions", "narrowing across callbacks"),
        workflow_tools=("tsc", "npm", "vitest", "biome", "eslint"),
    ),
    LanguageSpec(
        slug="html",
        display_name="HTML",
        tier="primary",
        extensions=(".html", ".htm"),
        shebang_tokens=(),
        line_comment=None,
        block_comment=("<!--", "-->"),
        typing_discipline="declarative DOM markup specification (HTML Living Standard)",
        memory_model="browser DOM tree allocation",
        concurrency_model="declarative defer/async resource loading",
        standard_library_pillars=("semantic landmarks (main, nav, article)", "forms & validation", "ARIA accessibility attributes", "canvas/dialog/template"),
        common_error_patterns=("missing accessible labels", "invalid interactive nesting (button inside a)", "unescaped XSS sinks"),
        workflow_tools=("html-validate", "axe", "playwright"),
    ),
    LanguageSpec(
        slug="css",
        display_name="CSS",
        tier="primary",
        extensions=(".css",),
        shebang_tokens=(),
        line_comment=None,
        block_comment=("/*", "*/"),
        typing_discipline="declarative cascade, specificity & layout rule system",
        memory_model="browser style/layout render tree",
        concurrency_model="compositor-thread animations & transitions",
        standard_library_pillars=("Flexbox", "CSS Grid", "custom properties (--var)", "@media / @container queries", "clamp/min/max"),
        common_error_patterns=("specificity wars", "flex/grid min-width:auto overflow", "z-index stacking context confusion", "layout thrashing"),
        workflow_tools=("stylelint", "postcss", "lightningcss"),
    ),
    LanguageSpec(
        slug="sql",
        display_name="SQL",
        tier="primary",
        extensions=(".sql",),
        shebang_tokens=(),
        line_comment="--",
        block_comment=("/*", "*/"),
        typing_discipline="static relational schema types with three-valued NULL logic",
        memory_model="RDBMS buffer pool, work_mem sort/hash arenas, WAL",
        concurrency_model="ACID transactions, MVCC snapshots, row/table locking, isolation levels",
        standard_library_pillars=("JOINs", "CTEs (WITH)", "window functions (OVER/PARTITION BY)", "aggregations", "indexes & EXPLAIN"),
        common_error_patterns=("NOT IN with NULLs returning empty set", "unintended Cartesian join fan-out", "N+1 query patterns", "SQL string-concatenation injection"),
        workflow_tools=("psql", "sqlite3", "sqlfluff", "alembic"),
    ),
    LanguageSpec(
        slug="c",
        display_name="C",
        tier="primary",
        extensions=(".c", ".h"),
        shebang_tokens=(),
        line_comment="//",
        block_comment=("/*", "*/"),
        typing_discipline="static nominal weak/low-level systems typing (C11/C17/C23)",
        memory_model="manual stack & heap management (malloc/calloc/realloc/free)",
        concurrency_model="POSIX threads (pthreads), C11 stdatomic, mutexes/condvars",
        standard_library_pillars=("stdlib.h", "stdint.h", "string.h", "stdio.h", "errno.h", "stdatomic.h"),
        common_error_patterns=("buffer overflows & missing NUL terminators", "use-after-free / double-free", "signed integer overflow UB", "uninitialized stack reads"),
        workflow_tools=("gcc", "clang", "make", "cmake", "valgrind", "asan/ubsan"),
    ),
    LanguageSpec(
        slug="cpp",
        display_name="C++",
        tier="primary",
        extensions=(".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"),
        shebang_tokens=(),
        line_comment="//",
        block_comment=("/*", "*/"),
        typing_discipline="static strong multiparadigm with templates & concepts (C++17/20/23)",
        memory_model="deterministic RAII destructors, move semantics, smart pointers (unique_ptr/shared_ptr)",
        concurrency_model="std::thread, std::jthread, std::atomic, futures, C++20 coroutines",
        standard_library_pillars=("STL containers (vector, unordered_map)", "algorithms & ranges", "std::optional/variant/span/string_view", "memory"),
        common_error_patterns=("dangling string_view/span/iterator invalidation", "object slicing", "data races on shared state", "exception-unsafe raw new"),
        workflow_tools=("g++", "clang++", "cmake", "ninja", "ctest", "clang-tidy"),
    ),
    LanguageSpec(
        slug="rust",
        display_name="Rust",
        tier="primary",
        extensions=(".rs",),
        shebang_tokens=(),
        line_comment="//",
        block_comment=("/*", "*/"),
        typing_discipline="static affine ownership, lifetimes & trait-based algebraic type system",
        memory_model="compile-time ownership & borrow checker (zero-GC deterministic drop)",
        concurrency_model="fearless concurrency via Send/Sync marker traits, channels, async/await futures",
        standard_library_pillars=("Option & Result", "Iterator", "Vec/HashMap/BTreeMap", "Arc/Mutex/RwLock", "std::fs/io/sync"),
        common_error_patterns=("holding std::sync::MutexGuard across .await points", "unnecessary .clone() churn", "panic via .unwrap() on untrusted input", "unsafe aliasing violations"),
        workflow_tools=("cargo", "rustc", "clippy", "rustfmt", "miri"),
    ),
    LanguageSpec(
        slug="go",
        display_name="Go",
        tier="primary",
        extensions=(".go",),
        shebang_tokens=(),
        line_comment="//",
        block_comment=("/*", "*/"),
        typing_discipline="static structural interfaces with generics (Go 1.18+)",
        memory_model="concurrent tri-color low-latency garbage collector + escape analysis",
        concurrency_model="M:N scheduled goroutines, typed channels, select, context.Context cancellation",
        standard_library_pillars=("context", "net/http", "io", "sync/errgroup", "encoding/json", "errors", "slices/maps"),
        common_error_patterns=("goroutine leaks on unbuffered channels without context cancellation", "nil interface vs interface holding nil pointer", "concurrent map read/write panic"),
        workflow_tools=("go test", "go vet", "gofmt", "golangci-lint"),
    ),
)


SECONDARY_LANGUAGES: tuple[LanguageSpec, ...] = (
    LanguageSpec(
        slug="java",
        display_name="Java",
        tier="secondary",
        extensions=(".java",),
        shebang_tokens=(),
        line_comment="//",
        block_comment=("/*", "*/"),
        typing_discipline="static nominal OOP + records/sealed classes (Java 17/21)",
        memory_model="JVM G1/ZGC garbage collector",
        concurrency_model="OS & Project Loom virtual threads, CompletableFuture, java.util.concurrent",
        standard_library_pillars=("java.util.stream", "Collections", "java.nio.file", "java.time", "Optional"),
        common_error_patterns=("NullPointerException", "equals/hashCode contract violations", "resource leaks without try-with-resources"),
        workflow_tools=("javac", "maven", "gradle", "junit"),
    ),
    LanguageSpec(
        slug="csharp",
        display_name="C#",
        tier="secondary",
        extensions=(".cs",),
        shebang_tokens=(),
        line_comment="//",
        block_comment=("/*", "*/"),
        typing_discipline="static nominal/generics with nullable reference types (.NET 8+)",
        memory_model=".NET generational GC + stack-allocated Span<T>/struct value types",
        concurrency_model="Task-based async/await, CancellationToken, Channels",
        standard_library_pillars=("LINQ", "System.Collections.Generic", "Span<T>/Memory<T>", "System.Text.Json", "HttpClient"),
        common_error_patterns=("async void exceptions", "sync-over-async .Result deadlocks", "HttpClient socket exhaustion"),
        workflow_tools=("dotnet", "xunit", "nuget"),
    ),
    LanguageSpec(
        slug="php",
        display_name="PHP",
        tier="secondary",
        extensions=(".php",),
        shebang_tokens=("php",),
        line_comment="//",
        block_comment=("/*", "*/"),
        typing_discipline="gradually typed dynamic/strict_types=1 (PHP 8.2+)",
        memory_model="per-request reference counting + cycle collector",
        concurrency_model="shared-nothing FPM workers, Fibers",
        standard_library_pillars=("PDO", "SPL", "DateTimeImmutable", "json_encode/decode", "mbstring"),
        common_error_patterns=("missing declare(strict_types=1)", "SQL injection without PDO bound params", "loose in_array without strict=true"),
        workflow_tools=("php -l", "composer", "phpunit", "phpstan"),
    ),
    LanguageSpec(
        slug="bash",
        display_name="Bash",
        tier="secondary",
        extensions=(".sh", ".bash"),
        shebang_tokens=("bash", "sh"),
        line_comment="#",
        block_comment=None,
        typing_discipline="string/word-expansion command shell",
        memory_model="process environment & subshell fork model",
        concurrency_model="POSIX pipelines, background jobs (&), wait, xargs -P",
        standard_library_pillars=("parameter expansion", "set -euo pipefail", "trap", "getopts", "arrays"),
        common_error_patterns=("unquoted variable word-splitting/globbing", "masked pipeline exit codes without pipefail", "command injection via eval"),
        workflow_tools=("bash -n", "shellcheck", "bats"),
    ),
    LanguageSpec(
        slug="powershell",
        display_name="PowerShell",
        tier="secondary",
        extensions=(".ps1", ".psm1"),
        shebang_tokens=("pwsh",),
        line_comment="#",
        block_comment=("<#", "#>"),
        typing_discipline="object-pipeline dynamic/.NET typed shell",
        memory_model=".NET CLR garbage collection",
        concurrency_model="ForEach-Object -Parallel, Start-Job, ThreadJob",
        standard_library_pillars=("CmdletBinding", "Where-Object/Select-Object", "Invoke-RestMethod", "ErrorActionPreference"),
        common_error_patterns=("non-terminating errors ignored by try/catch without -ErrorAction Stop", "single-item pipeline array unrolling"),
        workflow_tools=("pwsh", "PSScriptAnalyzer", "Pester"),
    ),
    LanguageSpec(
        slug="lua",
        display_name="Lua",
        tier="secondary",
        extensions=(".lua",),
        shebang_tokens=("lua", "luajit"),
        line_comment="--",
        block_comment=("--[[", "]]"),
        typing_discipline="dynamic table/metatable-centric scripting",
        memory_model="incremental/generational mark-and-sweep GC",
        concurrency_model="stackful asymmetric coroutines",
        standard_library_pillars=("table", "string", "coroutine", "math", "io", "pcall/xpcall"),
        common_error_patterns=("1-based indexing off-by-one", "accidental globals without `local`", "`#t` length undefined on sparse tables with nil holes"),
        workflow_tools=("luac -p", "luacheck", "busted"),
    ),
    LanguageSpec(
        slug="kotlin",
        display_name="Kotlin",
        tier="secondary",
        extensions=(".kt", ".kts"),
        shebang_tokens=("kotlin",),
        line_comment="//",
        block_comment=("/*", "*/"),
        typing_discipline="static null-safe JVM/Multiplatform type system",
        memory_model="JVM / Kotlin Native GC",
        concurrency_model="structured concurrency coroutines (CoroutineScope, Flow, Dispatchers)",
        standard_library_pillars=("collections & sequences", "kotlinx.coroutines", "Result", "data/sealed classes"),
        common_error_patterns=("unsafe `!!` non-null assertions", "GlobalScope coroutine leaks", "CancellationException swallowed in catch(Exception)"),
        workflow_tools=("kotlinc", "gradle", "detekt"),
    ),
    LanguageSpec(
        slug="swift",
        display_name="Swift",
        tier="secondary",
        extensions=(".swift",),
        shebang_tokens=("swift",),
        line_comment="//",
        block_comment=("/*", "*/"),
        typing_discipline="static protocol-oriented value-semantic type system (Swift 5.10/6)",
        memory_model="Automatic Reference Counting (ARC) + copy-on-write value types",
        concurrency_model="async/await, Actors, Sendable data-race safety, TaskGroup",
        standard_library_pillars=("Foundation", "Array/Dictionary/Set", "Result", "Codable", "AsyncSequence"),
        common_error_patterns=("strong reference retain cycles in closures without [weak self]", "force-unwrapping `!` crashes", "actor reentrancy assumptions"),
        workflow_tools=("swift build", "swift test", "swiftlint"),
    ),
    LanguageSpec(
        slug="ruby",
        display_name="Ruby",
        tier="secondary",
        extensions=(".rb",),
        shebang_tokens=("ruby",),
        line_comment="#",
        block_comment=("=begin", "=end"),
        typing_discipline="dynamic pure object-oriented duck typing (+ Sorbet/RBS optional)",
        memory_model="generational incremental mark-and-sweep/compacting GC",
        concurrency_model="Threads, Fibers, Ractors",
        standard_library_pillars=("Enumerable", "Hash/Array", "Struct/Data", "File/Pathname", "Net::HTTP", "JSON"),
        common_error_patterns=("NoMethodError on nil", "symbol/hash key mismatch (string vs symbol)", "monkey-patching global namespace collisions"),
        workflow_tools=("ruby -c", "bundle", "rspec", "rubocop"),
    ),
)


ALL_SUPPORTED_LANGUAGES: dict[str, LanguageSpec] = {
    spec.slug: spec for spec in (*PRIMARY_LANGUAGES, *SECONDARY_LANGUAGES)
}

EXTENSION_TO_LANGUAGE: dict[str, LanguageSpec] = {
    ext: spec
    for spec in (*PRIMARY_LANGUAGES, *SECONDARY_LANGUAGES)
    for ext in spec.extensions
}

KNOWN_UNSUPPORTED_EXTENSIONS: dict[str, str] = {
    ".hs": "haskell",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".zig": "zig",
    ".jl": "julia",
    ".scala": "scala",
    ".clj": "clojure",
    ".r": "r",
    ".R": "r",
    ".dart": "dart",
    ".cob": "cobol",
    ".v": "verilog",
    ".sv": "systemverilog",
    ".sol": "solidity",
    ".ml": "ocaml",
    ".nim": "nim",
}


@dataclass(frozen=True)
class LanguageScopeAssessment:
    """Epistemic scope classification for language routing and honest user communication."""

    language: str
    tier: LanguageTier
    scope_token: str
    in_primary_or_secondary_scope: bool
    calibration_policy: str


def classify_language_scope(language_or_path: str) -> LanguageScopeAssessment:
    """
    Classify a language name or file path into `primary`, `secondary`, or `unsupported` (Step 34).
    """
    raw = language_or_path.strip()
    lower = raw.lower()

    aliases = {
        "c++": "cpp",
        "cxx": "cpp",
        "c#": "csharp",
        "cs": "csharp",
        "js": "javascript",
        "ts": "typescript",
        "py": "python",
        "sh": "bash",
        "shell": "bash",
        "golang": "go",
        "ps1": "powershell",
        "rb": "ruby",
        "rs": "rust",
        "kt": "kotlin",
    }
    normalized = aliases.get(lower, lower)
    if normalized in ALL_SUPPORTED_LANGUAGES:
        spec = ALL_SUPPORTED_LANGUAGES[normalized]
        tok = "<|lang_scope:primary|>" if spec.tier == "primary" else "<|lang_scope:secondary|>"
        return LanguageScopeAssessment(
            language=spec.slug,
            tier=spec.tier,
            scope_token=tok,
            in_primary_or_secondary_scope=True,
            calibration_policy=(
                f"{spec.display_name} is a supported {spec.tier} Rootism language; "
                "apply full idiomatic, stdlib, framework, and testing competence."
            ),
        )

    suffix = Path(raw).suffix
    if suffix.lower() in EXTENSION_TO_LANGUAGE:
        spec = EXTENSION_TO_LANGUAGE[suffix.lower()]
        tok = "<|lang_scope:primary|>" if spec.tier == "primary" else "<|lang_scope:secondary|>"
        return LanguageScopeAssessment(
            language=spec.slug,
            tier=spec.tier,
            scope_token=tok,
            in_primary_or_secondary_scope=True,
            calibration_policy=(
                f"{spec.display_name} is a supported {spec.tier} Rootism language."
            ),
        )

    detected_unsupported = KNOWN_UNSUPPORTED_EXTENSIONS.get(suffix, lower or "unknown")
    return LanguageScopeAssessment(
        language=detected_unsupported,
        tier="unsupported",
        scope_token="<|lang_scope:unsupported|>",
        in_primary_or_secondary_scope=False,
        calibration_policy=(
            f"'{detected_unsupported}' is outside Rootism's 19 primary/secondary languages. "
            "Provide best-effort assistance grounded in general computer science principles, "
            "explicitly note reduced ecosystem familiarity, and verify syntax conservatively "
            "rather than fabricating unverified library APIs."
        ),
    )


@dataclass(frozen=True)
class SyntaxCheckResult:
    """Outcome of automated source-code syntax verification."""

    language: str
    valid: bool
    validator_used: str
    error_message: str | None = None


def check_balanced_delimiters(source: str, line_comment: str | None = "//") -> tuple[bool, str | None]:
    """Comment- and string-literal-aware balanced delimiter & unclosed-string checker."""
    stack: list[tuple[str, int]] = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    closers = {v: k for k, v in pairs.items()}
    i = 0
    n = len(source)
    line = 1

    while i < n:
        ch = source[i]
        if ch == "\n":
            line += 1
            i += 1
            continue
        if line_comment and source.startswith(line_comment, i):
            nl = source.find("\n", i)
            if nl == -1:
                break
            i = nl
            continue
        if source.startswith("/*", i):
            end = source.find("*/", i + 2)
            if end == -1:
                return False, f"Unclosed block comment starting at line {line}"
            line += source[i : end + 2].count("\n")
            i = end + 2
            continue
        if ch in ("'", '"', "`"):
            quote = ch
            start_line = line
            i += 1
            closed = False
            while i < n:
                c2 = source[i]
                if c2 == "\\":
                    i += 2
                    continue
                if c2 == "\n":
                    if quote != "`":
                        break
                    line += 1
                if c2 == quote:
                    closed = True
                    i += 1
                    break
                i += 1
            if not closed:
                return False, f"Unclosed string literal ({quote}) starting at line {start_line}"
            continue
        if ch in pairs:
            stack.append((ch, line))
        elif ch in closers:
            if not stack or stack[-1][0] != closers[ch]:
                return False, f"Mismatched closing delimiter '{ch}' at line {line}"
            stack.pop()
        i += 1

    if stack:
        open_ch, open_line = stack[-1]
        return False, f"Unclosed delimiter '{open_ch}' opened at line {open_line}"
    return True, None


def validate_code_syntax(language: str, source: str) -> SyntaxCheckResult:
    """
    Validate code syntax using real language parsers/compilers when installed in the runtime
    (`ast.parse` for Python, `node --check` for JS, `gcc`/`g++ -fsyntax-only` for C/C++,
    `bash -n` for Bash) and deterministic lexical/structural validators for remaining languages.
    """
    scope = classify_language_scope(language)
    lang = scope.language

    if not source.strip():
        return SyntaxCheckResult(lang, False, "non_empty_check", "Empty source snippet")

    if lang == "python":
        try:
            ast.parse(source)
            return SyntaxCheckResult("python", True, "python_ast")
        except SyntaxError as exc:
            return SyntaxCheckResult("python", False, "python_ast", f"{exc.msg} (line {exc.lineno})")

    if lang == "bash" and shutil.which("bash"):
        proc = subprocess.run(
            ["bash", "-n"],
            input=source,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0:
            return SyntaxCheckResult("bash", True, "bash_-n")
        return SyntaxCheckResult("bash", False, "bash_-n", proc.stderr.strip())

    if lang == "javascript" and shutil.which("node"):
        with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False, encoding="utf-8") as tmp:
            tmp.write(source)
            tmp_path = Path(tmp.name)
        try:
            proc = subprocess.run(
                ["node", "--check", str(tmp_path)],
                text=True,
                capture_output=True,
                timeout=5,
                check=False,
            )
            if proc.returncode == 0:
                return SyntaxCheckResult("javascript", True, "node_--check")
            return SyntaxCheckResult("javascript", False, "node_--check", proc.stderr.strip()[:240])
        finally:
            tmp_path.unlink(missing_ok=True)

    if lang in ("c", "cpp"):
        compiler = "gcc" if lang == "c" else "g++"
        ext = ".c" if lang == "c" else ".cpp"
        if shutil.which(compiler):
            with tempfile.NamedTemporaryFile("w", suffix=ext, delete=False, encoding="utf-8") as tmp:
                tmp.write(source)
                tmp_path = Path(tmp.name)
            try:
                cmd = [compiler, "-fsyntax-only", "-Wall"]
                if lang == "cpp":
                    cmd.append("-std=c++20")
                cmd.append(str(tmp_path))
                proc = subprocess.run(
                    cmd,
                    text=True,
                    capture_output=True,
                    timeout=8,
                    check=False,
                )
                if proc.returncode == 0:
                    return SyntaxCheckResult(lang, True, f"{compiler}_-fsyntax-only")
                return SyntaxCheckResult(lang, False, f"{compiler}_-fsyntax-only", proc.stderr.strip()[:240])
            finally:
                tmp_path.unlink(missing_ok=True)

    spec = ALL_SUPPORTED_LANGUAGES.get(lang)
    line_comment = spec.line_comment if spec else "//"
    if lang == "html":
        has_angle = "<" in source and ">" in source
        if not has_angle:
            return SyntaxCheckResult("html", False, "html_structural", "No HTML element tags found")
        if source.count("<") < 1 or ("<!--" in source and "-->" not in source):
            return SyntaxCheckResult("html", False, "html_structural", "Malformed HTML comment/tag structure")
        return SyntaxCheckResult("html", True, "html_structural")

    ok, err = check_balanced_delimiters(source, line_comment=line_comment)
    return SyntaxCheckResult(lang, ok, "structural_delimiter_lexer", err)
