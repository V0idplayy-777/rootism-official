"""
Automated Test Execution, Test Generation & Structured Failure Diagnosis (Steps 73 & 74).

Diagnoses failures across:
1. `SYNTAX_OR_COMPILER_ERROR`: SyntaxError, IndentationError, gcc/g++/rustc/tsc compile errors
2. `MISSING_DEPENDENCY_OR_IMPORT`: ModuleNotFoundError, ImportError, Cannot find module
3. `TYPE_MISMATCH`: TypeError, mypy/tsc type errors
4. `FAILING_ASSERTION_OR_LOGIC_BUG`: AssertionError, pytest/jest assertion mismatch
5. `RUNTIME_EXCEPTION`: ZeroDivisionError, IndexError, KeyError, ValueError, segfault
6. `TIMEOUT_OR_RESOURCE_LIMIT`: Infinite loop / sandbox timeout
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rootism.agent.sandbox import SandboxedCommandRunner, SandboxExecutionResult


@dataclass(frozen=True)
class FailureDiagnosis:
    passed: bool
    category: str
    error_type: str
    file_location: str | None
    line_number: int | None
    summary: str
    suggested_repair_focus: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PY_TRACEBACK_LOC_RE = re.compile(r'File "([^"]+)", line (\d+)')
_GCC_TSC_LOC_RE = re.compile(r"([A-Za-z0-9_./-]+\.[A-Za-z0-9]+):(\d+)(?::\d+)?:?\s*(?:error|Error)")


class FailureDiagnosticsEngine:
    """
    Classifies compiler, test-runner, and runtime failure logs into actionable engineering
    root-cause categories (Step 74).
    """

    @staticmethod
    def diagnose(exec_res: SandboxExecutionResult) -> FailureDiagnosis:
        if not exec_res.executed:
            return FailureDiagnosis(
                passed=False,
                category="BLOCKED_OR_AWAITING_CONFIRMATION",
                error_type="SafetyPolicyGate",
                file_location=None,
                line_number=None,
                summary=exec_res.safety_decision.reason,
                suggested_repair_focus="Request explicit user approval or choose a non-destructive operation.",
            )

        if exec_res.timed_out:
            return FailureDiagnosis(
                passed=False,
                category="TIMEOUT_OR_RESOURCE_LIMIT",
                error_type="TimeoutExpired",
                file_location=None,
                line_number=None,
                summary=f"Execution timed out ({exec_res.duration_ms:.0f} ms). Likely non-terminating loop or blocking I/O.",
                suggested_repair_focus="Check loop termination invariants, base cases, and algorithmic complexity.",
            )

        if exec_res.exit_code == 0:
            return FailureDiagnosis(
                passed=True,
                category="PASSED",
                error_type="None",
                file_location=None,
                line_number=None,
                summary="All executed checks/tests exited with code 0.",
                suggested_repair_focus="None required.",
            )

        combined = f"{exec_res.stdout}\n{exec_res.stderr}".strip()
        file_loc: str | None = None
        line_no: int | None = None

        py_matches = _PY_TRACEBACK_LOC_RE.findall(combined)
        if py_matches:
            file_loc, line_s = py_matches[-1]
            line_no = int(line_s)
        else:
            c_match = _GCC_TSC_LOC_RE.search(combined)
            if c_match:
                file_loc, line_no = c_match.group(1), int(c_match.group(2))

        if any(tok in combined for tok in ("SyntaxError", "IndentationError", "error: expected", "TS1005")):
            return FailureDiagnosis(
                passed=False,
                category="SYNTAX_OR_COMPILER_ERROR",
                error_type="SyntaxOrCompilerError",
                file_location=file_loc,
                line_number=line_no,
                summary="Source failed lexical/syntactic or compiler parsing.",
                suggested_repair_focus="Inspect brackets, indentation, semicolons, and type declarations near reported line.",
            )

        if any(tok in combined for tok in ("ModuleNotFoundError", "ImportError", "Cannot find module", "cannot find package")):
            return FailureDiagnosis(
                passed=False,
                category="MISSING_DEPENDENCY_OR_IMPORT",
                error_type="ImportOrDependencyError",
                file_location=file_loc,
                line_number=line_no,
                summary="Unresolved module import or missing package dependency.",
                suggested_repair_focus="Verify module path spelling, relative imports, and workspace manifests.",
            )

        if any(tok in combined for tok in ("TypeError", "TS2322", "TS2345", "mismatched types")):
            return FailureDiagnosis(
                passed=False,
                category="TYPE_MISMATCH",
                error_type="TypeError",
                file_location=file_loc,
                line_number=line_no,
                summary="Argument, return value, or operator type mismatch.",
                suggested_repair_focus="Check function signatures, None/null handling, and container element types.",
            )

        if any(tok in combined for tok in ("AssertionError", "assert ", "FAILED ", "Expected:", "Assertion `")):
            return FailureDiagnosis(
                passed=False,
                category="FAILING_ASSERTION_OR_LOGIC_BUG",
                error_type="AssertionError",
                file_location=file_loc,
                line_number=line_no,
                summary="Program ran without crashing, but violated an expected output or invariant assertion.",
                suggested_repair_focus="Trace boundary conditions, off-by-one indexing, and formula logic against test expectations.",
            )

        # Extract concrete exception class if present
        exc_match = re.search(r"\b([A-Za-z]+(?:Error|Exception|Panic))\b", combined)
        err_name = exc_match.group(1) if exc_match else f"ExitCode({exec_res.exit_code})"
        return FailureDiagnosis(
            passed=False,
            category="RUNTIME_EXCEPTION",
            error_type=err_name,
            file_location=file_loc,
            line_number=line_no,
            summary=f"Unhandled runtime failure ({err_name}) during execution.",
            suggested_repair_focus="Guard edge cases (empty inputs, zero divisors, missing keys, bounds).",
        )


class AgentTestVerifier:
    """
    Detects project test suites, executes them inside `SandboxedCommandRunner`,
    diagnoses failures, and synthesizes baseline smoke/unit tests when none exist (Step 73).
    """

    def __init__(self, sandbox: SandboxedCommandRunner) -> None:
        self.sandbox = sandbox

    def detect_default_test_command(self) -> str:
        root = self.sandbox.safety_guard.workspace_root
        if (root / "pyproject.toml").exists() or list(root.glob("**/test_*.py")) or list(root.glob("**/*_test.py")):
            py_bin = sys.executable
            return f'"{py_bin}" -m pytest -q'
        if (root / "package.json").exists():
            return "npm test"
        if (root / "Cargo.toml").exists():
            return "cargo test"
        if (root / "go.mod").exists():
            return "go test ./..."
        return f'"{sys.executable}" -m unittest discover'

    def run_verification(
        self,
        test_command: str | None = None,
        *,
        timeout_sec: float = 25.0,
    ) -> tuple[SandboxExecutionResult, FailureDiagnosis]:
        cmd = test_command or self.detect_default_test_command()
        res = self.sandbox.run(cmd, timeout_sec=timeout_sec)
        diag = FailureDiagnosticsEngine.diagnose(res)
        return res, diag

    def synthesize_python_smoke_test(self, target_rel_path: str) -> tuple[str, str]:
        """
        Inspect a Python module via `ast` and generate a real importable unit test file
        verifying module importability and discovered pure callables/classes.
        Returns `(suggested_test_rel_path, test_source_code)`.
        """
        src_path: Path = self.sandbox.safety_guard.resolve_confined_path(target_rel_path)
        source = src_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        rel = src_path.relative_to(self.sandbox.safety_guard.workspace_root)
        mod_dotted = ".".join(rel.with_suffix("").parts)
        func_names = [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
        ]
        class_names = [
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
        ]

        lines = [
            f'"""Auto-generated verification test for `{mod_dotted}`."""',
            "",
            f"import {mod_dotted} as target_mod",
            "",
            "",
            "def test_module_exports_exist() -> None:",
        ]
        if not func_names and not class_names:
            lines.append("    assert target_mod is not None")
        else:
            for fn in func_names:
                lines.append(f"    assert callable(getattr(target_mod, {fn!r}, None))")
            for cls in class_names:
                lines.append(f"    assert isinstance(getattr(target_mod, {cls!r}, None), type)")
        lines.append("")

        test_rel = f"tests/test_auto_{src_path.stem}.py"
        return test_rel, "\n".join(lines)
