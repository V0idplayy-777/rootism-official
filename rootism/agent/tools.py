"""
Structured Coding Agent Tool Suite with Unified Diffs & Reversible Rollback (Step 69).

Implements structured, schema-validated tools:
- `inspect_project`: Repository overview, manifests, symbol & import map
- `list_files`: List workspace files
- `read_file`: Read file contents with line numbering and slice support
- `search_project`: Regex/literal code search across workspace files
- `create_file`: Create or overwrite file with automatic rollback snapshot + unified diff
- `edit_file`: Exact substring block replacement with unified diff + rollback snapshot
- `delete_file`: Gated by `SafetyPolicyGuard` (requires user confirmation token!)
- `revert_last_edit`: Atomically roll back the latest file mutation
- `run_command`: Execute shell command inside `SandboxedCommandRunner`
- `run_tests`: Execute test suite and return structured `FailureDiagnosis`
"""

from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rootism.agent.safety import SafetyPolicyGuard
from rootism.agent.sandbox import SandboxedCommandRunner
from rootism.agent.testing import AgentTestVerifier
from rootism.agent.workspace import WorkspaceAnalyzer


@dataclass(frozen=True)
class FileMutationRollbackEntry:
    relative_path: str
    existed_before: bool
    previous_content: str | None
    new_content: str | None
    unified_diff: str


@dataclass(frozen=True)
class AgentToolResult:
    tool_name: str
    ok: bool
    requires_confirmation: bool
    confirmation_token: str | None
    summary: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentToolRegistry:
    """
    Executes structured coding-agent tool calls against a confined workspace,
    recording unified diffs and reversible rollback history for every file mutation.
    """

    def __init__(self, workspace_root: str | Path, *, default_timeout_sec: float = 15.0) -> None:
        self.safety = SafetyPolicyGuard(workspace_root)
        self.sandbox = SandboxedCommandRunner(self.safety, default_timeout_sec=default_timeout_sec)
        self.analyzer = WorkspaceAnalyzer(self.safety)
        self.verifier = AgentTestVerifier(self.sandbox)
        self.rollback_stack: list[FileMutationRollbackEntry] = []

    @staticmethod
    def tool_schemas() -> list[dict[str, Any]]:
        return [
            {"name": "inspect_project", "args": {}},
            {"name": "list_files", "args": {"glob_pattern": "optional str"}},
            {"name": "read_file", "args": {"path": "str", "start_line": "int=1", "end_line": "int=250"}},
            {"name": "search_project", "args": {"pattern": "str", "is_regex": "bool=False"}},
            {"name": "create_file", "args": {"path": "str", "content": "str"}},
            {"name": "edit_file", "args": {"path": "str", "old_text": "str", "new_text": "str"}},
            {"name": "delete_file", "args": {"path": "str", "confirmation_token": "optional str"}},
            {"name": "revert_last_edit", "args": {}},
            {"name": "run_command", "args": {"command": "str", "confirmation_token": "optional str"}},
            {"name": "run_tests", "args": {"test_command": "optional str"}},
        ]

    def invoke(self, tool_name: str, args: dict[str, Any] | None = None) -> AgentToolResult:
        params = args or {}
        dispatch = {
            "inspect_project": self._tool_inspect_project,
            "list_files": self._tool_list_files,
            "read_file": self._tool_read_file,
            "search_project": self._tool_search_project,
            "create_file": self._tool_create_file,
            "edit_file": self._tool_edit_file,
            "delete_file": self._tool_delete_file,
            "revert_last_edit": self._tool_revert_last_edit,
            "run_command": self._tool_run_command,
            "run_tests": self._tool_run_tests,
        }
        handler = dispatch.get(tool_name)
        if handler is None:
            return AgentToolResult(
                tool_name=tool_name,
                ok=False,
                requires_confirmation=False,
                confirmation_token=None,
                summary=f"Unknown tool '{tool_name}'.",
                payload={"available_tools": sorted(dispatch.keys())},
            )
        try:
            return handler(params)
        except Exception as exc:
            return AgentToolResult(
                tool_name=tool_name,
                ok=False,
                requires_confirmation=False,
                confirmation_token=None,
                summary=f"{type(exc).__name__}: {exc}",
                payload={"error_type": type(exc).__name__, "message": str(exc)},
            )

    def _tool_inspect_project(self, _params: dict[str, Any]) -> AgentToolResult:
        overview = self.analyzer.inspect_repository()
        return AgentToolResult(
            tool_name="inspect_project",
            ok=True,
            requires_confirmation=False,
            confirmation_token=None,
            summary=f"Indexed {overview.total_files} files ({len(overview.symbol_to_files)} symbols).",
            payload=overview.to_dict(),
        )

    def _tool_list_files(self, params: dict[str, Any]) -> AgentToolResult:
        files = self.analyzer.iter_workspace_files()
        root = self.safety.workspace_root
        rels = [str(f.relative_to(root)) for f in files]
        glob_pat = params.get("glob_pattern")
        if glob_pat:
            rels = [r for r in rels if Path(r).match(str(glob_pat))]
        return AgentToolResult(
            tool_name="list_files",
            ok=True,
            requires_confirmation=False,
            confirmation_token=None,
            summary=f"Listed {len(rels)} workspace file(s).",
            payload={"files": rels},
        )

    def _tool_read_file(self, params: dict[str, Any]) -> AgentToolResult:
        target = self.safety.resolve_confined_path(str(params["path"]))
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {params['path']}")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, int(params.get("start_line", 1)))
        end = min(len(lines), int(params.get("end_line", max(250, start + 249))))
        sliced = lines[start - 1 : end]
        numbered = "\n".join(f"{i:4d} | {line}" for i, line in enumerate(sliced, start=start))
        rel = str(target.relative_to(self.safety.workspace_root))
        return AgentToolResult(
            tool_name="read_file",
            ok=True,
            requires_confirmation=False,
            confirmation_token=None,
            summary=f"Read {rel} lines {start}..{end} (of {len(lines)}).",
            payload={
                "path": rel,
                "total_lines": len(lines),
                "start_line": start,
                "end_line": end,
                "content": "\n".join(sliced),
                "numbered_content": numbered,
            },
        )

    def _tool_search_project(self, params: dict[str, Any]) -> AgentToolResult:
        pattern = str(params["pattern"])
        is_regex = bool(params.get("is_regex", False))
        rx = re.compile(pattern) if is_regex else re.compile(re.escape(pattern))
        matches: list[dict[str, Any]] = []
        root = self.safety.workspace_root
        for f in self.analyzer.iter_workspace_files():
            if f.stat().st_size > 512_000:
                continue
            try:
                txt = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(f.relative_to(root))
            for idx, line in enumerate(txt.splitlines(), start=1):
                if rx.search(line):
                    matches.append({"path": rel, "line": idx, "snippet": line.strip()[:200]})
                    if len(matches) >= 100:
                        break
            if len(matches) >= 100:
                break
        return AgentToolResult(
            tool_name="search_project",
            ok=True,
            requires_confirmation=False,
            confirmation_token=None,
            summary=f"Found {len(matches)} match(es) for {pattern!r}.",
            payload={"matches": matches},
        )

    def _tool_create_file(self, params: dict[str, Any]) -> AgentToolResult:
        rel_arg = str(params["path"])
        new_content = str(params["content"])
        target = self.safety.resolve_confined_path(rel_arg)
        rel = str(target.relative_to(self.safety.workspace_root))

        existed = target.is_file()
        old_content = target.read_text(encoding="utf-8", errors="replace") if existed else None
        diff = self._compute_unified_diff(rel, old_content or "", new_content)

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_content, encoding="utf-8")
        self.rollback_stack.append(
            FileMutationRollbackEntry(rel, existed, old_content, new_content, diff)
        )
        return AgentToolResult(
            tool_name="create_file",
            ok=True,
            requires_confirmation=False,
            confirmation_token=None,
            summary=f"{'Updated' if existed else 'Created'} {rel} ({len(new_content)} chars).",
            payload={"path": rel, "unified_diff": diff},
        )

    def _tool_edit_file(self, params: dict[str, Any]) -> AgentToolResult:
        rel_arg = str(params["path"])
        old_text = str(params["old_text"])
        new_text = str(params["new_text"])
        target = self.safety.resolve_confined_path(rel_arg)
        if not target.is_file():
            raise FileNotFoundError(f"Cannot edit missing file: {rel_arg}")
        rel = str(target.relative_to(self.safety.workspace_root))
        current = target.read_text(encoding="utf-8")

        occurrences = current.count(old_text)
        if occurrences == 0:
            raise ValueError(f"`old_text` not found in {rel}.")
        if occurrences > 1:
            raise ValueError(
                f"`old_text` matched {occurrences} locations in {rel}; include more surrounding context."
            )

        updated = current.replace(old_text, new_text, 1)
        diff = self._compute_unified_diff(rel, current, updated)
        target.write_text(updated, encoding="utf-8")
        self.rollback_stack.append(
            FileMutationRollbackEntry(rel, True, current, updated, diff)
        )
        return AgentToolResult(
            tool_name="edit_file",
            ok=True,
            requires_confirmation=False,
            confirmation_token=None,
            summary=f"Applied targeted patch to {rel}.",
            payload={"path": rel, "unified_diff": diff},
        )

    def _tool_delete_file(self, params: dict[str, Any]) -> AgentToolResult:
        rel_arg = str(params["path"])
        token = params.get("confirmation_token")
        decision = self.safety.evaluate_file_mutation(
            operation="delete",
            paths=[rel_arg],
            provided_confirmation_token=token,
        )
        if not decision.allowed_immediately:
            return AgentToolResult(
                tool_name="delete_file",
                ok=False,
                requires_confirmation=decision.requires_user_confirmation,
                confirmation_token=decision.confirmation_token,
                summary=decision.reason,
                payload={"safety_decision": decision.to_dict(), "path": rel_arg},
            )

        target = self.safety.resolve_confined_path(rel_arg)
        if not target.is_file():
            raise FileNotFoundError(f"File does not exist: {rel_arg}")
        rel = str(target.relative_to(self.safety.workspace_root))
        old_content = target.read_text(encoding="utf-8", errors="replace")
        diff = self._compute_unified_diff(rel, old_content, "")
        target.unlink()
        self.rollback_stack.append(
            FileMutationRollbackEntry(rel, True, old_content, None, diff)
        )
        return AgentToolResult(
            tool_name="delete_file",
            ok=True,
            requires_confirmation=False,
            confirmation_token=decision.confirmation_token,
            summary=f"Deleted {rel} (rollback snapshot preserved).",
            payload={"path": rel, "unified_diff": diff},
        )

    def _tool_revert_last_edit(self, _params: dict[str, Any]) -> AgentToolResult:
        if not self.rollback_stack:
            return AgentToolResult(
                tool_name="revert_last_edit",
                ok=False,
                requires_confirmation=False,
                confirmation_token=None,
                summary="Rollback stack is empty; no file mutations to revert.",
                payload={},
            )
        entry = self.rollback_stack.pop()
        target = self.safety.resolve_confined_path(entry.relative_path)
        if entry.existed_before and entry.previous_content is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(entry.previous_content, encoding="utf-8")
            action = f"Restored previous content of {entry.relative_path}."
        else:
            if target.exists():
                target.unlink()
            action = f"Removed newly created file {entry.relative_path}."
        return AgentToolResult(
            tool_name="revert_last_edit",
            ok=True,
            requires_confirmation=False,
            confirmation_token=None,
            summary=action,
            payload={"reverted_path": entry.relative_path},
        )

    def _tool_run_command(self, params: dict[str, Any]) -> AgentToolResult:
        cmd = str(params["command"])
        token = params.get("confirmation_token")
        res = self.sandbox.run(cmd, confirmation_token=token)
        ok = res.executed and res.exit_code == 0 and not res.timed_out
        return AgentToolResult(
            tool_name="run_command",
            ok=ok,
            requires_confirmation=res.safety_decision.requires_user_confirmation,
            confirmation_token=res.safety_decision.confirmation_token,
            summary=(
                res.safety_decision.reason
                if not res.executed
                else f"Command exited with code {res.exit_code} in {res.duration_ms:.1f} ms."
            ),
            payload=res.to_dict(),
        )

    def _tool_run_tests(self, params: dict[str, Any]) -> AgentToolResult:
        cmd = params.get("test_command")
        res, diag = self.verifier.run_verification(str(cmd) if cmd else None)
        return AgentToolResult(
            tool_name="run_tests",
            ok=diag.passed,
            requires_confirmation=res.safety_decision.requires_user_confirmation,
            confirmation_token=res.safety_decision.confirmation_token,
            summary=diag.summary,
            payload={"execution": res.to_dict(), "diagnosis": diag.to_dict()},
        )

    @staticmethod
    def _compute_unified_diff(rel_path: str, before: str, after: str) -> str:
        diff_lines = difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            n=3,
        )
        return "".join(diff_lines)
