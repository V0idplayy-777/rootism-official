"""
Agent Safety Boundaries, Path Confinement & Human Confirmation Gate (Steps 70 & 76).

Enforces:
1. Strict workspace root confinement (preventing `../` traversal and symlink escapes).
2. Unconditional blocklist of catastrophic system-level operations (`sudo`, `mkfs`, `rm -rf /`,
   fork bombs, raw device writes, credential exfiltration).
3. Human-in-the-loop confirmation gate (`ConfirmationRequirement`) for destructive workspace
   actions (`rm`, `git clean`, `git reset --hard`, `DROP TABLE`, mass file deletion, bulk overwrites).
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class WorkspaceSecurityViolation(PermissionError):
    """Raised when an operation attempts to escape workspace confinement or run a forbidden command."""


@dataclass(frozen=True)
class SafetyCheckDecision:
    allowed_immediately: bool
    blocked_unconditionally: bool
    requires_user_confirmation: bool
    risk_level: str  # "safe" | "destructive_confirm" | "blocked"
    reason: str
    confirmation_token: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_UNCONDITIONAL_BLOCK_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:^|\s)sudo(?:\s|$)"), "Privilege escalation (`sudo`) is strictly prohibited."),
    (re.compile(r":\(\)\s*\{\s*:\|:&\s*\};:"), "Fork-bomb pattern blocked."),
    (re.compile(r"rm\s+-[a-zA-Z]*r[a-zA-Z]*f?\s+/(?:\s|$)"), "Root filesystem deletion blocked."),
    (re.compile(r"(?:mkfs|fdisk|dd\s+if=)"), "Raw block device manipulation blocked."),
    (re.compile(r"(?:curl|wget)\s+[^|]*\|\s*(?:ba)?sh"), "Remote pipe-to-shell execution blocked."),
    (re.compile(r"/etc/(?:shadow|passwd|sudoers)"), "System credential file access blocked."),
)

_CONFIRMATION_REQUIRED_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:^|\s|;|&)rm\s+"), "File/directory deletion via `rm` requires explicit user confirmation."),
    (re.compile(r"git\s+reset\s+--hard"), "Destructive `git reset --hard` requires explicit user confirmation."),
    (re.compile(r"git\s+clean\s+-[a-zA-Z]*f"), "Destructive `git clean -f` requires explicit user confirmation."),
    (re.compile(r"\bDROP\s+(?:TABLE|DATABASE)\b", re.IGNORECASE), "SQL `DROP` operation requires confirmation."),
    (re.compile(r"\bTRUNCATE\s+TABLE\b", re.IGNORECASE), "SQL `TRUNCATE` operation requires confirmation."),
)

_SENSITIVE_ENV_SUBSTRINGS: tuple[str, ...] = (
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "API_KEY",
    "PRIVATE_KEY",
    "AWS_",
    "GITHUB_TOKEN",
    "GH_TOKEN",
)


class SafetyPolicyGuard:
    """
    Validates file paths, shell commands, and destructive workspace mutations.
    """

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        if not self.workspace_root.exists():
            self.workspace_root.mkdir(parents=True, exist_ok=True)
        self._approved_tokens: set[str] = set()

    def resolve_confined_path(self, relative_or_abs_path: str | Path) -> Path:
        """
        Resolve `relative_or_abs_path` and guarantee that both its lexical resolution
        and real symlink resolution stay strictly inside `self.workspace_root`.
        """
        raw = Path(relative_or_abs_path)
        candidate = (raw if raw.is_absolute() else (self.workspace_root / raw)).resolve()
        try:
            candidate.relative_to(self.workspace_root)
        except ValueError as exc:
            raise WorkspaceSecurityViolation(
                f"Path '{relative_or_abs_path}' resolves outside workspace root '{self.workspace_root}'."
            ) from exc

        # Guard against touching `.git/config` or `.git/hooks` executable hooks
        rel_parts = candidate.relative_to(self.workspace_root).parts
        if len(rel_parts) >= 2 and rel_parts[0] == ".git" and rel_parts[1] in ("config", "hooks", "credentials"):
            raise WorkspaceSecurityViolation(
                f"Modifying sensitive git internals ('{'/'.join(rel_parts[:2])}') is forbidden."
            )
        return candidate

    @staticmethod
    def compute_confirmation_token(action_type: str, target_summary: str) -> str:
        digest = hashlib.sha256(f"{action_type}::{target_summary}".encode("utf-8")).hexdigest()[:16]
        return f"confirm-{digest}"

    def grant_confirmation(self, token: str) -> None:
        self._approved_tokens.add(token)

    def evaluate_command(
        self,
        command: str,
        *,
        provided_confirmation_token: str | None = None,
    ) -> SafetyCheckDecision:
        stripped = command.strip()
        if not stripped:
            return SafetyCheckDecision(
                allowed_immediately=False,
                blocked_unconditionally=True,
                requires_user_confirmation=False,
                risk_level="blocked",
                reason="Empty command.",
            )

        for pat, msg in _UNCONDITIONAL_BLOCK_PATTERNS:
            if pat.search(stripped):
                return SafetyCheckDecision(
                    allowed_immediately=False,
                    blocked_unconditionally=True,
                    requires_user_confirmation=False,
                    risk_level="blocked",
                    reason=msg,
                )

        # Block absolute parent traversal tokens (`../`) targeting outside workspace in shell args
        if "../" in stripped and (self.workspace_root.parent.as_posix() in stripped or stripped.count("../") >= 3):
            return SafetyCheckDecision(
                allowed_immediately=False,
                blocked_unconditionally=True,
                requires_user_confirmation=False,
                risk_level="blocked",
                reason="Suspicious multi-level parent directory traversal (`../`) in shell command.",
            )

        for pat, msg in _CONFIRMATION_REQUIRED_PATTERNS:
            if pat.search(stripped):
                tok = self.compute_confirmation_token("command", stripped)
                if provided_confirmation_token == tok or tok in self._approved_tokens:
                    return SafetyCheckDecision(
                        allowed_immediately=True,
                        blocked_unconditionally=False,
                        requires_user_confirmation=False,
                        risk_level="safe",
                        reason="Destructive command explicitly confirmed by user.",
                        confirmation_token=tok,
                    )
                return SafetyCheckDecision(
                    allowed_immediately=False,
                    blocked_unconditionally=False,
                    requires_user_confirmation=True,
                    risk_level="destructive_confirm",
                    reason=msg,
                    confirmation_token=tok,
                )

        return SafetyCheckDecision(
            allowed_immediately=True,
            blocked_unconditionally=False,
            requires_user_confirmation=False,
            risk_level="safe",
            reason="Command passes workspace safety policy.",
        )

    def evaluate_file_mutation(
        self,
        *,
        operation: str,  # "delete" | "bulk_overwrite" | "edit" | "create"
        paths: list[str],
        provided_confirmation_token: str | None = None,
    ) -> SafetyCheckDecision:
        for p in paths:
            self.resolve_confined_path(p)

        if operation == "delete" or (operation == "bulk_overwrite" and len(paths) > 3):
            summary = f"{operation}:{','.join(sorted(paths))}"
            tok = self.compute_confirmation_token("file_mutation", summary)
            if provided_confirmation_token == tok or tok in self._approved_tokens:
                return SafetyCheckDecision(
                    allowed_immediately=True,
                    blocked_unconditionally=False,
                    requires_user_confirmation=False,
                    risk_level="safe",
                    reason=f"Destructive file operation ({operation}) confirmed by user.",
                    confirmation_token=tok,
                )
            return SafetyCheckDecision(
                allowed_immediately=False,
                blocked_unconditionally=False,
                requires_user_confirmation=True,
                risk_level="destructive_confirm",
                reason=f"Operation '{operation}' affecting {len(paths)} file(s) requires user confirmation.",
                confirmation_token=tok,
            )

        return SafetyCheckDecision(
            allowed_immediately=True,
            blocked_unconditionally=False,
            requires_user_confirmation=False,
            risk_level="safe",
            reason="Standard workspace file edit allowed.",
        )

    @staticmethod
    def build_sanitized_environment(extra_env: dict[str, str] | None = None) -> dict[str, str]:
        """
        Strip credentials, tokens, and secret environment variables before spawning
        any agent subprocess (Step 70).
        """
        safe_env: dict[str, str] = {}
        for k, v in os.environ.items():
            upper_k = k.upper()
            if any(secret_sub in upper_k for secret_sub in _SENSITIVE_ENV_SUBSTRINGS):
                continue
            safe_env[k] = v
        safe_env["PYTHONDONTWRITEBYTECODE"] = "1"
        safe_env["CI"] = "true"
        if extra_env:
            for k, v in extra_env.items():
                if not any(sub in k.upper() for sub in _SENSITIVE_ENV_SUBSTRINGS):
                    safe_env[k] = v
        return safe_env
