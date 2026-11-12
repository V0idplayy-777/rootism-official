"""
Sandboxed Command Execution Runtime (Step 70).

Enforces:
1. Workspace directory confinement (`cwd` strictly inside `workspace_root`)
2. Hard wall-clock execution timeouts (`timeout_sec`) with process-group termination
3. Output size caps (`max_output_bytes`) with explicit truncation telemetry
4. POSIX CPU time (`RLIMIT_CPU`) and file-write size (`RLIMIT_FSIZE`) resource limits
5. Sanitized environment stripping sensitive tokens/keys
6. Integration with `SafetyPolicyGuard` (blocking forbidden commands and gating destructive ones)
"""

from __future__ import annotations

import os
import resource
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rootism.agent.safety import SafetyCheckDecision, SafetyPolicyGuard


@dataclass(frozen=True)
class SandboxExecutionResult:
    command: str
    cwd: str
    executed: bool
    timed_out: bool
    exit_code: int | None
    stdout: str
    stderr: str
    output_truncated: bool
    duration_ms: float
    safety_decision: SafetyCheckDecision

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["safety_decision"] = self.safety_decision.to_dict()
        return d


class SandboxedCommandRunner:
    """
    Executes build, lint, compiler, and test commands safely inside `workspace_root`.
    """

    def __init__(
        self,
        safety_guard: SafetyPolicyGuard,
        *,
        default_timeout_sec: float = 15.0,
        max_output_bytes: int = 65_536,
        cpu_time_limit_sec: int = 30,
        max_file_write_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.safety_guard = safety_guard
        self.default_timeout_sec = float(default_timeout_sec)
        self.max_output_bytes = int(max_output_bytes)
        self.cpu_time_limit_sec = int(cpu_time_limit_sec)
        self.max_file_write_bytes = int(max_file_write_bytes)

    def run(
        self,
        command: str,
        *,
        relative_cwd: str = ".",
        timeout_sec: float | None = None,
        confirmation_token: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> SandboxExecutionResult:
        confined_cwd: Path = self.safety_guard.resolve_confined_path(relative_cwd)
        if not confined_cwd.is_dir():
            raise NotADirectoryError(f"Sandbox working directory does not exist: {confined_cwd}")

        decision = self.safety_guard.evaluate_command(
            command, provided_confirmation_token=confirmation_token
        )
        if not decision.allowed_immediately:
            return SandboxExecutionResult(
                command=command,
                cwd=str(confined_cwd.relative_to(self.safety_guard.workspace_root)),
                executed=False,
                timed_out=False,
                exit_code=None,
                stdout="",
                stderr=decision.reason,
                output_truncated=False,
                duration_ms=0.0,
                safety_decision=decision,
            )

        effective_timeout = float(timeout_sec if timeout_sec is not None else self.default_timeout_sec)
        env = self.safety_guard.build_sanitized_environment(extra_env)
        cpu_lim = max(1, int(max(self.cpu_time_limit_sec, effective_timeout + 2)))
        fsize_lim = self.max_file_write_bytes

        def _apply_posix_limits() -> None:
            os.setsid()
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_lim, cpu_lim + 2))
            except (ValueError, OSError):
                pass
            try:
                resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_lim, fsize_lim))
            except (ValueError, OSError):
                pass

        t0 = time.perf_counter()
        timed_out = False
        proc = subprocess.Popen(
            ["/bin/bash", "-c", command],
            cwd=str(confined_cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_apply_posix_limits if os.name == "posix" else None,
        )
        try:
            raw_out, raw_err = proc.communicate(timeout=effective_timeout)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "posix":
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                proc.kill()
            raw_out, raw_err = proc.communicate()
            exit_code = -signal.SIGKILL

        dur_ms = (time.perf_counter() - t0) * 1000.0
        truncated = False
        if len(raw_out) > self.max_output_bytes:
            raw_out = raw_out[: self.max_output_bytes] + b"\n...[STDOUT TRUNCATED BY SANDBOX LIMIT]..."
            truncated = True
        if len(raw_err) > self.max_output_bytes:
            raw_err = raw_err[: self.max_output_bytes] + b"\n...[STDERR TRUNCATED BY SANDBOX LIMIT]..."
            truncated = True

        rel_cwd_str = str(confined_cwd.relative_to(self.safety_guard.workspace_root)) or "."
        return SandboxExecutionResult(
            command=command,
            cwd=rel_cwd_str,
            executed=True,
            timed_out=timed_out,
            exit_code=exit_code,
            stdout=raw_out.decode("utf-8", errors="replace"),
            stderr=raw_err.decode("utf-8", errors="replace"),
            output_truncated=truncated,
            duration_ms=round(dur_ms, 2),
            safety_decision=decision,
        )
