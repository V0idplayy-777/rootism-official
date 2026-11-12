"""
Verified Original Synthetic Data Generation Engine (Steps 39 & 40).

Generates commercially clean (`Rootism-Proprietary-Original`) training tasks across:
1. Algorithmic & data-structure implementations with parameterized invariants
2. Bug diagnosis & repair pairs (buggy implementation + failing assertion + root-cause diagnosis + fixed solution)
3. Structured mathematical & constraint-satisfaction reasoning problems
4. Repository-level multi-file task serialization (`<|repo_start|> ... <|file_sep|> ... <|repo_end|>`)

CRITICAL QUALITY GATE:
Generated executable tasks are NEVER assumed valid. Every Python algorithmic, debugging,
and mathematical verification sample is executed inside an isolated subprocess with a strict
timeout (`python3 -I`) against its unit tests before admission!
"""

from __future__ import annotations

import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class VerifiedSyntheticSample:
    sample_id: str
    category: str
    language: str
    licence: str
    verified_by_execution: bool
    formatted_training_text: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_isolated_python_verification(script: str, timeout_sec: float = 4.0) -> tuple[bool, str]:
    """Execute self-contained Python verification script in isolated mode (`-I`)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        if proc.returncode == 0:
            return True, ""
        return False, (proc.stderr or proc.stdout).strip()[:400]
    except subprocess.TimeoutExpired:
        return False, "Verification timed out"


class VerifiedSyntheticGenerator:
    """
    Produces execution-verified original training examples for Rootism.
    """

    def __init__(self, seed: int = 20261112) -> None:
        self.rng = random.Random(seed)
        self.generated_count = 0
        self.rejected_verification_count = 0

    def generate_sliding_window_task(self, idx: int) -> VerifiedSyntheticSample | None:
        """Parameterized bounded-subarray / sliding-window algorithmic task + unit tests."""
        threshold = self.rng.randint(12, 85)
        length = self.rng.randint(8, 20)
        arr = [self.rng.randint(1, 25) for _ in range(length)]

        solution_code = (
            "def longest_bounded_subarray(nums: list[int], limit: int) -> int:\n"
            "    \"\"\"Return length of the longest contiguous subarray with sum <= limit (nums_i > 0).\"\"\"\n"
            "    left = 0\n"
            "    window_sum = 0\n"
            "    best = 0\n"
            "    for right, val in enumerate(nums):\n"
            "        window_sum += val\n"
            "        while window_sum > limit and left <= right:\n"
            "            window_sum -= nums[left]\n"
            "            left += 1\n"
            "        best = max(best, right - left + 1)\n"
            "    return best\n"
        )

        # Brute-force oracle inside verification harness confirms algorithmic correctness!
        verify_harness = (
            f"{solution_code}\n"
            "def _oracle(nums, limit):\n"
            "    n = len(nums)\n"
            "    ans = 0\n"
            "    for i in range(n):\n"
            "        for j in range(i, n):\n"
            "            if sum(nums[i:j+1]) <= limit:\n"
            "                ans = max(ans, j - i + 1)\n"
            "    return ans\n"
            f"sample = {arr!r}\n"
            f"lim = {threshold}\n"
            "assert longest_bounded_subarray(sample, lim) == _oracle(sample, lim)\n"
            "assert longest_bounded_subarray([], lim) == 0\n"
            "assert longest_bounded_subarray([lim + 5], lim) == 0\n"
        )
        ok, _ = _run_isolated_python_verification(verify_harness)
        if not ok:
            self.rejected_verification_count += 1
            return None

        expected_val_script = (
            f"{solution_code}\nprint(longest_bounded_subarray({arr!r}, {threshold}))"
        )
        ok2, _ = _run_isolated_python_verification(expected_val_script)
        if not ok2:
            self.rejected_verification_count += 1
            return None

        text = (
            "<|user_start|><|reason_mode:structured|>\n"
            f"Implement `longest_bounded_subarray(nums: list[int], limit: int) -> int` in Python "
            f"for strictly positive integers `nums` in O(n) time and O(1) auxiliary space, "
            f"and verify on `nums={arr!r}`, `limit={threshold}`.\n"
            "<|user_end|><|assistant_start|>"
            "<|reason_summary_start|>\n"
            "Because every element `nums[i] > 0`, prefix sums are strictly monotonic. "
            "A two-pointer sliding window `[left, right]` maintains `window_sum <= limit` "
            "with each index advancing at most `n` times, yielding O(n) time and O(1) space.\n"
            "<|reason_summary_end|>\n"
            "<|code_start|><|lang_tag|>python\n"
            f"{solution_code}"
            "<|code_end|><|assistant_end|>"
        )
        self.generated_count += 1
        return VerifiedSyntheticSample(
            sample_id=f"syn-algo-window-{idx:05d}",
            category="algorithmic_coding_and_reasoning",
            language="python",
            licence="Rootism-Proprietary-Original",
            verified_by_execution=True,
            formatted_training_text=text,
            metadata={"threshold": threshold, "input_length": length},
        )

    def generate_debugging_repair_task(self, idx: int) -> VerifiedSyntheticSample | None:
        """
        Generate a buggy implementation, verify the bug ACTUALLY fails the test,
        and verify the repaired code PASSES all tests!
        """
        multiplier = self.rng.randint(2, 9)
        buggy_code = (
            f"def accumulate_scaled(items: list[int], acc: list[int] = []) -> list[int]:\n"
            f"    for x in items:\n"
            f"        acc.append(x * {multiplier})\n"
            f"    return acc\n"
        )
        fixed_code = (
            f"def accumulate_scaled(items: list[int], acc: list[int] | None = None) -> list[int]:\n"
            f"    target = [] if acc is None else acc\n"
            f"    for x in items:\n"
            f"        target.append(x * {multiplier})\n"
            f"    return target\n"
        )

        # Verify buggy version genuinely fails on second call AND fixed version passes!
        check_script = (
            f"{buggy_code}\n"
            "r1 = accumulate_scaled([1, 2])\n"
            "r2 = accumulate_scaled([3])\n"
            f"assert r2 != [3 * {multiplier}], 'Buggy code unexpectedly passed!'\n"
            f"{fixed_code}\n"
            f"assert accumulate_scaled([1, 2]) == [{1 * multiplier}, {2 * multiplier}]\n"
            f"assert accumulate_scaled([3]) == [{3 * multiplier}]\n"
        )
        ok, _ = _run_isolated_python_verification(check_script)
        if not ok:
            self.rejected_verification_count += 1
            return None

        text = (
            "<|user_start|><|reason_mode:explain|>\n"
            "Diagnose why consecutive calls to `accumulate_scaled` corrupt state across requests "
            "and provide a corrected implementation:\n"
            "<|code_start|><|lang_tag|>python\n"
            f"{buggy_code}"
            "<|code_end|><|user_end|><|assistant_start|>"
            "<|reason_summary_start|>\n"
            "Root cause: Python evaluates default parameter expressions once at function definition time. "
            "The mutable list `acc: list[int] = []` persists across all calls that omit `acc`, leaking prior "
            "elements into subsequent invocations. Replace the default with `None` and allocate a fresh list per call.\n"
            "<|reason_summary_end|>\n"
            "<|code_start|><|lang_tag|>python\n"
            f"{fixed_code}"
            "<|code_end|><|assistant_end|>"
        )
        self.generated_count += 1
        return VerifiedSyntheticSample(
            sample_id=f"syn-debug-mutdefault-{idx:05d}",
            category="debugging_and_error_diagnosis",
            language="python",
            licence="Rootism-Proprietary-Original",
            verified_by_execution=True,
            formatted_training_text=text,
            metadata={"multiplier": multiplier, "bug_class": "mutable_default_argument"},
        )

    def generate_modular_constraint_reasoning_task(self, idx: int) -> VerifiedSyntheticSample | None:
        """
        Generate a novel discrete mathematics / CRT constraint satisfaction task
        and solve + verify via executable solver.
        """
        mod_a, mod_b = self.rng.choice([(5, 7), (7, 11), (8, 13), (9, 14), (11, 15)])
        secret = self.rng.randint(mod_a * mod_b + 1, 4 * mod_a * mod_b)
        rem_a = secret % mod_a
        rem_b = secret % mod_b
        lower_bound = (secret // (mod_a * mod_b)) * (mod_a * mod_b)
        upper_bound = lower_bound + (mod_a * mod_b) - 1

        solver_script = (
            f"matches = [x for x in range({lower_bound}, {upper_bound + 1}) "
            f"if x % {mod_a} == {rem_a} and x % {mod_b} == {rem_b}]\n"
            f"assert matches == [{secret}], f'Expected unique [{secret}], got {{matches}}'\n"
        )
        ok, _ = _run_isolated_python_verification(solver_script)
        if not ok:
            self.rejected_verification_count += 1
            return None

        lcm_val = mod_a * mod_b
        base_smallest = secret % lcm_val
        text = (
            "<|user_start|><|reason_mode:structured|>\n"
            f"Find the unique integer N in the closed interval [{lower_bound}, {upper_bound}] "
            f"satisfying N ≡ {rem_a} (mod {mod_a}) and N ≡ {rem_b} (mod {mod_b}).\n"
            "<|user_end|><|assistant_start|>"
            "<|reason_summary_start|>\n"
            f"1. Since gcd({mod_a}, {mod_b}) = 1, by the Chinese Remainder Theorem there is a unique residue "
            f"modulo {mod_a} * {mod_b} = {lcm_val}, which equals {base_smallest} (mod {lcm_val}).\n"
            f"2. Shifting into [{lower_bound}, {upper_bound}] of width {lcm_val} yields N = {lower_bound} + {base_smallest} = {secret}.\n"
            f"3. Verification: {secret} % {mod_a} = {secret % mod_a} and {secret} % {mod_b} = {secret % mod_b}.\n"
            "<|reason_summary_end|>\n"
            f"The unique integer is **{secret}**.<|assistant_end|>"
        )
        self.generated_count += 1
        return VerifiedSyntheticSample(
            sample_id=f"syn-reason-crt-{idx:05d}",
            category="mathematical_and_constraint_reasoning",
            language="python",
            licence="Rootism-Proprietary-Original",
            verified_by_execution=True,
            formatted_training_text=text,
            metadata={"mod_a": mod_a, "mod_b": mod_b, "answer": secret},
        )

    @staticmethod
    def format_repository_task(
        repo_name: str,
        files: dict[str, str],
        task_prompt: str,
        solution_patch_or_answer: str,
    ) -> str:
        """Serialize a multi-file repository task using Rootism's structured repo special tokens."""
        parts = [f"<|repo_start|>{repo_name}\n"]
        for rel_path in sorted(files.keys()):
            parts.append(f"<|file_sep|><|file_path|>{rel_path}\n{files[rel_path]}\n")
        parts.append("<|repo_end|>\n")
        parts.append(f"<|user_start|>{task_prompt}<|user_end|>")
        parts.append(f"<|assistant_start|>{solution_patch_or_answer}<|assistant_end|>")
        return "".join(parts)

    def generate_batch(self, count_per_type: int = 4) -> list[VerifiedSyntheticSample]:
        samples: list[VerifiedSyntheticSample] = []
        for i in range(count_per_type):
            for gen_fn in (
                self.generate_sliding_window_task,
                self.generate_debugging_repair_task,
                self.generate_modular_constraint_reasoning_task,
            ):
                item = gen_fn(i)
                if item is not None:
                    samples.append(item)
        return samples
