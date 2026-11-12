"""
User-Visible Reasoning Presentation Modes & Privacy Boundary (Steps 66 & 67).

Supports all user-selectable reasoning presentation modes:
- `concise`: Direct answer / code without extended narrative.
- `explain`: Clear explanation alongside solution.
- `structured`: Structured solution with approach, trade-offs, and verification notes.
- `approach_summary`: High-level summary of the algorithmic approach + final answer.
- `assumptions_and_intermediates`: Relevant assumptions, key intermediate invariants, and solution.

Privacy & Honesty Invariant (Steps 66 & 84):
- Raw internal deliberation blocks (`<|reason_plan_start|>...<|reason_plan_end|>`,
  `<|reason_verify_start|>...<|reason_verify_end|>`) are NEVER exposed as raw transcripts.
- Only faithful `<|reason_summary_start|>...<|reason_summary_end|>` summaries are surfaced,
  and when the model emits no summary block, none is fabricated!
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

ReasoningMode = Literal[
    "concise",
    "explain",
    "structured",
    "approach_summary",
    "assumptions_and_intermediates",
]

REASONING_MODE_SPECS: dict[ReasoningMode, dict[str, str]] = {
    "concise": {
        "control_token": "<|reason_mode:concise|>",
        "label": "Concise Answer",
        "system_hint": "Provide the direct, accurate code or answer with minimal prose.",
    },
    "explain": {
        "control_token": "<|reason_mode:explain|>",
        "label": "Explanation + Solution",
        "system_hint": "Provide a clear, technically rigorous explanation alongside the implementation.",
    },
    "structured": {
        "control_token": "<|reason_mode:structured|>",
        "label": "Structured Solution",
        "system_hint": "Present a faithful reasoning summary, trade-off evaluation, verification check, and solution.",
    },
    "approach_summary": {
        "control_token": "<|reason_mode:structured|>",
        "label": "Summary of Approach",
        "system_hint": "Summarize the high-level algorithmic strategy briefly before giving the solution.",
    },
    "assumptions_and_intermediates": {
        "control_token": "<|reason_mode:structured|>",
        "label": "Assumptions & Intermediates",
        "system_hint": "State explicit assumptions and key intermediate invariants alongside the final result.",
    },
}

_INTERNAL_PLAN_RE = re.compile(
    r"<\|reason_plan_start\|>[\s\S]*?<\|reason_plan_end\|>", flags=re.MULTILINE
)
_INTERNAL_VERIFY_RE = re.compile(
    r"<\|reason_verify_start\|>[\s\S]*?<\|reason_verify_end\|>", flags=re.MULTILINE
)
_UNCLOSED_INTERNAL_RE = re.compile(
    r"<\|reason_(?:plan|verify)_start\|>[\s\S]*$", flags=re.MULTILINE
)
_SUMMARY_BLOCK_RE = re.compile(
    r"<\|reason_summary_start\|>([\s\S]*?)<\|reason_summary_end\|>", flags=re.MULTILINE
)
_CONTROL_TAGS_CLEANUP_RE = re.compile(
    r"<\|(?:assistant_start|assistant_end|user_start|user_end|system_start|system_end|bos|eos|pad"
    r"|reason_mode:[a-z_]+|reason_summary_start|reason_summary_end)\|>"
)
_CODE_TAG_START_RE = re.compile(r"<\|code_start\|><\|lang_tag\|>([a-zA-Z0-9_+#-]+)\n?")
_CODE_TAG_PLAIN_START_RE = re.compile(r"<\|code_start\|>\n?")
_CODE_TAG_END_RE = re.compile(r"<\|code_end\|>")


@dataclass(frozen=True)
class FormattedReasoningResponse:
    mode: ReasoningMode
    visible_answer: str
    reasoning_summary: str | None
    internal_deliberation_redacted: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReasoningOutputFormatter:
    """
    Formats prompts for a chosen `ReasoningMode` and sanitizes raw model token streams
    so internal deliberations stay private while faithful summaries are cleanly structured.
    """

    @staticmethod
    def build_structured_prompt_segments(
        *,
        user_message: str,
        mode: ReasoningMode = "structured",
        system_prompt: str | None = None,
        workspace_context: str | None = None,
    ) -> list[tuple[str, bool]]:
        """
        Return `[(text, is_trusted_special_token)]` ready for `RootismTokenizer.encode_structured(...)`
        so user and workspace strings remain strictly isolated from special tokens!
        """
        spec = REASONING_MODE_SPECS[mode]
        sys_body = system_prompt or (
            "You are Rootism, an independently engineered software engineering and reasoning model."
        )
        sys_full = f"{sys_body}\nPresentation policy: {spec['system_hint']}"

        segments: list[tuple[str, bool]] = [
            ("<|bos|>", True),
            ("<|system_start|>", True),
            (sys_full, False),
            ("<|system_end|>", True),
        ]
        if workspace_context:
            segments.extend(
                [
                    ("<|repo_start|>", True),
                    (workspace_context, False),
                    ("<|repo_end|>", True),
                ]
            )
        segments.extend(
            [
                ("<|user_start|>", True),
                (spec["control_token"], True),
                ("\n" + user_message, False),
                ("<|user_end|>", True),
                ("<|assistant_start|>", True),
            ]
        )
        return segments

    @staticmethod
    def sanitize_and_extract(
        raw_decoded_text: str,
        mode: ReasoningMode = "structured",
    ) -> FormattedReasoningResponse:
        had_internal = bool(
            _INTERNAL_PLAN_RE.search(raw_decoded_text)
            or _INTERNAL_VERIFY_RE.search(raw_decoded_text)
            or _UNCLOSED_INTERNAL_RE.search(raw_decoded_text)
        )

        # 1. Redact private internal scratchpad spans unconditionally
        scrubbed = _INTERNAL_PLAN_RE.sub("", raw_decoded_text)
        scrubbed = _INTERNAL_VERIFY_RE.sub("", scrubbed)
        scrubbed = _UNCLOSED_INTERNAL_RE.sub("", scrubbed)

        # 2. Extract genuine `<|reason_summary_start|>...<|reason_summary_end|>` if present (never fabricate!)
        summaries = [m.strip() for m in _SUMMARY_BLOCK_RE.findall(scrubbed) if m.strip()]
        reasoning_summary = "\n\n".join(summaries) if summaries else None
        without_summary = _SUMMARY_BLOCK_RE.sub("", scrubbed)

        # 3. Convert `<|code_start|><|lang_tag|>lang ... <|code_end|>` into clean Markdown fences
        rendered = _CODE_TAG_START_RE.sub(r"```\1\n", without_summary)
        rendered = _CODE_TAG_PLAIN_START_RE.sub("```\n", rendered)
        rendered = _CODE_TAG_END_RE.sub("\n```", rendered)
        rendered = _CONTROL_TAGS_CLEANUP_RE.sub("", rendered).strip()

        if mode == "concise":
            # In concise mode, suppress optional summary card so user gets direct output
            return FormattedReasoningResponse(
                mode=mode,
                visible_answer=rendered or (reasoning_summary or ""),
                reasoning_summary=None,
                internal_deliberation_redacted=had_internal,
            )

        return FormattedReasoningResponse(
            mode=mode,
            visible_answer=rendered,
            reasoning_summary=reasoning_summary,
            internal_deliberation_redacted=had_internal,
        )
