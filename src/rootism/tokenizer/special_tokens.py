"""Fixed, permanently ordered special tokens (requirement 25).

IDs 0..N-1 are reserved and NEVER produced by BPE merges. Changing this list
changes the tokenizer fingerprint, which invalidates every checkpoint. That is
intentional: token mappings may not change after training begins (requirement 28).

Semantics are normative; the agent, data pipeline and inference runtime all use
these exact strings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialToken:
    id: int
    token: str
    kind: str
    semantics: str


def _build() -> tuple[SpecialToken, ...]:
    rows = [
        ("<pad>", "padding", "Padding slot. Excluded from loss and from attention; never counted as a training token."),
        ("<bos>", "boundary", "Beginning of a sequence. Prepended by the input builder, never by raw corpus text."),
        ("<eos>", "boundary", "End of a sequence / end of an assistant turn. Terminates generation."),
        ("<unk>", "fallback", "Fallback for out-of-range integer ids. Byte-level BPE never emits it for valid UTF-8 text."),
        ("<sep>", "boundary", "Separator between independent documents inside a packed training sequence."),
        ("<fim_prefix>", "code", "Fill-in-the-middle prefix: code before the cursor."),
        ("<fim_middle>", "code", "Fill-in-the-middle middle: code after the cursor."),
        ("<fim_suffix>", "code", "Fill-in-the-middle suffix: the span to be predicted."),
        ("<code>", "code", "Opens a code span with an explicit language attribute in the preceding text."),
        ("</code>", "code", "Closes a code span."),
        ("<file>", "file", "Opens a file payload in workspace context."),
        ("</file>", "file", "Closes a file payload."),
        ("<file_path>", "file", "Opens a repository-relative file path."),
        ("</file_path>", "file", "Closes a repository-relative file path."),
        ("<diff>", "diff", "Opens a unified diff proposed by the agent."),
        ("</diff>", "diff", "Closes a unified diff."),
        ("<workspace>", "context", "Opens the workspace context block sent to the model."),
        ("</workspace>", "context", "Closes the workspace context block."),
        ("<system>", "role", "Opens a system-level instruction."),
        ("</system>", "role", "Closes a system-level instruction."),
        ("<user>", "role", "Opens a user message."),
        ("</user>", "role", "Closes a user message."),
        ("<assistant>", "role", "Opens an assistant message."),
        ("</assistant>", "role", "Closes an assistant message."),
        ("<instruction>", "instruction", "Opens an instruction boundary."),
        ("</instruction>", "instruction", "Closes an instruction boundary."),
        ("<reasoning>", "reasoning", "Opens a USER-VISIBLE structured reasoning summary (approach, assumptions, intermediate results). It is generated text, never a raw dump of private internal deliberation."),
        ("</reasoning>", "reasoning", "Closes a user-visible reasoning summary."),
        ("<task>", "reasoning", "Opens task metadata (objective and constraints) for a reasoning-heavy request."),
        ("</task>", "reasoning", "Closes task metadata."),
        ("<plan>", "agent", "Opens the agent's planned action list."),
        ("</plan>", "agent", "Closes the agent's planned action list."),
        ("<observation>", "agent", "Opens an observation returned by a tool or the environment."),
        ("</observation>", "agent", "Closes an observation."),
        ("<agent>", "agent", "Opens an agent-to-agent / orchestrator message."),
        ("</agent>", "agent", "Closes an agent message."),
        ("<tool_call>", "tool", "Opens a structured tool call. Payload is JSON with validated arguments."),
        ("</tool_call>", "tool", "Closes a structured tool call."),
        ("<tool_name>", "tool", "Opens the tool name field of a tool call."),
        ("</tool_name>", "tool", "Closes the tool name field."),
        ("<tool_args>", "tool", "Opens the JSON arguments field of a tool call."),
        ("</tool_args>", "tool", "Closes the JSON arguments field."),
        ("<tool_result>", "tool", "Opens the result returned by a tool."),
        ("</tool_result>", "tool", "Closes a tool result."),
        ("<test_output>", "tool", "Opens captured test/command stdout and stderr."),
        ("</test_output>", "tool", "Closes captured test output."),
        ("<error>", "tool", "Opens a structured error report."),
        ("</error>", "tool", "Closes a structured error report."),
        ("<mask>", "training", "Reserved masked-token marker for auxiliary objectives; ignored by causal LM loss."),
    ]
    tokens = tuple(SpecialToken(i, t, k, s) for i, (t, k, s) in enumerate(rows))
    return tokens


SPECIAL_TOKENS: tuple[SpecialToken, ...] = _build()
SPECIAL_BY_TOKEN = {s.token: s for s in SPECIAL_TOKENS}
SPECIAL_BY_ID = {s.id: s for s in SPECIAL_TOKENS}
NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)


def special_id(token: str) -> int:
    try:
        return SPECIAL_BY_TOKEN[token].id
    except KeyError as exc:
        raise KeyError(f"unknown special token {token!r}") from exc


def validate_special_tokens() -> None:
    ids = [s.id for s in SPECIAL_TOKENS]
    assert ids == list(range(len(ids))), "special token ids must be contiguous from 0"
    assert len(set(s.token for s in SPECIAL_TOKENS)) == len(ids), "duplicate special token strings"
    assert all(s.token.startswith("<") and s.token.endswith(">") for s in SPECIAL_TOKENS)


validate_special_tokens()
