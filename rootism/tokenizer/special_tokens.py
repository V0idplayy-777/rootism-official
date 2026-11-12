"""
Rootism Special Tokens & Formal Semantics Registry (Step 25).

Reserves IDs 0..63 permanently so special-token semantics are frozen across
all tokenizer vocabulary sizes and checkpoint generations.
"""

from __future__ import annotations

from dataclasses import dataclass

RESERVED_SPECIAL_TOKEN_SLOTS: int = 64


@dataclass(frozen=True)
class SpecialTokenSpec:
    """Formal definition and exact operational semantics of a Rootism special token."""

    token_id: int
    surface: str
    category: str
    semantics: str


CORE_SPECIAL_TOKEN_SPECS: tuple[SpecialTokenSpec, ...] = (
    # 0..3: Sequence & fallback primitives
    SpecialTokenSpec(0, "<|pad|>", "sequence", "Non-attending right/tail sequence padding token; masked in loss."),
    SpecialTokenSpec(1, "<|bos|>", "sequence", "Beginning-of-sequence anchor starting every independent document or session."),
    SpecialTokenSpec(2, "<|eos|>", "sequence", "End-of-sequence terminator signaling complete generation or document boundary."),
    SpecialTokenSpec(3, "<|unk|>", "sequence", "Reserved unknown fallback guard (byte-level BPE covers all 256 raw bytes without emitting <unk>)."),
    # 4..10: Instruction & role boundaries
    SpecialTokenSpec(4, "<|system_start|>", "instruction", "Opens trusted system-level policy and environment specification block."),
    SpecialTokenSpec(5, "<|system_end|>", "instruction", "Closes system policy block."),
    SpecialTokenSpec(6, "<|user_start|>", "instruction", "Opens user prompt or task request block."),
    SpecialTokenSpec(7, "<|user_end|>", "instruction", "Closes user prompt block."),
    SpecialTokenSpec(8, "<|assistant_start|>", "instruction", "Opens Rootism assistant response turn."),
    SpecialTokenSpec(9, "<|assistant_end|>", "instruction", "Closes Rootism assistant response turn."),
    SpecialTokenSpec(10, "<|turn_sep|>", "instruction", "Explicit delimiter separating multi-turn conversational exchanges."),
    # 11..19: Reasoning control & structured summary boundaries (Phase VII)
    SpecialTokenSpec(11, "<|reason_mode:concise|>", "reasoning", "Control directive requesting direct, minimal-prose final solution."),
    SpecialTokenSpec(12, "<|reason_mode:explain|>", "reasoning", "Control directive requesting clear pedagogical explanation alongside code."),
    SpecialTokenSpec(13, "<|reason_mode:structured|>", "reasoning", "Control directive requesting assumptions, approach trade-offs, verification summary, and solution."),
    SpecialTokenSpec(14, "<|reason_plan_start|>", "reasoning", "Opens structured problem decomposition and algorithmic plan block."),
    SpecialTokenSpec(15, "<|reason_plan_end|>", "reasoning", "Closes structured plan block."),
    SpecialTokenSpec(16, "<|reason_verify_start|>", "reasoning", "Opens self-verification, invariant check, and edge-case audit block."),
    SpecialTokenSpec(17, "<|reason_verify_end|>", "reasoning", "Closes self-verification block."),
    SpecialTokenSpec(18, "<|reason_summary_start|>", "reasoning", "Opens user-facing faithful reasoning summary (distinct from raw internal deliberation)."),
    SpecialTokenSpec(19, "<|reason_summary_end|>", "reasoning", "Closes user-facing faithful reasoning summary."),
    # 20..31: Code, Fill-in-the-Middle (FIM), Diff & Repository Context Boundaries
    SpecialTokenSpec(20, "<|code_start|>", "code", "Opens delimited source code block."),
    SpecialTokenSpec(21, "<|code_end|>", "code", "Closes delimited source code block."),
    SpecialTokenSpec(22, "<|lang_tag|>", "code", "Precedes canonical language identifier (e.g. python, rust, typescript)."),
    SpecialTokenSpec(23, "<|repo_start|>", "code", "Opens multi-file repository context envelope."),
    SpecialTokenSpec(24, "<|repo_end|>", "code", "Closes multi-file repository context envelope."),
    SpecialTokenSpec(25, "<|file_sep|>", "code", "Separates distinct files inside a repository context or packed multi-file sample."),
    SpecialTokenSpec(26, "<|file_path|>", "code", "Marks workspace-relative file path header."),
    SpecialTokenSpec(27, "<|fim_prefix|>", "code", "Fill-in-the-Middle prefix span marker preceding cursor."),
    SpecialTokenSpec(28, "<|fim_suffix|>", "code", "Fill-in-the-Middle suffix span marker following cursor."),
    SpecialTokenSpec(29, "<|fim_middle|>", "code", "Fill-in-the-Middle target span marker to synthesize."),
    SpecialTokenSpec(30, "<|diff_start|>", "code", "Opens unified patch/edit block."),
    SpecialTokenSpec(31, "<|diff_end|>", "code", "Closes unified patch/edit block."),
    # 32..41: Structured Tool Calling & Tool Execution Results (Phase VIII)
    SpecialTokenSpec(32, "<|tool_defs_start|>", "tool", "Opens JSON schema registry of callable workspace tools."),
    SpecialTokenSpec(33, "<|tool_defs_end|>", "tool", "Closes callable tool schema registry."),
    SpecialTokenSpec(34, "<|tool_call_start|>", "tool", "Opens structured, schema-validated tool invocation payload."),
    SpecialTokenSpec(35, "<|tool_call_end|>", "tool", "Closes structured tool invocation payload."),
    SpecialTokenSpec(36, "<|tool_name|>", "tool", "Marks canonical tool identifier within a tool call."),
    SpecialTokenSpec(37, "<|tool_args|>", "tool", "Marks JSON-serialized validated arguments object within a tool call."),
    SpecialTokenSpec(38, "<|tool_result_start|>", "tool", "Opens sandboxed execution output returned by an invoked tool."),
    SpecialTokenSpec(39, "<|tool_result_end|>", "tool", "Closes tool execution output block."),
    SpecialTokenSpec(40, "<|tool_status:ok|>", "tool", "Indicates tool invocation completed with zero exit/validation error."),
    SpecialTokenSpec(41, "<|tool_status:error|>", "tool", "Indicates tool invocation failed, timed out, or was denied by policy."),
    # 42..51: Coding Agent Orchestration & Language Epistemic Scope Metadata
    SpecialTokenSpec(42, "<|agent_task|>", "agent", "Marks top-level software engineering objective assigned to the coding agent."),
    SpecialTokenSpec(43, "<|agent_state|>", "agent", "Marks current state-machine transition (planning, inspecting, editing, testing, awaiting_confirmation)."),
    SpecialTokenSpec(44, "<|agent_step|>", "agent", "Marks numbered iterative execution step in multi-step agent loop."),
    SpecialTokenSpec(45, "<|agent_confirm_req|>", "agent", "Signals destructive/external action requiring explicit user confirmation before execution."),
    SpecialTokenSpec(46, "<|agent_confirm_res|>", "agent", "Carries user approval or rejection decision for a confirmation gate."),
    SpecialTokenSpec(47, "<|agent_obs_start|>", "agent", "Opens structured workspace/test/diagnostic observation block."),
    SpecialTokenSpec(48, "<|agent_obs_end|>", "agent", "Closes agent observation block."),
    SpecialTokenSpec(49, "<|lang_scope:primary|>", "language_scope", "Tags task in one of Rootism's 10 primary programming languages."),
    SpecialTokenSpec(50, "<|lang_scope:secondary|>", "language_scope", "Tags task in one of Rootism's 9 secondary programming languages."),
    SpecialTokenSpec(51, "<|lang_scope:unsupported|>", "language_scope", "Tags out-of-scope language so model applies calibrated best-effort assistance without overclaiming."),
)


def build_special_token_table(reserved_slots: int = RESERVED_SPECIAL_TOKEN_SLOTS) -> list[SpecialTokenSpec]:
    """Return full table of `reserved_slots` special tokens (named + forward-compatible reserved slots)."""
    if reserved_slots < len(CORE_SPECIAL_TOKEN_SPECS):
        raise ValueError(
            f"reserved_slots ({reserved_slots}) must be >= {len(CORE_SPECIAL_TOKEN_SPECS)}"
        )
    specs = list(CORE_SPECIAL_TOKEN_SPECS)
    for idx in range(len(CORE_SPECIAL_TOKEN_SPECS), reserved_slots):
        specs.append(
            SpecialTokenSpec(
                token_id=idx,
                surface=f"<|reserved_{idx:02d}|>",
                category="reserved",
                semantics="Reserved slot preserving permanent token-ID stability for future protocol extensions.",
            )
        )
    return specs


SPECIAL_TOKENS_BY_ID: dict[int, SpecialTokenSpec] = {
    spec.token_id: spec for spec in build_special_token_table()
}
SPECIAL_TOKENS_BY_SURFACE: dict[str, SpecialTokenSpec] = {
    spec.surface: spec for spec in build_special_token_table()
}
