"""
Pydantic v2 Request & Response Schemas for the Rootism Serving API (Steps 79, 80, 89).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from rootism.reasoning.modes import ReasoningMode


class ChatMessageInput(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1, max_length=250_000)


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessageInput] = Field(..., min_length=1, max_length=128)
    reasoning_mode: ReasoningMode = "structured"
    stream: bool = False
    max_tokens: int = Field(default=96, ge=1, le=8192)
    temperature: float = Field(default=0.25, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=1024)
    repetition_penalty: float = Field(default=1.08, ge=1.0, le=2.0)
    stop: list[str] = Field(default_factory=list, max_length=8)
    workspace_context: str | None = Field(default=None, max_length=120_000)
    seed: int | None = None
    request_id: str | None = Field(default=None, max_length=96)


class CodeCompletionRequest(BaseModel):
    prefix: str = Field(..., max_length=120_000)
    suffix: str = Field(default="", max_length=120_000)
    max_tokens: int = Field(default=64, ge=1, le=4096)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=32, ge=0, le=1024)
    stop: list[str] = Field(default_factory=list, max_length=8)
    seed: int | None = None


class WorkspaceFileSaveRequest(BaseModel):
    path: str = Field(..., min_length=1, max_length=512)
    content: str = Field(..., max_length=500_000)


class AgentToolInvokeRequest(BaseModel):
    tool_name: str = Field(..., min_length=1, max_length=64)
    args: dict[str, Any] = Field(default_factory=dict)


class AgentConfirmationDecisionRequest(BaseModel):
    confirmation_token: str = Field(..., min_length=8, max_length=128)
    approve: bool
    pending_tool_name: str | None = None
    pending_tool_args: dict[str, Any] = Field(default_factory=dict)


class AgentTaskRunRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=16_000)
    max_iterations: int = Field(default=8, ge=1, le=20)
    reasoning_mode: ReasoningMode = "structured"
    confirmation_token: str | None = None
