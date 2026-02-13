"""Shared interaction types for BFCL implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(Enum):
    """Interaction status."""

    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


class DecisionKind(Enum):
    """Kind of parsed agent decision."""

    ERROR = "error"
    ANSWER = "answer"
    TOOL_CALL = "tool_call"


@dataclass(slots=True)
class ErrorDecision:
    """Decision representing a parser/format error path."""

    error: str
    kind: DecisionKind = field(default=DecisionKind.ERROR, init=False)


@dataclass(slots=True)
class AnswerDecision:
    """Decision representing an assistant final answer for current turn."""

    answer: str
    kind: DecisionKind = field(default=DecisionKind.ANSWER, init=False)


@dataclass(slots=True)
class ToolCallDecision:
    """Decision representing one or multiple parsed tool calls."""

    tool_calls: list[str]
    assistant_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    kind: DecisionKind = field(default=DecisionKind.TOOL_CALL, init=False)


AgentDecision = ErrorDecision | AnswerDecision | ToolCallDecision


@dataclass(slots=True)
class EnvTransition:
    """Environment adapter step output consumed by the shared driver."""

    observation_messages: list[dict[str, Any]] = field(default_factory=list)
    reward_delta: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    metric_updates: dict[str, float | int | list[Any]] = field(default_factory=dict)
    add_generation_prompt: bool = True


@dataclass(slots=True)
class DriverStepResult:
    """Single interaction step snapshot for optional hook consumption."""

    step_index: int
    response: str
    decision: AgentDecision
    transition: EnvTransition


@dataclass
class InteractionMetrics:
    """Structured metrics for interaction tracking."""

    tool_call_successes: int = 0
    tool_call_errors: int = 0
    format_errors: int = 0
    user_turn_scores: list[float] = field(default_factory=list)

    @property
    def total_tool_calls(self) -> int:
        return self.tool_call_successes + self.tool_call_errors

    @property
    def total_interactions(self) -> int:
        return (
            self.tool_call_successes
            + self.tool_call_errors
            + self.format_errors
            + len(self.user_turn_scores)
        )


@dataclass
class InteractionResult:
    """Result of a complete agent-environment interaction."""

    prompt: str
    reward: float
    messages: list[dict[str, Any]]
    info: dict[str, Any]
    response: str = ""
    loss_mask: list[int] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    response_length: int = 0
    status: Status = Status.COMPLETED
    turn_rewards: list[float] = field(default_factory=list)
    metrics: InteractionMetrics = field(default_factory=InteractionMetrics)
    rollout_log_probs: list[float] = field(default_factory=list)


@dataclass
class TokenTrajectory:
    """Internal token trajectory tracker for TITO alignment."""

    accumulated_tokens: list[int]
    response_token_ids: list[int] = field(default_factory=list)
    loss_masks: list[int] = field(default_factory=list)
    rollout_log_probs: list[float] = field(default_factory=list)

    def append_response(
        self,
        token_ids: list[int],
        *,
        loss_mask: int,
        log_probs: list[float] | None = None,
    ) -> None:
        if not token_ids:
            return

        self.accumulated_tokens.extend(token_ids)
        self.response_token_ids.extend(token_ids)
        self.loss_masks.extend([loss_mask] * len(token_ids))

        if log_probs is None:
            self.rollout_log_probs.extend([0.0] * len(token_ids))
            return

        if len(log_probs) != len(token_ids):
            raise ValueError(
                f"log_probs length ({len(log_probs)}) must match token_ids length ({len(token_ids)})"
            )
        self.rollout_log_probs.extend(log_probs)
