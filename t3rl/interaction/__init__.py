"""Shared interaction abstractions and contracts."""

from t3rl.interaction.driver import InteractionDriver
from t3rl.interaction.types import (
    AgentDecision,
    AnswerDecision,
    DecisionKind,
    DriverStepResult,
    EnvTransition,
    ErrorDecision,
    InteractionMetrics,
    InteractionResult,
    Status,
    TokenTrajectory,
    ToolCallDecision,
)

__all__ = [
    "InteractionDriver",
    "AgentDecision",
    "DecisionKind",
    "ErrorDecision",
    "AnswerDecision",
    "ToolCallDecision",
    "EnvTransition",
    "DriverStepResult",
    "Status",
    "InteractionMetrics",
    "InteractionResult",
    "TokenTrajectory",
]
