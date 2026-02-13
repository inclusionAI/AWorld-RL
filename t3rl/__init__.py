"""t3rl: multi-turn tool-use RL integration package."""

from __future__ import annotations

from t3rl.agents.bfcl import BFCLTrainableAgent
from t3rl.parsers.reasoning import (
    DeepSeekR1ReasoningParser,
    Instruct2507ReasoningParser,
    TaggedReasoningParser,
    create_reasoning_parser,
)
from t3rl.parsers.tool_call import (
    Glm47MoeToolCallParser,
    HermesToolCallParser,
    create_tool_call_parser,
)


async def generate_bfcl(*args, **kwargs):
    """Lazy BFCL rollout entrypoint."""
    from t3rl.rollout.bfcl import generate

    return await generate(*args, **kwargs)


__all__ = [
    "generate_bfcl",
    "BFCLTrainableAgent",
    "HermesToolCallParser",
    "Glm47MoeToolCallParser",
    "TaggedReasoningParser",
    "Instruct2507ReasoningParser",
    "DeepSeekR1ReasoningParser",
    "create_tool_call_parser",
    "create_reasoning_parser",
]
