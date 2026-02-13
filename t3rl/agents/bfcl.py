"""Trainable agent for BFCL multi-turn function calling with shared driver."""

from __future__ import annotations

import json
from typing import Any

from t3rl.agents.base import BaseTrainableAgent
from t3rl.interaction.driver import InteractionDriver
from t3rl.interaction.types import (
    AgentDecision,
    AnswerDecision,
    EnvTransition,
    ErrorDecision,
    ToolCallDecision,
)
from t3rl.envs.bfcl_gym import BFCLGymAdapter
from t3rl.parsers.reasoning import ReasoningParser
from t3rl.parsers.tool_call import ToolCallParser, ToolCallParseResult


class BFCLTrainableAgent(BaseTrainableAgent):
    """BFCL policy hooks on top of the shared interaction driver."""

    def __init__(
        self,
        tools_info: list[dict[str, Any]],
        tool_parser: ToolCallParser | None = None,
        reasoning_parser: ReasoningParser | None = None,
        max_step_limit: int = 10,
        tito_drift_check_enabled: bool = False,
    ):
        super().__init__(
            tools_info=tools_info,
            tool_parser=tool_parser,
            reasoning_parser=reasoning_parser,
            max_step_limit=max_step_limit,
            tito_drift_check_enabled=tito_drift_check_enabled,
        )
        self._env_adapter: BFCLGymAdapter | None = None

    async def asolve(
        self,
        env,
        rollout_args,
        sampling_params: dict[str, Any],
        initial_messages: list[dict[str, Any]],
        follow_up_questions: list[str],
        max_num_steps: int = 30,
    ):
        self._env_adapter = BFCLGymAdapter(
            env,
            follow_up_questions=follow_up_questions,
            max_step_limit=self.max_step_limit,
        )
        self._env_adapter.reset()

        driver = InteractionDriver(agent=self, hooks=self)
        return await driver.run(
            rollout_args=rollout_args,
            sampling_params=sampling_params,
            initial_messages=initial_messages,
            max_num_steps=max_num_steps,
        )

    def parse_decision(self, response: str) -> AgentDecision:
        is_valid, error_msg = self.reasoning_parser.validate_reasoning_tags(response)
        if not is_valid:
            return ErrorDecision(error=f"Invalid think tag format: {error_msg}")

        tool_parse_text = self.reasoning_parser.strip_reasoning(response)

        is_valid, error_msg = self.tool_parser.validate_tool_call_tags(tool_parse_text)
        if not is_valid:
            return ErrorDecision(error=f"Invalid tool_call tag format: {error_msg}")

        if self.tool_parser.has_tool_call(tool_parse_text):
            parsed_tool_calls = self.tool_parser.parse(tool_parse_text)
            if not parsed_tool_calls:
                return ErrorDecision(
                    error=(
                        "Found <tool_call> tags but failed to parse "
                        "any valid tool calls."
                    )
                )

            errors = [
                tool_call for tool_call in parsed_tool_calls if tool_call.is_error
            ]
            if errors:
                error_messages = [
                    f"Tool '{tool_call.name}': {tool_call.raw}" for tool_call in errors
                ]
                return ErrorDecision(
                    error=f"Tool call parse errors: {'; '.join(error_messages)}"
                )

            bfcl_calls = self._convert_to_bfcl_format(parsed_tool_calls)
            assistant_tool_calls = self._convert_to_assistant_tool_calls(
                parsed_tool_calls
            )
            return ToolCallDecision(
                tool_calls=bfcl_calls,
                assistant_tool_calls=assistant_tool_calls,
            )

        answer = self.reasoning_parser.extract_final_answer(response)
        return AnswerDecision(answer=answer)

    def env_step(self, decision: AgentDecision) -> EnvTransition:
        if self._env_adapter is None:
            raise RuntimeError("BFCL env adapter is not initialized")
        return self._env_adapter.step(decision)

    def on_episode_end(
        self,
        *,
        result,
        total_reward: float,
        turn_rewards: list[float],
        **_: Any,
    ) -> None:
        result.reward = total_reward / max(len(turn_rewards), 1)

    def _convert_to_bfcl_format(
        self,
        tool_calls: list[ToolCallParseResult],
    ) -> list[str]:
        result: list[str] = []
        for tool_call in tool_calls:
            if tool_call.is_error:
                continue
            args_str = ", ".join(
                f"{key}={repr(value)}" for key, value in tool_call.input.items()
            )
            result.append(f"{tool_call.name}({args_str})")
        return result

    def _convert_to_assistant_tool_calls(
        self,
        tool_calls: list[ToolCallParseResult],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            if tool_call.is_error:
                continue
            result.append(
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.input, ensure_ascii=False),
                    },
                }
            )
        return result
