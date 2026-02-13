"""Gym-style adapter for canonical BFCL env orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from t3rl.interaction.types import (
    AgentDecision,
    AnswerDecision,
    EnvTransition,
    ErrorDecision,
    ToolCallDecision,
)


@dataclass(slots=True)
class BFCLGymState:
    """Runtime state for BFCL turn orchestration."""

    current_turn_index: int = 0
    current_turn_attempts: int = 0


class BFCLGymAdapter:
    """Adapter-only gym facade over canonical :class:`t3rl.envs.bfcl.BFCLEnv`."""

    def __init__(
        self,
        env,
        *,
        follow_up_questions: list[str] | None = None,
        max_step_limit: int = 10,
    ):
        self.env = env
        self.max_step_limit = max_step_limit

        self.state = BFCLGymState()
        self._initial_question_queue: list[str] = list(follow_up_questions or [])
        self.question_queue: list[str] = list(self._initial_question_queue)

    def reset(self) -> EnvTransition:
        self.state = BFCLGymState()
        self.question_queue = list(self._initial_question_queue)
        return EnvTransition(observation_messages=[], add_generation_prompt=True)

    def step(self, decision: AgentDecision) -> EnvTransition:
        if isinstance(decision, ErrorDecision):
            return self._step_on_error(decision)
        if isinstance(decision, ToolCallDecision):
            return self._step_on_tool_call(decision)
        if isinstance(decision, AnswerDecision):
            return self._step_on_answer(decision)

        raise TypeError(f"Unsupported decision type: {type(decision)}")

    def _step_on_error(self, decision: ErrorDecision) -> EnvTransition:
        self.state.current_turn_attempts += 1
        metric_updates: dict[str, float | int | list[Any]] = {"format_errors": 1}

        if self.state.current_turn_attempts > self.max_step_limit:
            transition = self._advance_turn_with_zero_score()
            transition.metric_updates.update(metric_updates)
            return transition

        return EnvTransition(
            observation_messages=[{"role": "user", "content": decision.error}],
            add_generation_prompt=True,
            metric_updates=metric_updates,
        )

    def _step_on_tool_call(self, decision: ToolCallDecision) -> EnvTransition:
        execution_results = self.env.execute_calls(decision.tool_calls)
        self.state.current_turn_attempts += 1

        has_error = self._has_execution_error(execution_results)
        metric_updates: dict[str, float | int | list[Any]] = {
            "tool_call_errors" if has_error else "tool_call_successes": 1
        }

        tool_messages = self._build_tool_messages(decision.tool_calls, execution_results)

        if self.state.current_turn_attempts > self.max_step_limit:
            transition = self._advance_turn_with_zero_score()
            transition.metric_updates.update(metric_updates)
            if tool_messages:
                transition.observation_messages = (
                    tool_messages + transition.observation_messages
                )
            transition.info = {
                **transition.info,
                "execution_results": execution_results,
                "tool_execution_error": has_error,
            }
            return transition

        return EnvTransition(
            observation_messages=tool_messages,
            add_generation_prompt=True,
            metric_updates=metric_updates,
            info={
                "execution_results": execution_results,
                "tool_execution_error": has_error,
            },
        )

    def _step_on_answer(self, _: AnswerDecision) -> EnvTransition:
        turn_score = self.env.calculate_turn_score(self.state.current_turn_index)
        score = float(turn_score.get("turn_score", 0.0))

        transition = self._advance_turn_with_score(score)
        transition.reward_delta = score
        transition.info = {**transition.info, "turn_score": turn_score}
        return transition

    def _advance_turn_with_zero_score(self) -> EnvTransition:
        return self._advance_turn(score=0.0, record_user_turn_score=False)

    def _advance_turn_with_score(self, score: float) -> EnvTransition:
        return self._advance_turn(score=score, record_user_turn_score=True)

    def _advance_turn(
        self,
        *,
        score: float,
        record_user_turn_score: bool,
    ) -> EnvTransition:
        self.env.reset_turn()

        metric_updates: dict[str, float | int | list[Any]] = {
            "turn_rewards": [score],
        }
        if record_user_turn_score:
            metric_updates["user_turn_scores"] = [score]

        if self.question_queue:
            next_question = self.question_queue.pop(0)
            self.state.current_turn_index += 1
            self.state.current_turn_attempts = 0
            return EnvTransition(
                observation_messages=[{"role": "user", "content": next_question}],
                add_generation_prompt=True,
                metric_updates=metric_updates,
                info={"advanced_turn": True},
            )

        return EnvTransition(
            observation_messages=[],
            terminated=True,
            add_generation_prompt=False,
            metric_updates=metric_updates,
            info={"advanced_turn": False},
        )

    def _has_execution_error(self, execution_results: list[str]) -> bool:
        error_prefix = "Error during execution:"
        return any(
            isinstance(result, str) and result.startswith(error_prefix)
            for result in execution_results
        )

    def _build_tool_messages(
        self,
        tool_calls: list[str],
        execution_results: list[str],
    ) -> list[dict[str, Any]]:
        tool_messages: list[dict[str, Any]] = []
        for tool_call, result in zip(tool_calls, execution_results, strict=False):
            func_name = tool_call.split("(")[0] if "(" in tool_call else tool_call
            tool_messages.append(
                {
                    "role": "tool",
                    "name": func_name,
                    "content": (
                        json.dumps(result, ensure_ascii=False)
                        if not isinstance(result, str)
                        else result
                    ),
                }
            )
        return tool_messages
