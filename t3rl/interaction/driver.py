"""Shared multi-turn interaction driver."""

from __future__ import annotations

import logging
from typing import Any, Protocol

try:
    from slime.rollout.sglang_rollout import GenerateState
except ModuleNotFoundError:
    GenerateState = None

try:
    from slime.utils.http_utils import post
except ModuleNotFoundError:

    async def post(*args, **kwargs):
        raise ImportError(
            "slime.utils.http_utils.post is unavailable. "
            "Please ensure slime runtime dependencies are installed."
        )

from t3rl.interaction.types import (
    AgentDecision,
    DecisionKind,
    DriverStepResult,
    EnvTransition,
    InteractionMetrics,
    InteractionResult,
    Status,
    TokenTrajectory,
)

logger = logging.getLogger(__name__)


class DriverHooks(Protocol):
    """Hooks required by the shared interaction driver."""

    def parse_decision(self, response: str) -> AgentDecision:
        """Parse raw assistant response into a typed decision."""

    def env_step(self, decision: AgentDecision) -> EnvTransition:
        """Apply parsed decision to environment adapter and return transition."""


class InteractionDriver:
    """Generic interaction loop with shared TITO/logprob bookkeeping."""

    def __init__(self, agent, hooks: DriverHooks):
        self.agent = agent
        self.hooks = hooks

    async def run(
        self,
        *,
        rollout_args,
        sampling_params: dict[str, Any],
        initial_messages: list[dict[str, Any]],
        max_num_steps: int,
    ) -> InteractionResult:
        if GenerateState is None:
            raise ImportError(
                "slime.rollout.sglang_rollout.GenerateState is unavailable. "
                "Please ensure slime runtime dependencies are installed."
            )

        state = GenerateState(rollout_args)
        url = f"http://{rollout_args.sglang_router_ip}:{rollout_args.sglang_router_port}/generate"

        token_messages = list(initial_messages)
        log_messages = list(initial_messages)
        if not token_messages:
            raise ValueError("initial_messages cannot be empty")

        prompt_text, prompt_token_ids = self.agent._prepare_prompt_tokens(
            state,
            token_messages,
        )

        trajectory = TokenTrajectory(accumulated_tokens=list(prompt_token_ids))
        turn_rewards: list[float] = []
        metrics = InteractionMetrics()
        total_reward = 0.0
        last_info: dict[str, Any] = {}

        result = InteractionResult(
            prompt=prompt_text,
            reward=0.0,
            messages=[],
            info={},
        )

        for step_index in range(max_num_steps):
            payload = {
                "input_ids": trajectory.accumulated_tokens,
                "sampling_params": sampling_params,
                "return_logprob": True,
            }

            try:
                output = await post(url, payload)
            except Exception as exc:
                logger.warning("LLM call failed: %s", exc)
                result.status = Status.ABORTED
                break

            if (
                output.get("meta_info", {}).get("finish_reason", {}).get("type")
                == "abort"
            ):
                result.status = Status.ABORTED
                break

            response = output.get("text", "")
            if response.endswith("<|im_end|>"):
                response = response[:-10]

            output_token_logprobs = output.get("meta_info", {}).get(
                "output_token_logprobs"
            )
            if output_token_logprobs is None:
                raise ValueError(
                    "No output_token_logprobs in SGLang response with return_logprob=True"
                )

            assistant_tokens: list[int] = []
            assistant_log_probs: list[float] = []
            for token_meta in output_token_logprobs:
                if not isinstance(token_meta, (list, tuple)) or len(token_meta) < 2:
                    raise ValueError(
                        f"Invalid output_token_logprobs item: {token_meta!r}"
                    )
                assistant_log_probs.append(float(token_meta[0]))
                assistant_tokens.append(int(token_meta[1]))

            trajectory.append_response(
                assistant_tokens,
                loss_mask=1,
                log_probs=assistant_log_probs,
            )

            decision = self.hooks.parse_decision(response)

            assistant_message = {"role": "assistant", "content": response}
            token_messages.append(assistant_message)

            assistant_log_message = dict(assistant_message)
            if decision.kind == DecisionKind.TOOL_CALL and getattr(
                decision, "assistant_tool_calls", None
            ):
                assistant_log_message["tool_calls"] = decision.assistant_tool_calls
            log_messages.append(assistant_log_message)

            on_assistant_logged = getattr(self.hooks, "on_assistant_logged", None)
            if callable(on_assistant_logged):
                on_assistant_logged(
                    response=response,
                    decision=decision,
                    step_index=step_index,
                )

            transition = self.hooks.env_step(decision)

            self._apply_metric_updates(metrics, turn_rewards, transition.metric_updates)
            total_reward += float(transition.reward_delta)
            if transition.info:
                last_info = dict(transition.info)

            if transition.observation_messages:
                env_token_ids = self.agent._encode_messages_delta(
                    state.tokenizer,
                    token_messages,
                    transition.observation_messages,
                    add_generation_prompt=transition.add_generation_prompt,
                )
                token_messages.extend(transition.observation_messages)
                log_messages.extend(
                    dict(msg) for msg in transition.observation_messages
                )
                trajectory.append_response(env_token_ids, loss_mask=0)
            elif transition.add_generation_prompt and not (
                transition.terminated or transition.truncated
            ):
                gen_prompt_tokens = self.agent._get_generation_prompt_tokens(
                    state.tokenizer,
                    token_messages,
                )
                trajectory.append_response(gen_prompt_tokens, loss_mask=0)

            step_result = DriverStepResult(
                step_index=step_index,
                response=response,
                decision=decision,
                transition=transition,
            )
            on_step = getattr(self.hooks, "on_step", None)
            if callable(on_step):
                on_step(step_result)

            if transition.terminated:
                result.status = Status.COMPLETED
                break
            if transition.truncated:
                result.status = Status.TRUNCATED
                break

        if result.status not in {Status.COMPLETED, Status.ABORTED, Status.TRUNCATED}:
            result.status = Status.TRUNCATED

        if self.agent.tito_drift_check_enabled:
            self.agent._validate_token_trajectory(
                state.tokenizer,
                token_messages,
                trajectory.accumulated_tokens,
            )

        result.reward = total_reward
        result.messages = log_messages
        result.info = last_info
        result.loss_mask = trajectory.loss_masks
        result.tokens = trajectory.accumulated_tokens
        result.response_length = len(trajectory.loss_masks)
        result.turn_rewards = turn_rewards
        result.metrics = metrics
        result.rollout_log_probs = trajectory.rollout_log_probs

        assert len(trajectory.rollout_log_probs) == len(trajectory.loss_masks), (
            f"TITO violation: rollout_log_probs({len(trajectory.rollout_log_probs)}) != "
            f"loss_mask({len(trajectory.loss_masks)})"
        )
        assert len(trajectory.response_token_ids) == len(trajectory.loss_masks), (
            f"TITO violation: response_token_ids({len(trajectory.response_token_ids)}) != "
            f"loss_mask({len(trajectory.loss_masks)})"
        )

        result.response = "".join(
            message.get("content", "")
            for message in log_messages
            if message.get("role") == "assistant"
        )

        on_episode_end = getattr(self.hooks, "on_episode_end", None)
        if callable(on_episode_end):
            on_episode_end(
                result=result,
                total_reward=total_reward,
                turn_rewards=turn_rewards,
                metrics=metrics,
                last_info=last_info,
            )

        return result

    def _apply_metric_updates(
        self,
        metrics: InteractionMetrics,
        turn_rewards: list[float],
        updates: dict[str, float | int | list[Any]],
    ) -> None:
        if not updates:
            return

        for key, value in updates.items():
            if key == "tool_call_successes":
                metrics.tool_call_successes += int(value)
            elif key == "tool_call_errors":
                metrics.tool_call_errors += int(value)
            elif key == "format_errors":
                metrics.format_errors += int(value)
            elif key == "user_turn_scores":
                if isinstance(value, list):
                    metrics.user_turn_scores.extend(float(item) for item in value)
                else:
                    metrics.user_turn_scores.append(float(value))
            elif key == "turn_rewards":
                if isinstance(value, list):
                    turn_rewards.extend(float(item) for item in value)
                else:
                    turn_rewards.append(float(value))
