"""
BFCL-Slime Integration Entry Point

This module provides the main entry point for slime training with BFCL environment.
It implements the custom_generate_function interface expected by slime.

Usage:
    In training script, specify:
    --custom-generate-function-path t3rl.rollout.bfcl.generate
"""

from __future__ import annotations

import json
import logging
from typing import Any

from slime.utils.types import Sample

from t3rl.agents.bfcl import BFCLTrainableAgent
from t3rl.interaction.types import InteractionResult
from t3rl.envs.bfcl import BFCLEnv
from t3rl.eval.common import log_rollout_metric_dict
from t3rl.parsers import create_reasoning_parser, create_tool_call_parser
from t3rl.rollout.sample_adapters import map_status

logger = logging.getLogger(__name__)


def res_to_sample(res: InteractionResult, sample: Sample, ground_truth: list) -> Sample:
    """
    Convert InteractionResult to Sample format for slime training.

    Computes reward as 'progress' (sum of relevant user turn scores / count)
    and stores additional metrics for tensorboard logging.

    Args:
        res: InteractionResult from BFCL agent.
        sample: Original sample to update.
        ground_truth: Ground truth for computing metrics.

    Returns:
        Updated Sample object for slime training.
    """
    sample.status = map_status(res.status)

    # Extract metrics from InteractionResult
    metrics = res.metrics
    total_interaction_rounds = metrics.total_interactions

    # Compute progress (main reward signal) from user turn scores
    user_scores = metrics.user_turn_scores
    progress = (
        sum(1 for s in user_scores if s > 0) / len(user_scores) if user_scores else 0.0
    )

    # Compute format_reward
    error_format = metrics.format_errors
    format_reward = (
        ((total_interaction_rounds - error_format) / total_interaction_rounds)
        if total_interaction_rounds > 0
        else 0.0
    )

    # Accuracy: 1.0 if all user turns are correct (all scores > 0)
    accuracy = 1.0 if user_scores and all(s > 0 for s in user_scores) else 0.0

    # Update sample fields
    sample.tokens = res.tokens
    sample.response = res.response
    sample.reward = progress  # Use progress as the main reward
    sample.loss_mask = res.loss_mask
    sample.response_length = res.response_length

    # Set rollout_log_probs for TIS/mismatch metrics
    # This enables --get-mismatch-metrics and --custom-tis-function-path
    if res.rollout_log_probs:
        sample.rollout_log_probs = res.rollout_log_probs

    # Store additional info in metadata for logging
    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["turn_rewards"] = res.turn_rewards
    sample.metadata["user_turn_scores"] = user_scores  # Store actual scores
    sample.metadata["messages"] = res.messages

    # Store metrics for tensorboard (will be aggregated by custom log function)
    sample.metadata["bfcl_metrics"] = {
        "progress": progress,
        "accuracy": accuracy,
        "format_reward": format_reward,
    }

    logger.debug(
        f"res_to_sample: response_length={res.response_length}, "
        f"loss_mask_len={len(res.loss_mask)}, "
        f"tokens_len={len(res.tokens)}, "
        f"reward={progress}, accuracy={accuracy}"
    )

    return sample


async def generate(args, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """
    Generate a complete agent-environment interaction trajectory for BFCL.

    This is the main entry point for slime training. It creates a BFCL
    environment, initializes a trainable agent, and executes a full interaction
    trajectory. The result is converted to slime's Sample format for training.

    Args:
        args: Rollout arguments from slime training pipeline.
              Custom config parameters (from --custom-config-path) are available as attributes:
              - reasoning_parser: Reasoning parser key (default: "instruct2507")
              - tool_call_parser: Tool-call parser key (default: "hermes")
              - max_step_limit: Maximum steps per turn (default: 10)
              - max_num_steps: Maximum total interaction steps (default: 30)
        sample: Sample containing task data in metadata.bfcl_task field.
        sampling_params: LLM sampling parameters.

    Returns:
        Sample object containing the complete interaction trajectory.

    Raises:
        AssertionError: If partial rollout is requested (not supported).
    """
    # Validate arguments
    assert not args.partial_rollout, (
        "Partial rollout is not supported for BFCL interactions."
    )

    # Read configuration from args (loaded via --custom-config-path)
    reasoning_parser_name = getattr(args, "reasoning_parser", "instruct2507")
    tool_call_parser_name = getattr(args, "tool_call_parser", "hermes")
    max_step_limit = getattr(args, "max_step_limit", 10)
    max_num_steps = getattr(args, "max_num_steps", 30)
    tito_drift_check_enabled = getattr(args, "tito_drift_check_enabled", False)

    # Extract task data from sample
    task_data = _extract_task_data(sample)

    logger.info(f"Starting BFCL interaction for task {task_data.get('id', 'unknown')}")

    # Parse initial_config
    initial_config = task_data.get("initial_config", {})
    if isinstance(initial_config, str):
        initial_config = json.loads(initial_config)

    # Create BFCLEnv instance
    env = BFCLEnv(
        initial_config=initial_config,
        involved_classes=task_data.get("involved_classes", []),
        ground_truth_calls=task_data.get("ground_truth", []),
        long_context="long_context" in task_data.get("id", "")
        or "composite" in task_data.get("id", ""),
    )

    # Get tools info from metadata (single source of truth)
    # Note: tools are stored in metadata.tools, not in bfcl_task.tools
    tools_info = []
    if sample.metadata and isinstance(sample.metadata, dict):
        tools_info = sample.metadata.get("tools", [])

    reasoning_parser = create_reasoning_parser(reasoning_parser_name)
    tool_parser = create_tool_call_parser(
        tool_call_parser_name, defined_tools=tools_info
    )

    # Create trainable agent with config from args
    agent = BFCLTrainableAgent(
        tools_info=tools_info,
        tool_parser=tool_parser,
        reasoning_parser=reasoning_parser,
        max_step_limit=max_step_limit,
        tito_drift_check_enabled=tito_drift_check_enabled,
    )

    # Extract initial messages from sample.prompt
    initial_messages = _extract_initial_messages(sample)

    # Get follow-up questions (questions after the first one)
    follow_up_questions = list(task_data.get("processed_question", []))

    # Execute agent-environment interaction
    interaction_result = await agent.asolve(
        env=env,
        rollout_args=args,
        sampling_params=sampling_params,
        initial_messages=initial_messages,
        follow_up_questions=follow_up_questions,
        max_num_steps=max_num_steps,
    )

    # Get ground truth for metrics computation
    ground_truth = task_data.get("ground_truth", [])

    # Convert to slime Sample format
    result_sample = res_to_sample(interaction_result, sample, ground_truth)

    logger.info(
        f"Finished BFCL interaction for task {task_data.get('id', 'unknown')}, "
        f"reward={result_sample.reward:.2f}, status={interaction_result.status.name}"
    )

    return result_sample


def _extract_task_data(sample: Sample) -> dict[str, Any]:
    """
    Extract BFCL task data from sample.

    The task data is stored in sample.metadata.bfcl_task.
    Note: slime's Dataset class only loads prompt, label, metadata, and tools fields.
    The extra_info field is NOT loaded, so all task data must be in metadata.

    Args:
        sample: Sample object.

    Returns:
        Task data dictionary.
    """
    # Check metadata.bfcl_task (primary location)
    if sample.metadata and isinstance(sample.metadata, dict):
        metadata_dict: dict[str, Any] = sample.metadata
        if "bfcl_task" in metadata_dict:
            task = metadata_dict["bfcl_task"]
            if isinstance(task, dict):
                return task
        # Check if metadata itself contains task data (fallback)
        if "initial_config" in metadata_dict:
            return metadata_dict

    # Try extra_info as fallback (for backward compatibility)
    if hasattr(sample, "extra_info") and sample.extra_info:
        extra_info = sample.extra_info
        if isinstance(extra_info, dict):
            extra_dict = {str(k): v for k, v in extra_info.items()}
            if "bfcl_task" in extra_dict:
                task = extra_dict["bfcl_task"]
                if isinstance(task, dict):
                    return {str(k): v for k, v in task.items()}
            if "initial_config" in extra_dict:
                return extra_dict

    # Try parsing prompt as JSON (last resort)
    if sample.prompt:
        try:
            if isinstance(sample.prompt, str):
                data = json.loads(sample.prompt)
                if isinstance(data, dict) and "initial_config" in data:
                    return data
        except json.JSONDecodeError:
            pass

    # Return empty dict if nothing found
    logger.warning("Could not extract BFCL task data from sample")
    return {}


def _extract_initial_messages(sample: Sample) -> list[dict[str, Any]]:
    """
    Extract initial conversation messages from sample.prompt.

    The prompt field contains the initial messages (system + first user question)
    that were set during data preprocessing.

    Args:
        sample: Sample object.

    Returns:
        List of initial messages.

    Raises:
        ValueError: If no valid messages can be extracted.
    """
    if sample.prompt:
        # If prompt is already a list of messages
        if isinstance(sample.prompt, list):
            messages = sample.prompt
            if messages and isinstance(messages[0], dict):
                return messages

        # If prompt is a JSON string
        if isinstance(sample.prompt, str):
            try:
                parsed = json.loads(sample.prompt)
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    return parsed
            except json.JSONDecodeError:
                # Treat as plain text user message
                return [{"role": "user", "content": sample.prompt}]

    raise ValueError(
        "Could not extract initial messages from sample.prompt. "
        "Expected a list of message dicts or a JSON string."
    )


def eval_rollout_with_metrics(args, rollout_id, data_source, evaluation=False):
    """Wrap slime eval rollout and guarantee a mutable metrics dict.

    This follows slime's intended logging flow: custom eval log hook only mutates
    `extra_metrics`, and the framework performs unified logging once.
    """
    from slime.rollout.sglang_rollout import generate_rollout

    output = generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
    if evaluation and getattr(output, "metrics", None) is None:
        output.metrics = {}
    return output


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """
    Reward function for BFCL tasks.

    This function is called by slime to compute rewards. For BFCL,
    the reward is already computed during generation and stored in sample.reward.
    The reward is computed as 'progress' = sum(successful_user_turns) / total_user_turns.

    Args:
        args: Rollout arguments.
        sample: Sample with computed reward.
        **kwargs: Additional arguments.

    Returns:
        Reward value (progress score).
    """
    # Reward is already computed during generation as 'progress'
    if sample.reward is not None:
        return sample.reward

    # Fallback: compute from user_turn_scores if available
    if sample.metadata and "user_turn_scores" in sample.metadata:
        user_scores = sample.metadata["user_turn_scores"]
        if user_scores:
            return sum(1 for s in user_scores if s > 0) / len(user_scores)

    # Default reward for failed samples
    return 0.0


def log_rollout_data(
    rollout_id, args, samples, rollout_extra_metrics, rollout_time
) -> bool:
    """Aggregate BFCL metrics for rollout tensorboard logging."""
    return log_rollout_metric_dict(
        samples,
        rollout_extra_metrics,
        metrics_key="bfcl_metrics",
        metric_keys=["progress", "accuracy", "format_reward"],
    )
