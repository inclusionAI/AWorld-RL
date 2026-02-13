"""Base trainable agent for shared TITO/tokenization logic."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

try:
    from slime.rollout.sglang_rollout import GenerateState
except ModuleNotFoundError:
    GenerateState = Any

from t3rl.interaction.types import InteractionResult
from t3rl.parsers.reasoning import ReasoningParser, create_reasoning_parser
from t3rl.parsers.tool_call import ToolCallParser, create_tool_call_parser

logger = logging.getLogger(__name__)


DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


class BaseTrainableAgent(ABC):
    """Shared trainable-agent base with tokenizer/TITO utilities."""

    def __init__(
        self,
        tools_info: list[dict[str, Any]],
        tool_parser: ToolCallParser | None = None,
        reasoning_parser: ReasoningParser | None = None,
        max_step_limit: int = 10,
        tito_drift_check_enabled: bool = False,
    ):
        self.tools_info = tools_info
        self.tool_parser = tool_parser or create_tool_call_parser("hermes")
        self.reasoning_parser = reasoning_parser or create_reasoning_parser(
            "instruct2507"
        )
        self.max_step_limit = max_step_limit
        self.think_start_token, self.think_end_token = self.reasoning_parser.tag_pair
        self.tito_drift_check_enabled = tito_drift_check_enabled

    @abstractmethod
    async def asolve(self, *args, **kwargs) -> InteractionResult:
        """Execute one full environment interaction rollout."""
        raise NotImplementedError

    def _prepare_prompt_tokens(
        self, state: GenerateState, messages: list[dict[str, Any]]
    ) -> tuple[str, list[int]]:
        """Prepare prompt text and tokenize it.

        Args:
            state: GenerateState with tokenizer.
            messages: Initial messages (system + user).

        Returns:
            Tuple of (prompt_text, prompt_token_ids).
        """
        prompt_text = state.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=self.tools_info,
        )
        prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)[
            "input_ids"
        ]
        return prompt_text, prompt_token_ids

    def _encode_single_message(
        self,
        tokenizer,
        message: dict[str, Any],
        add_generation_prompt: bool = False,
    ) -> list[int]:
        """
        Encode a single message to tokens using a dummy-prefix trim setup.

        NOTE: This helper is retained for compatibility. Main rollout code uses
        `_encode_messages_delta` for context-aware token delta extraction.
        """
        message_tokens = self._encode_messages_delta(
            tokenizer,
            DUMMY_MESSAGES,
            [message],
            add_generation_prompt=add_generation_prompt,
        )

        # Remove BOS token if present at the beginning
        bos_id = tokenizer.bos_token_id
        if bos_id is not None and message_tokens and message_tokens[0] == bos_id:
            message_tokens = message_tokens[1:]

        return message_tokens

    def _get_generation_prompt_tokens(
        self,
        tokenizer,
        messages: list[dict[str, Any]],
    ) -> list[int]:
        """
        Get the generation prompt tokens (e.g., "<|im_start|>assistant\n").

        This is added after tool results to prepare for the next assistant turn.

        Args:
            tokenizer: Tokenizer instance.
            messages: Current conversation messages used as context.

        Returns:
            List of token IDs for the generation prompt.
        """
        return self._encode_messages_delta(
            tokenizer,
            messages,
            [],
            add_generation_prompt=True,
        )

    def _encode_messages_delta(
        self,
        tokenizer,
        base_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> list[int]:
        """Encode token delta from `base_messages` to `base_messages + new_messages`.

        This method is context-aware and works for templates where adding new
        messages changes rendering details across message boundaries (e.g.,
        consecutive tool-message grouping).
        """
        base_text = tokenizer.apply_chat_template(
            base_messages,
            tools=self.tools_info,
            tokenize=False,
            add_generation_prompt=False,
        )
        full_text = tokenizer.apply_chat_template(
            base_messages + new_messages,
            tools=self.tools_info,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

        base_tokens = tokenizer.encode(base_text, add_special_tokens=False)
        full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

        return self._extract_suffix_tokens(
            tokenizer,
            prefix_text=base_text,
            full_text=full_text,
            prefix_tokens=base_tokens,
            full_tokens=full_tokens,
        )

    def _extract_suffix_tokens(
        self,
        tokenizer,
        *,
        prefix_text: str,
        full_text: str,
        prefix_tokens: list[int],
        full_tokens: list[int],
    ) -> list[int]:
        """Extract suffix tokens from a (prefix, full) template pair."""
        if (
            len(full_tokens) >= len(prefix_tokens)
            and full_tokens[: len(prefix_tokens)] == prefix_tokens
        ):
            return full_tokens[len(prefix_tokens) :]

        if full_text.startswith(prefix_text):
            suffix_text = full_text[len(prefix_text) :]
            if not suffix_text:
                return []
            return tokenizer.encode(suffix_text, add_special_tokens=False)

        # Last-resort token fallback: strip common token prefix.
        common_prefix_length = 0
        for prefix_token, full_token in zip(prefix_tokens, full_tokens, strict=False):
            if prefix_token != full_token:
                break
            common_prefix_length += 1

        logger.warning(
            "Chat-template prefix mismatch while extracting token delta "
            "(prefix_tokens=%d, full_tokens=%d, common_prefix=%d).",
            len(prefix_tokens),
            len(full_tokens),
            common_prefix_length,
        )
        return full_tokens[common_prefix_length:]

    def _validate_token_trajectory(
        self,
        tokenizer,
        messages: list[dict[str, Any]],
        accumulated_tokens: list[int],
    ) -> None:
        """Validate trajectory tokens against chat-template reconstruction.

        This check is optional and controlled by tito_drift_check_* config.
        """
        expected_text_without_prompt = tokenizer.apply_chat_template(
            messages,
            tools=self.tools_info,
            tokenize=False,
            add_generation_prompt=False,
        )
        expected_tokens_without_prompt = tokenizer.encode(
            expected_text_without_prompt,
            add_special_tokens=False,
        )

        if expected_tokens_without_prompt == accumulated_tokens:
            return

        # Some trajectories may intentionally end with generation prompt tokens
        # after a tool call. Validate that variant as a compatibility path.
        expected_text_with_prompt = tokenizer.apply_chat_template(
            messages,
            tools=self.tools_info,
            tokenize=False,
            add_generation_prompt=True,
        )
        expected_tokens_with_prompt = tokenizer.encode(
            expected_text_with_prompt,
            add_special_tokens=False,
        )

        if expected_tokens_with_prompt == accumulated_tokens:
            return

        def count_mismatches(expected_tokens: list[int]) -> int:
            pair_mismatches = sum(
                1
                for actual_token, expected_token in zip(
                    accumulated_tokens,
                    expected_tokens,
                    strict=False,
                )
                if actual_token != expected_token
            )
            return pair_mismatches + abs(len(accumulated_tokens) - len(expected_tokens))

        reference_label, reference_tokens = min(
            (
                ("add_generation_prompt=False", expected_tokens_without_prompt),
                ("add_generation_prompt=True", expected_tokens_with_prompt),
            ),
            key=lambda item: count_mismatches(item[1]),
        )

        mismatch_details: list[str] = []
        max_report = 32
        mismatch_count = 0
        for index, (actual_token, expected_token) in enumerate(
            zip(accumulated_tokens, reference_tokens, strict=False)
        ):
            if actual_token == expected_token:
                continue
            mismatch_count += 1
            if len(mismatch_details) < max_report:
                mismatch_details.append(
                    f"idx={index}: actual={actual_token}, expected={expected_token}"
                )

        if len(accumulated_tokens) != len(reference_tokens):
            mismatch_count += abs(len(accumulated_tokens) - len(reference_tokens))
            if len(mismatch_details) < max_report:
                mismatch_details.append(
                    f"length mismatch: actual={len(accumulated_tokens)}, expected={len(reference_tokens)}"
                )

        summary = (
            "TITO drift detected between accumulated trajectory tokens and "
            "chat-template reconstruction "
            f"(closest_reference={reference_label}, mismatches={mismatch_count})."
        )
        detail_text = (
            " | ".join(mismatch_details)
            if mismatch_details
            else "(no per-index details captured)"
        )

        logger.warning("%s %s", summary, detail_text)

    def _get_token_delta(
        self, tokenizer, messages: list[dict[str, Any]]
    ) -> tuple[list[int], list[int]]:
        """
        Calculate token delta for multi-turn conversations.

        NOTE: This method is kept for backward compatibility but is no longer
        used in the main TITO flow. Use _encode_single_message instead.

        Args:
            tokenizer: Tokenizer instance.
            messages: Conversation messages.

        Returns:
            Tuple of (token_ids, loss_mask).
        """
        curr = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False, tools=self.tools_info
        )
        token_ids = []
        loss_mask = []

        if messages[-1]["role"] == "assistant":
            # Assistant response: loss_mask=1
            prev = tokenizer.apply_chat_template(
                messages[:-1],
                add_generation_prompt=True,
                tokenize=False,
                tools=self.tools_info,
            )
            new_tokens = tokenizer.encode(curr[len(prev) :], add_special_tokens=False)
            token_ids = new_tokens
            loss_mask = [1] * len(new_tokens)
        else:
            # User/Tool response: loss_mask=0
            prev = tokenizer.apply_chat_template(
                messages[:-1],
                add_generation_prompt=False,
                tokenize=False,
                tools=self.tools_info,
            )
            new_tokens = tokenizer.encode(curr[len(prev) :], add_special_tokens=False)
            token_ids = new_tokens
            loss_mask = [0] * len(new_tokens)

        return token_ids, loss_mask
