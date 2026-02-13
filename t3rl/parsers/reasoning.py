"""Reasoning parser interfaces, implementations, and registry."""

from __future__ import annotations

import re
from abc import ABC
from typing import Callable


ReasoningParserFactory = Callable[..., "ReasoningParser"]
_REASONING_PARSER_REGISTRY: dict[str, ReasoningParserFactory] = {}


def register_reasoning_parser(
    name: str,
    parser_factory: ReasoningParserFactory | None = None,
    aliases: tuple[str, ...] = (),
):
    """Register a reasoning parser by name.

    Can be used as either:
    - decorator: `@register_reasoning_parser("key", aliases=(...))`
    - direct call: `register_reasoning_parser("key", ParserClass, aliases=(...))`
    """

    def _register(factory: ReasoningParserFactory):
        keys = {name.lower(), *(alias.lower() for alias in aliases)}
        for key in keys:
            _REASONING_PARSER_REGISTRY[key] = factory
        return factory

    if parser_factory is not None:
        return _register(parser_factory)
    return _register


def get_registered_reasoning_parsers() -> list[str]:
    return sorted(_REASONING_PARSER_REGISTRY.keys())


def create_reasoning_parser(name: str | None = None, **kwargs) -> "ReasoningParser":
    parser_key = (name or "instruct2507").lower()
    factory = _REASONING_PARSER_REGISTRY.get(parser_key)
    if factory is None:
        available = ", ".join(get_registered_reasoning_parsers())
        raise ValueError(
            f"Unknown reasoning parser '{name}'. Available parsers: {available}"
        )
    return factory(**kwargs)


class ReasoningParser(ABC):
    """Base class for reasoning-thought parsing and stripping."""

    @property
    def tag_pair(self) -> tuple[str | None, str | None]:
        return None, None

    def validate_reasoning_tags(self, text: str) -> tuple[bool, str | None]:
        return True, None

    def strip_reasoning(self, text: str) -> str:
        return text

    def extract_reasoning(self, text: str) -> str | None:
        return None

    def has_reasoning(self, text: str) -> bool:
        return self.extract_reasoning(text) is not None

    def extract_final_answer(self, text: str) -> str:
        return self.strip_reasoning(text)


class TaggedReasoningParser(ReasoningParser):
    """Parse and strip reasoning content enclosed by explicit tag pairs."""

    def __init__(self, start_token: str | None = "<thinking>", end_token: str | None = "</thinking>"):
        self.start_token = start_token
        self.end_token = end_token

        if start_token and end_token:
            self._reasoning_pattern: re.Pattern[str] | None = re.compile(
                rf"{re.escape(start_token)}(.*?){re.escape(end_token)}",
                re.DOTALL,
            )
            self._reasoning_block_pattern: re.Pattern[str] | None = re.compile(
                rf"{re.escape(start_token)}.*?{re.escape(end_token)}",
                re.DOTALL,
            )
        else:
            self._reasoning_pattern = None
            self._reasoning_block_pattern = None

    @property
    def tag_pair(self) -> tuple[str | None, str | None]:
        return self.start_token, self.end_token

    def validate_reasoning_tags(self, text: str) -> tuple[bool, str | None]:
        if not self.start_token or not self.end_token:
            return True, None

        start_count = text.count(self.start_token)
        end_count = text.count(self.end_token)

        if start_count != end_count:
            return False, f"Mismatched think tags: {start_count} opening, {end_count} closing"

        if start_count > 0:
            first_start = text.find(self.start_token)
            first_end = text.find(self.end_token)
            if first_end < first_start:
                return False, "Think end tag appears before start tag"

        return True, None

    def strip_reasoning(self, text: str) -> str:
        if self._reasoning_block_pattern is None:
            return text
        return self._reasoning_block_pattern.sub("", text)

    def extract_reasoning(self, text: str) -> str | None:
        if self._reasoning_pattern is None:
            return None
        match = self._reasoning_pattern.search(text)
        return match.group(1).strip() if match else None

    def extract_final_answer(self, text: str) -> str:
        if not self.start_token or not self.end_token:
            return text

        if self.start_token in text and self.end_token in text:
            think_end_pos = text.rfind(self.end_token)
            if think_end_pos != -1:
                return text[think_end_pos + len(self.end_token) :].strip()

        return text


@register_reasoning_parser(
    name="instruct2507",
    aliases=("i2507", "qwen2507", "thinking", "default"),
)
class Instruct2507ReasoningParser(TaggedReasoningParser):
    """Reasoning parser for Qwen3 Instruct-2507 style prompts using `<thinking>` tags."""

    def __init__(self, start_token: str = "<thinking>", end_token: str = "</thinking>"):
        super().__init__(start_token=start_token, end_token=end_token)


@register_reasoning_parser(
    name="deepseek_r1",
    aliases=("deepseek-r1", "r1", "dsr1"),
)
class DeepSeekR1ReasoningParser(ReasoningParser):
    """Reasoning parser for DeepSeek-R1 style outputs.

    DeepSeek-R1 may emit reasoning without a leading `<think>` tag, but still end with
    `</think>`. Examples:
      - `reasoning ... </think> final answer`
      - `<think>reasoning ... </think> final answer`
    """

    def __init__(self, start_token: str = "<think>", end_token: str = "</think>"):
        self.start_token = start_token
        self.end_token = end_token

    @property
    def tag_pair(self) -> tuple[str | None, str | None]:
        return self.start_token, self.end_token

    def validate_reasoning_tags(self, text: str) -> tuple[bool, str | None]:
        if not self.end_token:
            return True, None

        end_count = text.count(self.end_token)
        start_count = text.count(self.start_token) if self.start_token else 0

        if end_count == 0:
            if start_count > 0:
                return False, f"Mismatched think tags: {start_count} opening, 0 closing"
            return True, None

        if self.start_token and start_count not in {end_count, end_count - 1}:
            return False, f"Mismatched think tags: {start_count} opening, {end_count} closing"

        if self.start_token and start_count == end_count and start_count > 0:
            first_start = text.find(self.start_token)
            first_end = text.find(self.end_token)
            if first_end < first_start:
                return False, "Think end tag appears before start tag"

        return True, None

    def extract_reasoning(self, text: str) -> str | None:
        end_pos = text.find(self.end_token)
        if end_pos < 0:
            return None

        reasoning_part = text[:end_pos]
        if self.start_token and self.start_token in reasoning_part:
            reasoning_part = reasoning_part.split(self.start_token, maxsplit=1)[1]

        reasoning_part = reasoning_part.strip()
        return reasoning_part if reasoning_part else None

    def strip_reasoning(self, text: str) -> str:
        end_pos = text.find(self.end_token)
        if end_pos < 0:
            return text
        return text[end_pos + len(self.end_token) :].lstrip()

    def extract_final_answer(self, text: str) -> str:
        return self.strip_reasoning(text)


@register_reasoning_parser(
    name="qwen_think",
    aliases=("qwen3-thinking", "think"),
)
def _create_qwen_think_parser(**kwargs) -> ReasoningParser:
    start_token = kwargs.pop("start_token", "<think>")
    end_token = kwargs.pop("end_token", "</think>")
    if kwargs:
        unknown = ", ".join(sorted(kwargs.keys()))
        raise ValueError(f"Unsupported kwargs for qwen_think parser: {unknown}")
    return TaggedReasoningParser(start_token=start_token, end_token=end_token)
