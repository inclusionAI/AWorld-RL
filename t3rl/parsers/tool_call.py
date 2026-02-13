"""Tool-call parser interfaces, implementations, and registry."""

from __future__ import annotations

import ast
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

UNKNOWN_TOOL_NAME = "unknown_tool"

ToolCallParserFactory = Callable[..., "ToolCallParser"]
_TOOL_CALL_PARSER_REGISTRY: dict[str, ToolCallParserFactory] = {}


def register_tool_call_parser(
    name: str,
    parser_factory: ToolCallParserFactory | None = None,
    aliases: tuple[str, ...] = (),
):
    """Register a tool-call parser by name.

    Can be used as either:
    - decorator: `@register_tool_call_parser("key", aliases=(...))`
    - direct call: `register_tool_call_parser("key", ParserClass, aliases=(...))`
    """

    def _register(factory: ToolCallParserFactory):
        keys = {name.lower(), *(alias.lower() for alias in aliases)}
        for key in keys:
            _TOOL_CALL_PARSER_REGISTRY[key] = factory
        return factory

    if parser_factory is not None:
        return _register(parser_factory)
    return _register


def get_registered_tool_call_parsers() -> list[str]:
    return sorted(_TOOL_CALL_PARSER_REGISTRY.keys())


def create_tool_call_parser(name: str | None = None, **kwargs) -> "ToolCallParser":
    parser_key = (name or "hermes").lower()
    factory = _TOOL_CALL_PARSER_REGISTRY.get(parser_key)
    if factory is None:
        available = ", ".join(get_registered_tool_call_parsers())
        raise ValueError(
            f"Unknown tool-call parser '{name}'. Available parsers: {available}"
        )
    return factory(**kwargs)


@dataclass(frozen=True, slots=True)
class ToolCallParseResult:
    """A parsed tool call request."""

    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    raw: str | None = None

    @property
    def is_error(self) -> bool:
        return self.raw is not None

    @property
    def payload(self) -> str:
        if self.is_error:
            return self.raw or ""
        return json.dumps(self.input)


class ToolCallParser(ABC):
    """Base class for tool-call parsers."""

    @property
    def message_separator(self) -> str:
        return ""

    @abstractmethod
    def parse(self, text: str) -> list[ToolCallParseResult]:
        ...

    @abstractmethod
    def has_tool_call(self, text: str) -> bool:
        ...

    def validate_tool_call_tags(self, text: str) -> tuple[bool, str | None]:
        """Validate tag balance/shape for tool-call markers."""
        return True, None

    def __call__(self, text: str) -> list[dict[str, Any]]:
        results = self.parse(text)
        return [
            {"id": tool_call.id, "name": tool_call.name, "input": tool_call.input}
            for tool_call in results
            if not tool_call.is_error
        ]


def infer_type_from_python_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return "string"


def infer_type_from_json_schema(schema: dict[str, Any]) -> str | None:
    """Infer canonical JSON type from a schema object."""
    if not isinstance(schema, dict):
        return None

    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return schema_type
    if isinstance(schema_type, list):
        for candidate in schema_type:
            if isinstance(candidate, str) and candidate != "null":
                return candidate

    for compound_key in ("anyOf", "oneOf"):
        variants = schema.get(compound_key)
        if isinstance(variants, list):
            for variant in variants:
                inferred = infer_type_from_json_schema(variant)
                if inferred:
                    return inferred

    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for variant in all_of:
            inferred = infer_type_from_json_schema(variant)
            if inferred:
                return inferred

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return infer_type_from_python_value(enum_values[0])

    if isinstance(schema.get("properties"), dict):
        return "object"

    if "items" in schema:
        return "array"

    return None


def _extract_tool_name_and_parameters(tool: Any) -> tuple[str | None, dict[str, Any] | None]:
    if isinstance(tool, dict):
        function_block = tool.get("function") if isinstance(tool.get("function"), dict) else None
        if function_block is not None:
            name = function_block.get("name")
            parameters = function_block.get("parameters")
            return name if isinstance(name, str) else None, parameters if isinstance(parameters, dict) else None

        name = tool.get("name")
        parameters = tool.get("parameters")
        return name if isinstance(name, str) else None, parameters if isinstance(parameters, dict) else None

    function_obj = getattr(tool, "function", None)
    if function_obj is not None:
        name = getattr(function_obj, "name", None)
        parameters = getattr(function_obj, "parameters", None)
        return name if isinstance(name, str) else None, parameters if isinstance(parameters, dict) else None

    name = getattr(tool, "name", None)
    parameters = getattr(tool, "parameters", None)
    return name if isinstance(name, str) else None, parameters if isinstance(parameters, dict) else None


def get_argument_type(
    func_name: str,
    arg_key: str,
    defined_tools: list[Any],
) -> str | None:
    """Get expected argument type from tool definitions."""
    name_to_parameters: dict[str, dict[str, Any]] = {}
    for tool in defined_tools:
        name, parameters = _extract_tool_name_and_parameters(tool)
        if name and isinstance(parameters, dict):
            name_to_parameters[name] = parameters

    params = name_to_parameters.get(func_name)
    if not params:
        return None

    properties = params.get("properties")
    if not isinstance(properties, dict):
        return None

    arg_spec = properties.get(arg_key)
    if not isinstance(arg_spec, dict):
        return None

    return infer_type_from_json_schema(arg_spec)


def _convert_to_number(value: str) -> Any:
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except (ValueError, AttributeError):
        return value


def parse_arguments(json_value: str, arg_type: str | None = None) -> tuple[Any, bool]:
    """Parse argument values with multi-step fallback."""
    try:
        parsed_value = json.loads(json_value)
        if arg_type == "number" and isinstance(parsed_value, str):
            parsed_value = _convert_to_number(parsed_value)
        return parsed_value, True
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        wrapped = json.loads('{"tmp": "' + json_value + '"}')
        parsed_value = json.loads(wrapped["tmp"])
        if arg_type == "number" and isinstance(parsed_value, str):
            parsed_value = _convert_to_number(parsed_value)
        return parsed_value, True
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    try:
        parsed_value = ast.literal_eval(json_value)
        return parsed_value, True
    except (ValueError, SyntaxError):
        pass

    try:
        quoted_value = json.dumps(str(json_value))
        return json.loads(quoted_value), True
    except (json.JSONDecodeError, ValueError):
        return json_value, False


@register_tool_call_parser(
    name="hermes",
    aliases=("xml", "tool_xml", "qwen_xml", "default", "h"),
)
class HermesToolCallParser(ToolCallParser):
    """Parser for `<tool_call>{...}</tool_call>` responses."""

    DEFAULT_BOT_TOKEN = "<tool_call>"
    DEFAULT_EOT_TOKEN = "</tool_call>"
    _NAME_PATTERN = re.compile(r'"name"\s*:\s*"([^"]+)"')

    def __init__(
        self,
        bot_token: str = DEFAULT_BOT_TOKEN,
        eot_token: str = DEFAULT_EOT_TOKEN,
        **_: Any,
    ) -> None:
        self.bot_token = bot_token
        self.eot_token = eot_token

        self._pattern = re.compile(
            rf"{re.escape(self.bot_token)}\s*(.*?)\s*{re.escape(self.eot_token)}",
            re.DOTALL,
        )

    @property
    def message_separator(self) -> str:
        return "\n"

    def parse(self, text: str) -> list[ToolCallParseResult]:
        tool_calls: list[ToolCallParseResult] = []
        for index, match in enumerate(self._pattern.finditer(text)):
            raw_content = match.group(1).strip()
            tool_call_id = f"call_{index:04d}"

            try:
                call_json = json.loads(raw_content)
            except json.JSONDecodeError as exc:
                tool_calls.append(self._make_error_tool_call(raw_content, tool_call_id, exc))
                continue

            if isinstance(call_json, dict):
                name = call_json.get("name")
                arguments = call_json.get("arguments", {})
            else:
                name = None
                arguments = {}

            if not name or not isinstance(name, str):
                tool_calls.append(
                    self._make_error_tool_call(raw_content, tool_call_id, ValueError("missing name"))
                )
                continue

            tool_calls.append(
                ToolCallParseResult(
                    id=tool_call_id,
                    name=name,
                    input=arguments if isinstance(arguments, dict) else {},
                )
            )

        return tool_calls

    def _make_error_tool_call(
        self,
        raw_content: str,
        tool_call_id: str,
        error: Exception,
    ) -> ToolCallParseResult:
        name_match = self._NAME_PATTERN.search(raw_content)
        name = name_match.group(1) if name_match else UNKNOWN_TOOL_NAME
        logger.warning("Tool call parse error: %s", error)
        return ToolCallParseResult(id=tool_call_id, name=name, input={}, raw=raw_content)

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text and self.eot_token in text

    def validate_tool_call_tags(self, text: str) -> tuple[bool, str | None]:
        start_count = text.count(self.bot_token)
        end_count = text.count(self.eot_token)

        if start_count != end_count:
            return False, f"Mismatched tool_call tags: {start_count} opening, {end_count} closing"

        return True, None


@register_tool_call_parser(
    name="glm47_moe",
    aliases=("glm47", "glm4.7", "glm", "glm_xml"),
)
class Glm47MoeToolCallParser(ToolCallParser):
    """Parser for GLM-4.7 style XML-ish tool calls.

    Supported format:
      <tool_call>func<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
    """

    def __init__(
        self,
        defined_tools: list[Any] | None = None,
        bot_token: str = "<tool_call>",
        eot_token: str = "</tool_call>",
        **_: Any,
    ) -> None:
        self.defined_tools = defined_tools or []
        self.bot_token = bot_token
        self.eot_token = eot_token

        self._func_call_regex = re.compile(
            rf"{re.escape(self.bot_token)}(.*?){re.escape(self.eot_token)}",
            re.DOTALL,
        )
        self._func_name_regex = re.compile(r"^\s*([^<\s][^<]*?)(?=(?:<arg_key>|$))", re.DOTALL)
        self._func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>(?:\n|\s)*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def validate_tool_call_tags(self, text: str) -> tuple[bool, str | None]:
        start_count = text.count(self.bot_token)
        end_count = text.count(self.eot_token)
        if start_count != end_count:
            return False, f"Mismatched tool_call tags: {start_count} opening, {end_count} closing"
        return True, None

    def parse(self, text: str) -> list[ToolCallParseResult]:
        calls: list[ToolCallParseResult] = []

        for index, match in enumerate(self._func_call_regex.finditer(text)):
            inner = (match.group(1) or "").strip()
            tool_call_id = f"call_{index:04d}"
            if not inner:
                calls.append(self._make_error_tool_call(inner, tool_call_id, ValueError("empty tool_call")))
                continue

            func_match = self._func_name_regex.search(inner)
            if func_match is None:
                calls.append(self._make_error_tool_call(inner, tool_call_id, ValueError("missing function name")))
                continue

            func_name = func_match.group(1).strip()
            arg_section = inner[func_match.end() :]

            has_arg_markup = "<arg_key>" in arg_section or "<arg_value>" in arg_section
            pairs = self._func_arg_regex.findall(arg_section)
            if has_arg_markup and not pairs:
                calls.append(self._make_error_tool_call(inner, tool_call_id, ValueError("malformed arguments")))
                continue

            arguments = self._parse_argument_pairs(pairs, func_name)
            calls.append(
                ToolCallParseResult(
                    id=tool_call_id,
                    name=func_name,
                    input=arguments,
                )
            )

        return calls

    def _parse_argument_pairs(
        self,
        pairs: list[tuple[str, str]],
        func_name: str,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {}
        for raw_key, raw_value in pairs:
            key = raw_key.strip()
            value_text = raw_value.strip()

            arg_type = get_argument_type(func_name, key, self.defined_tools)
            parsed_value, _ = parse_arguments(value_text, arg_type=arg_type)
            arguments[key] = parsed_value

        return arguments

    def _make_error_tool_call(
        self,
        raw_content: str,
        tool_call_id: str,
        error: Exception,
    ) -> ToolCallParseResult:
        logger.warning("GLM47 tool call parse error: %s", error)
        return ToolCallParseResult(id=tool_call_id, name=UNKNOWN_TOOL_NAME, input={}, raw=raw_content)
