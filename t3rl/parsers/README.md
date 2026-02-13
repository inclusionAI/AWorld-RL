# t3rl Parsers

This folder keeps parser logic split into two layers:

- `tool_call.py`: tool-call parsing (format detection + argument extraction)
- `reasoning.py`: reasoning-span parsing (think block handling)

## Parser role in the new interaction stack

Parsers are policy-layer components. The runtime call chain is:

1. `InteractionDriver` gets assistant raw text from model output.
2. Agent policy calls parser(s) and builds typed `AgentDecision`.
3. Env adapter consumes `AgentDecision` and returns `EnvTransition`.
4. Driver appends transition messages and maintains token/logprob invariants.

Parsers stay text-level only and do not mutate env state.

## Why `Instruct2507` uses `<thinking>` instead of `<think>`

For Qwen3 Instruct-2507, the tokenizer shares vocabulary behavior with thinking models.
`<think>` is treated as a special token with its own embedding. In thinking-model post-training,
that token appears in RL traces, so the model learns to produce it.

In Instruct post-training, the model is trained with instruction tuning and typically does **not**
see `<think>` traces. If prompts force `<think>...</think>`, the model may receive an embedding
pattern it does not reliably use for tagged output. In rollouts this often shows up as:

- reasoning text is present,
- but `<think>` / `</think>` tags are missing.

Using `<thinking>...</thinking>` avoids this tokenizer-specific mismatch for Instruct-2507 in practice,
while preserving explicit reasoning boundary markers required by strict multi-turn formatting.

## Registry with decorators

Both parser modules support decorator-style registration:

```python
from t3rl.parsers.reasoning import register_reasoning_parser
from t3rl.parsers.tool_call import register_tool_call_parser

@register_reasoning_parser("my_reasoning", aliases=("mr",))
class MyReasoningParser:
    ...

@register_tool_call_parser("my_tool_parser", aliases=("mtp",))
class MyToolCallParser:
    ...
```

## Registered parser keys

### Reasoning parsers (`reasoning.py`)

- `instruct2507` (aliases: `i2507`, `qwen2507`, `thinking`, `default`)
- `deepseek_r1` (aliases: `deepseek-r1`, `r1`, `dsr1`)
- `qwen_think` (aliases: `qwen3-thinking`, `think`)

### Tool-call parsers (`tool_call.py`)

- `hermes` (aliases: `xml`, `tool_xml`, `qwen_xml`, `default`, `h`)
- `glm47_moe` (aliases: `glm47`, `glm4.7`, `glm`, `glm_xml`)

`glm47_moe` is implemented as a non-streaming parser for XML-like payloads:

```text
<tool_call>fn<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
```

It includes compact argument parsing with JSON-schema-aware type inference.

## Config usage

Set parser keys in `configs/*/default.yaml`:

```yaml
reasoning_parser: "instruct2507"
tool_call_parser: "hermes"
```

Short-key examples:

- `reasoning_parser: "i2507"`
- `reasoning_parser: "r1"`
- `tool_call_parser: "h"`
- `tool_call_parser: "glm47"`

## Invariants and safety principles

Parser behavior is intentionally text-level only and does **not** alter token trajectory construction.
TITO/TIS invariants remain enforced by driver/agent-side checks:

- `len(rollout_log_probs) == len(loss_mask) == response_length`
- optional end-of-rollout drift check remains available via `tito_drift_check_enabled`

This follows the multi-turn training principles documented in `CLAUDE.md`.
