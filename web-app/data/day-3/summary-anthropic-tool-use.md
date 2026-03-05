# Anthropic Tool Use (Claude)

**Tool use** lets Claude interact with external functions you define, extending its capabilities beyond text generation. You provide tool definitions; Claude decides when and how to call them. The model never executes anything directly—it returns structured requests that your code fulfills.

The flow mirrors OpenAI's but with different message formats. Define tools with `name`, `description`, and `input_schema` (JSON Schema for parameters). When Claude wants to use a tool, it returns a `tool_use` content block with an `id`, the tool `name`, and `input` arguments.

```python
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"]
    }
}]
```

**Key difference from OpenAI:** Tool results use `role: "user"` with a `tool_result` content block (not a separate "tool" role). Each result must include the `tool_use_id` matching the original request.

```python
messages.append({
    "role": "user",
    "content": [{
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": json.dumps(result)
    }]
})
```

Claude supports **parallel tool calls**—multiple `tool_use` blocks in a single response when operations are independent. Return all results in one user message before continuing.

For schema validation, add **`strict: true`** to tool definitions. This guarantees Claude's tool calls match your schema exactly—no type mismatches, no missing required fields. Essential for production agents where invalid parameters would cause failures.

**Tool choice** controls behavior: `auto` (default) lets Claude decide, `any` requires at least one tool call, `none` prevents calls, or specify a tool name to force it.

Tool access is high-leverage for agents. On benchmarks like SWE-bench (software engineering) and LAB-Bench (scientific interpretation), adding even simple tools produces outsized capability gains. The contract is clear: you define what's available, Claude decides when to use it, your code executes safely.
