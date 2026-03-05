# Anthropic Tool Use (Claude)

Claude's **tool use** capability allows the model to interact with external functions you define. Like OpenAI's function calling, Claude doesn't execute tools directly—it returns structured requests that your code handles. You define the schema, Claude decides when to use it, and you control execution.

Anthropic distinguishes between **client tools** (functions you implement on your systems) and **server tools** (built-in capabilities like web search that execute on Anthropic's infrastructure automatically).

The flow for client tools: send a request with tool definitions and user message. If Claude decides to use a tool, it returns `stop_reason: "tool_use"` with one or more **`tool_use` blocks** containing ID, tool name, and arguments. Execute the tool and send back a **`tool_result` block** referencing the original ID. Claude generates its final response incorporating results.

```python
# Tool result must reference original tool_use_id
{"type": "tool_result", "tool_use_id": "toolu_123", "content": "72°F, sunny"}
```

Key differences from OpenAI: use **`input_schema`** instead of `parameters`, tool definitions are direct objects (not wrapped in `{"type": "function"}`), and tool results are content blocks within a `user` role message—not a separate `tool` role.

Claude supports **parallel tool calls**, returning multiple `tool_use` blocks. Execute all and return all results in a single message. Enable **`strict: true`** for schema compliance.

**Server tools** like web search execute automatically on Anthropic's infrastructure. Include them in the tools array and results are incorporated without a tool result exchange.

Different Claude models behave differently: **Opus** tends to ask for clarification when parameters are missing, while **Sonnet** may guess values. Add chain-of-thought instructions asking Claude to reason about tool relevance before calling.

Tool definitions count against input tokens (~346 tokens overhead). For **MCP integrations**, rename `inputSchema` to `input_schema` when converting definitions.
