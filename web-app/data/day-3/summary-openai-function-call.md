# OpenAI Function Calling for Agents

**Function calling** is how agents connect to external capabilities. The model decides when a tool would help, returns a structured request describing the call, and your code executes it. This separation—model requests, code executes—is the foundation of safe tool-using agents.

When building agents from scratch, function calling provides the structured interface between LLM reasoning and real-world actions. The conversation follows five steps: send tool definitions with the user message, receive tool call requests, execute functions in your code, return results with matching `tool_call_id`, and continue the conversation.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search for information",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        },
        "strict": True
    }
}]
```

**Strict mode** (`strict: true`) guarantees the model's output matches your schema—essential for agents where malformed parameters would break the execution loop. Without it, you're parsing best-effort JSON that might fail unpredictably.

**Tool choice** controls model behavior: `"auto"` lets it decide, `"required"` forces a call, `"none"` prevents calls, or specify a function name to force it. For agents, `"auto"` is typical—the model decides when tools help versus when to provide a final answer.

Modern models support **parallel tool calls**—requesting multiple executions simultaneously. When building your agent loop, check for multiple tool calls in each response and execute all before continuing.

For agents built from scratch, function calling replaces manual parsing of "Action: search(query)" formats with guaranteed structured output. The model's tool call is already valid JSON; you just route it to the right function.
