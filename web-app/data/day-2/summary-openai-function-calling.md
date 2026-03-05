# OpenAI Function Calling

**Function calling** lets models request execution of functions you define. The critical insight: the model never executes anything—it returns a structured request describing what function to call with what arguments. Your code decides whether to execute, handles the call, and passes results back. This separation of intent from execution is what makes tool-using agents safe.

The conversation flow has five steps. Send a request with **tool definitions** alongside the user message. The model returns one or more **tool calls** specifying functions and parameters. Execute those functions with your implementations. Send results back in a `tool` role message with the matching `tool_call_id`. The model incorporates results and either responds or requests more tool calls.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        },
        "strict": True
    }
}]
```

**Strict mode** (`strict: true`) guarantees the model's output matches your JSON Schema exactly—no omitted required fields, no invalid types. Without it, you're parsing best-effort JSON that might not conform.

Control tool usage with **`tool_choice`**: `"auto"` lets the model decide (default), `"required"` forces at least one call, `"none"` prevents calls, or specify `{"type": "function", "function": {"name": "specific_tool"}}` to force that exact function.

Modern models support **parallel tool calls**—requesting multiple executions in a single response. When the model needs weather for both Paris and Tokyo, it returns both requests at once. Execute all calls, return all results, then continue.

For quality tool use: write descriptions that explain *when* to use a tool, not just what it does. Include format examples in parameter descriptions. Use enums to constrain choices. Apply the **"intern test"**—if someone unfamiliar with your system couldn't use the function correctly from your description alone, add more detail.

Tool definitions count against context limits and token costs. Each function's name, description, and schema consumes input tokens on every request.
