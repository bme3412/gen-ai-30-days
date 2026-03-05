# OpenAI Function Calling

**Function calling** is OpenAI's mechanism for letting models request the execution of functions you define. The key insight: the model never actually executes anything—it returns a structured request describing what function to call and with what arguments. Your code decides whether to execute, handles execution, and passes results back.

The flow works in five steps. First, send a request with **tool definitions** alongside the user's message. The model returns one or more **tool calls** specifying functions and parameters. Execute those functions with your real implementations. Send results back in a special message format. The model incorporates results and generates its final response—or requests additional tool calls.

When defining a function tool, provide a JSON object with key fields: `type` (always "function"), `name` (like "get_weather"), `description` (critical for guiding decisions), and `parameters` (JSON Schema for inputs). Setting **`strict: true`** guarantees schema compliance.

```python
tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
    "strict": True
}]
```

Control tool usage with **`tool_choice`**: `"auto"` lets the model decide, `"required"` forces at least one tool call, `"none"` prevents calls, or specify a function name to force that exact function.

Modern models support **parallel tool calls**—requesting multiple function executions in a single response. Iterate through all calls, execute each, and return all results before continuing. Disable with `parallel_tool_calls=False` if needed.

For best results: write clear descriptions explaining when and why to use each function, include parameter documentation with format examples, use **enums** to constrain values, and keep function count under 20. Apply the **"intern test"**: if an intern couldn't correctly use the function from your description alone, add more detail.

Function definitions count against **context limits and token billing**. Minimize usage by limiting exposed functions and keeping descriptions concise but complete.
