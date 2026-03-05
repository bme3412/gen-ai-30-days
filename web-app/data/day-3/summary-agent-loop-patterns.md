# Agent Loop Implementation Patterns

**Focus:** Common patterns for implementing agent execution loops in production

The agent loop is where theory meets execution. While the concept is simple—call LLM, parse response, execute action, repeat—production implementations require careful handling of several concerns.

**The Basic Loop** follows a predictable structure: send messages to the LLM, check if the response contains an action directive, execute the action if present, append the observation to messages, and repeat. Termination occurs when the model produces a final answer or maximum iterations are reached.

```python
while iterations < max_turns:
    response = llm(messages)
    if "Answer:" in response:
        return extract_answer(response)
    action, args = parse_action(response)
    result = execute_tool(action, args)
    messages.append({"role": "user", "content": f"Observation: {result}"})
    iterations += 1
```

**Parsing strategies** range from simple regex to structured outputs. Regex works for prototypes: `re.search(r"Action: (\w+)\((.*)\)", response)`. Production systems often use structured outputs (Day 2) to guarantee valid action formats, eliminating parsing failures entirely.

**Error handling** must address several failure modes. Unknown tool names require graceful messages like "Tool 'foo' not found. Available: search, calculate." Malformed arguments need validation before execution. Tool execution failures (network errors, timeouts) should be caught and reported as observations so the agent can adapt.

**Stop conditions** prevent infinite loops and runaway costs. Maximum turn limits (typically 5-15) cap iterations. Answer detection checks for explicit completion signals. Token budget limits prevent expensive conversations. Timeout limits catch stuck executions.

**Message history management** affects context quality. Full history provides complete context but grows expensive. Sliding windows keep recent turns only. Summarization compresses old context. The right strategy depends on task complexity and cost constraints.

**Observation formatting** influences how well the agent interprets results. Consistent formats like `Observation: {tool_name} returned: {result}` help the model understand what happened. Error observations should be distinguishable: `Observation: Error - {error_message}`.
