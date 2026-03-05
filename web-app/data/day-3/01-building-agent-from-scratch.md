# Building an Agent from Scratch

**Sources:** DeepLearning.AI "Agentic AI" Course, "AI Agents in LangGraph" Course
**Focus:** Implementing a complete agent using only Python and LLM APIs

---

## Table of Contents

1. [Introduction: Why Build Without Frameworks?](#1-introduction-why-build-without-frameworks)
2. [The Minimal Agent: Components Overview](#2-the-minimal-agent-components-overview)
3. [Agent Class Design](#3-agent-class-design)
4. [Tool Registry Pattern](#4-tool-registry-pattern)
5. [System Prompt Engineering](#5-system-prompt-engineering)
6. [Response Parsing Strategies](#6-response-parsing-strategies)
7. [The Execution Loop](#7-the-execution-loop)
8. [Error Handling & Edge Cases](#8-error-handling--edge-cases)
9. [Stop Conditions & Termination](#9-stop-conditions--termination)
10. [Putting It All Together](#10-putting-it-all-together)
11. [Testing Your Agent](#11-testing-your-agent)
12. [Common Pitfalls](#12-common-pitfalls)
13. [When to Use Frameworks Instead](#13-when-to-use-frameworks-instead)

---

## 1. Introduction: Why Build Without Frameworks?

Frameworks abstract away the mechanics of agents. That's their value—and their cost. When your LangChain agent produces unexpected output, you need to understand what's happening at the raw loop level: Is parsing failing? Is context overflowing? Is the tool silently erroring? Without that understanding, debugging is guesswork.

Building from scratch gives you three things:

**Complete control.** You decide how actions are parsed, how errors surface, how history accumulates, when execution stops. No fighting framework assumptions.

**Minimal dependencies.** One HTTP client, one JSON library. No version conflicts, no framework churn.

**Transferable knowledge.** Every framework wraps the same primitives: message threading, tool dispatch, loop control. Master the raw form and you'll use any framework more effectively.

### What We'll Build

A working agent in under 150 lines of Python:

- Maintains conversation history (the OBSERVE from Day 1)
- Routes to tools based on LLM decisions (connecting Day 2's function calling)
- Parses action directives from free-form output
- Handles errors without crashing
- Terminates when appropriate

This implements the OBSERVE-THINK-ACT-REFLECT loop from Day 1 using the structured tool calling patterns from Day 2—the bridge from theory to working code.

---

## 2. The Minimal Agent: Components Overview

Strip away the abstractions and an agent is just a loop: ask the LLM what to do, do it, repeat. Four components make this work.

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         AGENT                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │   System     │    │   Message    │    │    Tool      │     │
│   │   Prompt     │    │   History    │    │   Registry   │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    EXECUTION LOOP                        │   │
│   │                                                          │   │
│   │   ┌─────────┐    ┌─────────┐    ┌─────────┐            │   │
│   │   │  Call   │───▶│  Parse  │───▶│ Execute │            │   │
│   │   │  LLM    │    │ Action  │    │  Tool   │            │   │
│   │   └─────────┘    └─────────┘    └────┬────┘            │   │
│   │        ▲                              │                  │   │
│   │        │         ┌─────────┐          │                  │   │
│   │        └─────────│  Add    │◀─────────┘                  │   │
│   │                  │ Result  │                             │   │
│   │                  └─────────┘                             │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### System Prompt

Defines persona, capabilities, and—critically—output format. The model needs to know exactly how to structure responses so you can parse them. This is where the "contract" between your code and the LLM gets established.

### Message History

The agent's working memory. Each turn appends the LLM's response and any tool observations. On every API call, the model sees the full history—this is how it maintains context (the OBSERVE phase).

### Tool Registry

A dictionary mapping names to callables. When the LLM says "search", this tells your code which Python function to run. Tools are ordinary functions; the intelligence is in how the LLM decides to invoke them (connecting to Day 2's function calling).

### Execution Loop

The agent's heartbeat: call LLM → parse response → execute tool → append observation → repeat. Continues until the model produces a final answer or hits a stop condition. This is the THINK-ACT-REFLECT cycle made concrete.

---

## 3. Agent Class Design

We'll build the Agent class incrementally. Start minimal, add complexity only as needed.

### Minimal Structure

```python
from openai import OpenAI

class Agent:
    def __init__(self, system_prompt: str = ""):
        self.client = OpenAI()
        self.system = system_prompt
        self.messages = []

    def __call__(self, user_message: str) -> str:
        """Process a user message and return the agent's response."""
        self.messages.append({"role": "user", "content": user_message})
        response = self._call_llm()
        self.messages.append({"role": "assistant", "content": response})
        return response

    def _call_llm(self) -> str:
        """Make an API call to the LLM."""
        messages = [{"role": "system", "content": self.system}] + self.messages
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0  # Deterministic for consistent behavior
        )
        return completion.choices[0].message.content
```

This is a conversational assistant—it maintains history for multi-turn conversations but can't act on the world. To become an agent, it needs tools.

### Adding State

Track execution metadata for debugging:

```python
class Agent:
    def __init__(self, system_prompt: str = ""):
        self.client = OpenAI()
        self.system = system_prompt
        self.messages = []
        self.turn_count = 0
        self.last_action = None
        self.last_observation = None
```

### Constructor Parameters

In production, you'll want more flexibility:

```python
def __init__(
    self,
    system_prompt: str = "",
    model: str = "gpt-4o",
    temperature: float = 0,
    max_tokens: int = 1024,
    tools: dict = None
):
    self.client = OpenAI()
    self.system = system_prompt
    self.model = model
    self.temperature = temperature
    self.max_tokens = max_tokens
    self.tools = tools or {}
    self.messages = []
```

---

## 4. Tool Registry Pattern

Tools transform a text generator into an agent that acts on the world. Day 2 covered how to define tools for LLM APIs; here we implement the execution side. The registry is just a dictionary—name to callable.

### Basic Registry

```python
def search_web(query: str) -> str:
    """Search the web for information."""
    # In production, call a real search API
    return f"Search results for: {query}"

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)  # Safe only for math!
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # In production, call a weather API
    return f"Weather in {city}: 72°F, sunny"

# The registry
tools = {
    "search": search_web,
    "calculate": calculate,
    "weather": get_weather,
}
```

### Tool Execution

```python
def execute_tool(self, action: str, args: str) -> str:
    """Execute a tool and return its result."""
    if action not in self.tools:
        return f"Error: Unknown tool '{action}'. Available: {list(self.tools.keys())}"

    try:
        result = self.tools[action](args)
        return str(result)
    except Exception as e:
        return f"Error executing {action}: {e}"
```

### Tool Documentation

The LLM needs to know what tools are available and how to use them. Include this in the system prompt:

```python
def format_tools_for_prompt(tools: dict) -> str:
    """Generate tool documentation for the system prompt."""
    lines = ["Available tools:"]
    for name, func in tools.items():
        doc = func.__doc__ or "No description"
        lines.append(f"- {name}: {doc}")
    return "\n".join(lines)
```

### Structured Tool Definitions

For better reliability, define tools with explicit schemas:

```python
TOOL_DEFINITIONS = {
    "search": {
        "description": "Search the web for current information",
        "parameters": {
            "query": {"type": "string", "description": "The search query"}
        },
        "function": search_web
    },
    "calculate": {
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "expression": {"type": "string", "description": "Math expression like '2 + 2'"}
        },
        "function": calculate
    }
}
```

---

## 5. System Prompt Engineering

The system prompt is the contract between your code and the LLM. It must specify output format precisely—ambiguity here causes parsing failures downstream.

### The ReAct Format

ReAct (Reasoning + Acting) structures output into parseable segments:

```python
SYSTEM_PROMPT = """You are a helpful assistant that can use tools to answer questions.

You run in a loop of Thought, Action, Observation.
At the end of the loop you output an Answer.

Use Thought to describe your reasoning about the question.
Use Action to run one of the available tools.
Observation will be the result of running the tool.

Your available tools are:

search: Search the web for information
    Example: Action: search(current weather in Tokyo)

calculate: Evaluate a mathematical expression
    Example: Action: calculate(25 * 4 + 10)

weather: Get current weather for a city
    Example: Action: weather(London)

Always use this exact format:

Thought: [your reasoning]
Action: [tool_name(arguments)]

When you have the final answer:

Thought: [your reasoning]
Answer: [your final response to the user]

Begin!
"""
```

### Key Elements

**Format specification**: Show the exact structure, then examples. The model mirrors what it sees.

**Tool documentation**: Every available tool with description and example invocation. Models can't use undocumented tools.

**Termination signal**: How the agent signals completion. Here, `Answer:` marks the final response.

**Few-shot examples**: For complex behaviors, include complete thought-action-observation cycles in the prompt.

### Alternative Formats

JSON actions trade readability for parsing reliability:

```python
SYSTEM_PROMPT = """When you want to use a tool, output JSON:

{"action": "tool_name", "args": {"param1": "value1"}}

When you have the final answer:

{"answer": "your response"}
"""
```

Combined with Day 2's structured outputs, JSON format eliminates parsing failures entirely.

### Prompt Length Considerations

System prompts consume tokens on every call. Include only:
- Core behavior instructions
- Tool definitions with brief examples
- Output format specification
- Essential constraints

Detailed documentation belongs in retrieval systems, not system prompts.

---

## 6. Response Parsing Strategies

Parsing bridges free-form LLM output to structured execution. This is where most agent failures originate—models deviate from format, and brittle parsers break.

### Regex Parsing

Fast and sufficient when format compliance is high:

```python
import re

def parse_action(response: str) -> tuple[str, str] | None:
    """Extract action and arguments from response.

    Returns (action_name, arguments) or None if no action found.
    """
    # Match "Action: tool_name(arguments)"
    pattern = r"Action:\s*(\w+)\((.+?)\)"
    match = re.search(pattern, response, re.IGNORECASE)

    if match:
        action = match.group(1)
        args = match.group(2).strip()
        return (action, args)

    return None

def parse_answer(response: str) -> str | None:
    """Extract final answer from response."""
    pattern = r"Answer:\s*(.+)"
    match = re.search(pattern, response, re.DOTALL)

    if match:
        return match.group(1).strip()

    return None
```

### Handling Variations

Models drift from format. Robust parsers anticipate common deviations:

```python
def parse_action_flexible(response: str) -> tuple[str, str] | None:
    """Parse action with flexibility for format variations."""

    # Try standard format first
    patterns = [
        r"Action:\s*(\w+)\((.+?)\)",           # Action: search(query)
        r"Action:\s*(\w+)\s*\((.+?)\)",        # Action: search (query)
        r"\*\*Action\*\*:\s*(\w+)\((.+?)\)",   # **Action**: search(query)
        r"action:\s*(\w+)\((.+?)\)",           # action: search(query)
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            return (match.group(1).lower(), match.group(2).strip())

    return None
```

### JSON Parsing

JSON format trades natural output for reliable parsing:

```python
import json

def parse_json_response(response: str) -> dict | None:
    """Parse JSON action from response."""
    try:
        # Find JSON object in response
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except json.JSONDecodeError:
        pass
    return None
```

### Using Structured Outputs

For guaranteed schema conformance, use Day 2's structured outputs:

```python
from pydantic import BaseModel

class AgentAction(BaseModel):
    thought: str
    action: str | None = None
    args: str | None = None
    answer: str | None = None

def call_llm_structured(self) -> AgentAction:
    """Call LLM with structured output."""
    messages = [{"role": "system", "content": self.system}] + self.messages
    completion = self.client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=messages,
        response_format=AgentAction
    )
    return completion.choices[0].message.parsed
```

This eliminates parsing failures entirely—the response is guaranteed to match the schema.

---

## 7. The Execution Loop

This is the core—where OBSERVE-THINK-ACT-REFLECT becomes running code.

### Basic Loop

```python
def run(self, user_message: str, max_turns: int = 10) -> str:
    """Run the agent until it produces an answer or hits max turns."""

    # Add user message to history
    self.messages.append({"role": "user", "content": user_message})

    for turn in range(max_turns):
        # Get LLM response
        response = self._call_llm()
        self.messages.append({"role": "assistant", "content": response})

        # Check for final answer
        answer = parse_answer(response)
        if answer:
            return answer

        # Check for action
        action_result = parse_action(response)
        if action_result:
            action, args = action_result
            observation = self.execute_tool(action, args)

            # Add observation to history
            self.messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })
        else:
            # No action or answer—model might be confused
            self.messages.append({
                "role": "user",
                "content": "Please respond with either an Action or an Answer."
            })

    return "Max turns reached without answer."
```

### Connecting to Day 1 Concepts

This loop implements the OBSERVE-THINK-ACT-REFLECT pattern from Day 1:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   ┌─────────┐         ┌─────────┐         ┌─────────┐          │
│   │ OBSERVE │────────▶│  THINK  │────────▶│   ACT   │          │
│   │         │         │         │         │         │          │
│   │ Read    │         │ LLM     │         │ Execute │          │
│   │ history │         │ decides │         │ tool    │          │
│   └─────────┘         └─────────┘         └────┬────┘          │
│        ▲                                       │                │
│        │              ┌─────────┐              │                │
│        │              │ REFLECT │              │                │
│        │              │         │              │                │
│        └──────────────│ Format  │◀─────────────┘                │
│                       │ result  │                               │
│                       └─────────┘                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

- **OBSERVE**: The LLM sees the full message history including previous observations
- **THINK**: The LLM generates a "Thought:" explaining its reasoning
- **ACT**: The agent executes the specified tool
- **REFLECT**: The observation is formatted and added to history for the next iteration

### Streaming Execution

Long-running agents benefit from progress callbacks:

```python
def run_with_callbacks(
    self,
    user_message: str,
    on_thought: callable = None,
    on_action: callable = None,
    on_observation: callable = None,
    max_turns: int = 10
) -> str:
    """Run agent with callbacks for each step."""

    self.messages.append({"role": "user", "content": user_message})

    for turn in range(max_turns):
        response = self._call_llm()
        self.messages.append({"role": "assistant", "content": response})

        # Extract and report thought
        if on_thought:
            thought = extract_thought(response)
            if thought:
                on_thought(thought)

        answer = parse_answer(response)
        if answer:
            return answer

        action_result = parse_action(response)
        if action_result:
            action, args = action_result

            if on_action:
                on_action(action, args)

            observation = self.execute_tool(action, args)

            if on_observation:
                on_observation(observation)

            self.messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })

    return "Max turns reached."
```

---

## 8. Error Handling & Edge Cases

Production agents don't crash—they surface errors as observations so the LLM can adapt.

### Tool Errors

Informative error messages let the model recover:

```python
def execute_tool(self, action: str, args: str) -> str:
    """Execute tool with comprehensive error handling."""

    # Unknown tool
    if action not in self.tools:
        available = ", ".join(self.tools.keys())
        return f"Error: Unknown tool '{action}'. Available tools: {available}"

    # Execute with error catching
    try:
        result = self.tools[action](args)
        return str(result)
    except TypeError as e:
        return f"Error: Invalid arguments for {action}. {e}"
    except TimeoutError:
        return f"Error: {action} timed out. Try a simpler query."
    except Exception as e:
        return f"Error: {action} failed with: {type(e).__name__}: {e}"
```

### Parse Failures

Guide the model back to expected format rather than crashing:

```python
def run(self, user_message: str, max_turns: int = 10) -> str:
    parse_failures = 0
    max_parse_failures = 3

    for turn in range(max_turns):
        response = self._call_llm()
        self.messages.append({"role": "assistant", "content": response})

        answer = parse_answer(response)
        if answer:
            return answer

        action_result = parse_action(response)
        if action_result:
            parse_failures = 0  # Reset on successful parse
            action, args = action_result
            observation = self.execute_tool(action, args)
            self.messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })
        else:
            parse_failures += 1
            if parse_failures >= max_parse_failures:
                return "Unable to parse agent response. Please try rephrasing."

            self.messages.append({
                "role": "user",
                "content": (
                    "I couldn't parse your response. Please use the exact format:\n"
                    "Thought: [reasoning]\n"
                    "Action: tool_name(arguments)\n\n"
                    "Or if you have the answer:\n"
                    "Thought: [reasoning]\n"
                    "Answer: [your response]"
                )
            })

    return "Max turns reached."
```

### Malformed Arguments

Validate before execution to provide actionable feedback:

```python
def validate_and_execute(self, action: str, args: str) -> str:
    """Validate arguments and execute tool."""

    tool_def = self.tool_definitions.get(action)
    if not tool_def:
        return f"Error: Unknown tool '{action}'"

    # Check required parameters
    params = tool_def.get("parameters", {})
    if not args and params:
        param_list = ", ".join(params.keys())
        return f"Error: {action} requires parameters: {param_list}"

    # Execute
    try:
        return tool_def["function"](args)
    except Exception as e:
        return f"Error: {e}"
```

### Infinite Loops

Detect repetition patterns that indicate the agent is stuck:

```python
def detect_loop(self, window: int = 4) -> bool:
    """Detect if agent is repeating itself."""
    if len(self.messages) < window * 2:
        return False

    recent = [m["content"] for m in self.messages[-window:] if m["role"] == "assistant"]
    earlier = [m["content"] for m in self.messages[-window*2:-window] if m["role"] == "assistant"]

    # Check for repeated patterns
    return recent == earlier
```

---

## 9. Stop Conditions & Termination

An agent without stop conditions runs until it exhausts resources. Defense in depth: layer multiple termination checks.

### Multiple Stop Conditions

```python
def run(
    self,
    user_message: str,
    max_turns: int = 10,
    max_tokens: int = 50000,
    timeout_seconds: int = 300
) -> str:
    """Run with multiple stop conditions."""

    import time
    start_time = time.time()
    total_tokens = 0

    self.messages.append({"role": "user", "content": user_message})

    for turn in range(max_turns):
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            return f"Timeout after {elapsed:.0f} seconds."

        # Check token budget
        if total_tokens > max_tokens:
            return f"Token budget exceeded ({total_tokens} tokens)."

        # Check for loops
        if self.detect_loop():
            return "Agent appears stuck. Please try a different approach."

        # Normal execution
        response, tokens_used = self._call_llm_with_usage()
        total_tokens += tokens_used

        self.messages.append({"role": "assistant", "content": response})

        answer = parse_answer(response)
        if answer:
            return answer

        # ... continue with action execution

    return "Max turns reached."
```

### Graceful Termination

When stopping early, extract partial value rather than returning empty-handed:

```python
def terminate(self, reason: str) -> str:
    """Terminate execution with a summary."""

    # Try to get a partial answer from the LLM
    self.messages.append({
        "role": "user",
        "content": f"We need to stop here ({reason}). Based on what you've learned so far, please provide your best answer."
    })

    response = self._call_llm()
    answer = parse_answer(response)

    if answer:
        return f"{answer}\n\n(Note: {reason})"

    return f"Unable to complete. {reason}"
```

---

## 10. Putting It All Together

The complete implementation—everything from sections 3-9 unified:

```python
import re
from openai import OpenAI

class Agent:
    """A minimal but complete agent implementation."""

    def __init__(
        self,
        system_prompt: str,
        tools: dict = None,
        model: str = "gpt-4o",
        temperature: float = 0
    ):
        self.client = OpenAI()
        self.system = system_prompt
        self.tools = tools or {}
        self.model = model
        self.temperature = temperature
        self.messages = []

    def _call_llm(self) -> str:
        """Make LLM API call."""
        messages = [{"role": "system", "content": self.system}] + self.messages
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        return completion.choices[0].message.content

    def _parse_action(self, response: str) -> tuple[str, str] | None:
        """Extract action from response."""
        match = re.search(r"Action:\s*(\w+)\((.+?)\)", response, re.IGNORECASE)
        if match:
            return (match.group(1).lower(), match.group(2).strip())
        return None

    def _parse_answer(self, response: str) -> str | None:
        """Extract final answer from response."""
        match = re.search(r"Answer:\s*(.+)", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _execute_tool(self, action: str, args: str) -> str:
        """Execute a tool and return result."""
        if action not in self.tools:
            available = ", ".join(self.tools.keys())
            return f"Error: Unknown tool '{action}'. Available: {available}"

        try:
            return str(self.tools[action](args))
        except Exception as e:
            return f"Error: {action} failed: {e}"

    def run(self, user_message: str, max_turns: int = 10) -> str:
        """Run the agent until completion."""
        self.messages.append({"role": "user", "content": user_message})

        for turn in range(max_turns):
            response = self._call_llm()
            self.messages.append({"role": "assistant", "content": response})

            # Check for final answer
            answer = self._parse_answer(response)
            if answer:
                return answer

            # Check for action
            action_result = self._parse_action(response)
            if action_result:
                action, args = action_result
                observation = self._execute_tool(action, args)
                self.messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })
            else:
                self.messages.append({
                    "role": "user",
                    "content": "Please respond with an Action or Answer."
                })

        return "Max turns reached without answer."

    def reset(self):
        """Clear conversation history."""
        self.messages = []
```

### Example Usage

```python
# Define tools
def search(query: str) -> str:
    return f"Top result for '{query}': Wikipedia article about {query}"

def calculate(expr: str) -> str:
    return str(eval(expr))

# Create system prompt
PROMPT = """You are a helpful assistant with access to tools.

Available tools:
- search(query): Search the web
- calculate(expression): Do math

Use this format:
Thought: [your reasoning]
Action: tool_name(arguments)

When done:
Thought: [your reasoning]
Answer: [final response]
"""

# Create and run agent
agent = Agent(
    system_prompt=PROMPT,
    tools={"search": search, "calculate": calculate}
)

result = agent.run("What is 25 * 47, and who invented calculus?")
print(result)
```

---

## 11. Testing Your Agent

Agents are non-deterministic—the same input can produce different execution paths. Test at multiple levels: unit tests for deterministic components, integration tests for behavioral properties.

### Unit Testing Tools

```python
import pytest

def test_calculate_tool():
    result = calculate("2 + 2")
    assert result == "4"

def test_calculate_error():
    result = calculate("invalid")
    assert "Error" in result

def test_search_returns_string():
    result = search("python")
    assert isinstance(result, str)
    assert len(result) > 0
```

### Testing Parsing

```python
def test_parse_action():
    response = "Thought: I need to search\nAction: search(python tutorials)"
    action, args = parse_action(response)
    assert action == "search"
    assert args == "python tutorials"

def test_parse_action_no_match():
    response = "I'm not sure what to do."
    result = parse_action(response)
    assert result is None

def test_parse_answer():
    response = "Thought: I have all info\nAnswer: The answer is 42."
    answer = parse_answer(response)
    assert answer == "The answer is 42."
```

### Integration Testing

```python
def test_agent_simple_calculation():
    agent = Agent(system_prompt=PROMPT, tools={"calculate": calculate})
    result = agent.run("What is 100 / 4?")
    assert "25" in result

def test_agent_uses_correct_tool():
    agent = Agent(system_prompt=PROMPT, tools={"search": search, "calculate": calculate})
    agent.run("Search for information about Python")

    # Check that search was called
    actions = [m for m in agent.messages if "Observation:" in m.get("content", "")]
    assert len(actions) > 0

def test_agent_handles_unknown_tool():
    agent = Agent(system_prompt=PROMPT, tools={"calculate": calculate})
    # Force an attempt to use unknown tool
    agent.messages = [
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "Action: unknown_tool(test)"}
    ]
    # Agent should handle gracefully
    result = agent.run("continue")
    # Should not crash
```

### Behavioral Testing

For complex behaviors, use eval datasets:

```python
TEST_CASES = [
    {
        "input": "What is 15% of 200?",
        "expected_tool": "calculate",
        "expected_contains": "30"
    },
    {
        "input": "Who wrote Romeo and Juliet?",
        "expected_tool": "search",
        "expected_contains": "Shakespeare"
    }
]

def test_agent_behaviors():
    for case in TEST_CASES:
        agent = Agent(system_prompt=PROMPT, tools=tools)
        result = agent.run(case["input"])

        assert case["expected_contains"].lower() in result.lower(), \
            f"Expected '{case['expected_contains']}' in response to '{case['input']}'"
```

---

## 12. Common Pitfalls

These failure modes appear repeatedly in agent implementations.

### Pitfall 1: Overly Complex Prompts

Verbose prompts dilute signal. Start minimal:

```python
# Bad: Too much detail
PROMPT = """You are an extremely helpful assistant that always thinks carefully
before responding. You have access to the following tools which you should use
whenever they would be helpful. Make sure to explain your reasoning thoroughly
and consider multiple perspectives before arriving at your final answer..."""

# Good: Clear and focused
PROMPT = """You are a helpful assistant with tools.

Tools:
- search(query): Search the web
- calculate(expr): Do math

Format:
Thought: [reasoning]
Action: tool(args)

When done:
Answer: [response]
"""
```

### Pitfall 2: Poor Error Messages

Generic errors don't help the model recover:

```python
# Bad
return "Error"

# Good
return f"Error: Tool '{action}' not found. Available tools: {list(tools.keys())}"
```

### Pitfall 3: Unbounded Context

Message history grows indefinitely:

```python
# Bad: No limit
self.messages.append(response)

# Good: Sliding window
MAX_HISTORY = 20
self.messages = self.messages[-MAX_HISTORY:]
```

### Pitfall 4: Missing Stop Conditions

Agents can loop forever:

```python
# Bad: Only max_turns
while not done:
    # Could run forever

# Good: Multiple conditions
while (
    turns < max_turns and
    tokens < max_tokens and
    not timeout_reached and
    not loop_detected
):
    # Bounded execution
```

### Pitfall 5: Ignoring Tool Latency

Network calls are slow. Handle appropriately:

```python
# Bad: No timeout
result = api_call()

# Good: Timeout handling
try:
    result = api_call(timeout=30)
except TimeoutError:
    result = "Tool timed out. Try a simpler query."
```

---

## 13. When to Use Frameworks Instead

Raw implementation teaches fundamentals. Frameworks trade customization for velocity and battle-tested patterns.

### Use Frameworks When:

**Complex state management**: LangGraph's checkpointing handles state persistence across sessions, something that's tedious to implement from scratch.

**Multi-agent coordination**: CrewAI and AutoGen have tested patterns for agent-to-agent communication and task delegation.

**Production observability**: LangSmith and similar tools integrate seamlessly with their frameworks for tracing and debugging.

**Rapid prototyping**: Frameworks let you test ideas quickly before investing in custom implementation.

**Team collaboration**: Standardized abstractions help teams work together consistently.

### Continue Building From Scratch When:

**Learning**: Understanding internals helps you use any tool more effectively.

**Simple agents**: A search-and-answer agent doesn't need LangGraph's complexity.

**Maximum control**: Custom requirements might fight framework assumptions.

**Minimal dependencies**: Production systems benefit from fewer moving parts.

### Hybrid Approach

Often the best path combines both:

```python
# Use framework for orchestration, raw code for custom tools
from langgraph import StateGraph

def custom_tool(state):
    # Your from-scratch implementation
    agent = Agent(system=PROMPT, tools=tools)
    return agent.run(state["query"])

# Framework handles the workflow
graph = StateGraph()
graph.add_node("custom_agent", custom_tool)
```

Understanding the raw loop makes you effective with any tool—and helps you know when you don't need one.

---

## Summary

An agent is simpler than the frameworks suggest:

1. **Message history** → the OBSERVE mechanism from Day 1
2. **System prompt** → defines output contract for parsing
3. **Tool registry** → executes Day 2's function calls
4. **Parse logic** → bridges free-form output to structured execution
5. **Execution loop** → THINK-ACT-REFLECT made concrete

Under 150 lines for a working implementation. Add error handling, stop conditions, and tests for production.

Master the raw form and frameworks become tools you choose rather than abstractions you depend on.
