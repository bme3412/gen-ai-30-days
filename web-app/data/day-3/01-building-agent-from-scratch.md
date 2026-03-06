# Building an Agent from Scratch

**Sources:** DeepLearning.AI "Agentic AI" Course, "AI Agents in LangGraph" Course
**Focus:** Implementing a complete agent using only Python and LLM APIs

---

## Why Build Without Frameworks?

Frameworks like LangChain, LangGraph, and CrewAI have made agent development accessible to developers who might otherwise never attempt it. They provide convenient abstractions, pre-built components, and established patterns that can get a prototype running in minutes. But this convenience extracts a price that becomes apparent the moment something goes wrong.

When your LangChain agent produces unexpected output, you're left guessing at the cause. Is the output parser failing to extract the action? Is the context window overflowing, causing the model to lose track of earlier instructions? Is a tool silently throwing an exception that gets swallowed somewhere in the chain? The framework's abstractions, which seemed so helpful during the happy path, now obscure the very information you need to diagnose the problem. You find yourself adding print statements, reading framework source code, and performing trial-and-error debugging that consumes hours.

Building from scratch strips away these layers of abstraction. You see exactly what messages go to the API, exactly what comes back, exactly how parsing happens, and exactly where errors originate. This transparency isn't just useful for debugging—it fundamentally changes your understanding of what agents are and how they work.

The benefits compound across several dimensions. You gain **complete control** over every aspect of agent behavior: how actions are parsed from model output, how errors are surfaced and handled, how conversation history accumulates over time, and precisely when execution should terminate. There are no framework-imposed limitations or opinionated defaults fighting against your requirements.

You work with **minimal dependencies**—just an HTTP client for API calls and a JSON library for parsing responses. This simplicity eliminates version conflicts between framework components, avoids the churn of keeping up with rapidly evolving framework APIs, and reduces the surface area for security vulnerabilities. Your agent's behavior depends only on code you wrote and can inspect.

Most importantly, you develop **transferable knowledge** that makes you effective with any framework. Every agent framework, regardless of its specific API, wraps the same fundamental primitives: message construction, API calls, response parsing, tool execution, and history management. Once you understand these primitives through direct implementation, frameworks become tools you choose deliberately rather than abstractions you depend on blindly. You can read framework source code and immediately understand what it's doing. You can debug framework behavior by recognizing which primitive is misbehaving. You can evaluate new frameworks by assessing how they handle each primitive.

By the end of this lesson, you'll have a working agent in under 150 lines of Python. It will maintain conversation history (implementing Day 1's OBSERVE pattern), route to tools based on LLM decisions (using Day 2's function calling concepts), parse action directives from model output, handle errors gracefully, and terminate when appropriate. This is the bridge from theory to working code.

---

## The Anatomy of an Agent

Strip away the abstractions and marketing language, and an agent reveals itself as a remarkably simple construct: ask the LLM what to do, do it, ask again. This loop continues until the model decides it has accomplished the task and provides a final answer. Four components make this work, each with a specific responsibility that directly maps to the concepts we explored in Days 1 and 2.

### The System Prompt

The **system prompt** defines the agent's persona, capabilities, and—most critically—its output format. Think of it as a contract between your code and the model. Your parsing logic expects responses in a specific structure; the system prompt is where you establish that expectation.

A well-crafted system prompt accomplishes several things simultaneously. It tells the model what role it's playing ("You are a helpful research assistant"). It enumerates the tools available and explains when each should be used ("Use search when you need current information; use calculate for mathematical operations"). It specifies the exact output format the model should follow, often with examples to eliminate ambiguity. And it establishes termination conditions—how the model should signal that it's finished and ready to deliver a final answer.

The precision of your format specification directly determines parsing reliability. Vague instructions like "respond with your reasoning and then your action" invite the model to improvise, producing outputs that your regex or JSON parser can't handle. Explicit specifications like "Always format actions as `Action: tool_name(arguments)`" give the model a clear target that's easy to match consistently.

### Message History

The **message history** serves as the agent's working memory. Each turn of the conversation appends new entries: the user's request, the assistant's reasoning and action, the observation from tool execution, and so on. When you make an API call, you send the full history, giving the model complete context about what has happened so far.

This is the OBSERVE phase from Day 1 made concrete. The model doesn't retain state between API calls—it has no memory of previous interactions unless you explicitly include them in the message list. By accumulating history, you create the illusion of a continuous conversation with an agent that remembers what it has done and what it has learned.

History management involves trade-offs that become apparent in longer conversations. Full history provides complete context but grows expensive as token counts increase. Eventually, you'll exceed context limits or find that important early instructions get "lost" as the model pays more attention to recent messages. Production agents often implement sliding windows (keeping only the last N turns), summarization (compressing old context into summaries), or retrieval (storing history externally and fetching relevant portions). For learning purposes, full history is simplest and works well for short interactions.

### The Tool Registry

The **tool registry** maps tool names to Python functions. When the LLM says "search," your registry tells the code which function to execute. The registry is typically implemented as a simple dictionary: `{"search": search_function, "calculate": calc_function}`.

Tools themselves are ordinary Python functions with nothing special about them. They take arguments, perform some operation, and return a string result. The "intelligence" lies entirely in how the LLM decides when to invoke them—the tool functions are just the capabilities you're offering.

This separation of concerns is powerful. You can develop and test tools independently of the agent. You can swap tool implementations without changing agent logic. You can add new tools by simply registering new functions. The agent doesn't need to know anything about what tools do; it only needs to know their names and how to pass the results back to the model.

### The Execution Loop

The **execution loop** is the agent's heartbeat, the code that makes the OBSERVE-THINK-ACT-REFLECT cycle actually run. It orchestrates all the other components: calling the LLM with current history, parsing the response to determine what action was requested, executing that action through the tool registry, formatting the result as an observation, appending everything to history, and deciding whether to continue or terminate.

The loop continues until one of several conditions is met: the model produces a final answer (indicating it believes the task is complete), the loop hits a maximum iteration limit (preventing runaway execution), or some error condition makes continuation impossible. Each iteration represents one full cycle of the agentic pattern we explored in Day 1.

Understanding the execution loop is essential because it's where most agent failures originate. Parsing errors, infinite loops, context overflow, tool failures—all of these manifest in the execution loop. When you've built the loop yourself, debugging these issues becomes straightforward: you can log each step, inspect message contents, and pinpoint exactly where things go wrong.

---

## Building the Agent Class

We'll construct the Agent class incrementally, starting with the absolute minimum viable implementation and adding complexity only when the simpler version proves insufficient. This approach mirrors how you should build agents in practice: start simple, test, and elaborate only in response to actual failures.

### The Foundation

The simplest possible agent needs just three things: a client for making API calls, a system prompt defining behavior, and a list to hold message history. With these components, we can implement a basic conversational assistant that maintains context across turns.

```python
from openai import OpenAI

class Agent:
    def __init__(self, system_prompt: str = ""):
        self.client = OpenAI()
        self.system = system_prompt
        self.messages = []

    def __call__(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})
        response = self._call_llm()
        self.messages.append({"role": "assistant", "content": response})
        return response

    def _call_llm(self) -> str:
        messages = [{"role": "system", "content": self.system}] + self.messages
        completion = self.client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=0
        )
        return completion.choices[0].message.content
```

Let's trace through what happens when you call this agent. The `__call__` method receives a user message and appends it to history with the "user" role. It then calls `_call_llm`, which constructs the full message list by prepending the system prompt to the accumulated history. The OpenAI API receives this complete context and generates a response. That response is appended to history with the "assistant" role and returned to the caller.

Notice that we set `temperature=0` for deterministic responses. In production agents, you typically want reproducible behavior—the same inputs should produce the same outputs. Temperature > 0 introduces randomness that makes debugging difficult and can cause the agent to behave inconsistently across runs.

This implementation creates a conversational assistant that remembers previous exchanges—but it can't act on the world. It can discuss what it might do, but it has no mechanism to actually do anything. To transform this chatbot into an agent, we need tools.

---

## The Tool Registry

**Tools** are what transform a text generator into an agent that can take action. Day 2 covered how to define tools for LLM APIs using JSON schemas and function descriptions. Here we implement the other side of that contract: the execution machinery that runs when the LLM requests a tool call.

### Defining Tools

The registry is simply a dictionary mapping names to callables. Each tool is an ordinary Python function. Best practice is to include a docstring describing what the tool does—you'll use this description in the system prompt to help the LLM understand when to invoke each tool.

```python
def search_web(query: str) -> str:
    """Search the web for current information on any topic."""
    # In production, this would call a real search API
    return f"Search results for: {query}"

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = {
    "search": search_web,
    "calculate": calculate,
    "time": get_current_time
}
```

A few important observations about these tool implementations:

**Tools always return strings.** Even when a tool performs a calculation that produces a number, convert it to a string before returning. The result will be inserted into the conversation as text, and the LLM will interpret it as part of the ongoing dialogue. Returning non-strings can cause type errors or unexpected behavior.

**Tools should handle their own errors.** Rather than raising exceptions that crash the agent, catch errors internally and return descriptive error messages. This allows the LLM to see what went wrong and potentially adapt—maybe it misformatted the arguments and can try again, or maybe it needs to approach the problem differently.

**Tools should be stateless when possible.** If a tool depends on external state, it becomes harder to test and reason about. When state is necessary (like maintaining a session with an external service), document it clearly.

### Executing Tools

The `execute_tool` method looks up the requested action in the registry and calls the corresponding function. It handles two error cases gracefully: unknown tool names and execution failures.

```python
def execute_tool(self, action: str, args: str) -> str:
    if action not in self.tools:
        available = ", ".join(self.tools.keys())
        return f"Error: Unknown tool '{action}'. Available tools: {available}"

    try:
        result = self.tools[action](args)
        return str(result)
    except Exception as e:
        return f"Error executing {action}: {type(e).__name__}: {e}"
```

Notice how error messages are designed to help the LLM recover. When an unknown tool is requested, we list all available tools—giving the model the information it needs to correct its mistake. When execution fails, we include both the exception type and message, providing diagnostic information the model can use to understand what went wrong.

This error-as-observation pattern is crucial for robust agents. Rather than crashing on errors, we feed error information back into the conversation. The LLM can then reason about what happened and try a different approach. Many agent failures stem from treating errors as fatal rather than as information the agent can learn from.

---

## Crafting the System Prompt

The **system prompt** is the contract between your code and the LLM. Every word matters because ambiguity here causes parsing failures downstream. The model will do its best to follow your instructions, but "its best" can vary significantly based on how precisely you specify the expected format.

### The ReAct Format

The **ReAct format** (Reasoning + Acting) structures agent output into parseable segments. The model produces a Thought explaining its reasoning, then an Action specifying which tool to call with what arguments. After you execute the tool and provide an Observation, this cycle repeats until the model is ready to provide a final Answer.

ReAct emerged from research showing that models perform better on complex tasks when they externalize their reasoning before acting. The "Thought" step isn't just for human readability—it actually improves the model's decision-making by forcing it to articulate its plan before committing to an action.

```python
SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

Available tools:
- search(query): Search the web for current information on any topic
- calculate(expression): Evaluate a mathematical expression (e.g., "2 + 2", "sqrt(16)")
- time(): Get the current date and time

IMPORTANT: Always respond using EXACTLY one of these two formats:

FORMAT 1 - When you need to use a tool:
Thought: [explain your reasoning about what information you need]
Action: tool_name(arguments)

FORMAT 2 - When you have enough information to answer:
Thought: [explain how you reached your conclusion]
Answer: [your complete response to the user]

Examples:

User: What is 15% of 230?
Thought: I need to calculate 15% of 230. I'll use the calculate tool.
Action: calculate(230 * 0.15)

User: What time is it?
Thought: The user wants to know the current time. I'll use the time tool.
Action: time()

After receiving an observation, continue with another Thought/Action or provide your final Answer.
"""
```

Several elements make this prompt effective:

**Explicit format specification.** Rather than describing the format in prose ("respond with your reasoning followed by an action"), we show the exact structure with labeled components. The model has a clear template to follow.

**Enumerated tools with examples.** Each tool is listed with its name, arguments, and a brief description of what it does. Including usage examples (like "2 + 2", "sqrt(16)") helps the model understand the expected argument format.

**Termination signal.** The `Answer:` prefix tells our parsing code that the agent is done and ready to return a final response. Without this explicit signal, we'd have to guess whether the model's output is a final answer or just intermediate reasoning.

**Few-shot examples.** Complete examples of the expected format dramatically improve compliance. The model sees exactly what a valid Thought/Action sequence looks like and mimics that structure.

### JSON Actions (Alternative Approach)

For production systems where parsing reliability is paramount, you can require JSON-formatted actions instead of the natural-language format shown above:

```python
SYSTEM_PROMPT_JSON = """You are a helpful assistant with access to tools.

When you need to use a tool, respond with ONLY a JSON object:
{"thought": "your reasoning", "action": "tool_name", "args": "arguments"}

When you have the final answer:
{"thought": "your reasoning", "answer": "your response to the user"}
"""
```

JSON actions trade natural-sounding output for parsing reliability. Combined with Day 2's structured outputs, the model's response is guaranteed to be valid JSON matching your schema—no regex needed, no edge cases to handle. The downside is that the agent's reasoning becomes less readable when inspecting conversation history.

The choice between natural language and JSON depends on your priorities. For learning and debugging, natural language makes it easier to understand what the agent is thinking. For production systems, JSON provides the reliability you need.

---

## Parsing Model Output

**Parsing** bridges free-form LLM output to structured execution. This is where the rubber meets the road—and where most agent failures originate. Models drift from specified formats, add unexpected flourishes, or structure their output slightly differently than your parser expects.

### Regex Parsing

Regex parsing is fast, requires no additional dependencies, and works well when format compliance is high. The key is designing patterns that are specific enough to match valid actions but tolerant enough to handle common variations.

```python
import re

def parse_action(response: str) -> tuple[str, str] | None:
    """Extract action name and arguments from model response.

    Matches patterns like:
    - Action: search(climate change effects)
    - Action: calculate(2 + 2)
    - **Action**: search(query)  (markdown formatting)
    """
    pattern = r"Action:\s*(\w+)\((.+?)\)"
    match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
    if match:
        action_name = match.group(1).lower()
        arguments = match.group(2).strip()
        return (action_name, arguments)
    return None

def parse_answer(response: str) -> str | None:
    """Extract final answer from model response.

    Matches the Answer: prefix and captures everything after it.
    """
    pattern = r"Answer:\s*(.+)"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None
```

Let's break down the regex patterns:

For `parse_action`, the pattern `r"Action:\s*(\w+)\((.+?)\)"` matches:
- `Action:` - The literal prefix
- `\s*` - Any amount of whitespace
- `(\w+)` - One or more word characters, captured as group 1 (the action name)
- `\(` - A literal opening parenthesis
- `(.+?)` - One or more characters, non-greedy, captured as group 2 (the arguments)
- `\)` - A literal closing parenthesis

The `re.IGNORECASE` flag handles variations like "action:" or "ACTION:". The `re.DOTALL` flag allows the pattern to work even if there are newlines in unexpected places.

### Handling Format Drift

Models drift from format over time and across different inputs. A model might add markdown formatting (`**Action**:`), use slightly different wording (`I'll use Action:`), or structure multi-line arguments differently than expected. Robust parsers anticipate these common deviations.

```python
def parse_action_robust(response: str) -> tuple[str, str] | None:
    """More tolerant action parser that handles common variations."""

    # Try multiple patterns in order of specificity
    patterns = [
        r"Action:\s*(\w+)\((.+?)\)",           # Standard format
        r"\*\*Action\*\*:\s*(\w+)\((.+?)\)",   # Markdown bold
        r"Tool:\s*(\w+)\((.+?)\)",             # "Tool" instead of "Action"
        r"Use:\s*(\w+)\((.+?)\)",              # "Use" prefix
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            return (match.group(1).lower(), match.group(2).strip())

    return None
```

This cascading approach tries the expected format first, then falls back to common variations. You can extend the pattern list based on what deviations you observe in practice.

### Structured Outputs (Best Reliability)

For guaranteed reliability, use **structured outputs** from Day 2. Define a Pydantic model specifying exactly what fields you expect, and the API returns a parsed object that's guaranteed to match your schema.

```python
from pydantic import BaseModel
from typing import Optional

class AgentResponse(BaseModel):
    thought: str
    action: Optional[str] = None
    args: Optional[str] = None
    answer: Optional[str] = None

# With OpenAI's structured output support:
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    response_format={"type": "json_object"},
)
response = AgentResponse.model_validate_json(completion.choices[0].message.content)
```

With structured outputs, parsing failures become impossible—the API guarantees valid responses matching your schema. The trade-off is that you're locked into JSON format, which some find less readable for debugging.

### Parsing Strategy Comparison

| Strategy | Reliability | Flexibility | Complexity | Best For |
|----------|-------------|-------------|------------|----------|
| Basic Regex | Medium | High | Low | Prototypes, controlled settings |
| Robust Regex | Medium-High | High | Medium | Production with natural language |
| JSON Extraction | High | Medium | Medium | Balance of reliability/readability |
| Structured Outputs | Guaranteed | Low | Low | Production, high reliability needs |

---

## The Execution Loop

This is the core of the agent—where the **OBSERVE-THINK-ACT-REFLECT** pattern from Day 1 becomes running code. Every concept we've discussed converges in this loop: message history, tool execution, parsing, and termination conditions.

### Loop Structure

The loop starts by adding the user's message to history, establishing the task. Then it enters a cycle: call the LLM to get a response, append that response to history, check for termination or action, and continue accordingly.

```python
def run(self, user_message: str, max_turns: int = 10) -> str:
    """Execute the agent loop until completion or max turns."""

    # Initialize with the user's request
    self.messages.append({"role": "user", "content": user_message})

    for turn in range(max_turns):
        # THINK: Get the model's response
        response = self._call_llm()
        self.messages.append({"role": "assistant", "content": response})

        # Check for final answer (termination)
        if answer := parse_answer(response):
            return answer

        # ACT: Check for and execute action
        if action_result := parse_action(response):
            action, args = action_result

            # Execute the tool
            observation = self.execute_tool(action, args)

            # OBSERVE/REFLECT: Add observation to history
            self.messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })
        else:
            # No action or answer - nudge the model back on track
            self.messages.append({
                "role": "user",
                "content": "Please respond with an Action or Answer."
            })

    return "Max turns reached without answer."
```

Let's trace through this loop to understand how the Day 1 concepts manifest:

1. **OBSERVE**: When `_call_llm()` is called, the model receives the full message history including all previous observations. It "sees" everything that has happened so far.

2. **THINK**: The model generates a response with a "Thought:" component, articulating its reasoning about what to do next.

3. **ACT**: If the response contains an "Action:", we parse it and execute the corresponding tool. The tool performs some operation on the external world.

4. **REFLECT**: We format the tool's output as an "Observation:" and append it to history. This observation becomes part of what the model sees on the next iteration, allowing it to incorporate new information into its reasoning.

### Why Observations Use "user" Role

You might notice that observations are appended with `role: "user"` rather than some hypothetical "system" or "tool" role. This is intentional and worth understanding.

The conversation model that LLMs use has a simple structure: messages alternate between "user" (the human) and "assistant" (the model). When we inject an observation, we're simulating a user providing new information. From the model's perspective, it said "I want to search for X" and then received a response "Here's what I found."

This role choice affects how the model interprets the observation. Using "user" role signals that this is new input requiring a response. The model will naturally continue the conversation by generating another assistant message, which is exactly what we want—either another action or a final answer.

### Handling Parse Failures

When the model's response doesn't match either the action or answer format, we need to handle it gracefully. Simply crashing would end the agent; ignoring it would create an infinite loop. The solution is to nudge the model back toward the expected format:

```python
else:
    # Response contained neither action nor answer
    self.messages.append({
        "role": "user",
        "content": "Please respond with an Action or Answer."
    })
```

This message reminds the model of the expected format and gives it another chance to comply. In practice, one nudge usually suffices—the model "remembers" the format from the system prompt and corrects itself.

For persistent failures, you might track consecutive parse failures and terminate after a threshold:

```python
if consecutive_failures > 3:
    return "Unable to generate valid response format."
```

---

## Handling Errors Gracefully

Production agents don't crash—they surface errors as observations so the LLM can adapt. The philosophy is simple: errors are information, and information should flow to the model so it can make better decisions.

### Informative Error Messages

When a tool fails, the error message should give the model everything it needs to understand what went wrong and potentially recover. Compare these two error responses:

**Bad:** `"Error"`

**Good:** `"Error: Tool 'serch' not found. Available tools: search, calculate, time. Did you mean 'search'?"`

The first tells the model nothing. The second tells it exactly what happened, lists valid alternatives, and even suggests a correction. The model can use this information to retry with the correct tool name.

```python
def execute_tool(self, action: str, args: str) -> str:
    """Execute a tool with comprehensive error handling."""

    # Unknown tool
    if action not in self.tools:
        available = ", ".join(sorted(self.tools.keys()))
        # Check for close matches
        close_matches = [t for t in self.tools if t.startswith(action[0])]
        suggestion = f" Did you mean '{close_matches[0]}'?" if close_matches else ""
        return f"Error: Unknown tool '{action}'. Available: {available}.{suggestion}"

    # Execution failure
    try:
        result = self.tools[action](args)
        return str(result)
    except TypeError as e:
        return f"Error: Invalid arguments for {action}. {e}"
    except TimeoutError:
        return f"Error: {action} timed out. Try a simpler query."
    except Exception as e:
        return f"Error: {action} failed with {type(e).__name__}: {e}"
```

### Recovery Strategies

Different error types suggest different recovery approaches:

**Unknown tools**: List available tools. The model usually just misremembered or mistyped the name.

**Invalid arguments**: Explain what arguments are expected. The model can reformulate its request.

**Timeout errors**: Suggest simplifying the query. A search for "everything about climate change" might time out, while "climate change temperature impact 2024" might succeed.

**Rate limits**: Indicate when the tool will be available again. The model might decide to use cached information or try a different approach.

**Network errors**: Distinguish between temporary and persistent failures. A temporary glitch might resolve on retry; a persistent failure might require a different tool.

### Parse Failure Handling

When parsing fails, treat it as another form of error observation:

```python
if not action_result and not answer_result:
    self.messages.append({
        "role": "user",
        "content": (
            "I couldn't parse your response. Please use exactly one of these formats:\n"
            "- Thought: [reasoning]\n  Action: tool_name(arguments)\n"
            "- Thought: [reasoning]\n  Answer: [final response]"
        )
    })
```

This explicit reminder of the expected format usually gets the model back on track. The key is being specific—don't just say "invalid format," show exactly what's expected.

---

## Stop Conditions

An agent without **stop conditions** runs until it exhausts resources—your API budget, your patience, or your context window. Defense in depth means layering multiple termination checks, each catching different failure modes.

### Turn Limits

The most basic stop condition is a maximum turn count. After N iterations without reaching an answer, give up. Typical values range from 5-15 depending on task complexity.

```python
for turn in range(max_turns):
    # ... agent loop ...
    pass

return "Max turns reached without answer."
```

But turns alone aren't sufficient. A malicious or confused model could burn through 15 turns in seconds, making expensive API calls each time. You need additional checks.

### Token Budget

Track cumulative token usage and terminate when approaching a budget:

```python
def __init__(self, ..., max_tokens: int = 50000):
    self.max_tokens = max_tokens
    self.tokens_used = 0

def _call_llm(self) -> str:
    completion = self.client.chat.completions.create(...)
    self.tokens_used += completion.usage.total_tokens

    if self.tokens_used > self.max_tokens:
        raise TokenBudgetExceeded(f"Used {self.tokens_used} tokens")

    return completion.choices[0].message.content
```

### Timeout

A wall-clock timeout catches agents that get stuck in slow tool executions or network delays:

```python
import time

def run(self, user_message: str, max_turns: int = 10, timeout: int = 300):
    start_time = time.time()

    for turn in range(max_turns):
        if time.time() - start_time > timeout:
            return "Execution timeout reached."
        # ... rest of loop ...
```

### Loop Detection

Sophisticated agents can get stuck in loops, repeating the same actions without making progress. Detect this by comparing recent messages:

```python
def _detect_loop(self) -> bool:
    """Check if the agent is repeating itself."""
    if len(self.messages) < 6:
        return False

    recent_assistant_msgs = [
        m["content"] for m in self.messages[-6:]
        if m["role"] == "assistant"
    ]

    # Check for exact repetition
    if len(recent_assistant_msgs) >= 2:
        if recent_assistant_msgs[-1] == recent_assistant_msgs[-2]:
            return True

    return False
```

### Graceful Termination

When you must stop early, don't just return an error message. Ask the model to provide its best answer given what it's learned:

```python
if turn >= max_turns - 1:
    # Final turn - force an answer
    self.messages.append({
        "role": "user",
        "content": (
            "You've reached the maximum number of turns. "
            "Please provide your best answer based on what you've learned so far."
        )
    })
    response = self._call_llm()
    if answer := parse_answer(response):
        return answer
    # Extract whatever content we can
    return response

return "Unable to complete task."
```

---

## The Complete Implementation

Here's everything unified—a working agent in under 100 lines that incorporates all the concepts we've discussed. This implementation is deliberately minimal but fully functional. You can use it as-is for simple tasks or extend it for more complex requirements.

```python
import re
from openai import OpenAI

class Agent:
    def __init__(self, system_prompt: str, tools: dict = None, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.system = system_prompt
        self.tools = tools or {}
        self.model = model
        self.messages = []

    def _call_llm(self) -> str:
        messages = [{"role": "system", "content": self.system}] + self.messages
        return self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0
        ).choices[0].message.content

    def _parse_action(self, text: str) -> tuple[str, str] | None:
        match = re.search(r"Action:\s*(\w+)\((.+?)\)", text, re.IGNORECASE)
        return (match.group(1).lower(), match.group(2).strip()) if match else None

    def _parse_answer(self, text: str) -> str | None:
        match = re.search(r"Answer:\s*(.+)", text, re.DOTALL)
        return match.group(1).strip() if match else None

    def _execute_tool(self, action: str, args: str) -> str:
        if action not in self.tools:
            return f"Error: Unknown tool '{action}'. Available: {list(self.tools.keys())}"
        try:
            return str(self.tools[action](args))
        except Exception as e:
            return f"Error: {action} failed: {e}"

    def run(self, user_message: str, max_turns: int = 10) -> str:
        self.messages.append({"role": "user", "content": user_message})

        for _ in range(max_turns):
            response = self._call_llm()
            self.messages.append({"role": "assistant", "content": response})

            if answer := self._parse_answer(response):
                return answer

            if result := self._parse_action(response):
                action, args = result
                obs = self._execute_tool(action, args)
                self.messages.append({"role": "user", "content": f"Observation: {obs}"})
            else:
                self.messages.append({"role": "user", "content": "Respond with Action or Answer."})

        return "Max turns reached."
```

### Using the Agent

To use this agent, define your tools, write a system prompt specifying the format, create an instance, and call `run()`:

```python
# Define tools
def search(query: str) -> str:
    return f"Results for '{query}': [relevant information here]"

def calculate(expression: str) -> str:
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

tools = {"search": search, "calculate": calculate}

# System prompt (use the SYSTEM_PROMPT from earlier)
prompt = SYSTEM_PROMPT

# Create and run
agent = Agent(system_prompt=prompt, tools=tools)
result = agent.run("What is the square root of 144?")
print(result)
```

---

## Testing Your Agent

Agents are **non-deterministic**—the same input can produce different execution paths depending on subtle variations in model output. This makes testing challenging but not impossible. Test at multiple levels to build confidence.

### Unit Testing Tools

Test each tool function independently of the agent. These tests are deterministic and fast, giving you high confidence in the building blocks:

```python
def test_calculate_valid():
    assert calculate("2 + 2") == "4"
    assert calculate("10 * 5") == "50"
    assert calculate("100 / 4") == "25.0"

def test_calculate_invalid():
    result = calculate("not math")
    assert "Error" in result or "error" in result.lower()

def test_search_returns_string():
    result = search("test query")
    assert isinstance(result, str)
    assert len(result) > 0
```

### Unit Testing Parsers

Test your parsing functions with known inputs. Create a suite of test cases covering normal operation and edge cases:

```python
def test_parse_action_standard():
    response = "Thought: I need to search\nAction: search(climate change)"
    result = parse_action(response)
    assert result == ("search", "climate change")

def test_parse_action_with_whitespace():
    response = "Action:  search( query with spaces )"
    result = parse_action(response)
    assert result == ("search", "query with spaces")

def test_parse_action_missing():
    response = "Thought: I'm thinking about this"
    result = parse_action(response)
    assert result is None

def test_parse_answer():
    response = "Thought: I have the info\nAnswer: The answer is 42."
    result = parse_answer(response)
    assert result == "The answer is 42."
```

### Integration Testing

Full agent tests are inherently less deterministic. Focus on behavioral properties rather than exact outputs:

```python
def test_agent_calculation():
    agent = Agent(system_prompt=SYSTEM_PROMPT, tools=tools)
    result = agent.run("What is 15 * 7?")
    assert "105" in result

def test_agent_uses_tool():
    agent = Agent(system_prompt=SYSTEM_PROMPT, tools=tools)
    agent.run("Search for information about Python")

    # Verify search tool was called by checking history
    messages = [m["content"] for m in agent.messages]
    assert any("Observation:" in m for m in messages)

def test_agent_terminates():
    agent = Agent(system_prompt=SYSTEM_PROMPT, tools=tools)
    result = agent.run("Hello", max_turns=5)
    # Agent should respond without timing out
    assert result != "Max turns reached."
```

### Logging for Debugging

Add logging to understand agent behavior during development:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def run(self, user_message: str, max_turns: int = 10) -> str:
    logger.debug(f"Starting agent with message: {user_message}")

    for turn in range(max_turns):
        logger.debug(f"Turn {turn + 1}/{max_turns}")
        response = self._call_llm()
        logger.debug(f"LLM response: {response[:200]}...")

        # ... rest of loop
```

---

## Common Pitfalls

Several failure modes appear repeatedly in agent implementations. Knowing them in advance helps you avoid the traps.

### Overly Complex Prompts

When your agent misbehaves, the temptation is to add more instructions: "Think carefully before acting," "Consider multiple perspectives," "Double-check your reasoning." These additions rarely help and often hurt.

Long, complex prompts dilute the signal. The model has to parse through paragraphs of instructions to find the format specification it actually needs. Important directives get lost among the verbiage. The model may become confused about priorities.

**Fix:** Start minimal. You can always add complexity if the simple version fails, but start with the bare essentials: available tools, required format, and termination conditions. If the agent fails, add one thing at a time and test.

### Poor Error Messages

Error messages are instructions to the agent about what went wrong. "Error" tells it nothing. "Tool failed" is barely better. The agent needs specific information to recover.

**Fix:** Every error message should answer three questions: What happened? Why did it happen? What should the agent do instead?

### Unbounded Context

Message history grows indefinitely. Eventually you'll exceed the model's context limit, causing API errors. Before that, you'll hit performance degradation as the model struggles to pay attention to earlier messages.

**Fix:** Implement a context management strategy: sliding windows (keep last N turns), summarization (compress old context), or retrieval (store history externally). For learning purposes, full history is fine, but production agents need bounds.

### Missing Stop Conditions

An agent without stop conditions loops forever, consuming your API credits until you manually kill it. This happens more often than you'd expect—a subtle parsing failure can cause the agent to never recognize its own answers.

**Fix:** Layer multiple stop conditions: turn limits, token budgets, timeouts, and loop detection. Any one of these should be sufficient; together they're robust.

### Ignoring Tool Latency

Network calls take time. A search API might respond in 100ms or 10 seconds depending on load. Without timeouts, your agent hangs indefinitely on a slow tool.

**Fix:** Add timeouts to tool execution. Consider providing progress callbacks for long-running agents so users know something is happening. Test your agent with artificially slow tools to understand the experience.

---

## When to Use Frameworks Instead

Building from scratch teaches fundamentals that make you effective with any tool. But frameworks exist for good reasons. Understanding when to use each approach is part of the expertise you're developing.

### Use Frameworks When:

**You need complex state management.** LangGraph's checkpointing handles persistence across conversations, resumption after failures, and branching execution paths. Building these features yourself is tedious and error-prone.

**You're building multi-agent systems.** CrewAI and AutoGen have battle-tested patterns for agent-to-agent communication, task delegation, and coordination. Getting these interactions right from scratch requires significant effort.

**You need production observability.** LangSmith integrates with LangChain to provide tracing, debugging, and monitoring. Building equivalent tooling yourself is a major undertaking.

**You're prototyping rapidly.** Frameworks let you test ideas quickly, exploring different architectures before committing to a custom implementation.

**Your team needs standardization.** Shared abstractions help teams collaborate effectively. When everyone uses the same patterns, code reviews and knowledge transfer become easier.

### Build from Scratch When:

**You're learning.** There's no substitute for implementing the fundamentals yourself. The understanding you gain makes you effective with any framework.

**The agent is simple.** A search-and-answer agent doesn't need LangGraph's complexity. Adding a framework would introduce overhead without benefit.

**You need maximum control.** Custom requirements might fight framework assumptions. If you're constantly working around framework limitations, you'd be better off without it.

**You're minimizing dependencies.** Production systems benefit from fewer moving parts. Each dependency is a potential source of bugs, security issues, and maintenance burden.

### Decision Framework

| Factor | Build from Scratch | Use Framework |
|--------|-------------------|---------------|
| Agent complexity | Simple, single-purpose | Multi-agent, stateful |
| Control requirements | Maximum customization | Defaults acceptable |
| Dependency tolerance | Minimal preferred | Framework churn acceptable |
| Team context | Solo or small team | Larger team needs standardization |
| Use case | Learning, production core | Rapid prototyping, complex orchestration |

### The Hybrid Approach

Often the best path combines both approaches: use a framework for orchestration while implementing custom tools with raw code. LangGraph handles state management and execution flow while your tools make direct API calls without framework wrapping.

This hybrid approach gives you framework benefits where they matter (state, coordination, observability) while maintaining control where you need it (tool implementation, error handling, custom logic).

Understanding the raw loop makes you effective with any tool—and helps you know when you don't need one.

---

## Summary

An agent is simpler than the frameworks suggest. Strip away the abstractions and you find a loop that calls an API, parses responses, executes tools, and repeats.

**Message history** implements the OBSERVE mechanism from Day 1. By accumulating conversation context, the agent "remembers" what it has done and learned.

**The system prompt** defines the output contract. Precise format specifications make parsing reliable. Ambiguous prompts produce ambiguous outputs.

**The tool registry** executes Day 2's function calls. Tools are just Python functions; the intelligence lies in how the LLM decides when to invoke them.

**Parse logic** bridges free-form output to structured execution. Regex works for prototypes; structured outputs guarantee reliability.

**The execution loop** makes THINK-ACT-REFLECT concrete. Each iteration adds to history, calls the model, parses the response, executes actions, and checks for termination.

Under 100 lines for a working implementation. Add error handling for graceful failure recovery. Add stop conditions to prevent runaway execution. Add tests for confidence in your components.

Master the raw form and frameworks become tools you choose deliberately rather than abstractions you depend on blindly. You'll know when a framework helps and when it's unnecessary overhead. You'll debug framework issues by recognizing which primitive is misbehaving. You'll extend frameworks with custom components because you understand what they need to provide.

This is the foundation. Day 4 builds on it, exploring how LangGraph adds state management and sophisticated control flow to these fundamental patterns.
