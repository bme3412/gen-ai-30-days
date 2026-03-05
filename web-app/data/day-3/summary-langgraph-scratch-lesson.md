# Build an Agent from Scratch - LangGraph Lesson

**Source:** DeepLearning.AI "AI Agents in LangGraph" - Lesson: Build an Agent from Scratch
**Approach:** Pure Python + OpenAI API, no framework dependencies

This lesson demonstrates building a functional agent using nothing but the raw LLM API and Python code. The core pattern is **ReAct** (Reasoning + Acting), where the model cycles through thinking, deciding on actions, executing them, and observing results until it reaches an answer.

The **Agent class** is minimal but complete. It stores a system prompt defining the agent's behavior and maintains a message history list for conversation context. The `__call__` method adds user messages and generates LLM responses, while `execute` runs the full agentic loop.

```python
class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result
```

**Tool functions** are regular Python functions mapped in a dictionary. The lesson demonstrates `calculate()` for math expressions and `average_dog_weight()` for domain lookups. The key insight: tools are just functions—the "magic" is in how the agent decides to call them.

```python
known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight,
}
```

**Response parsing** uses regex to extract action directives from LLM output. When the model outputs `Action: calculate(2 + 2)`, the parser extracts the function name and arguments, looks up the function in the registry, and executes it.

The **execution loop** continues until no more actions are detected (meaning the model has produced a final answer) or a maximum turn limit is reached. Each tool result is formatted as an "Observation" and fed back to the model for the next iteration.

The lesson emphasizes that frameworks like LangGraph provide conveniences—state management, checkpointing, visualization—but understanding this raw loop is essential for debugging, customization, and knowing what's actually happening inside any agent system.
