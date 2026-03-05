# LangChain Agents Overview

**LangChain** is an open-source framework with pre-built agent architecture and integrations for 15+ providers—OpenAI, Anthropic, Google, and more. You can build agents in under 10 lines of code and swap providers without code changes.

The framework offers three abstraction levels. **Deep Agents** is the recommended starting point—batteries-included with automatic compression of long conversations, a virtual filesystem, and subagent-spawning for context management. **LangChain agents** offer customization for context engineering and tool configurations. **LangGraph** is the low-level orchestration framework for combining deterministic and agentic workflows.

```python
from langchain.agents import create_agent

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[get_weather],
    system_prompt="You are a helpful assistant"
)
response = agent.invoke({"messages": [{"role": "user", "content": "Weather in SF?"}]})
```

The **standard model interface** solves the problem of different providers having unique APIs. Switch between OpenAI, Anthropic, and Google with no code changes. The `@tool` decorator turns any Python function into a tool the agent can use, with the docstring guiding the model's decisions.

LangChain agents inherit **LangGraph capabilities**: durable execution that survives failures and resumes, **human-in-the-loop** workflows that pause for approval, persistence for saving agent state, and streaming for real-time output.

**LangSmith** provides observability and debugging. Enable tracing to see execution paths, capture state transitions, view runtime metrics, and debug agent decisions—essential for production systems.

```python
os.environ["LANGSMITH_TRACING"] = "true"
```

Install with your provider package (`langchain[openai]`, `langchain[anthropic]`), define tools with `@tool`, create an agent, and invoke it. LangChain handles the tool execution loop, error cases, and response synthesis automatically.
