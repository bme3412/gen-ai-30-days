# LangChain Agents Overview

**Source:** LangChain Documentation
**Topics:** Agents, LangGraph, Deep Agents, Model Integrations

---

## Table of Contents

1. [What is LangChain?](#1-what-is-langchain)
2. [Creating Your First Agent](#2-creating-your-first-agent)
3. [LangChain vs LangGraph vs Deep Agents](#3-langchain-vs-langgraph-vs-deep-agents)
4. [Core Benefits](#4-core-benefits)
5. [Getting Started](#5-getting-started)

---

## 1. What is LangChain?

LangChain is an **open-source framework** with a pre-built agent architecture and integrations for any model or tool—allowing you to build agents that adapt as fast as the ecosystem evolves.

### Key Capabilities

- Connect to **15+ providers** (OpenAI, Anthropic, Google, and more) with under 10 lines of code
- Pre-built agent architecture that handles reasoning, tool use, and memory
- Seamlessly swap providers without code changes
- Built on top of **LangGraph** for durable execution, streaming, and human-in-the-loop

---

## 2. Creating Your First Agent

```python
# pip install -qU langchain "langchain[anthropic]"
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[get_weather],
    system_prompt="You are a helpful assistant"
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]}
)
```

That's it—a working agent in under 10 lines.

---

## 3. LangChain vs LangGraph vs Deep Agents

### Decision Guide

| Framework | When to Use | Complexity |
|-----------|-------------|------------|
| **Deep Agents** | Start here for most agent use cases | Batteries-included |
| **LangChain** | Need to customize agent behavior | Medium |
| **LangGraph** | Complex deterministic + agentic workflows | Advanced |

### Deep Agents

The recommended starting point. Comes with modern features:
- **Automatic compression** of long conversations
- **Virtual filesystem** for agent operations
- **Subagent-spawning** for managing and isolating context

Deep Agents are implementations of LangChain agents with sensible defaults.

### LangChain Agents

For when you need customization beyond Deep Agents:
- Custom context engineering
- Specific tool configurations
- Modified reasoning patterns

### LangGraph

The low-level agent orchestration framework for:
- Combining **deterministic** and **agentic** workflows
- Heavy customization requirements
- Complex state management needs

**Note:** LangChain agents are built on LangGraph internally. You don't need to know LangGraph for basic agent usage.

---

## 4. Core Benefits

### Standard Model Interface

Different providers have unique APIs. LangChain standardizes interactions so you can:
- Switch between OpenAI, Anthropic, Google with no code changes
- Avoid vendor lock-in
- Compare models easily

```python
# Same code works with different providers
from langchain.llms import ChatOpenAI, ChatAnthropic

# Switch provider by changing one line
llm = ChatOpenAI(model="gpt-4o")
# llm = ChatAnthropic(model="claude-sonnet-4-20250514")
```

### Easy to Use, Highly Flexible

- **Quick start:** Build a simple agent in under 10 lines
- **Customizable:** Full control over context engineering when needed

### Built on LangGraph

LangChain agents inherit LangGraph capabilities:
- **Durable execution** - Survive failures and resume
- **Human-in-the-loop** - Pause for human approval
- **Persistence** - Save and restore agent state
- **Streaming** - Real-time output as the agent works

### Debug with LangSmith

Gain visibility into agent behavior:
- Trace execution paths
- Capture state transitions
- View detailed runtime metrics
- Debug complex agent decisions

```python
# Enable tracing
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-api-key"
```

---

## 5. Getting Started

### Installation

```bash
pip install langchain
pip install langchain[openai]      # For OpenAI
pip install langchain[anthropic]   # For Anthropic
pip install langchain[google]      # For Google
```

### Quick Example: Agent with Tools

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def search_database(query: str) -> str:
    """Search the company database for information."""
    # Your implementation
    return f"Results for: {query}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    # Your implementation
    return f"Email sent to {to}"

agent = create_agent(
    model="gpt-4o",
    tools=[search_database, send_email],
    system_prompt="""You are a helpful assistant that can search
    databases and send emails on behalf of users."""
)

response = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Find John's email and send him a meeting request"
    }]
})
```

### Next Steps

1. **Quickstart Guide** - Build your first agent step by step
2. **Tool Documentation** - Learn to create custom tools
3. **LangSmith Setup** - Enable tracing for debugging
4. **LangGraph Docs** - For advanced orchestration needs

---

## Summary

| Concept | Description |
|---------|-------------|
| **LangChain** | High-level framework for building agents |
| **LangGraph** | Low-level orchestration (used internally) |
| **Deep Agents** | Batteries-included agent implementations |
| **LangSmith** | Observability and debugging platform |
| **Tools** | Functions that agents can call |
