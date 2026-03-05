# AI Agentic Design Patterns with AutoGen

**Source:** DeepLearning.AI Course
**Instructors:** Chi Wang (Microsoft), Qingyun Wu (Penn State)
**Topics:** Multi-Agent Systems, Conversable Agents, Agent Patterns

---

## Table of Contents

1. [What is AutoGen?](#1-what-is-autogen)
2. [Core Concept: Conversable Agent](#2-core-concept-conversable-agent)
3. [Two-Agent Conversations](#3-two-agent-conversations)
4. [Sequential Chats](#4-sequential-chats)
5. [Reflection & Nested Chats](#5-reflection--nested-chats)
6. [Tool Use](#6-tool-use)
7. [Code Execution](#7-code-execution)
8. [Group Chats](#8-group-chats)

---

## 1. What is AutoGen?

AutoGen is a **multi-agent conversational framework** that enables you to:
- Create multiple agents with different roles and capabilities
- Implement complex AI applications using agentic design patterns
- Build systems where agents collaborate, critique, and iterate

### Example Use Case

Analyzing financial data might require:
- Writing code to collect share prices
- Analyzing the data
- Synthesizing findings into a report

A multi-agent system creates specialized agents (researcher, data collector, writer, executor) that work together, iteratively reviewing and improving results.

---

## 2. Core Concept: Conversable Agent

### What is a Conversable Agent?

The foundational building block in AutoGen—an entity that can:
- Act on behalf of human intent
- Send and receive messages
- Perform actions and generate replies
- Interact with other agents

### Creating Your First Agent

```python
from autogen import ConversableAgent

llm_config = {"model": "gpt-4o-mini"}

agent = ConversableAgent(
    name="chatbot",
    llm_config=llm_config,
    human_input_mode="NEVER"  # Never ask for human input
)

# Generate a single reply
reply = agent.generate_reply(
    messages=[{"content": "Tell me a joke", "role": "user"}]
)
```

### Human Input Modes

| Mode | Behavior |
|------|----------|
| `"NEVER"` | Agent uses LLM only, no human input |
| `"ALWAYS"` | Always asks human before generating reply |
| `"TERMINATE"` | Asks human only when conversation ends |

### Key Insight: Stateless vs Stateful

`generate_reply()` is **stateless**—each call is fresh. For maintaining conversation history, use `initiate_chat()`.

---

## 3. Two-Agent Conversations

### Creating a Conversation

Define two agents with different personas and let them interact:

```python
cathy = ConversableAgent(
    name="cathy",
    system_message="Your name is Cathy and you're a stand-up comedian.",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

joe = ConversableAgent(
    name="joe",
    system_message="""Your name is Joe and you're a stand-up comedian.
    Start the next joke from the punchline of the previous joke.""",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# Start conversation (Joe initiates, Cathy responds)
chat_result = joe.initiate_chat(
    recipient=cathy,
    message="I'm Joe. Cathy, let's keep the jokes rolling.",
    max_turns=2
)
```

### Inspecting Results

```python
# View chat history
print(chat_result.chat_history)

# Check token usage and cost
print(chat_result.cost)

# Get summary (default: last message)
print(chat_result.summary)
```

### Customizing Summaries

```python
chat_result = joe.initiate_chat(
    recipient=cathy,
    message="Let's tell jokes",
    max_turns=2,
    summary_method="reflection_with_llm",
    summary_prompt="Summarize the conversation"
)
# LLM generates a summary instead of using last message
```

### Termination Conditions

Instead of fixed `max_turns`, use dynamic termination:

```python
joe = ConversableAgent(
    name="joe",
    system_message="...",
    llm_config=llm_config,
    is_termination_msg=lambda msg: "goodbye" in msg["content"].lower()
)
```

---

## 4. Sequential Chats

### Pattern: Step-by-Step Task Execution

Run multiple conversations in sequence, with context carrying forward.

```
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Agent A │───▶│ Agent B │───▶│ Agent C │
│(Task 1) │    │(Task 2) │    │(Task 3) │
└─────────┘    └─────────┘    └─────────┘
     │              │              │
     └──────────────┴──────────────┘
            Context carries forward
```

### Example: Customer Onboarding

```python
from autogen import initiate_chats

# Define sequential chat configuration
chat_queue = [
    {
        "recipient": info_agent,
        "message": "Gather customer information",
        "summary_method": "reflection_with_llm",
        "carryover": "This is the first step."
    },
    {
        "recipient": topic_agent,
        "message": "What topics interest you?",
        "carryover": "Use previous context"
    },
    {
        "recipient": engagement_agent,
        "message": "Create personalized engagement plan",
        "carryover": "Use all previous context"
    }
]

# Execute all chats in sequence
results = initiate_chats(chat_queue)
```

### Carryover Mechanism

Each subsequent chat receives a "carryover" from previous chats—typically the summary. This enables:
- Information gathered in step 1 informs step 2
- Progressive context building
- Task decomposition with state

---

## 5. Reflection & Nested Chats

### Pattern: Iterative Improvement

One agent generates content, another critiques it, and they iterate.

```
┌───────────┐     ┌───────────┐
│  Writer   │────▶│  Critic   │
│           │◀────│           │
└───────────┘     └───────────┘
      │                 │
      └────── Loop ─────┘
```

### Example: Blog Post Writing

```python
writer = ConversableAgent(
    name="writer",
    system_message="You are a writer. Write content based on feedback.",
    llm_config=llm_config
)

critic = ConversableAgent(
    name="critic",
    system_message="""You are a critic. Provide feedback on writing.
    When satisfied, say 'APPROVED'.""",
    llm_config=llm_config,
    is_termination_msg=lambda msg: "APPROVED" in msg["content"]
)

# Writer initiates, critic reviews, iterate until APPROVED
chat_result = writer.initiate_chat(
    recipient=critic,
    message="Write a blog post about AI agents",
    max_turns=5
)
```

### Nested Chats: Agents Managing Agents

A nested chat triggers when an agent receives a message, allowing it to consult other agents before responding.

```
┌─────────────────────────────────────┐
│           Main Agent                │
│  ┌─────────────────────────────┐    │
│  │     Nested Conversation     │    │
│  │  ┌─────┐  ┌─────┐  ┌─────┐ │    │
│  │  │Sub 1│──│Sub 2│──│Sub 3│ │    │
│  │  └─────┘  └─────┘  └─────┘ │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

```python
# Register a nested chat for the writer
writer.register_nested_chats(
    [
        {
            "recipient": seo_expert,
            "message": "Review for SEO",
            "summary_method": "last_msg"
        },
        {
            "recipient": legal_reviewer,
            "message": "Check for compliance",
            "summary_method": "last_msg"
        }
    ],
    trigger=critic  # Triggers when receiving from critic
)
```

---

## 6. Tool Use

### Pattern: Agents with Functions

Provide agents with callable functions they can use during conversations.

```python
from autogen import register_function

def get_stock_price(symbol: str) -> float:
    """Get current stock price for a symbol."""
    # Implementation
    return 150.25

def calculate_return(buy_price: float, sell_price: float) -> float:
    """Calculate percentage return."""
    return ((sell_price - buy_price) / buy_price) * 100

# Create agents
analyst = AssistantAgent(
    name="analyst",
    system_message="You analyze stocks. Use available tools.",
    llm_config=llm_config
)

executor = UserProxyAgent(
    name="executor",
    human_input_mode="NEVER",
    code_execution_config=False
)

# Register tools with both agents
register_function(
    get_stock_price,
    caller=analyst,      # Suggests tool use
    executor=executor,   # Executes the tool
    description="Get current stock price"
)

register_function(
    calculate_return,
    caller=analyst,
    executor=executor,
    description="Calculate investment return"
)
```

### How Tool Use Works

```
┌─────────────┐    1. Suggest tool    ┌────────────┐
│   Analyst   │──────────────────────▶│  Executor  │
│  (Caller)   │                       │            │
│             │◀──────────────────────│            │
└─────────────┘    3. Return result   └────────────┘
                                            │
                                     2. Execute tool
                                            │
                                            ▼
                                     ┌────────────┐
                                     │  Function  │
                                     └────────────┘
```

---

## 7. Code Execution

### Pattern: Write and Execute Code

Agents can write code, execute it (in a sandbox), and use results.

```python
from autogen import AssistantAgent, UserProxyAgent

# Agent that writes code
coder = AssistantAgent(
    name="coder",
    system_message="""You are a helpful assistant.
    Write Python code to solve tasks. Put code in ```python blocks.""",
    llm_config=llm_config
)

# Agent that executes code
executor = UserProxyAgent(
    name="executor",
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": True  # Sandboxed execution
    }
)

# Coder writes code, executor runs it
executor.initiate_chat(
    coder,
    message="Plot the stock price of NVDA for the last year"
)
```

### Execution Flow

```
User Request → Coder writes Python → Executor runs code → Results back to Coder
                                          │
                               ┌──────────┴──────────┐
                               │  Docker Container   │
                               │  (Safe execution)   │
                               └─────────────────────┘
```

### Safety Considerations

| Setting | Purpose |
|---------|---------|
| `use_docker=True` | Run code in isolated container |
| `work_dir` | Specify where files are created |
| `timeout` | Limit execution time |

---

## 8. Group Chats

### Pattern: Multiple Agents Collaborating

For complex tasks requiring multiple specialists:

```python
from autogen import GroupChat, GroupChatManager

# Create specialized agents
planner = AssistantAgent(name="planner", ...)
researcher = AssistantAgent(name="researcher", ...)
writer = AssistantAgent(name="writer", ...)
reviewer = AssistantAgent(name="reviewer", ...)

# Create group chat
group_chat = GroupChat(
    agents=[planner, researcher, writer, reviewer],
    messages=[],
    max_round=10,
    speaker_selection_method="auto"  # LLM decides who speaks next
)

# Manager orchestrates the conversation
manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config
)

# Start the group chat
user_proxy.initiate_chat(
    manager,
    message="Create a report on AI trends for 2024"
)
```

### Speaker Selection Methods

| Method | Behavior |
|--------|----------|
| `"auto"` | LLM decides who speaks next |
| `"round_robin"` | Agents take turns in order |
| `"random"` | Random agent selection |
| Custom function | Your logic determines speaker |

### Adding a Planning Agent

```python
planner = AssistantAgent(
    name="planner",
    system_message="""You are a planning agent.
    Given a task:
    1. Break it into subtasks
    2. Assign each subtask to the appropriate agent
    3. Specify the order of execution

    Available agents: researcher, writer, reviewer"""
)
```

### Controlling Flow

```python
def custom_speaker_selection(last_speaker, groupchat):
    """Custom logic for who speaks next."""
    messages = groupchat.messages

    if last_speaker.name == "planner":
        return researcher
    elif last_speaker.name == "researcher":
        return writer
    elif last_speaker.name == "writer":
        return reviewer
    else:
        return planner

group_chat = GroupChat(
    agents=[planner, researcher, writer, reviewer],
    speaker_selection_method=custom_speaker_selection
)
```

---

## Design Patterns Summary

| Pattern | When to Use | Example |
|---------|-------------|---------|
| **Two-Agent Chat** | Simple back-and-forth | Debate, interview |
| **Sequential Chat** | Step-by-step pipeline | Customer onboarding |
| **Reflection** | Iterative improvement | Writing with review |
| **Nested Chat** | Agent needs sub-consultations | Complex research |
| **Tool Use** | Need external functions | API calls, calculations |
| **Code Execution** | Dynamic computation | Data analysis, plotting |
| **Group Chat** | Multi-expert collaboration | Report generation |

---

## Best Practices

1. **Clear System Messages** - Define each agent's role and capabilities precisely
2. **Termination Conditions** - Always define when conversations should end
3. **Appropriate Patterns** - Match the pattern to your task complexity
4. **Sandboxed Execution** - Use Docker for code execution
5. **Summary Methods** - Use LLM reflection for better context carryover
6. **Token Management** - Monitor costs with `chat_result.cost`
