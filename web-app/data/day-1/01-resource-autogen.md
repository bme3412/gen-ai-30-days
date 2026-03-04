# AI Agentic Design Patterns with AutoGen

**Course:** DeepLearning.AI in partnership with Microsoft
**Instructors:** Chi Wang (Principal Researcher, Microsoft) & Qingyun Wu (Assistant Professor, Penn State University)

---

## Table of Contents

1. [Introduction & Course Overview](#1-introduction--course-overview)
2. [Lesson 1: Conversable Agents](#2-lesson-1-conversable-agents)
3. [Lesson 2: Sequential Chats](#3-lesson-2-sequential-chats)
4. [Lesson 3: Reflection & Nested Chats](#4-lesson-3-reflection--nested-chats)
5. [Lesson 4: Tool Use](#5-lesson-4-tool-use)
6. [Lesson 5: Coding & Code Execution](#6-lesson-5-coding--code-execution)
7. [Lesson 6: Group Chats & Planning](#7-lesson-6-group-chats--planning)

---

## 1. Introduction & Course Overview

This course is an introduction to **AutoGen**, a multi-agent conversational framework that enables you to quickly create multiple agents with different roles, persona tasks, and capabilities to implement complex AI applications using different AI agentic design patterns.

### Motivating Example: Financial Analysis

Let's say you're interested in analyzing financial data. The task may require:
- Writing code to collect and analyze share prices
- Synthesizing findings into a report

This might take a person days of research, coding, and writing. A multi-agent system can streamline this process by enabling you to create and hire agents that work for you as a researcher, data collector, co-writer, and executor. Your agents can also iteratively review, critique, and improve the results until it meets your standard.

### Course Lessons Overview

| Lesson | Topic | Key Concept |
|--------|-------|-------------|
| 1 | Conversable Agent | Building block agent, two-agent standup comedy chat |
| 2 | Sequential Chats | Multi-step tasks in sequence, customer onboarding |
| 3 | Reflection & Nested Chats | Agent reflection framework, blog post with reviewers |
| 4 | Tool Use | User-defined functions, conversational chess game |
| 5 | Coding & Code Execution | Free-form code generation, financial analysis |
| 6 | Group Chats & Planning | Multi-agent collaboration, planning agent |

---

## 2. Lesson 1: Conversable Agents

### What is an Agent in AutoGen?

In AutoGen, an **agent** is an entity that can:
- Act on behalf of human intent
- Send and receive messages
- Perform actions and generate replies
- Interact with other agents

### The Conversable Agent Class

AutoGen has a built-in agent class called `ConversableAgent` that unifies different types of agents in the same programming abstraction.

**Built-in capabilities:**
- LLM configurations for generating replies
- Code execution
- Function and tool execution
- Human-in-the-loop support
- Termination condition checking

You can switch each component on and off and customize it to suit your application needs.

### Basic Setup

```python
from autogen import ConversableAgent
from utils import get_openai_api_key

# Get API key and define LLM config
api_key = get_openai_api_key()
llm_config = {"model": "gpt-3.5-turbo"}

# Create a basic conversable agent
agent = ConversableAgent(
    name="chatbot",
    llm_config=llm_config,
    human_input_mode="never"  # Agent won't ask for human input
)
```

### Human Input Modes

| Mode | Behavior |
|------|----------|
| `"never"` | Agent never asks for human input |
| `"always"` | Agent always asks for human input before generating reply |
| `"terminate"` | Agent asks for input only at termination |

### Generating Replies

```python
# Generate a single reply (stateless)
reply = agent.generate_reply(
    messages=[{"content": "Tell me a joke", "role": "user"}]
)
```

**Important:** The `generate_reply` function doesn't alter the internal state of the agent. Each call is independent—the agent won't remember previous interactions.

### Two-Agent Conversation: Standup Comedy Example

To maintain state and have ongoing conversations, use the `initiate_chat` method:

```python
# Create two comedian agents
cathy = ConversableAgent(
    name="cathy",
    system_message="Your name is Cathy and you're a stand-up comedian.",
    llm_config=llm_config,
    human_input_mode="never"
)

joe = ConversableAgent(
    name="joe",
    system_message="Your name is Joe and you're a stand-up comedian. Start the next joke from the punchline of the previous joke.",
    llm_config=llm_config,
    human_input_mode="never"
)

# Start the conversation
chat_result = joe.initiate_chat(
    recipient=cathy,
    message="I'm Joe. Cathy, let's keep the jokes rolling.",
    max_turns=2
)
```

### Inspecting Chat Results

```python
import pprint

# View chat history
pprint.pprint(chat_result.chat_history)

# Check token usage and cost
pprint.pprint(chat_result.cost)

# Get summary (default: last message)
print(chat_result.summary)
```

### Custom Summary Methods

```python
chat_result = joe.initiate_chat(
    recipient=cathy,
    message="Let's keep the jokes rolling.",
    max_turns=2,
    summary_method="reflection_with_llm",
    summary_prompt="Summarize the conversation."
)
```

### Custom Termination Conditions

Instead of using `max_turns`, you can define custom stopping conditions:

```python
def is_termination_msg(msg):
    return "I gotta go" in msg.get("content", "").lower()

cathy = ConversableAgent(
    name="cathy",
    system_message="Your name is Cathy... When you're ready to end the conversation, say 'I gotta go.'",
    llm_config=llm_config,
    human_input_mode="never",
    is_termination_msg=is_termination_msg
)
```

### Key Takeaways

- **ConversableAgent** is the building block for all AutoGen agents
- Use **system_message** to define agent behavior and persona
- Use **generate_reply** for stateless single responses
- Use **initiate_chat** for stateful conversations between agents
- Conversations can be controlled via `max_turns` or custom termination functions

---

## 3. Lesson 2: Sequential Chats

Sequential chats allow you to build conversational agents that work step by step to carry out a list of tasks in a sequence.

### Use Case: Customer Onboarding

A typical onboarding procedure:
1. **Information Collecting** - Gather customer personal information
2. **Interest Survey** - Survey customer's interests
3. **Customer Engagement** - Engage based on collected information

### Creating Specialized Agents

```python
from autogen import ConversableAgent

# Agent 1: Collect personal information
onboarding_personal_info_agent = ConversableAgent(
    name="onboarding_personal_info_agent",
    system_message="""You are a customer onboarding agent.
    Your job is to collect the customer's name and location.""",
    llm_config=llm_config,
    human_input_mode="never"
)

# Agent 2: Survey topic preferences
onboarding_topic_agent = ConversableAgent(
    name="onboarding_topic_agent",
    system_message="""You are a customer onboarding agent.
    Your job is to ask the customer about their interests and topic preferences.""",
    llm_config=llm_config,
    human_input_mode="never"
)

# Agent 3: Customer engagement
engagement_agent = ConversableAgent(
    name="engagement_agent",
    system_message="""You are a customer engagement agent.
    Provide fun facts, jokes, or stories based on the customer's
    personal information and topic preferences.""",
    llm_config=llm_config,
    human_input_mode="never"
)

# Customer proxy (represents the real user)
customer_proxy = ConversableAgent(
    name="customer_proxy",
    llm_config=llm_config,
    human_input_mode="always"  # Always asks for real user input
)
```

### Defining the Chat Sequence

```python
chats = [
    {
        "sender": onboarding_personal_info_agent,
        "recipient": customer_proxy,
        "message": "Hello! What is your name and location?",
        "max_turns": 2,
        "summary_method": "reflection_with_llm",
        "summary_prompt": "Return the customer information as JSON: {name: '', location: ''}"
    },
    {
        "sender": onboarding_topic_agent,
        "recipient": customer_proxy,
        "message": "Great! Could you tell me what topics you're interested in reading about?",
        "max_turns": 1,
        "summary_method": "reflection_with_llm"
    },
    {
        "sender": customer_proxy,
        "recipient": engagement_agent,
        "message": "Let's find something to read.",
        "max_turns": 1
    }
]
```

### Running Sequential Chats

```python
# Execute the entire onboarding process
results = customer_proxy.initiate_chats(chats)

# Inspect results from each chat
for i, result in enumerate(results):
    print(f"Chat {i+1} Summary: {result.summary}")
    print(f"Chat {i+1} Cost: {result.cost}")
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Summary carryover** | Information from previous chats is passed to subsequent chats |
| **Flexible summary methods** | Use `"last_msg"` or `"reflection_with_llm"` |
| **Custom summary prompts** | Control how information is extracted and formatted |
| **Max turns per chat** | Control conversation length at each step |

### Key Takeaways

- Sequential chats enable **multi-step workflows** with dependent tasks
- Each chat can have different agents, messages, and configurations
- **Summary methods** extract and pass information between chat sessions
- The `human_input_mode="always"` setting enables real user participation

---

## 4. Lesson 3: Reflection & Nested Chats

The agent reflection framework uses multiple agents to produce higher-quality outputs through iterative feedback and improvement.

### Use Case: Writing a High-Quality Blog Post

**Task:** Write a concise but engaging blog post about DeepLearning.AI (within 100 words).

### Basic Reflection with Two Agents

```python
from autogen import AssistantAgent

# Writer agent
writer = AssistantAgent(
    name="writer",
    llm_config=llm_config
)

# Critic agent
critic = ConversableAgent(
    name="critic",
    system_message="You are a critic. Investigate the writing and provide feedback.",
    llm_config=llm_config,
    human_input_mode="never"
)

# Two-agent reflection
chat_result = critic.initiate_chat(
    recipient=writer,
    message=task,
    max_turns=2
)
```

### Advanced Reflection with Nested Chats

For more sophisticated reflection, use **nested chats** to have multiple specialized reviewers examine different aspects of the work.

#### Creating Specialized Reviewers

```python
# SEO Reviewer
seo_reviewer = ConversableAgent(
    name="seo_reviewer",
    system_message="""You are an SEO reviewer responsible for optimizing
    content for search engines to ensure it ranks well and attracts organic traffic.
    Provide concise suggestions. Begin your review with 'SEO Review:'""",
    llm_config=llm_config,
    human_input_mode="never"
)

# Legal Reviewer
legal_reviewer = ConversableAgent(
    name="legal_reviewer",
    system_message="""You are a legal reviewer responsible for ensuring
    the content is legally compliant.
    Provide concise suggestions. Begin your review with 'Legal Review:'""",
    llm_config=llm_config,
    human_input_mode="never"
)

# Ethics Reviewer
ethics_reviewer = ConversableAgent(
    name="ethics_reviewer",
    system_message="""You are an ethics reviewer responsible for ensuring
    the content is ethically sound and free from potential ethical issues.
    Provide concise suggestions. Begin your review with 'Ethics Review:'""",
    llm_config=llm_config,
    human_input_mode="never"
)

# Meta Reviewer (aggregates all reviews)
meta_reviewer = ConversableAgent(
    name="meta_reviewer",
    system_message="""You are a meta reviewer. Aggregate all reviews
    from the other reviewers and provide final suggestions.""",
    llm_config=llm_config,
    human_input_mode="never"
)
```

#### Defining the Review Chat Sequence

```python
def reflection_message(recipient, messages, sender, config):
    """Extract content from outer chat for inner reviewers."""
    return sender.chat_messages_for_summary(recipient)[-1]["content"]

review_chats = [
    {
        "recipient": seo_reviewer,
        "message": reflection_message,
        "summary_method": "reflection_with_llm",
        "summary_prompt": "Return review as JSON: {reviewer: 'SEO', review: '...'}",
        "max_turns": 1
    },
    {
        "recipient": legal_reviewer,
        "message": reflection_message,
        "summary_method": "reflection_with_llm",
        "summary_prompt": "Return review as JSON: {reviewer: 'Legal', review: '...'}",
        "max_turns": 1
    },
    {
        "recipient": ethics_reviewer,
        "message": reflection_message,
        "summary_method": "reflection_with_llm",
        "summary_prompt": "Return review as JSON: {reviewer: 'Ethics', review: '...'}",
        "max_turns": 1
    },
    {
        "recipient": meta_reviewer,
        "message": "Aggregate all reviews and provide final suggestions.",
        "max_turns": 1
    }
]
```

#### Registering Nested Chat

```python
# Register the review sequence as a nested chat within the critic
critic.register_nested_chats(
    review_chats,
    trigger=writer  # Triggered when critic receives message from writer
)

# Now initiate the main chat
chat_result = critic.initiate_chat(
    recipient=writer,
    message=task,
    max_turns=2
)
```

### How Nested Chats Work

```
Writer sends draft → Critic receives draft
                          ↓
              [Nested Chat Triggered]
                          ↓
         SEO Reviewer → Legal Reviewer → Ethics Reviewer → Meta Reviewer
                          ↓
              Aggregated feedback returned
                          ↓
         Critic sends feedback → Writer revises
```

### Key Takeaways

- **Reflection** improves output quality through iterative feedback
- **Nested chats** enable complex workflows within a single agent's response
- Use **specialized reviewers** for different aspects (SEO, legal, ethics, etc.)
- The **meta reviewer** pattern aggregates multiple perspectives
- Nested chats are triggered based on the `trigger` parameter

---

## 5. Lesson 4: Tool Use

Tool use allows agents to call user-defined functions to perform specific actions.

### Use Case: Conversational Chess Game

Build a chess game where two AI players make legal moves on a chessboard.

### Defining Tools (Functions)

```python
import chess
from typing import Annotated

# Initialize the chess board
board = chess.Board()
made_move = False

def get_legal_moves() -> Annotated[str, "List of legal moves in UCI format"]:
    """Returns a string of all legal moves available."""
    return f"Possible moves are: {', '.join([str(m) for m in board.legal_moves])}"

def make_move(move: Annotated[str, "A move in UCI format"]) -> Annotated[str, "Result of the move"]:
    """Make a move on the chess board."""
    global made_move

    chess_move = chess.Move.from_uci(move)
    board.push(chess_move)
    made_move = True

    # Get piece information
    piece = board.piece_at(chess_move.to_square)
    piece_name = chess.piece_name(piece.piece_type).capitalize()

    return f"Moved {piece_name} from {chess.square_name(chess_move.from_square)} to {chess.square_name(chess_move.to_square)}"
```

### Creating Player Agents

```python
from autogen import ConversableAgent, register_function

# Player White
player_white = ConversableAgent(
    name="player_white",
    system_message="""You are a chess player and you play as white.
    First call get_legal_moves to get a list of legal moves,
    then call make_move to make a move.""",
    llm_config={"model": "gpt-4-turbo"}
)

# Player Black
player_black = ConversableAgent(
    name="player_black",
    system_message="""You are a chess player and you play as black.
    First call get_legal_moves to get a list of legal moves,
    then call make_move to make a move.""",
    llm_config={"model": "gpt-4-turbo"}
)

# Board Proxy (executes the moves)
def check_made_move(msg):
    return made_move

board_proxy = ConversableAgent(
    name="board_proxy",
    llm_config=False,  # No LLM needed
    is_termination_msg=check_made_move,
    default_auto_reply="Please make a move.",
    human_input_mode="never"
)
```

### Registering Tools

```python
from autogen import register_function

# Register tools for both players
for player in [player_white, player_black]:
    register_function(
        get_legal_moves,
        caller=player,      # Agent that proposes the function call
        executor=board_proxy,  # Agent that executes the function
        name="get_legal_moves",
        description="Get the list of legal chess moves"
    )

    register_function(
        make_move,
        caller=player,
        executor=board_proxy,
        name="make_move",
        description="Make a chess move"
    )
```

### Using Nested Chats for Tool Execution

```python
# Register nested chat for player white
player_white.register_nested_chats(
    trigger=player_black,
    chat_queue=[{
        "sender": board_proxy,
        "recipient": player_white,
        "summary_method": "last_msg"
    }]
)

# Register nested chat for player black
player_black.register_nested_chats(
    trigger=player_white,
    chat_queue=[{
        "sender": board_proxy,
        "recipient": player_black,
        "summary_method": "last_msg"
    }]
)
```

### Starting the Game

```python
# Initialize the board
board = chess.Board()

# Start the game
chat_result = player_black.initiate_chat(
    recipient=player_white,
    message="Let's play chess. Your move.",
    max_turns=2
)
```

### Tool Registration Details

When you register a function, AutoGen automatically populates the OpenAI function definition:

```python
# Inspect registered tools
print(player_black.llm_config["tools"])
```

Output:
```json
[
  {
    "type": "function",
    "function": {
      "name": "get_legal_moves",
      "description": "Get the list of legal chess moves",
      "parameters": {...}
    }
  },
  {
    "type": "function",
    "function": {
      "name": "make_move",
      "description": "Make a chess move",
      "parameters": {...}
    }
  }
]
```

### Adding Chitchat for Fun

```python
player_white = ConversableAgent(
    name="player_white",
    system_message="""You are a chess player and you play as white.
    First call get_legal_moves, then call make_move.
    After a move is made, chitchat to make the game fun.""",
    llm_config=llm_config
)
```

### Key Takeaways

- **Tool use** allows agents to call user-defined Python functions
- Use `register_function` to give agents access to tools
- The **caller** proposes the function call; the **executor** runs it
- **Type annotations** (using `Annotated`) help the agent understand function parameters
- Nested chats can encapsulate tool execution within agent conversations

---

## 6. Lesson 5: Coding & Code Execution

This lesson covers how to add coding capabilities to agents for tasks like financial analysis.

### Two Approaches

1. **Free-form code generation** - Agent writes code from scratch
2. **User-provided functions** - Agent uses pre-defined utility functions

### Setting Up the Code Executor

```python
from autogen.coding import LocalCommandLineCodeExecutor

executor = LocalCommandLineCodeExecutor(
    timeout=60,  # Timeout in seconds
    work_dir="coding"  # Directory for code files and outputs
)
```

### Creating Agents for Coding

```python
from autogen import ConversableAgent, AssistantAgent

# Code Executor Agent (runs the code)
code_executor_agent = ConversableAgent(
    name="code_executor_agent",
    llm_config=False,  # No LLM - just executes code
    code_execution_config={"executor": executor},
    human_input_mode="always",  # Always ask before executing
    default_auto_reply="Please continue. If everything is done, reply TERMINATE."
)

# Code Writer Agent (writes the code)
code_writer_agent = AssistantAgent(
    name="code_writer_agent",
    llm_config={"model": "gpt-4-turbo"},
    code_execution_config=False,  # Doesn't execute, only proposes
    human_input_mode="never"
)
```

### The AssistantAgent Default System Message

The `AssistantAgent` class comes with a comprehensive default system message that includes:
- Instructions for solving tasks using code
- Guidelines for when to use Python vs shell scripts
- Code formatting requirements (markdown code blocks)
- Error handling instructions
- Termination protocol

### Example Task: Financial Analysis

```python
from datetime import date

task = f"""Today is {date.today().strftime('%Y-%m-%d')}.
Create a plot showing stock gain year-to-date for NVIDIA and Tesla.
Make sure the code is in a markdown code block and save the figure to a file."""

# Start the coding conversation
chat_result = code_executor_agent.initiate_chat(
    recipient=code_writer_agent,
    message=task
)
```

### Workflow

```
1. Executor sends task → Writer generates code plan and Python code
2. User reviews code (human_input_mode="always")
3. User presses Enter to approve → Executor runs the code
4. Results sent back to Writer for verification
5. Writer replies TERMINATE when done
```

### Using User-Defined Functions

For more control, provide your own utility functions:

```python
def get_stock_prices(symbol: str, start_date: str, end_date: str) -> dict:
    """
    Get stock prices for a given symbol between start and end dates.

    Args:
        symbol: Stock ticker symbol (e.g., 'NVDA')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Dictionary with stock price data
    """
    import yfinance as yf
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data.to_dict()

def plot_stock_prices(stock_data: dict, filename: str) -> str:
    """
    Plot stock prices and save to a file.

    Args:
        stock_data: Dictionary with stock price data
        filename: Output filename for the plot

    Returns:
        Confirmation message
    """
    import matplotlib.pyplot as plt
    # ... plotting code ...
    plt.savefig(filename)
    return f"Plot saved to {filename}"
```

### Registering User Functions with Executor

```python
# Create executor with user-defined functions
executor = LocalCommandLineCodeExecutor(
    timeout=60,
    work_dir="coding",
    functions=[get_stock_prices, plot_stock_prices]  # Provide functions
)

# Update the code writer's system message
code_writer_agent.update_system_message(
    code_writer_agent.system_message +
    executor.format_functions_for_prompt()  # Auto-generates function docs
)
```

The system message now includes:
```
You have access to the following user-defined functions:
- from functions import get_stock_prices
- from functions import plot_stock_prices

[Function signatures and docstrings...]
```

### Code Execution vs Tool Execution

| Aspect | Tool Execution | Code Execution |
|--------|---------------|----------------|
| **Flexibility** | Restricted to predefined functions | Can write any code |
| **Determinism** | More deterministic | Less deterministic |
| **Model requirements** | Requires function calling support | Works with any model |
| **Security** | Safer (controlled functions) | Requires sandboxing |
| **Use case** | Known, specific operations | Creative problem-solving |

### Key Takeaways

- **LocalCommandLineCodeExecutor** runs code in a local directory
- **AssistantAgent** has a built-in system message for coding tasks
- Set `human_input_mode="always"` to review code before execution
- User-defined functions provide more control and cleaner code
- Code execution doesn't require the model to support function calling

---

## 7. Lesson 6: Group Chats & Planning

Group chats enable multiple agents to collaborate dynamically without pre-defined conversation sequences.

### Use Case: Stock Performance Report

**Task:** Write a detailed report on recent stock performance, including major news events, analysis of stock gains/losses, and interesting facts.

### Creating Specialized Agents

```python
import autogen
from autogen import ConversableAgent, AssistantAgent

# Admin (user proxy)
admin = ConversableAgent(
    name="Admin",
    system_message="Give the task and send instructions to writer to refine the blog post.",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="always"
)

# Planner (decomposes tasks)
planner = ConversableAgent(
    name="Planner",
    system_message="""Given a task, determine what information is needed to complete it.
    Information will be retrieved using Python code.
    After each step is done by others, check progress and instruct remaining steps.
    If a step fails, try to workaround.""",
    description="Given a task, determine what information is needed. Check progress and instruct remaining steps.",
    llm_config=llm_config
)

# Engineer (writes code)
engineer = AssistantAgent(
    name="Engineer",
    description="Writes code based on the plan provided by planner.",
    llm_config=llm_config
)

# Executor (runs code)
executor = ConversableAgent(
    name="Executor",
    description="Executes the code written by the engineer.",
    code_execution_config={"last_n_messages": 3, "work_dir": "coding", "use_docker": False},
    human_input_mode="never"
)

# Writer (creates content)
writer = ConversableAgent(
    name="Writer",
    system_message="""Write blogs in markdown format with relevant titles.
    Put content in markdown code block.
    Take feedback from Admin and refine the blog.""",
    description="Writes and refines the blog post.",
    llm_config=llm_config
)
```

### System Message vs Description

| Attribute | Purpose |
|-----------|---------|
| **system_message** | Internal instructions that only this agent sees |
| **description** | External description that other agents (and the manager) use to understand this agent's role |

### Creating the Group Chat

```python
from autogen import GroupChat, GroupChatManager

# Create the group chat
group_chat = GroupChat(
    agents=[admin, planner, engineer, executor, writer],
    messages=[],
    max_round=10
)

# Create the manager
manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config
)
```

### Starting the Group Chat

```python
task = """Write a blog post about recent stock performance.
Include analysis of gains/losses and major news events."""

# Initiate the group chat
chat_result = admin.initiate_chat(
    recipient=manager,
    message=task
)
```

### How Group Chat Works

1. **Admin** sends the task to the **Manager**
2. **Manager** broadcasts the message to all agents
3. **Manager** uses LLM to select the next speaker based on:
   - Current conversation history
   - Agent descriptions
   - Task requirements
4. Selected agent responds
5. Process repeats until `max_round` is reached or termination

### Typical Flow

```
Admin → [Task] → Manager
Manager selects → Planner (creates plan)
Manager selects → Engineer (writes code for step 1)
Manager selects → Executor (runs code)
Manager selects → Planner (reviews progress, suggests step 2)
Manager selects → Engineer (writes code for step 2)
Manager selects → Executor (runs code)
Manager selects → Writer (drafts blog post)
Manager selects → Admin (requests feedback)
Manager selects → Writer (refines based on feedback)
```

### Constraining Speaker Transitions

For more control, define allowed speaker transitions:

```python
allowed_transitions = {
    admin: [planner, writer],
    planner: [engineer, writer],
    engineer: [admin, executor],
    executor: [admin, engineer, planner],  # Note: writer not allowed after executor
    writer: [admin, planner]
}

group_chat = GroupChat(
    agents=[admin, planner, engineer, executor, writer],
    messages=[],
    max_round=10,
    allowed_or_disallowed_speaker_transitions=allowed_transitions,
    speaker_transitions_type="allowed"
)
```

This ensures the **planner** reviews progress before the **writer** starts writing, because the writer cannot directly follow the executor.

### Benefits of Constrained Transitions

- Prevents premature task completion
- Ensures proper review steps
- Still maintains flexibility compared to sequential chats
- Can simulate finite state machine behavior

### Key Takeaways

- **Group chat** provides dynamic multi-agent collaboration
- The **Manager** uses LLM to select speakers based on context
- Use **description** to help the manager understand agent roles
- A **Planner agent** helps with task decomposition
- **Speaker transition constraints** add control without losing flexibility
- Group chat is more flexible than sequential chat but less deterministic

---

## Summary: Agentic Design Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Conversable Agent** | Basic building block for agent interactions | Simple Q&A, two-agent chats |
| **Sequential Chats** | Predefined sequence of agent interactions | Multi-step workflows with known steps |
| **Reflection** | Agents critique and improve each other's work | Quality improvement, review processes |
| **Nested Chats** | Sub-conversations within a main conversation | Complex internal workflows, tool execution |
| **Tool Use** | Agents call user-defined functions | Specific operations, API calls, game moves |
| **Code Execution** | Agents write and execute code | Data analysis, file generation |
| **Group Chat** | Dynamic multi-agent collaboration | Complex tasks requiring planning |
| **Planning** | Agent decomposes tasks into subtasks | Task management, adaptive workflows |

---

## Additional Resources

- **AutoGen Documentation:** https://microsoft.github.io/autogen/
- **AutoGen GitHub:** https://github.com/microsoft/autogen

### Advanced Topics (not covered in this course)

- Teaching agents to improve over time
- Multimodal agents (vision capabilities)
- Using OpenAI Assistants as agent backends
- Agent-based evaluation and benchmarking tools
- Automated agent design for specific tasks
