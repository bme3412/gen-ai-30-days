# Building Agentic RAG with LlamaIndex

**Source:** DeepLearning.AI Course
**Instructor:** Jerry Liu (Co-founder & CEO, LlamaIndex)
**Topics:** Agentic RAG, Routing, Tool Use, Multi-Step Reasoning, Multi-Document Agents

---

## Table of Contents

1. [Introduction: Beyond Basic RAG](#1-introduction-beyond-basic-rag)
2. [Lesson 1: Router Query Engine](#2-lesson-1-router-query-engine)
3. [Lesson 2: Tool Calling with Metadata Filters](#3-lesson-2-tool-calling-with-metadata-filters)
4. [Lesson 3: Building an Agent Reasoning Loop](#4-lesson-3-building-an-agent-reasoning-loop)
5. [Lesson 4: Multi-Document Agents](#5-lesson-4-multi-document-agents)

---

## 1. Introduction: Beyond Basic RAG

### What is Agentic RAG?

**Agentic RAG** is a framework for building research agents capable of reasoning and decision-making over your data. Unlike standard RAG pipelines, agentic RAG can:

- Handle complex, multi-step queries
- Dynamically decide which tools or data sources to use
- Adjust its approach based on intermediate results
- Synthesize information across multiple documents

### Standard RAG vs Agentic RAG

| Aspect | Standard RAG | Agentic RAG |
|--------|-------------|-------------|
| Query handling | Single retrieval + synthesis | Multi-step reasoning |
| Decision making | None | Dynamic tool selection |
| Adaptability | Fixed pipeline | Adjusts based on results |
| Best for | Simple questions over small doc sets | Complex research questions |

### The Progression of Reasoning Capabilities

The course builds up reasoning capabilities incrementally:

1. **Routing** - Add decision-making to route requests to multiple tools
2. **Tool Use** - Create interfaces for agents to select tools and generate arguments
3. **Multi-Step Reasoning** - Use LLMs to perform multi-step reasoning with memory

---

## 2. Lesson 1: Router Query Engine

### Core Concept

A **router** picks one of several query engines to execute a query, providing dynamic query understanding capabilities.

### Building a Router Over a Single Document

```python
from llama_index.core import VectorStoreIndex, SummaryIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# Load and split document
documents = SimpleDirectoryReader(input_files=["metagpt.pdf"]).load_data()
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

# Create two different indexes
summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

# Create query engines from indexes
summary_query_engine = summary_index.as_query_engine(use_async=True)
vector_query_engine = vector_index.as_query_engine()
```

### Understanding Index Types

| Index Type | Behavior | Best For |
|------------|----------|----------|
| **Vector Index** | Returns most similar nodes by embedding | Specific questions, fact lookup |
| **Summary Index** | Returns ALL nodes in the index | Summarization, overview questions |

### Creating Query Tools with Descriptions

```python
# Wrap query engines as tools with descriptions
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description="Useful for summarization questions related to MetaGPT"
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for retrieving specific context from the MetaGPT paper"
)
```

### Building the Router

```python
# Create router with LLM-based selection
router_query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[summary_tool, vector_tool],
    verbose=True
)

# Query - router automatically picks the right tool
response = router_query_engine.query("What is a summary of the document?")
# Output: Selects query engine 0 (summary tool)

response = router_query_engine.query("How do agents share information?")
# Output: Selects query engine 1 (vector tool)
```

### How Selection Works

The LLM selector:
1. Receives the user query
2. Examines tool descriptions
3. Reasons about which tool best matches the query intent
4. Returns the selected tool index with reasoning

---

## 3. Lesson 2: Tool Calling with Metadata Filters

### Beyond Simple Routing

In Lesson 1, the router picked between tools. Now we extend this so the LLM can:
- **Pick a function** to execute
- **Infer arguments** to pass to that function

### The FunctionTool Abstraction

```python
from llama_index.core.tools import FunctionTool

def add(x: int, y: int) -> int:
    """Adds two integers together."""
    return x + y

def mystery(x: int, y: int) -> int:
    """Mystery function on two integers."""
    return (x + y) * (x + y)

# Wrap Python functions as tools
add_tool = FunctionTool.from_defaults(fn=add)
mystery_tool = FunctionTool.from_defaults(fn=mystery)
```

**Key Insight:** Type annotations and docstrings are critical—they become the prompt that guides the LLM's tool selection and argument inference.

### Using predict_and_call

```python
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")

# LLM picks the tool AND infers arguments
response = llm.predict_and_call(
    tools=[add_tool, mystery_tool],
    user_msg="What is the output of the mystery function on 2 and 9?"
)
# Output: Calls mystery(x=2, y=9) → 121
```

### Metadata Filtering with Vector Search

Documents have metadata (page numbers, sections, dates). We can let the LLM infer metadata filters:

```python
from llama_index.core.vector_stores import MetadataFilters, FilterCondition

# Manual metadata filtering
filters = MetadataFilters.from_dicts([
    {"key": "page_label", "value": "2"}
])

query_engine = vector_index.as_query_engine(
    similarity_top_k=2,
    filters=filters
)
```

### Auto-Retrieval: LLM Infers Filters

```python
def vector_query(query: str, page_numbers: list[str] = None) -> str:
    """
    Perform vector search over the index.

    Args:
        query: The search query string
        page_numbers: Optional list of page numbers to filter on

    Returns:
        Response string from the query
    """
    if page_numbers:
        filters = MetadataFilters.from_dicts([
            {"key": "page_label", "operator": "in", "value": page_numbers}
        ])
        query_engine = vector_index.as_query_engine(filters=filters)
    else:
        query_engine = vector_index.as_query_engine()

    return str(query_engine.query(query))

# Create tool from function
vector_query_tool = FunctionTool.from_defaults(fn=vector_query)

# LLM infers both query AND page filters
response = llm.predict_and_call(
    tools=[vector_query_tool],
    user_msg="What are the high-level results of MetaGPT as described on page 2?"
)
# LLM calls: vector_query(query="high-level results", page_numbers=["2"])
```

---

## 4. Lesson 3: Building an Agent Reasoning Loop

### From Single-Shot to Multi-Step

Previous lessons used single forward passes. An **agent** can:
- Reason over tools across multiple steps
- Handle complex questions requiring decomposition
- Maintain memory throughout the process

### Agent Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Agent Runner                      │
│  (Task dispatcher, orchestrates worker, returns     │
│   final response)                                   │
├─────────────────────────────────────────────────────┤
│              Function Calling Agent Worker           │
│  (Given context + history, decides next tool call,  │
│   executes, decides if done)                        │
└─────────────────────────────────────────────────────┘
```

### Creating a Function Calling Agent

```python
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner

# Create worker with tools
agent_worker = FunctionCallingAgentWorker.from_tools(
    tools=[vector_tool, summary_tool],
    llm=llm,
    verbose=True
)

# Create runner (the interface we interact with)
agent = AgentRunner(agent_worker)
```

### Multi-Step Reasoning in Action

```python
response = agent.query(
    "Tell me about the agent roles in MetaGPT, "
    "and then how they communicate with each other."
)
```

**Agent execution trace:**
1. Calls `summary_tool` with input "agent roles in MetaGPT"
2. Gets response about Product Manager, Architect, Engineer roles
3. Uses chain-of-thought to formulate next query
4. Calls tool again for "communication between agent roles"
5. Synthesizes final response combining both results

### Conversational Memory with chat()

```python
# First turn
response = agent.chat("Tell me about the evaluation datasets used.")

# Follow-up (agent remembers context)
response = agent.chat("Tell me the results over one of the above datasets.")
# Agent translates to: "results over the HumanEval dataset"
```

### Low-Level Debugging API

For debugging and control, use the step-by-step API:

```python
# Create a task
task = agent.create_task(
    "Tell me about agent roles and how they communicate."
)

# Execute one step at a time
step_output = agent.run_step(task.task_id)

# Inspect progress
completed = agent.get_completed_steps(task.task_id)
print(f"Completed steps: {len(completed)}")

# Inject human guidance mid-execution
step_output = agent.run_step(
    task.task_id,
    input="What about how agents share information?"
)

# Check if done
if step_output.is_last:
    response = agent.finalize_response(task.task_id)
```

### Benefits of Low-Level API

| Benefit | Description |
|---------|-------------|
| **Debugability** | Step through execution, inspect intermediate states |
| **Steerability** | Inject human feedback at any point |
| **Async UX** | Listen for human input during execution |
| **Pause/Resume** | Stop and continue later |

---

## 5. Lesson 4: Multi-Document Agents

### Scaling to Multiple Documents

Each document gets its own vector and summary tools:

```python
papers = ["metagpt.pdf", "longlora.pdf", "selfrag.pdf"]

paper_to_tools = {}
for paper in papers:
    vector_tool, summary_tool = get_doc_tools(paper)
    paper_to_tools[paper] = [vector_tool, summary_tool]

# Flatten all tools
all_tools = [tool for tools in paper_to_tools.values() for tool in tools]
# 3 papers × 2 tools = 6 tools total
```

### The Scaling Problem

With many documents, stuffing all tools into the prompt causes:
1. **Context overflow** - Tools may not fit
2. **Cost/latency spike** - More input tokens
3. **Selection confusion** - LLM struggles with too many choices

### Solution: Tool Retrieval

Instead of giving the agent ALL tools, retrieve relevant tools first:

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex

# Create an index over the tools themselves
tool_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex
)

# Create retriever
tool_retriever = tool_index.as_retriever(similarity_top_k=3)
```

### How Tool Retrieval Works

```
User Query: "Compare evaluation datasets in MetaGPT and SWE-bench"
                    ↓
            Tool Retriever
                    ↓
    Retrieved: [metagpt_summary_tool,
                swebench_summary_tool,
                metagpt_vector_tool]
                    ↓
         Agent uses only these 3 tools
```

### Creating an Agent with Tool Retrieval

```python
agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=tool_retriever,  # Instead of tools=
    llm=llm,
    system_prompt="""You are an agent designed to answer queries
    over a set of research papers. Use the tools to look up relevant
    information and synthesize your answer.""",
    verbose=True
)

agent = AgentRunner(agent_worker)
```

### Cross-Document Analysis

```python
response = agent.chat(
    "Compare and contrast the approaches in LongLoRA and LoftQ papers."
)
```

**Agent execution:**
1. Tool retriever finds LongLoRA and LoftQ tools
2. Calls `summary_tool_longlora` for approach details
3. Calls `summary_tool_loftq` for its approach
4. Synthesizes comparison from both responses

---

## Key Takeaways

### Agentic RAG Capabilities Progression

| Level | Capability | Implementation |
|-------|------------|----------------|
| 1 | **Routing** | Router picks between query engines |
| 2 | **Tool Calling** | LLM picks function + infers arguments |
| 3 | **Multi-Step Agent** | Reasoning loop with memory |
| 4 | **Multi-Doc Agent** | Tool retrieval + cross-document synthesis |

### Best Practices

1. **Write clear tool descriptions** - They guide the LLM's decisions
2. **Use type annotations** - Help with argument inference
3. **Enable verbose mode** - Debug agent decision-making
4. **Use tool retrieval at scale** - Don't stuff all tools into context
5. **Leverage low-level API** - For debugging and human-in-the-loop

### When to Use Each Pattern

| Pattern | Use When |
|---------|----------|
| Basic RAG | Simple Q&A over small document sets |
| Router | Need to choose between distinct query strategies |
| Tool Calling | Need structured arguments or metadata filters |
| Agent Loop | Complex queries requiring multiple steps |
| Multi-Doc Agent | Research across many documents |
