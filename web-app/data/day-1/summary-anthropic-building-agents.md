# Building Effective Agents (Anthropic)

**Source:** Anthropic Blog
**Authors:** Erik Schluntz, Barry Zhang
**Key Insight:** The most successful agent implementations use simple, composable patterns rather than complex frameworks.

---

## Table of Contents

1. [Philosophy: Simplicity Over Complexity](#1-philosophy-simplicity-over-complexity)
2. [Workflows vs Agents](#2-workflows-vs-agents)
3. [Building Block: The Augmented LLM](#3-building-block-the-augmented-llm)
4. [Workflow Patterns](#4-workflow-patterns)
5. [Autonomous Agents](#5-autonomous-agents)
6. [Practical Applications](#6-practical-applications)
7. [Tool Design Best Practices](#7-tool-design-best-practices)

---

## 1. Philosophy: Simplicity Over Complexity

### The Core Principle

*"The most successful agent implementations aren't using complex frameworks or specialized libraries. They're building with simple, composable patterns."*

### When NOT to Build Agents

Start with the simplest solution and add complexity only when needed:

| Approach | When to Use |
|----------|-------------|
| **Single LLM call** | Most applications—optimize with retrieval and in-context examples |
| **Workflows** | Well-defined tasks needing predictability and consistency |
| **Agents** | Flexibility and model-driven decision-making at scale |

**Key Question:** Does the latency/cost tradeoff for better task performance make sense?

### Framework Guidance

Frameworks like Claude Agent SDK, LangChain, and others simplify development but:
- Create abstraction layers that can obscure prompts and responses
- Make debugging harder
- Can tempt you to add unnecessary complexity

**Recommendation:** Start with direct LLM APIs. Many patterns need only a few lines of code.

---

## 2. Workflows vs Agents

### Definitions

| Type | Definition | Control |
|------|------------|---------|
| **Workflow** | LLMs and tools orchestrated through predefined code paths | Developer controls the flow |
| **Agent** | LLMs dynamically direct their own processes and tool usage | Model controls the flow |

### Visual Comparison

```
WORKFLOW (Predefined Path)
┌─────┐    ┌─────┐    ┌─────┐
│Step1│───▶│Step2│───▶│Step3│
└─────┘    └─────┘    └─────┘
   │          │          │
   ▼          ▼          ▼
 [LLM]      [LLM]      [LLM]

AGENT (Dynamic Path)
        ┌──────────────────┐
        │      Agent       │
        │  ┌────────────┐  │
        │  │ Reasoning  │  │
        │  │   Loop     │  │
        │  └────────────┘  │
        └────────┬─────────┘
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
 [Tool A]    [Tool B]    [Tool C]
```

---

## 3. Building Block: The Augmented LLM

### Core Concept

The basic building block is an LLM enhanced with:
- **Retrieval** - Access to external knowledge
- **Tools** - Ability to take actions
- **Memory** - Persistent context across interactions

```
                    ┌─────────────┐
                    │  Augmented  │
                    │     LLM     │
                    └─────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   ┌─────────┐      ┌──────────┐      ┌─────────┐
   │Retrieval│      │  Tools   │      │ Memory  │
   └─────────┘      └──────────┘      └─────────┘
```

### Implementation Focus

1. **Tailor capabilities to your use case** - Don't add everything
2. **Provide easy, well-documented interfaces** - Model Context Protocol (MCP) helps here

---

## 4. Workflow Patterns

### Pattern 1: Prompt Chaining

Decompose a task into sequential steps, where each LLM call processes the output of the previous one.

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  Input  │───▶│  LLM 1  │───▶│  Gate   │───▶│  LLM 2  │───▶ Output
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                               (check)
```

**When to use:**
- Task can be cleanly decomposed into fixed subtasks
- Willing to trade latency for higher accuracy

**Examples:**
- Generate marketing copy → Translate to another language
- Write document outline → Validate criteria → Write full document

---

### Pattern 2: Routing

Classify input and direct it to specialized handlers.

```
                    ┌─────────────┐
                    │   Router    │
                    │ (Classify)  │
                    └─────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   ┌─────────┐      ┌──────────┐      ┌─────────┐
   │ Handler │      │ Handler  │      │ Handler │
   │    A    │      │    B     │      │    C    │
   └─────────┘      └──────────┘      └─────────┘
```

**When to use:**
- Distinct categories that benefit from specialized handling
- Classification can be done accurately

**Examples:**
- Customer service: General questions vs. refunds vs. technical support
- Model selection: Easy questions → Claude Haiku, Hard questions → Claude Sonnet

---

### Pattern 3: Parallelization

Run LLM calls simultaneously and aggregate results.

#### Sectioning (Independent Subtasks)
```
                    ┌───────────┐
                    │   Input   │
                    └───────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   ┌─────────┐      ┌──────────┐      ┌─────────┐
   │ LLM 1   │      │  LLM 2   │      │  LLM 3  │
   │(Aspect1)│      │(Aspect 2)│      │(Aspect3)│
   └─────────┘      └──────────┘      └─────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
                    ┌───────────┐
                    │ Aggregate │
                    └───────────┘
```

**Examples:**
- Guardrails: One LLM processes query while another screens for inappropriate content
- Eval systems: Different LLMs evaluate different aspects

#### Voting (Same Task, Multiple Attempts)
```
                    ┌───────────┐
                    │Same Prompt│
                    └───────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   ┌─────────┐      ┌──────────┐      ┌─────────┐
   │  LLM    │      │   LLM    │      │   LLM   │
   │(Attempt)│      │(Attempt) │      │(Attempt)│
   └─────────┘      └──────────┘      └─────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
                    ┌───────────┐
                    │   Vote    │
                    └───────────┘
```

**Examples:**
- Code vulnerability review: Multiple prompts flag issues
- Content moderation: Different aspects with vote thresholds

---

### Pattern 4: Orchestrator-Workers

A central LLM dynamically breaks down tasks, delegates to workers, and synthesizes results.

```
                    ┌──────────────┐
                    │ Orchestrator │
                    │   (Plans)    │
                    └──────────────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
       ┌────────┐   ┌────────┐   ┌────────┐
       │Worker 1│   │Worker 2│   │Worker 3│
       └────────┘   └────────┘   └────────┘
            │             │             │
            └─────────────┼─────────────┘
                          ▼
                    ┌──────────────┐
                    │  Synthesize  │
                    └──────────────┘
```

**Key difference from parallelization:** Subtasks aren't pre-defined; orchestrator determines them based on input.

**When to use:**
- Can't predict subtasks needed upfront
- Number and nature of subtasks vary

**Examples:**
- Coding products making complex multi-file changes
- Search tasks gathering info from multiple sources

---

### Pattern 5: Evaluator-Optimizer

One LLM generates, another evaluates, iterate until good enough.

```
                    ┌───────────┐
                    │   Input   │
                    └───────────┘
                          │
                          ▼
         ┌───────────────────────────────┐
         │        ┌───────────┐          │
         │        │ Generator │          │
         │        └───────────┘          │
         │              │                │
         │              ▼                │
         │        ┌───────────┐          │
         │        │ Evaluator │──────────┤
         │        └───────────┘          │
         │              │                │
         │        Good enough?           │
         │         No ↑  │ Yes           │
         │            │  ▼               │
         └────────────┘  Output          │
                                         │
         └───────────────────────────────┘
```

**When to use:**
- Clear evaluation criteria exist
- LLM responses demonstrably improve with feedback
- Analogous to human iterative writing process

**Examples:**
- Literary translation with nuance refinement
- Complex search requiring multiple rounds

---

## 5. Autonomous Agents

### When Agents Make Sense

Agents work best when:
- Open-ended problems with unpredictable steps
- Can't hardcode a fixed path
- Have trust in the model's decision-making
- Need to scale tasks in trusted environments

### Agent Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     AGENT LOOP                          │
│                                                         │
│  ┌─────────┐    ┌──────────┐    ┌───────────────────┐  │
│  │ Receive │───▶│  Reason  │───▶│ Select Tool/Action│  │
│  │  Task   │    │          │    └───────────────────┘  │
│  └─────────┘    └──────────┘              │            │
│                                           ▼            │
│                                    ┌────────────┐      │
│                                    │  Execute   │      │
│                                    │   Tool     │      │
│                                    └────────────┘      │
│                                           │            │
│                                           ▼            │
│                                    ┌────────────┐      │
│                                    │  Observe   │      │
│  ┌─────────┐                       │  Result    │      │
│  │ Return  │◀──── Done? ◀──────────┴────────────┘      │
│  │ Result  │                                           │
│  └─────────┘                                           │
└─────────────────────────────────────────────────────────┘
```

### Critical Success Factors

1. **Ground truth at each step** - Tool call results, code execution outputs
2. **Human checkpoints** - Pause for feedback at critical points
3. **Stopping conditions** - Maximum iterations to maintain control
4. **Thoughtful tool design** - Clear interfaces and documentation

### Real-World Examples

| Application | Description |
|-------------|-------------|
| **SWE-bench Coding Agent** | Resolves GitHub issues by editing multiple files based on task description |
| **Computer Use Agent** | Claude uses a computer to accomplish tasks |

---

## 6. Practical Applications

### A. Customer Support

Natural fit for agents because:
- Follows conversation flow with external actions needed
- Tools can pull customer data, order history, knowledge base
- Actions (refunds, ticket updates) can be programmatic
- Success clearly measurable (resolution rate)

### B. Coding Agents

Excellent for agents because:
- Solutions verifiable through automated tests
- Can iterate using test results as feedback
- Problem space is well-defined and structured
- Output quality objectively measurable

---

## 7. Tool Design Best Practices

### The ACI (Agent-Computer Interface)

Invest as much effort in tool design as you would in human-computer interfaces:

### Principles

| Principle | Implementation |
|-----------|----------------|
| **Clarity** | Would a junior developer understand how to use this tool from description alone? |
| **Examples** | Include example usage, edge cases, format requirements |
| **Naming** | Clear, descriptive parameter names |
| **Testing** | Run many example inputs to see model mistakes |
| **Error-proofing** | Design arguments to prevent mistakes (poka-yoke) |

### Example: Good vs Bad Tool Definition

```python
# BAD - vague, error-prone
def edit_file(path, content):
    """Edit a file"""
    ...

# GOOD - clear, constrained, documented
def edit_file(
    absolute_path: str,
    new_content: str,
    create_if_missing: bool = False
) -> dict:
    """
    Replace entire file contents with new content.

    Args:
        absolute_path: Full path to file (e.g., /home/user/file.txt).
                      Must be absolute, not relative.
        new_content: Complete new file contents. Will replace existing.
        create_if_missing: If True, creates file if it doesn't exist.

    Returns:
        {"success": True, "bytes_written": int} or
        {"success": False, "error": str}

    Example:
        edit_file("/home/user/config.json", '{"key": "value"}')
    """
    ...
```

### Real Lesson from SWE-bench

*"We spent more time optimizing our tools than the overall prompt."*

**Specific fix:** Changed from relative to absolute filepaths—model made mistakes with relative paths after moving directories; absolute paths worked flawlessly.

---

## Summary: Three Core Principles

1. **Maintain simplicity** in your agent's design
2. **Prioritize transparency** by showing the agent's planning steps
3. **Carefully craft your ACI** through thorough tool documentation and testing

*"Success in the LLM space isn't about building the most sophisticated system. It's about building the right system for your needs."*
