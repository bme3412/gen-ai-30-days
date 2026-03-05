# ReAct: Synergizing Reasoning and Acting in LLMs

**Paper:** "ReAct: Synergizing Reasoning and Acting in Language Models"
**Authors:** Yao et al. (2022)
**Key Contribution:** A paradigm that interleaves reasoning traces and actions, enabling LLMs to reason about tasks while interacting with external environments.

---

## Table of Contents

1. [The Core Insight](#1-the-core-insight)
2. [The Problem with Existing Approaches](#2-the-problem-with-existing-approaches)
3. [How ReAct Works](#3-how-react-works)
4. [Key Results](#4-key-results)
5. [Practical Benefits](#5-practical-benefits)
6. [Implementation Details](#6-implementation-details)

---

## 1. The Core Insight

### Human Reasoning as Inspiration

Consider cooking a dish. Between actions, humans naturally reason:

*"Now that everything is cut, I should heat up the pot of water."*

*"I don't have salt, so let me use soy sauce and pepper instead."*

*"How do I prepare dough? Let me search on the Internet."*

This **tight synergy between acting and reasoning** is what makes humans effective at:
- Learning new tasks quickly
- Performing robust decision-making
- Handling unexpected situations

### The ReAct Paradigm

**ReAct combines:**
- **Reasoning traces** (chain-of-thought) - for planning, tracking progress, handling exceptions
- **Actions** (tool use) - for interacting with external environments, gathering information

Neither alone is sufficient. Together, they create a powerful feedback loop.

---

## 2. The Problem with Existing Approaches

### Chain-of-Thought Reasoning (Alone)

Chain-of-thought prompting lets models reason step-by-step, but:

| Limitation | Problem |
|------------|---------|
| **Static** | Uses only internal knowledge |
| **Not grounded** | Cannot verify against external sources |
| **Hallucination-prone** | Errors propagate through reasoning chain |
| **No feedback** | Cannot correct course based on new information |

### Action-Only Approaches (Alone)

Pure action models (like behavior cloning) can interact with environments, but:

| Limitation | Problem |
|------------|---------|
| **No abstract reasoning** | Cannot plan or strategize |
| **No working memory** | Cannot track multi-step progress |
| **Rigid** | Cannot adapt to exceptions |

---

## 3. How ReAct Works

### The Key Innovation

ReAct **augments the action space** to include language:

```
Standard Agent:    Action Space = {available_actions}
ReAct Agent:       Action Space = {available_actions} ∪ {language_thoughts}
```

A "thought" is an action that:
- Does NOT affect the external environment
- Helps the model reason over context
- Updates internal state for future reasoning/acting

### Interleaved Execution

```
Question: "Which magazine was started first, Arthur's Magazine or First for Women?"

Thought 1: I need to search Arthur's Magazine and First for Women,
           and find which was started first.
Action 1:  Search[Arthur's Magazine]
Observation 1: Arthur's Magazine was an American literary periodical
               published in Philadelphia from 1844 to 1846.

Thought 2: Arthur's Magazine started in 1844. I need to search
           First for Women next.
Action 2:  Search[First for Women]
Observation 2: First for Women is a woman's magazine published by
               Bauer Media Group... launched in 1989.

Thought 3: First for Women was started in 1989. 1844 < 1989,
           so Arthur's Magazine was started first.
Action 3:  Finish[Arthur's Magazine]
```

### Types of Useful Thoughts

| Type | Example | Purpose |
|------|---------|---------|
| **Decomposition** | "I need to find X then Y then compare" | Break down complex tasks |
| **Knowledge injection** | "Mammals are warm-blooded animals" | Add relevant commonsense |
| **Extraction** | "The key info from that search is..." | Highlight important observations |
| **Progress tracking** | "I've found X, now I need Y" | Maintain working memory |
| **Exception handling** | "That didn't work, let me try..." | Adapt to failures |

---

## 4. Key Results

### Question Answering (HotpotQA)

ReAct with Wikipedia API outperformed:
- Pure chain-of-thought (hallucination issues)
- Action-only approaches (no reasoning)

**Best result:** Combining ReAct + CoT (use internal knowledge AND external retrieval)

### Fact Verification (FEVER)

ReAct overcame:
- **Hallucination** - by grounding in Wikipedia
- **Error propagation** - by verifying each step

### Decision Making (ALFWorld, WebShop)

| Benchmark | ReAct Improvement | Notable |
|-----------|-------------------|---------|
| ALFWorld | +34% success rate | vs. imitation learning |
| WebShop | +10% success rate | vs. reinforcement learning |

**With only 1-2 in-context examples** - no training required.

---

## 5. Practical Benefits

### A) Intuitive Design

Creating ReAct prompts is straightforward:
- Annotators simply write down their thoughts alongside actions
- No special format or thought engineering required
- Natural human task-solving trajectories work as examples

### B) General and Flexible

Works across diverse tasks:
- Question answering
- Fact verification
- Text-based games
- Web navigation
- And more...

### C) Performant and Robust

- Strong generalization from 1-6 examples
- Consistently outperforms reasoning-only or acting-only
- Works with few-shot prompting (no fine-tuning needed)

### D) Human Aligned and Controllable

```
┌─────────────────────────────────────────────────────────┐
│                   INTERPRETABILITY                       │
│                                                         │
│  • Humans can distinguish internal vs external info     │
│  • Reasoning traces explain decision basis              │
│  • Can edit thoughts mid-execution to steer agent       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Implementation Details

### Prompting Structure

For **reasoning-heavy tasks** (QA, fact verification):
```
Thought → Action → Observation → Thought → Action → Observation → ...
```
Alternate between thinking and acting systematically.

For **action-heavy tasks** (games, navigation):
```
Action → Action → Thought → Action → Action → Action → Thought → ...
```
Let the model decide when thoughts are needed (sparse reasoning).

### Example Prompt Template

```
Solve a question answering task with interleaving Thought, Action,
Observation steps.

Thought: reasoning about current situation
Action: one of [Search[entity], Lookup[keyword], Finish[answer]]
Observation: result of the action

Here are some examples:

Question: What is the elevation range for the area that the eastern
sector of the Colorado orogeny extends into?

Thought 1: I need to search Colorado orogeny, find the area that the
eastern sector extends into, then find the elevation range.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building...
...

Question: {user_question}
```

### When to Use ReAct

| Scenario | Recommendation |
|----------|----------------|
| Simple factual questions | Single LLM call may suffice |
| Complex multi-hop questions | ReAct with retrieval |
| Tasks requiring verification | ReAct to ground in sources |
| Interactive environments | ReAct for reasoning + acting |
| Need interpretability | ReAct traces explain decisions |

---

## Summary

### The ReAct Formula

```
ReAct = Chain-of-Thought + Tool Use + Interleaving
```

### Key Takeaways

1. **Reasoning alone** is brittle (hallucinations, no grounding)
2. **Acting alone** is inflexible (no planning, no adaptation)
3. **ReAct combines both** in an interleaved manner
4. **Thoughts are actions** that don't affect the environment but update internal state
5. **Few-shot prompting** works—no fine-tuning required
6. **Human-interpretable** traces enable debugging and steering

### Why This Matters

ReAct is foundational to modern agentic AI:
- Most agent frameworks implement ReAct-style reasoning
- The thought-action-observation loop is now standard
- Interleaved reasoning + tool use is the dominant paradigm
