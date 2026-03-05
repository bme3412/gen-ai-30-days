# ReAct: Synergizing Reasoning and Acting in LLMs

The **ReAct** paper introduced a paradigm that interleaves reasoning traces and actions, enabling LLMs to reason while interacting with external environments. The insight comes from human behavior—when cooking, we think "now that everything is cut, I should heat the water" or "I don't have salt, let me use soy sauce instead." This **tight synergy between acting and reasoning** is what makes humans effective at handling unexpected situations.

**Chain-of-thought reasoning** alone is static—it uses only internal knowledge without grounding against external sources. Errors propagate with no way to correct course. **Action-only approaches** can interact with environments but lack abstract reasoning and working memory. ReAct combines both by augmenting the action space to include language thoughts alongside available actions.

A **thought** is an action that does not affect the external environment but helps the model reason and update internal state. The execution interleaves thinking and acting in a loop:

```
Thought 1: I need to search Arthur's Magazine and First for Women.
Action 1:  Search[Arthur's Magazine]
Observation 1: Published from 1844 to 1846.
Thought 2: Arthur's started 1844. Now search First for Women.
Action 2:  Search[First for Women]
Observation 2: Launched in 1989.
Thought 3: 1844 < 1989, so Arthur's Magazine was first.
Action 3:  Finish[Arthur's Magazine]
```

Different **thought types** serve different purposes: **decomposition** breaks down complex tasks, **knowledge injection** adds relevant commonsense, **extraction** highlights important observations, **progress tracking** maintains working memory, and **exception handling** adapts when something fails.

ReAct outperformed both pure chain-of-thought and action-only approaches across benchmarks. On question answering, it overcame **hallucination** by grounding in external sources. On decision-making tasks, it achieved significant improvements with only 1-2 in-context examples—no training required.

Creating ReAct prompts is intuitive: annotators simply write down their thoughts alongside actions, following natural human task-solving trajectories. Because reasoning traces are visible, humans can inspect the decision basis and even **edit thoughts mid-execution** to steer the agent.

ReAct is foundational to modern agentic AI—most frameworks implement ReAct-style reasoning, and the **thought-action-observation loop** is now the dominant paradigm for tool-using language models.
