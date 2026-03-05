# Agentic AI Course - Building Agents from First Principles

**Course:** DeepLearning.AI "Agentic AI" by Andrew Ng
**Focus:** Building AI agents using raw Python without framework abstractions

The course teaches agent development from first principles, emphasizing that understanding the underlying mechanics enables you to implement patterns in any framework—or none at all. The curriculum is vendor-neutral, using pure Python and standard LLM APIs to build each pattern.

**Four core design patterns** form the foundation of agentic systems:

**Reflection** enables agents to examine their own outputs and determine improvement strategies. An agent generates a response, then critiques it, identifying weaknesses before producing a refined version. This iterative self-improvement happens without human intervention.

**Tool Use** connects LLM applications to external capabilities—web search, database queries, code execution, API calls. The model decides which functions to invoke based on the task, transforming a language model into an acting system.

**Planning** breaks complex tasks into executable subtasks. Rather than attempting everything at once, a planning agent decomposes problems, determines ordering and dependencies, then executes systematically.

**Multi-Agent Collaboration** deploys specialized agents working together. A researcher agent gathers information, a writer agent drafts content, a critic agent reviews—each focused on its domain of expertise.

The course emphasizes that **evaluation and error analysis** are the strongest predictors of building success. Monitoring execution traces to identify workflow breakdowns, then systematically improving weak components, matters more than architectural sophistication. Start simple, measure everything, and improve iteratively based on evidence rather than intuition.

The practical capstone builds a deep research agent that searches for information across sources, synthesizes findings, and generates coherent reports—demonstrating how the four patterns combine in real applications.
