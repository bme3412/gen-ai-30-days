# Building Effective Agents (Anthropic)

Anthropic's guidance emphasizes **simplicity over complexity**. The most successful agent implementations use simple, composable patterns rather than complex frameworks. Start with the simplest solution: a single LLM call with good retrieval handles most applications. Add **workflows** for well-defined tasks needing predictability. Use full **agents** only when you need flexibility and model-driven decision-making at scale.

The distinction between **workflows** and **agents** is fundamental. In a workflow, you orchestrate LLMs through predefined code paths—the developer controls the flow. In an agent, the LLM dynamically directs its own processes—the model controls the flow. Workflows are more predictable; agents offer more flexibility.

The basic building block is the **augmented LLM**—a model enhanced with retrieval, tools, and memory. Rather than adding every capability, tailor augmentations to your specific use case.

**Prompt chaining** decomposes tasks into sequential steps where each LLM call processes the previous output. **Routing** classifies input and directs it to specialized handlers—different handlers for refunds, general questions, technical issues. **Parallelization** runs LLM calls simultaneously: sectioning handles independent subtasks in parallel, voting runs the same task multiple times for higher confidence.

The **orchestrator-workers** pattern uses a central LLM to dynamically break down tasks and delegate to workers. Unlike parallelization where subtasks are predefined, the orchestrator determines subtasks based on input. The **evaluator-optimizer** pattern has one LLM generate while another evaluates, iterating until quality meets a threshold.

**Autonomous agents** work best for open-ended problems with unpredictable steps. Critical success factors: ground truth at each step from tool results, human checkpoints for feedback, stopping conditions to maintain control, and thoughtful tool design.

**Tool design** deserves as much investment as UI design. Clear descriptions that a junior developer could understand, example usage with edge cases, descriptive parameter names. One SWE-bench lesson: changing from relative to absolute filepaths eliminated a whole class of errors.

```python
def edit_file(absolute_path: str, new_content: str) -> dict:
    """Replace file contents. Path must be absolute (e.g., /home/user/file.txt)."""
```

Core principles: maintain **simplicity**, prioritize **transparency** by showing planning steps, and carefully craft your **agent-computer interface** through documentation and testing.
