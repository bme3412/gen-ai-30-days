# AI Agentic Design Patterns with AutoGen

**AutoGen** is a multi-agent conversational framework for creating specialized agents that work together through conversation. Rather than a single monolithic system, you create agents for different roles—researcher, writer, executor—that collaborate, critique, and iterate.

The foundational building block is the **ConversableAgent**, an entity that sends/receives messages, performs actions, and generates replies. Configure each with a name, system message, and **human_input_mode**: `NEVER` (LLM only), `ALWAYS` (asks human before replying), or `TERMINATE` (asks only when conversation ends).

```python
agent = ConversableAgent(name="chatbot", llm_config={"model": "gpt-4o"}, human_input_mode="NEVER")
```

**Two-agent conversations** let agents with different personas interact. One initiates with `initiate_chat()`, specifying recipient, message, and max_turns. The result includes chat history, token cost, and summary. Define **termination conditions** based on message content rather than fixed turn counts.

**Sequential chats** run multiple conversations in sequence with context carrying forward. Each chat specifies a recipient and how to carry over context from previous chats—enabling step-by-step task execution where each step builds on previous results.

The **reflection pattern** enables iterative improvement: a writer generates content, a critic provides feedback, they iterate until the critic says "APPROVED". **Nested chats** extend this—an agent can consult other agents (SEO expert, legal reviewer) before responding.

```python
critic = ConversableAgent(
    name="critic",
    is_termination_msg=lambda msg: "APPROVED" in msg["content"]
)
```

**Tool use** provides agents with callable functions. Register functions with both a **caller** (suggests tool use) and **executor** (runs the code). **Code execution** lets agents write Python and run it in Docker for safe, sandboxed computation.

**Group chats** orchestrate multiple specialists. Create agents for different roles, add them to a `GroupChat` managed by a `GroupChatManager`. **Speaker selection** can be automatic (LLM decides), round-robin, random, or custom logic.

```python
group_chat = GroupChat(agents=[planner, researcher, writer], speaker_selection_method="auto")
```

Best practices: precise system messages for each role, always define termination conditions, use Docker for code execution, monitor costs, and match pattern complexity to task requirements.
