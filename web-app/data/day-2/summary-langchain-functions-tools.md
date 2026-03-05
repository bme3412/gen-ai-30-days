# Functions, Tools and Agents with LangChain

This DeepLearning.AI course teaches building LLM applications that use tools and maintain conversations—from basic function calling through full conversational agents with memory.

The course begins with **OpenAI function calling** fundamentals. The model doesn't execute functions—it tells you what to call and with what arguments. Define functions with name, description, and JSON Schema. The model can be set to auto-decide, always use tools, or use a specific one. Your code handles the execution loop: get the call, execute, pass result back, get final response.

**LangChain Expression Language (LCEL)** provides a clean way to compose LLM pipelines. Using the pipe operator, chain prompts, models, and parsers into reusable workflows. LCEL gives async, batch processing, and streaming out of the box. Add fallbacks for error handling. The `bind()` method attaches parameters like function definitions to a model.

```python
chain = prompt | model.bind(functions=functions) | parser
```

**Pydantic integration** makes schema definition cleaner than raw JSON. Define Python classes with type hints and Field descriptions—LangChain converts to OpenAI format automatically. Docstrings become function descriptions, field descriptions guide behavior, and schemas stay in sync with code.

**Tagging** classifies text into categories—define a schema with fields like sentiment and language, extract those values from text. **Extraction** pulls multiple entities from text (all people mentioned with names and ages). For long documents: split into chunks, extract from each, flatten results.

The **tools and routing** lesson shows the `@tool` decorator for creating LangChain tools with descriptions and typed arguments. With multiple tools, the model routes to the appropriate one. The output parser determines whether the model wants to call a tool or return a final response.

**Conversational agents** tie everything together with the agent loop. The model receives input, decides whether to call a tool, and if so, the result goes into a scratchpad for the next iteration. This continues until a final response. Adding **ConversationBufferMemory** enables multi-turn conversations where context carries forward. The **AgentExecutor** wrapper handles errors, retries, and logging automatically.
