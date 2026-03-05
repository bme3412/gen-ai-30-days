# Building Agentic RAG with LlamaIndex

**Agentic RAG** extends traditional retrieval-augmented generation by adding reasoning and decision-making capabilities. Rather than following a fixed retrieve-then-generate pipeline, an agentic system dynamically decides which tools to use, formulates queries on the fly, and adjusts its approach based on intermediate results. This enables handling complex, multi-step research questions that would fail with standard RAG.

The simplest form of agentic behavior is **routing**. A router query engine examines the user's question and selects the most appropriate retrieval strategy. You create different index types—a **vector index** for specific fact lookup and a **summary index** for overview questions—then wrap them as tools with descriptions. The router directs each request to the right engine based on query intent.

```python
router = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[summary_tool, vector_tool]
)
```

**Tool calling** adds another layer. Instead of picking between pre-built engines, the LLM can select functions and infer their arguments. You wrap Python functions as tools with type annotations and docstrings—these guide the model's decisions. Given "What are the results on page 2?", the model generates a call with both the query string and a page filter it inferred.

```python
def vector_query(query: str, page_numbers: list[str] = None) -> str:
    """Perform vector search with optional page filtering."""
    # LLM infers page_numbers=["2"] from the user's question
    ...
```

The **agent reasoning loop** enables multi-step problem solving. An agent can decompose complex questions, execute multiple tool calls in sequence, and maintain memory of what it learned. The `AgentRunner` wraps this loop, and `agent.chat()` maintains conversational memory across turns.

```python
agent = AgentRunner(FunctionCallingAgentWorker.from_tools(tools, llm=llm))
response = agent.chat("Tell me about agent roles and how they communicate.")
```

LlamaIndex provides a **low-level API** for debugging: create a task, step through execution one call at a time, inspect intermediate states, and inject human guidance mid-process.

Scaling to multiple documents introduces **context overflow**. The solution is **tool retrieval**—creating a vector index over the tools themselves. Retrieve the most relevant tools first, then give only those to the agent.

```python
tool_retriever = ObjectIndex.from_objects(all_tools).as_retriever(similarity_top_k=3)
```

The capability progression: **routing** → **tool calling** → **agent loop** → **tool retrieval**. Clear tool descriptions are critical at every level—the LLM's decisions are only as good as the information you provide.
