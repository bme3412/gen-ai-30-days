# OpenAI Structured Outputs

**Structured Outputs** ensures the model always generates responses adhering to your supplied JSON Schema—no omitted keys, no invalid enum values. This provides reliable type-safety without validation retries, programmatic detection of safety refusals, and simpler prompting without strongly-worded formatting instructions.

The OpenAI SDKs make schema definition easy using **Pydantic** (Python) or Zod (JavaScript). Define a class with typed fields, pass it to the API, and get back a fully parsed object with IDE autocomplete.

```python
from pydantic import BaseModel

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

response = client.responses.parse(
    model="gpt-4o-2024-08-06",
    input=[{"role": "user", "content": "Alice and Bob have a meeting Friday"}],
    text_format=CalendarEvent
)
event = response.output_parsed  # Typed CalendarEvent object
```

Structured Outputs comes in two forms. **Function calling** is for bridging models to application functions—querying databases, interacting with UIs. **Response format** with `json_schema` structures user-facing output, like a math tutoring UI displaying steps differently from the final answer.

The difference from **JSON mode**: both ensure valid JSON, but only Structured Outputs guarantees schema adherence. Always prefer Structured Outputs when available.

Schema requirements for **strict mode**: all fields must be in `required` (use null unions for optional), every object needs `additionalProperties: false`, root must be an object (not anyOf). Limits: 5000 properties, 10 nesting levels, 1000 enum values.

When the model refuses for safety reasons, it returns a **refusal message** instead of structured data—check for and handle this. **Streaming** works via `stream()`, processing fields as they're generated for progressive UI display.

```python
with client.responses.stream(model="gpt-4.1", text_format=MyModel) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="")
```

**Recursive schemas** using `$ref` are supported for hierarchical structures. First request with any new schema has additional latency for processing; subsequent requests are fast.

For user-generated input, include instructions for handling cases that can't produce valid responses—the model will try to adhere to schema even with unrelated input, which can cause hallucination. Use Pydantic/Zod to keep schemas in sync with your types.
