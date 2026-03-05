# Instructor Library

**Instructor** is the most popular Python library for extracting structured, validated data from LLMs. Built on **Pydantic**, it works identically across 15+ providers—OpenAI, Anthropic, Google, Ollama. Define a Pydantic model describing the data you want, and Instructor handles getting the LLM to produce valid, typed output.

The library's core value is **reliability**. When the LLM returns data that doesn't match your schema—wrong types, missing fields, constraint violations—Instructor automatically retries with feedback about what went wrong. Specify `max_retries` and the library handles the loop, validation, and error messaging.

```python
from instructor import patch
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

client = patch(OpenAI())
person = client.chat.completions.create(
    model="gpt-4o",
    response_model=Person,
    messages=[{"role": "user", "content": "John is 25 years old"}]
)
```

Pydantic's full **validation capabilities** are available: Field constraints (`min_length`, `max_length`, `gt`, `lt`), custom validators, enums for constrained values, and nested models for complex hierarchical data. When validation fails, Instructor sends the error back to the LLM as retry context.

For real-time applications, Instructor supports **streaming**. `create_partial()` streams partial objects as they're generated—display fields as they become available. `create_iterable()` streams complete objects one at a time for extracting multiple items.

The library includes **hooks** for logging and monitoring—attach callbacks to request start, completion, and errors. There's also support for Jinja templating in prompts and **LLM-powered validation** where the model itself validates semantic constraints (e.g., checking for objectionable content).

Same code works across providers—just change the provider string. This provider abstraction combined with automatic validation and retry makes Instructor essential for production structured output workflows.
