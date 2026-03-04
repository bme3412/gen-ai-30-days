"""
Core Agent Logic - Template

This file contains the day's core agentic AI implementation.
Customize this file based on the specific concept being demonstrated.

The run_agent function is called by the FastAPI endpoint and should:
1. Process the user input with the vertical context
2. Execute the agentic pattern being demonstrated
3. Return steps taken and final result
"""

import os
from typing import Any
from openai import AsyncOpenAI

# Initialize OpenAI client (uses OPENAI_API_KEY env var)
client = AsyncOpenAI()


async def run_agent(
    user_input: str,
    vertical_config: dict,
    sample_data: dict
) -> dict:
    """
    Run the agent with industry-specific context.

    Args:
        user_input: The user's request/prompt
        vertical_config: Configuration for the selected industry vertical
        sample_data: Sample data specific to the vertical

    Returns:
        dict with 'steps' (list of actions taken) and 'result' (final output)
    """
    steps = []

    # Step 1: Understand the context
    steps.append({
        "type": "Context",
        "content": f"Processing request for {vertical_config['name']} industry"
    })

    # Step 2: Build the prompt with vertical context
    system_prompt = build_system_prompt(vertical_config)
    user_prompt = build_user_prompt(user_input, vertical_config, sample_data)

    steps.append({
        "type": "Reasoning",
        "content": f"Applying {vertical_config['name']} domain knowledge to analyze the request"
    })

    # Step 3: Call the LLM
    # TODO: Replace with actual agentic pattern implementation
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        result = response.choices[0].message.content

        steps.append({
            "type": "Action",
            "content": "Generated response using LLM"
        })

        return {
            "steps": steps,
            "result": result
        }

    except Exception as e:
        steps.append({
            "type": "Error",
            "content": f"Failed to process request: {str(e)}"
        })

        return {
            "steps": steps,
            "result": f"Error: {str(e)}. Make sure OPENAI_API_KEY is set."
        }


def build_system_prompt(vertical_config: dict) -> str:
    """Build the system prompt with vertical-specific context."""

    vertical_name = vertical_config["name"]
    examples = vertical_config.get("examples", [])
    examples_text = "\n".join(f"- {ex}" for ex in examples[:3])

    return f"""You are an expert AI assistant specializing in {vertical_name} applications.

Your role is to help users with {vertical_name.lower()}-related tasks using agentic AI patterns.

Example use cases you can help with:
{examples_text}

Provide clear, actionable responses tailored to the {vertical_name.lower()} industry.
When analyzing data or making recommendations, explain your reasoning step by step.
"""


def build_user_prompt(user_input: str, vertical_config: dict, sample_data: dict) -> str:
    """Build the user prompt with context and sample data."""

    prompt = f"""Request: {user_input}

Industry Context: {vertical_config['name']}
{vertical_config.get('description', '')}

"""

    if sample_data:
        import json
        prompt += f"""Available Data:
```json
{json.dumps(sample_data, indent=2)}
```

Please analyze the request using the available data and provide a comprehensive response.
"""

    return prompt


# ============================================
# Day-Specific Implementation
# ============================================
# TODO: Add day-specific agentic patterns below
# Examples:
# - ReAct loop implementation
# - Reflection patterns
# - Multi-agent coordination
# - Tool calling logic
# - Memory management
# ============================================
