"""
Example: LLM observability with tool / MCP calls using Asymetry.

This example shows:
- How to initialize Asymetry for LLM observability
- How to call OpenAI with tools enabled
- How tool / MCP calls are captured on the LLM span (llm.tools.* attributes)

Requirements:
- ASYMETRY_API_KEY set in your environment
- OPENAI_API_KEY set in your environment
"""

import os
import time

import openai

from asymetry.main import init_observability, shutdown_observability


def run_agent_like_call_with_tools() -> None:
    """
    Make a single OpenAI chat completion that uses tools.

    Asymetry will automatically:
    - Capture the LLM request (messages, params, model, latency, tokens)
    - Attach tool usage metadata to the LLM span:
        - llm.tools.count
        - llm.tools.names
    """
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Define a simple "tool" that looks like an agent function (e.g. MCP/tool call)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in ISO format.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "Timezone name, e.g. 'UTC'.",
                        }
                    },
                    "required": ["timezone"],
                },
            },
        }
    ]

    messages = [
        {
            "role": "system",
            "content": "You are an assistant that uses tools when helpful.",
        },
        {
            "role": "user",
            "content": "What time is it right now in UTC? Use tools if needed.",
        },
    ]

    print("\n" + "=" * 60)
    print("Asymetry LLM Tools / MCP Example")
    print("=" * 60 + "\n")

    print("Calling OpenAI with tools enabled...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.2,
    )

    msg = response.choices[0].message
    print("Assistant message content:", getattr(msg, "content", None))
    print("Tool calls from OpenAI SDK:", getattr(msg, "tool_calls", None))
    print("\n✓ Request and any tool/MCP calls have been captured by Asymetry.")
    print("  - LLM span includes llm.tools.count and llm.tools.names attributes.")
    print("  - Full messages and output are available in the Asymetry backend.\n")


def main() -> None:
    # Initialize Asymetry – this enables LLM observability and instruments OpenAI.
    init_observability(
        enabled=True,
        log_level="INFO",
        service_name="asymetry-llm-tools-example",
    )

    try:
        run_agent_like_call_with_tools()

        # Give the exporter a moment to flush the batch
        print("Waiting a few seconds for Asymetry exporter to flush...")
        time.sleep(5)
    finally:
        shutdown_observability(timeout=10)


if __name__ == "__main__":
    main()





