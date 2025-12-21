"""
Multi-LLM / Multi-agent observability example with Asymetry.

This example demonstrates:
- Two "agents" (ResearchAgent and AnswerAgent) collaborating on a task
- Each agent calling different OpenAI models (using cheap models)
- Optional tool / MCP-like calls on the OpenAI side
- Clear hierarchy in traces:
    request -> orchestrator -> agent -> LLM call (+ tools)

You should see in Asymetry:
- Custom spans from @observe (orchestrator, agents, helper functions)
- LLM spans for OpenAI with tokens, latency, finish_reason
- LLM spans enriched with llm.tools.* when tools are used

Requirements:
- ASYMETRY_API_KEY set in your environment
- OPENAI_API_KEY set in your environment
"""

import json
import os
import time
from datetime import datetime

import openai

from asymetry.main import init_observability, shutdown_observability
from asymetry.tracing import observe


# --- Setup clients -----------------------------------------------------------

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# --- Tools for OpenAI agent --------------------------------------------------

TIME_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time in ISO format. Useful for answering questions about dates, times, or scheduling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone name, e.g. 'UTC' or 'America/New_York'. Defaults to UTC if not specified.",
                    }
                },
                "required": [],
            },
        },
    }
]


# Tool function implementations
@observe(name="tool.get_current_time", kind="internal")
def get_current_time(timezone: str = "UTC") -> str:
    """Execute the get_current_time tool."""
    try:
        # Simple implementation - in production you'd use pytz or zoneinfo
        now = datetime.utcnow() if timezone.upper() == "UTC" else datetime.now()
        return now.isoformat()
    except Exception as e:
        return f"Error getting time: {str(e)}"


# Map tool names to their implementations
TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
}


# --- Agents ------------------------------------------------------------------


@observe(name="agent.research", kind="internal")
def research_agent(topic: str) -> str:
    """
    Research agent that uses OpenAI (gpt-3.5-turbo) to gather context.
    """
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=512,
        messages=[
            {
                "role": "system",
                "content": "You are a research assistant. Provide concise, high-signal notes.",
            },
            {
                "role": "user",
                "content": f"Research the following topic and provide bullet points: {topic}",
            },
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content or ""


@observe(name="agent.answer", kind="internal")
def answer_agent(question: str, research_notes: str) -> str:
    """
    Answer agent that uses OpenAI, with tools enabled (for richer LLM spans).
    Handles tool calling loop: executes tools and sends results back to the model.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior engineer answering questions for a user. "
                "You are given prior research notes and may call tools when helpful. "
                "If the question involves dates, times, or scheduling, use the get_current_time tool."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Here are some research notes:\n{research_notes}\n\n"
                f"Now answer the user's question: {question}"
            ),
        },
    ]

    max_iterations = 5  # Prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TIME_TOOL,
            tool_choice="auto",
            temperature=0.4,
        )

        msg = response.choices[0].message

        # Convert message to dict format for the messages list
        assistant_message = {
            "role": msg.role,
            "content": msg.content,
        }
        if msg.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_message)

        # Check if the model wants to call tools
        if msg.tool_calls:
            print(f"\n[AnswerAgent] Tool calls detected: {len(msg.tool_calls)} call(s)")
            # Execute each tool call
            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_id = tool_call.id

                print(f"  - Executing tool: {tool_name} (id: {tool_id})")

                # Parse arguments
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                # Execute the tool function
                if tool_name in TOOL_FUNCTIONS:
                    tool_result = TOOL_FUNCTIONS[tool_name](**tool_args)
                    print(f"    Result: {tool_result}")
                else:
                    tool_result = f"Error: Tool '{tool_name}' not found"
                    print(f"    Error: {tool_result}")

                # Add tool result to messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "content": str(tool_result),
                    }
                )

            iteration += 1
            continue  # Loop back to get the model's response to tool results

        # No tool calls - we have the final answer
        print("\n[AnswerAgent] Final response received (no tool calls)")
        return msg.content or ""

    # Fallback if we hit max iterations
    print(f"\n[AnswerAgent] Warning: Reached max iterations ({max_iterations})")
    return messages[-1].get("content", "") if messages else ""


@observe(name="workflow.orchestrator", kind="server")
def orchestrate_question_answer(question: str) -> str:
    """
    Top-level orchestrator that coordinates multiple agents and LLMs.

    Trace hierarchy:
        orchestrator (custom span)
          ├─ research_agent (custom span)
          │    └─ OpenAI LLM span (llm.request, gpt-3.5-turbo)
          └─ answer_agent (custom span)
               └─ OpenAI LLM span (llm.request, gpt-4o-mini, + llm.tools.*)
    """
    print(f"\n[orchestrator] Starting workflow for question: {question!r}")

    # Step 1: Research with OpenAI (gpt-3.5-turbo)
    research_notes = research_agent(question)
    print("\n[orchestrator] Research notes from ResearchAgent (gpt-3.5-turbo):\n")
    print(research_notes[:400] + ("..." if len(research_notes) > 400 else ""))

    # Step 2: Answer with OpenAI (gpt-4o-mini) + tools
    answer = answer_agent(question, research_notes)

    print("\n[orchestrator] Final answer from AnswerAgent (gpt-4o-mini):\n")
    print(answer[:400] + ("..." if len(answer) > 400 else ""))

    return answer


def main() -> None:
    # Initialize Asymetry – enables LLM observability and custom tracing.
    init_observability(
        enabled=True,
        log_level="INFO",
        service_name="asymetry-multi-llm-multi-agent-example",
    )

    try:
        question = (
            "How could I design an observability system for LLM agents in production? "
            "Also, what is the current time in UTC? This will help me understand when to schedule deployments."
        )
        orchestrate_question_answer(question)

        # Allow background exporter to flush data
        print("\nWaiting a few seconds for Asymetry exporter to flush...")
        time.sleep(8)
    finally:
        shutdown_observability(timeout=10)


if __name__ == "__main__":
    main()
