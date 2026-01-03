import os
import json
import inspect
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
from asymetry.main import init_observability
from asymetry.tracing import observe

# Load environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY)
load_dotenv()
init_observability(log_level="DEBUG")

# ==========================================
# 1. MOCK DATA & TOOLS
# ==========================================

# Mock Database
USERS_DB = {
    "alice@example.com": {
        "name": "Alice Smith",
        "location": "San Francisco, CA",
        "role": "Engineer",
    },
    "bob@example.com": {
        "name": "Bob Jones",
        "location": "London, UK",
        "role": "Designer",
    },
}

WEATHER_DB = {
    "San Francisco, CA": {"temp": "15C", "condition": "Foggy"},
    "London, UK": {"temp": "8C", "condition": "Rainy"},
    "New York, NY": {"temp": "22C", "condition": "Sunny"},
}


@observe(name="tool.get_user_info", span_type="tool")
def get_user_info(email: str):
    """Fetches user profile data given an email address."""
    print(f"  [Tool] Accessing DB for: {email}...")
    result = USERS_DB.get(email)
    if result:
        return json.dumps(result)
    return json.dumps({"error": "User not found"})


@observe(name="tool.get_weather", span_type="tool")
def get_weather(location: str):
    """Fetches weather data for a specific city or location string."""
    print(f"  [Tool] Checking weather sensors in: {location}...")
    result = WEATHER_DB.get(location, {"temp": "Unknown", "condition": "Unknown"})
    return json.dumps(result)


# Tool Definitions (JSON Schema for LLMs)
tools_schema = [
    {
        "name": "get_user_info",
        "description": "Get personal details (name, location, role) for a user given their email.",
        "input_schema": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "The user's email address"}
            },
            "required": ["email"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for a given location.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state/country, e.g. 'Paris, France'",
                }
            },
            "required": ["location"],
        },
    },
]

# Map string names to actual functions for execution
available_functions = {"get_user_info": get_user_info, "get_weather": get_weather}

# ==========================================
# 2. LLM CLIENT WRAPPERS (OpenAI & Claude)
# ==========================================


class LLMClient:
    """Abstract base class to unify OpenAI and Anthropic calls."""

    def run(self, messages: List[Dict], tools: List[Dict]) -> Any:
        raise NotImplementedError


class OpenAIWrapper(LLMClient):
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def run(self, messages, tools):
        # Convert Anthropic-style tool schema to OpenAI format if needed
        # For simplicity, we assume we map them here on the fly
        openai_tools = []
        for t in tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                }
            )

        response = self.client.chat.completions.create(
            model="gpt-4o", messages=messages, tools=openai_tools, tool_choice="auto"
        )
        print(response.choices[0].message)
        return response.choices[0].message


class AnthropicWrapper(LLMClient):
    def __init__(self):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def run(self, messages, tools):
        # Anthropic expects 'user' and 'assistant' roles only in the messages list.
        # System prompt is a separate parameter.
        system_prompt = "You are a helpful assistant."
        filtered_msgs = [m for m in messages if m["role"] != "system"]

        response = self.client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=1024,
            system=system_prompt,
            messages=filtered_msgs,
            tools=tools,
        )
        return response


# ==========================================
# 3. THE AGENTIC WORKFLOW
# ==========================================


@observe(name="agent.run_agent", span_type="agent")
def run_agent(user_query: str, provider: str = "openai"):
    """
    Main Agent Loop:
    1. Accept User Input
    2. Send to LLM
    3. Check if LLM wants to use a Tool
    4. If yes -> Execute Tool -> Add result to History -> Iterate
    5. If no -> Return Final Answer
    """

    print(f"\n--- Starting Agent Task ({provider.upper()}) ---")
    print(f"User Query: {user_query}")

    # Initialize Client
    if provider == "openai":
        llm = OpenAIWrapper()
    elif provider == "anthropic":
        llm = AnthropicWrapper()
    else:
        raise ValueError("Provider must be 'openai' or 'anthropic'")

    # Initialize Context (Conversation History)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to user data and weather tools.",
        }
    ]
    messages.append({"role": "user", "content": user_query})

    # MAX ITERATIONS to prevent infinite loops
    MAX_ITERATIONS = 5

    for i in range(MAX_ITERATIONS):
        print(f"\n[Iteration {i+1}] Thinking...")

        # 1. CALL LLM
        response = llm.run(messages, tools_schema)

        # 2. HANDLE RESPONSE DIFFERENCES
        tool_calls = []

        if provider == "openai":
            # OpenAI returns a message object
            msg_content = response.content
            if response.tool_calls:
                tool_calls = response.tool_calls
                messages.append(
                    response
                )  # Add the assistant's thought process to history
            else:
                # No tools, just a reply
                print(f"Agent Reply: {msg_content}")
                return msg_content

        elif provider == "anthropic":
            # Anthropic returns a Message object with a .content list
            # We must reconstruct the message for history
            assistant_msg_content = []

            for block in response.content:
                if block.type == "text":
                    assistant_msg_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_msg_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
                    # Standardize for our internal logic
                    tool_calls.append(block)

            messages.append({"role": "assistant", "content": assistant_msg_content})

            # If no tools used, we are done
            if not tool_calls:
                text_reply = next(
                    (b.text for b in response.content if b.type == "text"), ""
                )
                print(f"Agent Reply: {text_reply}")
                return text_reply

        # 3. EXECUTE TOOLS (If any)
        if tool_calls:
            print(f"  -> Agent decided to call {len(tool_calls)} tool(s).")

            for tool_call in tool_calls:
                # Extract function name and args based on provider structure
                if provider == "openai":
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    call_id = tool_call.id
                else:  # Anthropic
                    func_name = tool_call.name
                    func_args = tool_call.input
                    call_id = tool_call.id

                # Execute the actual Python function
                if func_name in available_functions:
                    tool_result = available_functions[func_name](**func_args)
                else:
                    tool_result = f"Error: Tool {func_name} not found."

                # 4. APPEND RESULT TO HISTORY
                if provider == "openai":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": tool_result,
                        }
                    )
                else:  # Anthropic
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": call_id,
                                    "content": tool_result,
                                }
                            ],
                        }
                    )
        else:
            break


# ==========================================
# 4. EXAMPLES
# ==========================================


@observe(name="run_oai", span_type="workflow")
def run_oai(prompt):
    try:
        run_agent(prompt, provider="openai")
    except Exception as e:
        print(f"OpenAI skipped: {e}")


@observe(name="run_claude", span_type="workflow")
def run_claude(prompt):
    try:
        run_agent(prompt, provider="anthropic")
    except Exception as e:
        print(f"Anthropic skipped: {e}")


if __name__ == "__main__":
    # Ensure you have set OPENAI_API_KEY and ANTHROPIC_API_KEY in your env

    # Example 1: Simple Database Fetch
    # run_agent("Who is alice@example.com?", provider="openai")

    # Example 2: Complex Multi-step Dependency (Get Location -> Get Weather)
    # This proves the "Agentic" nature: it cannot get weather without first getting the location.

    prompt = "Can you check the weather where bob@example.com lives?"
    run_oai(prompt)
    print("\n" + "=" * 50 + "\n")
    import time
    time.sleep(5)
    # run_claude(prompt)
