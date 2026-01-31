import json
from openai import OpenAI
from typing import List, Dict, Any, Callable
import dotenv
from asymetry import init_observability

dotenv.load_dotenv()
init_observability()

# ==========================================
# MODULE 1: Tool Implementations (The Logic)
# ==========================================

def get_weather(location: str, unit: str = "fahrenheit") -> str:
    """Actual business logic to fetch weather."""
    loc = location.lower()
    if "san francisco" in loc:
        return f"65 degrees {unit} and sunny"
    elif "london" in loc:
        return f"15 degrees {unit} and rainy"
    return "Weather data unavailable"

def get_time(location: str) -> str:
    """Mock logic to get time."""
    return f"The current time in {location} is 2:00 PM"

# ==========================================
# MODULE 2: Tool Configuration (The Wiring)
# ==========================================

# 1. Dispatch Table
TOOL_MAP: Dict[str, Callable] = {
    "get_weather": get_weather,
    "get_time": get_time
}

# 2. Schema Definitions (OpenAI Format)
# Note: OpenAI uses "parameters" instead of "input_schema"
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current local time in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state"}
                },
                "required": ["location"]
            }
        }
    }
]

# ==========================================
# MODULE 3: Core Agent Functions (The Engine)
# ==========================================

def execute_tool_call(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """Executes a single tool safely and returns the string result."""
    if tool_name not in TOOL_MAP:
        return f"Error: Tool {tool_name} not found."
    
    try:
        print(f"🛠️  Executing: {tool_name} with args {tool_args}")
        result = TOOL_MAP[tool_name](**tool_args)
        return str(result)
    except Exception as e:
        return f"Error executing tool: {str(e)}"

def run_conversation(client: OpenAI, user_prompt: str, model: str = "gpt-4o"):
    """
    Orchestrates the conversation:
    1. Sends prompt
    2. Checks for tool_calls
    3. Executes tools
    4. Submits results back to OpenAI
    """
    
    # Initialize message history
    messages = [{"role": "user", "content": user_prompt}]

    # Step 1: Initial call
    print("🤖 Thinking...")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto" 
    )
    print("---Response---")
    print(response)
    print("---End Response---")

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # Step 2: Check if model wants to call tools
    if tool_calls:
        # OpenAI requires us to append the assistant's "intent" message first
        messages.append(response_message)

        # Step 3: Execute all requested tools
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Execute logic
            function_response = execute_tool_call(function_name, function_args)
            
            # Step 4: Append result message
            # Critical: 'role' must be 'tool' and 'tool_call_id' must match
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })

        # Step 5: Send results back to get final answer
        print("📤 Sending tool results back to OpenAI...")
        second_response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        print("---Second Response---")
        print(second_response)
        print("---End Second Response---")
        
        return second_response.choices[0].message.content

    # If no tool was used, return the text directly
    return response_message.content

# ==========================================
# MAIN ENTRY POINT
# ==========================================

if __name__ == "__main__":
    client = OpenAI() # Assumes OPENAI_API_KEY is in env
    
    prompt = "What is the weather in San Francisco?"
    
    final_answer = run_conversation(client, prompt)
    print("\n✅ Final Answer:")
    print(final_answer)
