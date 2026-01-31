import anthropic
from typing import List, Dict, Any, Callable
import dotenv

dotenv.load_dotenv()

# ==========================================
# MODULE 1: Tool Implementations (The Logic)
# ==========================================

def get_weather(location: str, unit: str = "fahrenheit") -> str:
    """Actual business logic to fetch weather."""
    # In a real app, this would call an external API like OpenWeatherMap
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

# 1. Dispatch Table: Maps the string names Claude sees to actual Python functions
TOOL_MAP: Dict[str, Callable] = {
    "get_weather": get_weather,
    "get_time": get_time
}

# 2. Schema Definitions: What we send to Claude so it knows how to use tools
TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    },
    {
        "name": "get_time",
        "description": "Get the current local time in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state"}
            },
            "required": ["location"]
        }
    }
]

# ==========================================
# MODULE 3: Core Agent Functions (The Engine)
# ==========================================

def execute_tool_call(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """Executes a single tool safely and returns the string result."""
    if tool_name not in TOOL_MAP:
        return f"Error: Tool {tool_name} not found."
    
    try:
        # Dynamically call the function from the map
        result = TOOL_MAP[tool_name](**tool_input)
        return str(result)
    except Exception as e:
        return f"Error executing tool: {str(e)}"

def run_conversation(client: anthropic.Anthropic, user_prompt: str, model: str = "claude-3-5-haiku-latest"):
    """
    Orchestrates the conversation:
    1. Sends prompt
    2. Checks for tool use
    3. Executes tools (if any)
    4. Sends results back to get final answer
    """
    
    # Initialize message history
    messages = [{"role": "user", "content": user_prompt}]

    # Step 1: Initial call to Claude
    print("🤖 Thinking...")
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        tools=TOOL_SCHEMAS,
        messages=messages
    )
    print("---Response---")
    print(response)
    print("---End Response---")

    # Check if Claude wants to stop because it needs to use a tool
    if response.stop_reason == "tool_use":
        # Append Claude's "intent" to history so it remembers it asked for a tool
        messages.append({"role": "assistant", "content": response.content})
        
        # Step 2: Extract and Execute Tools
        # We loop through content blocks because Claude might request multiple tools at once
        tool_result_content = []
        
        for block in response.content:
            if block.type == "tool_use":
                print(f"🛠️  Claude invoked: {block.name} with inputs {block.input}")
                
                # Execute logic
                result_text = execute_tool_call(block.name, block.input)
                
                # Create the result block for the API
                tool_result_content.append({
                    "type": "tool_result",
                    "tool_use_id": block.id, # CRITICAL: Must match the request ID
                    "content": result_text
                })

        # Step 3: Send results back to Claude
        if tool_result_content:
            messages.append({
                "role": "user",
                "content": tool_result_content
            })
            
            print("📤 Sending tool results back to Claude...")
            final_response = client.messages.create(
                model=model,
                max_tokens=1024,
                tools=TOOL_SCHEMAS,
                messages=messages
            )

            print("---Final Response---")
            print(final_response)
            print("---End Final Response---")
            
            return final_response.content[0].text

    # If no tool was used, just return the text
    return response.content[0].text

# ==========================================
# MAIN ENTRY POINT
# ==========================================

if __name__ == "__main__":
    client = anthropic.Anthropic() # Assumes ANTHROPIC_API_KEY is in env
    
    prompt = "What is the weather in San Francisco?"
    
    final_answer = run_conversation(client, prompt)
    print("\n✅ Final Answer:")
    print(final_answer)