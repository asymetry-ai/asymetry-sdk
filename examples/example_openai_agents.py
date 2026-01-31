"""
Example: Using Asymetry with OpenAI Agents SDK

This example demonstrates how to automatically trace OpenAI Agents
applications with Asymetry observability.

Prerequisites:
    1. Install asymetry with agents support:
       pip install asymetry[agents]
       # or: poetry install --extras agents

    2. Set environment variables:
       export ASYMETRY_API_KEY="your-asymetry-api-key"
       export OPENAI_API_KEY="your-openai-api-key"

Usage:
    python examples/example_openai_agents.py
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Asymetry observability
# This automatically instruments OpenAI Agents SDK if installed
from asymetry import init_observability

init_observability(log_level="DEBUG")

# Import OpenAI Agents SDK
from agents import Agent, Runner, function_tool


# Define a simple tool
@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Mock weather data
    weather_data = {
        "san francisco": "Foggy, 15°C",
        "new york": "Sunny, 22°C",
        "london": "Rainy, 8°C",
        "tokyo": "Clear, 18°C",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


# Define an agent with tools
weather_agent = Agent(
    name="WeatherAssistant",
    instructions="""You are a helpful weather assistant. 
    When asked about weather, use the get_weather tool to provide accurate information.
    Be friendly and concise in your responses.""",
    tools=[get_weather],
    model="gpt-4",
)


async def main():
    """Run the weather agent with a sample query."""
    print("=" * 60)
    print("Asymetry + OpenAI Agents SDK Integration Example")
    print("=" * 60)

    # Run the agent
    print("\nUser: What's the weather like in San Francisco?")
    result = await Runner.run(
        weather_agent,
        input="What's the weather like in San Francisco?",
    )

    print(f"\nAgent: {result.final_output}")

    # Run another query to demonstrate multi-turn
    # print("\nUser: How about Tokyo?")
    # result = await Runner.run(
    #     weather_agent,
    #     input="How about Tokyo?",
    # )

    # print(f"\nAgent: {result.final_output}")

    # print("\n" + "=" * 60)
    # print("Check your Asymetry dashboard to see the traces!")
    # print("=" * 60)
    import time

    time.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
