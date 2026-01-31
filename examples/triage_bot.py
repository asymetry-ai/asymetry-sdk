import asyncio
from agents import Agent, Runner, function_tool
from asymetry import instrument_openai_agents

instrument_openai_agents()

# --- 1. Tools (Pure Logic) ---


@function_tool
def analyze_damage_report(photo_url: str) -> str:
    """Simulates a Vision API check for hardware defects."""
    # Logic only: No spans or logging
    if "broken_screen" in photo_url:
        return "Analysis: Physical impact detected. Not a factory defect."
    return "Analysis: Internal hardware failure. Covered by warranty."


@function_tool
def flag_for_human_review(reason: str, amount: float) -> str:
    """Escalates high-value cases to a human queue."""
    return f"ACTION REQUIRED: Human approval needed for ${amount}. Reason: {reason}"


# --- 2. Agents (Defined by Instructions and Handoffs) ---

# Focuses purely on technical assessment
tech_inspector = Agent(
    name="TechInspector",
    instructions="Analyze the provided photo_url and report if the damage is a defect or user-caused.",
    tools=[analyze_damage_report],
)

# The "Brain" that manages the business logic and thresholds
policy_agent = Agent(
    name="PolicyAgent",
    instructions="""
    1. Coordinate with TechInspector to see if the damage is covered.
    2. If the refund amount is over $500, you MUST use the flag_for_human_review tool.
    3. If under $500 and covered, confirm the refund.
    """,
    tools=[flag_for_human_review],
    handoffs=[tech_inspector],
)

# The Entry Point
triage_agent = Agent(
    name="TriageAgent",
    instructions="Determine if the user wants a return. If so, hand off to PolicyAgent.",
    handoffs=[policy_agent],
)

# --- 3. Execution ---


async def main():
    # Scenario: A high-value laptop return request
    user_query = "I want to return my laptop (Order LAPTOP-99, Value $1200). Here is the photo: https://store.com/broken_screen.jpg"

    # The Runner handles the 'loops' between agents behind the scenes
    result = await Runner.run(triage_agent, user_query)

    print(f"Final System Response:\n{result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
