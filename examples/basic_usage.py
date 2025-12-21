"""
Basic usage example for Asymetry SDK.

This example shows how to:
1. Initialize Asymetry observability
2. Make OpenAI API calls (automatically tracked)
3. Handle errors (also tracked)
"""

import os
import openai

from asymetry.main import init_observability, shutdown_observability
from asymetry.tracing import observe

# Setup OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

print("asymetry api key", os.getenv("ASYMETRY_API_KEY"))


@observe()
def parse_response(json_data):
    return json_data


@observe()
def example_one():
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )

        return parse_response(response.choices[0].message.content)
    except:
        print("Error running example_one")


def main():
    # Initialize Asymetry - this one line enables observability!
    init_observability()

    print("\n" + "=" * 60)
    print("Asymetry Basic Example")
    print("=" * 60 + "\n")

    # Example 1: Simple chat completion
    print("Example 1: Simple chat completion")
    print("-" * 40)
    example_one()

    # # Example 2: Multiple requests
    # print("Example 2: Multiple requests (batch testing)")
    # print("-" * 40)

    # questions = [
    #     "What is 2+2?",
    #     "Name a color.",
    #     "What year is it?",
    # ]

    # for i, question in enumerate(questions, 1):
    #     response = client.chat.completions.create(
    #         model="gpt-5-mini", messages=[{"role": "user", "content": question}]
    #     )
    #     print(f"{i}. Q: {question}")
    #     print(f"   A: {response.choices[0].message.content.strip()}")

    # print(f"\n✓ All {len(questions)} requests tracked!\n")

    # # Example 3: Error handling (tracked automatically)
    # print("Example 3: Error handling")
    # print("-" * 40)

    # try:
    #     # This will fail - invalid model
    #     response = client.chat.completions.create(
    #         model="gpt-nonexistent-model", messages=[{"role": "user", "content": "Hello"}]
    #     )
    # except Exception as e:
    #     print(f"✓ Error caught: {type(e).__name__}: {e}")
    #     print(f"✓ This error was automatically tracked by Asymetry!\n")

    # # Example 4: Streaming (not yet supported in v0.1-alpha)
    # print("Example 4: Note about streaming")
    # print("-" * 40)
    # print("⚠️  Streaming is not yet supported in v0.1-alpha")
    # print("    Only sync chat.completions.create() is tracked.\n")

    # print("=" * 60)
    # print("Examples complete!")
    # print("Check your Asymetry dashboard to see the tracked requests.")
    # print("=" * 60 + "\n")

    import time

    time.sleep(6)

    print("✓ Done!\n")


if __name__ == "__main__":
    main()
