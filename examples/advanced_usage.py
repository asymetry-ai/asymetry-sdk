"""
Advanced usage example for Asymetry SDK.

This example shows:
1. Custom configuration
2. Programmatic API key setup
3. Different models
4. Token tracking
"""

import os
import time
from asymetry.main import init_observability, shutdown_observability
import openai

# Setup OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")


def simulate_production_workload():
    """Simulate a production workload with various models."""

    print("\n" + "=" * 60)
    print("Asymetry Advanced Example - Production Simulation")
    print("=" * 60 + "\n")

    # Initialize with custom settings
    print("Initializing Asymetry with custom settings...")
    init_observability(
        enabled=True,
        log_level="DEBUG",  # Show detailed logs
    )
    print()

    client = openai.OpenAI()

    # Simulate different use cases
    use_cases = [
        {
            "name": "Customer Support Bot",
            "model": "gpt-5-mini",
            "prompt": "How do I reset my password?",
            "temperature": 0.3,
        },
        {
            "name": "Creative Writing Assistant",
            "model": "gpt-4",
            "prompt": "Write a haiku about programming.",
            "temperature": 0.9,
        },
        {
            "name": "Code Generator",
            "model": "gpt-4",
            "prompt": "Write a Python function to reverse a string.",
            "temperature": 0.2,
        },
        {
            "name": "Data Analysis",
            "model": "gpt-5-mini",
            "prompt": "Explain what a p-value means in statistics.",
            "temperature": 0.5,
        },
    ]

    print("Running simulated workload...\n")

    for i, use_case in enumerate(use_cases, 1):
        print(f"[{i}/{len(use_cases)}] {use_case['name']}")
        print(f"  Model: {use_case['model']}")
        print(f"  Temperature: {use_case['temperature']}")

        try:
            start = time.time()
            response = client.chat.completions.create(
                model=use_case["model"],
                messages=[
                    {"role": "system", "content": f"You are a {use_case['name']}."},
                    {"role": "user", "content": use_case["prompt"]},
                ],
            )
            elapsed = time.time() - start

            # Show token usage
            if response.usage:
                print(
                    f"  Tokens: {response.usage.prompt_tokens} in, "
                    f"{response.usage.completion_tokens} out, "
                    f"{response.usage.total_tokens} total"
                )

            print(f"  Latency: {elapsed:.2f}s")
            print(f"  Response: {response.choices[0].message.content[:100]}...")
            print(f"  ✓ Tracked by Asymetry\n")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            print(f"  ✓ Error tracked by Asymetry\n")

    print("=" * 60)
    print(f"Workload complete! {len(use_cases)} requests tracked.")
    print("Data is being exported in batches to Asymetry backend.")
    print("=" * 60 + "\n")

    # Explicit shutdown
    print("Waiting 2 seconds for final batch flush...")
    time.sleep(2)
    shutdown_observability(timeout=10)


def demonstrate_batching():
    """Demonstrate batch export behavior."""

    print("\n" + "=" * 60)
    print("Batch Export Demonstration")
    print("=" * 60 + "\n")

    init_observability(log_level="INFO")

    client = openai.OpenAI()

    print("Sending 10 rapid requests to demonstrate batching...")
    print("(Batch size = 100, Flush interval = 5s)\n")

    for i in range(10):
        try:
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": f"Count to {i+1}"}],
            )
            print(f"  Request {i+1}/10 completed")
        except Exception as e:
            print(f"  Request {i+1}/10 failed: {e}")

    print("\n✓ All requests queued")
    print("  They will be batched and sent within 5 seconds")
    print("  or when batch reaches 100 requests.\n")

    time.sleep(6)  # Wait for time-based flush
    shutdown_observability()


if __name__ == "__main__":
    # Run simulations
    simulate_production_workload()

    # Optional: uncomment to see batching behavior
    # demonstrate_batching()
