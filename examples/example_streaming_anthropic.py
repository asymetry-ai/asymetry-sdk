"""
Example: Anthropic Streaming with Asymetry Instrumentation

This example demonstrates how Anthropic streaming responses are now fully
instrumented with the Asymetry SDK. The SDK wraps the streaming context
manager to:
1. Process all event types (message_start, content_block_delta, etc.)
2. Accumulate content as text deltas arrive
3. Track tool uses that are streamed in pieces
4. Extract real token usage from message_start and message_delta events
5. Emit a span with full telemetry when the stream completes

Usage:
    export ANTHROPIC_API_KEY="your-key"
    export ASYMETRY_API_KEY="your-key"
    poetry run python examples/example_streaming_anthropic.py
"""

import os
import dotenv
import anthropic
from asymetry import init_observability, shutdown_observability

dotenv.load_dotenv()


def main():
    print("=" * 60)
    print("Anthropic Streaming Example with Asymetry Instrumentation")
    print("=" * 60)

    # Initialize Asymetry observability
    print("\n📊 Initializing Asymetry...")
    init_observability(
        log_level="DEBUG",  # Set to DEBUG to see span creation logs
    )

    # Create Anthropic client
    client = anthropic.Anthropic()

    # Example: Basic streaming with context manager
    print("\n📤 Sending streaming request to Anthropic...")
    print("-" * 40)

    stream = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": "Count from 1 to 5, with a brief description for each number.",
            }
        ],
        stream=True,
    )

    print("📥 Streaming response:\n")

    # Anthropic streaming uses a context manager
    with stream as event_stream:
        for event in event_stream:
            # Handle text deltas
            if event.type == "content_block_delta":
                if hasattr(event.delta, "text"):
                    print(event.delta.text, end="", flush=True)

            # Show message events for debugging
            elif event.type == "message_start":
                pass  # SDK extracts input_tokens here
            elif event.type == "message_delta":
                pass  # SDK extracts output_tokens here

    print("\n\n✅ Stream completed!")
    print("-" * 40)
    print("The Asymetry SDK has captured:")
    print("  • Full accumulated response content")
    print("  • Token usage (real - extracted from message events)")
    print("  • Time to first token")
    print("  • Total latency")
    print("  • Event count")

    # Give exporter time to send spans
    print("\n⏳ Waiting for spans to be exported...")
    import time

    time.sleep(3)

    # Shutdown
    print("\n🛑 Shutting down Asymetry...")
    shutdown_observability(timeout=5)

    print("\n✅ Done! Check your Asymetry dashboard for the streaming span.")


if __name__ == "__main__":
    main()
