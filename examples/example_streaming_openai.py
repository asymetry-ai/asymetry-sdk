"""
Example: OpenAI Streaming with Asymetry Instrumentation

This example demonstrates how streaming responses are now fully instrumented
with the Asymetry SDK. The SDK wraps the streaming iterator to:
1. Accumulate content as chunks arrive
2. Track tool calls that are streamed in pieces
3. Extract token usage from the final chunk (if stream_options is set)
4. Emit a span with full telemetry when the stream completes

Usage:
    export OPENAI_API_KEY="your-key"
    export ASYMETRY_API_KEY="your-key"
    poetry run python examples/example_streaming_openai.py
"""

import os
import dotenv
from openai import OpenAI
from asymetry import init_observability, shutdown_observability

dotenv.load_dotenv()


def main():
    print("=" * 60)
    print("OpenAI Streaming Example with Asymetry Instrumentation")
    print("=" * 60)

    # Initialize Asymetry observability
    print("\n📊 Initializing Asymetry...")
    init_observability(
        log_level="DEBUG",  # Set to DEBUG to see span creation logs
    )

    # Create OpenAI client
    client = OpenAI()

    # Example 1: Basic streaming
    print("\n📤 Sending streaming request to OpenAI...")
    print("-" * 40)

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Count from 1 to 5, with a brief description for each number.",
            },
        ],
        stream=True,
        # Enable this to get real token usage in the final chunk:
        stream_options={"include_usage": True},
    )

    print("📥 Streaming response:\n")
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n\n✅ Stream completed!")
    print("-" * 40)
    print("The Asymetry SDK has captured:")
    print("  • Full accumulated response content")
    print("  • Token usage (real if stream_options used, estimated otherwise)")
    print("  • Time to first token")
    print("  • Total latency")
    print("  • Chunk count")

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
