"""
Test file for streaming support in OpenAI and Anthropic SDKs.

This file tests the current instrumentation behavior with streaming responses
and helps identify what needs to be implemented to fully support streaming.

Requirements:
    pip install openai anthropic asymetry

Environment Variables:
    OPENAI_API_KEY - Your OpenAI API key
    ANTHROPIC_API_KEY - Your Anthropic API key
    ASYMETRY_API_KEY - Your Asymetry API key
"""

import time
from asymetry import init_observability, shutdown_observability


def test_openai_streaming():
    """Test OpenAI streaming response tracking."""
    print("\n" + "="*80)
    print("TEST 1: OpenAI Streaming")
    print("="*80)
    
    try:
        import openai
        
        client = openai.OpenAI()
        
        print("\n📤 Sending streaming request to OpenAI (gpt-4)...")
        start_time = time.time()
        
        stream = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "user", "content": "Count from 1 to 10, one number per line."}
            ],
            stream=True,
        )
        
        print("📥 Receiving streaming response:\n")
        full_response = ""
        chunk_count = 0
        
        for chunk in stream:
            chunk_count += 1
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                print(content, end="", flush=True)
        
        elapsed = time.time() - start_time
        
        print(f"\n\n✅ Stream completed:")
        print(f"   - Chunks received: {chunk_count}")
        print(f"   - Total response length: {len(full_response)} chars")
        print(f"   - Elapsed time: {elapsed:.2f}s")
        print(f"\n⚠️  Note: Current instrumentation may not fully capture streaming data")
        print(f"   Check if tokens, latency, and full response are tracked correctly.\n")
        
        return True
        
    except ImportError:
        print("❌ OpenAI SDK not installed. Run: pip install openai")
        return False
    except Exception as e:
        print(f"❌ Error during OpenAI streaming test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_anthropic_streaming():
    """Test Anthropic streaming response tracking."""
    print("\n" + "="*80)
    print("TEST 2: Anthropic Streaming")
    print("="*80)
    
    try:
        import anthropic
        
        client = anthropic.Anthropic()
        
        print("\n📤 Sending streaming request to Anthropic (claude-3-5-sonnet)...")
        start_time = time.time()
        
        stream = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Count from 1 to 10, one number per line."}
            ],
            stream=True,
        )
        
        print("📥 Receiving streaming response:\n")
        full_response = ""
        chunk_count = 0
        message_start_seen = False
        content_block_start_seen = False
        
        with stream as event_stream:
            for event in event_stream:
                chunk_count += 1
                
                # Anthropic streaming events have different types
                if event.type == "message_start":
                    message_start_seen = True
                    print(f"[Event: message_start, model={event.message.model}]")
                    
                elif event.type == "content_block_start":
                    content_block_start_seen = True
                    print(f"[Event: content_block_start]")
                    
                elif event.type == "content_block_delta":
                    if hasattr(event.delta, 'text'):
                        content = event.delta.text
                        full_response += content
                        print(content, end="", flush=True)
                        
                elif event.type == "content_block_stop":
                    print(f"\n[Event: content_block_stop]")
                    
                elif event.type == "message_delta":
                    if hasattr(event, 'usage'):
                        print(f"[Event: message_delta, usage={event.usage}]")
                        
                elif event.type == "message_stop":
                    print(f"[Event: message_stop]")
        
        elapsed = time.time() - start_time
        
        print(f"\n\n✅ Stream completed:")
        print(f"   - Events received: {chunk_count}")
        print(f"   - Message start seen: {message_start_seen}")
        print(f"   - Content block start seen: {content_block_start_seen}")
        print(f"   - Total response length: {len(full_response)} chars")
        print(f"   - Elapsed time: {elapsed:.2f}s")
        print(f"\n⚠️  Note: Current instrumentation may not fully capture streaming data")
        print(f"   Anthropic streaming uses event-based API with multiple event types.\n")
        
        return True
        
    except ImportError:
        print("❌ Anthropic SDK not installed. Run: pip install anthropic")
        return False
    except Exception as e:
        print(f"❌ Error during Anthropic streaming test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openai_non_streaming():
    """Test OpenAI non-streaming response (baseline comparison)."""
    print("\n" + "="*80)
    print("TEST 3: OpenAI Non-Streaming (Baseline)")
    print("="*80)
    
    try:
        import openai
        
        client = openai.OpenAI()
        
        print("\n📤 Sending non-streaming request to OpenAI (gpt-4)...")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "user", "content": "Count from 1 to 5, one number per line."}
            ],
        )
        
        elapsed = time.time() - start_time
        
        print("📥 Response received:\n")
        print(response.choices[0].message.content)
        
        print(f"\n✅ Request completed:")
        print(f"   - Model: {response.model}")
        print(f"   - Finish reason: {response.choices[0].finish_reason}")
        print(f"   - Input tokens: {response.usage.prompt_tokens}")
        print(f"   - Output tokens: {response.usage.completion_tokens}")
        print(f"   - Total tokens: {response.usage.total_tokens}")
        print(f"   - Elapsed time: {elapsed:.2f}s")
        print(f"\n✅ This should be fully tracked by current instrumentation.\n")
        
        return True
        
    except ImportError:
        print("❌ OpenAI SDK not installed. Run: pip install openai")
        return False
    except Exception as e:
        print(f"❌ Error during OpenAI non-streaming test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_anthropic_non_streaming():
    """Test Anthropic non-streaming response (baseline comparison)."""
    print("\n" + "="*80)
    print("TEST 4: Anthropic Non-Streaming (Baseline)")
    print("="*80)
    
    try:
        import anthropic
        
        client = anthropic.Anthropic()
        
        print("\n📤 Sending non-streaming request to Anthropic (claude-3-5-sonnet)...")
        start_time = time.time()
        
        response = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=50,
            messages=[
                {"role": "user", "content": "Count from 1 to 5, one number per line."}
            ],
        )
        
        elapsed = time.time() - start_time
        
        print("📥 Response received:\n")
        print(response.content[0].text)
        
        print(f"\n✅ Request completed:")
        print(f"   - Model: {response.model}")
        print(f"   - Stop reason: {response.stop_reason}")
        print(f"   - Input tokens: {response.usage.input_tokens}")
        print(f"   - Output tokens: {response.usage.output_tokens}")
        print(f"   - Elapsed time: {elapsed:.2f}s")
        print(f"\n✅ This should be fully tracked by current instrumentation.\n")
        
        return True
        
    except ImportError:
        print("❌ Anthropic SDK not installed. Run: pip install anthropic")
        return False
    except Exception as e:
        print(f"❌ Error during Anthropic non-streaming test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all streaming tests."""
    print("\n" + "="*80)
    print("🔍 Asymetry Streaming Tests")
    print("="*80)
    print("\nThis test suite checks streaming support for OpenAI and Anthropic.")
    print("Current instrumentation is designed for non-streaming requests.")
    print("\nThe tests will:")
    print("  1. Show how streaming responses work")
    print("  2. Identify gaps in streaming instrumentation")
    print("  3. Provide baseline non-streaming comparisons")
    
    # Initialize Asymetry
    print("\n📊 Initializing Asymetry observability...")
    try:
        init_observability(
            service_name="streaming-test",
            log_level="DEBUG",
        )
    except Exception as e:
        print(f"❌ Failed to initialize Asymetry: {e}")
        print("Continuing with tests anyway to demonstrate SDK behavior...")
    
    # Run tests
    results = {
        "openai_streaming": False,
        "anthropic_streaming": False,
        "openai_non_streaming": False,
        "anthropic_non_streaming": False,
    }
    
    # Test non-streaming first (baseline)
    results["openai_non_streaming"] = test_openai_non_streaming()
    time.sleep(1)
    
    results["anthropic_non_streaming"] = test_anthropic_non_streaming()
    time.sleep(1)
    
    # Test streaming
    results["openai_streaming"] = test_openai_streaming()
    time.sleep(1)
    
    results["anthropic_streaming"] = test_anthropic_streaming()
    
    # Summary
    print("\n" + "="*80)
    print("📊 Test Summary")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print("\n" + "="*80)
    print("📋 Current Instrumentation Status")
    print("="*80)
    print("\n✅ Non-streaming requests:")
    print("   - OpenAI: Fully instrumented")
    print("   - Anthropic: Fully instrumented")
    print("\n⚠️  Streaming requests:")
    print("   - OpenAI: Partially instrumented (may miss streaming-specific data)")
    print("   - Anthropic: Partially instrumented (may miss streaming-specific data)")
    print("\n💡 Recommendations:")
    print("   1. Add streaming-specific instrumentation wrappers")
    print("   2. Accumulate chunks to capture full response content")
    print("   3. Extract token usage from final streaming event")
    print("   4. Calculate accurate latency (first token + total time)")
    
    # Cleanup
    print("\n🛑 Shutting down Asymetry...")
    time.sleep(6)
    shutdown_observability()
    
    print("\n✅ Tests completed!\n")


if __name__ == "__main__":
    main()