import pytest
from unittest.mock import MagicMock, patch, Mock
import time
import json
from asymetry.instrumentation import (
    _instrumented_chat_create,
    _instrumented_messages_create,
    set_span_queue,
)
from asymetry.spans import SpanContext, LLMRequest

# Mock OpenTelemetry
sys_modules_patch = patch.dict(
    "sys.modules", {"opentelemetry": MagicMock(), "opentelemetry.trace": MagicMock()}
)
sys_modules_patch.start()


class MockQueue:
    def __init__(self):
        self.items = []

    def put_nowait(self, item):
        self.items.append(item)

    def get_nowait(self):
        if not self.items:
            return None
        return self.items.pop(0)


@pytest.fixture
def span_queue():
    queue = MockQueue()
    set_span_queue(queue)
    return queue


# --- OpenAI Tests ---


def test_openai_chat_create_non_streaming(span_queue):
    # Setup mock response
    mock_choice = MagicMock()
    mock_choice.message.content = "Hello world"
    mock_choice.message.function_call = None
    mock_choice.message.tool_calls = None
    mock_choice.finish_reason = "stop"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    # Mock original create method
    with patch(
        "asymetry.instrumentation._original_chat_create", return_value=mock_response
    ) as mock_create:
        # Call instrumented method
        _instrumented_chat_create(
            None, model="gpt-4", messages=[{"role": "user", "content": "Hi"}], stream=False
        )

        # Verify span was enqueued
        assert len(span_queue.items) == 1
        span_ctx = span_queue.items[0]

        # Verify span details
        assert isinstance(span_ctx, SpanContext)
        assert span_ctx.request.provider == "openai"
        assert span_ctx.request.model == "gpt-4"
        assert span_ctx.request.status == "success"
        assert span_ctx.tokens.input_tokens == 10
        assert span_ctx.tokens.output_tokens == 5
        assert span_ctx.request.output[0]["content"] == "Hello world"


def test_openai_chat_create_streaming(span_queue):
    # Setup mock stream chunks
    chunk1 = MagicMock()
    chunk1.choices = [
        MagicMock(delta=MagicMock(content="Hello", tool_calls=None), finish_reason=None)
    ]
    chunk1.usage = None

    chunk2 = MagicMock()
    chunk2.choices = [
        MagicMock(delta=MagicMock(content=" world", tool_calls=None), finish_reason="stop")
    ]
    # Verify OpenAI v2 style usage in last chunk (if include_usage=True) or implicitly
    chunk2.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    mock_stream = iter([chunk1, chunk2])

    # Mock original create method
    with patch("asymetry.instrumentation._original_chat_create", return_value=mock_stream):
        # Call instrumented method
        response_stream = _instrumented_chat_create(
            None, model="gpt-4", messages=[{"role": "user", "content": "Hi"}], stream=True
        )

        # Consume stream
        for _ in response_stream:
            pass

        # Verify span was enqueued
        assert len(span_queue.items) == 1
        span_ctx = span_queue.items[0]

        # Verify accumulated content
        assert span_ctx.request.output[0]["content"] == "Hello world"
        assert span_ctx.tokens.total_tokens == 15


# --- Anthropic Tests ---


def test_anthropic_messages_create_non_streaming(span_queue):
    # Setup mock response
    mock_content = MagicMock()
    mock_content.type = "text"
    mock_content.text = "Hello from Claude"

    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_response.stop_reason = "end_turn"
    mock_response.usage.input_tokens = 20
    mock_response.usage.output_tokens = 8

    # Mock original create method
    with patch("asymetry.instrumentation._original_messages_create", return_value=mock_response):
        # Call instrumented method
        _instrumented_messages_create(
            None,
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": "Hi"}],
            stream=False,
        )

        # Verify span
        assert len(span_queue.items) == 1
        span_ctx = span_queue.items[0]

        assert span_ctx.request.provider == "anthropic"
        assert span_ctx.request.output[0]["content"] == "Hello from Claude"
        assert span_ctx.tokens.input_tokens == 20


def test_anthropic_messages_create_streaming(span_queue):
    # Setup mock stream events
    event1 = MagicMock()
    event1.type = "message_start"
    event1.message.usage.input_tokens = 10

    event2 = MagicMock()
    event2.type = "content_block_delta"
    event2.delta.type = "text_delta"
    event2.delta.text = "Hello"

    event3 = MagicMock()
    event3.type = "message_delta"
    event3.usage.output_tokens = 5
    event3.delta.stop_reason = "end_turn"

    event4 = MagicMock()
    event4.type = "message_stop"

    mock_stream = MagicMock()
    mock_stream.__enter__.return_value = iter([event1, event2, event3, event4])
    mock_stream.__exit__.return_value = None

    with patch("asymetry.instrumentation._original_messages_create", return_value=mock_stream):
        # Call instrumented method
        stream_wrapper = _instrumented_messages_create(
            None, model="claude-3-sonnet", messages=[{"role": "user", "content": "Hi"}], stream=True
        )

        # Consume stream
        with stream_wrapper as stream:
            for _ in stream:
                pass

        # Verify span
        assert len(span_queue.items) == 1
        span_ctx = span_queue.items[0]

        assert span_ctx.request.output[0]["content"] == "Hello"
        assert span_ctx.tokens.input_tokens == 10
        assert span_ctx.tokens.output_tokens == 5
