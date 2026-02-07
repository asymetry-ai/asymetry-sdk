import pytest
from unittest.mock import MagicMock, patch
import time
import uuid
from asymetry.openai_agents import AsymetryTracingProcessor, instrument_openai_agents

# Mock agents library
sys_modules_patch = patch.dict("sys.modules", {"agents": MagicMock()})
sys_modules_patch.start()


@pytest.fixture
def span_queue():
    queue = MagicMock()
    return queue


@pytest.fixture
def processor(span_queue):
    return AsymetryTracingProcessor(span_queue=span_queue)


def test_trace_lifecycle(processor):
    # Mock trace object
    trace = MagicMock()
    trace.trace_id = "trace_123"
    trace.name = "test_agent"

    # Start trace
    processor.on_trace_start(trace)
    assert "123" in processor._active_traces

    # End trace
    processor.on_trace_end(trace)
    assert "123" not in processor._active_traces


def test_generation_span_processing(processor, span_queue):
    # Mock generation span
    span = MagicMock()
    span.trace_id = "trace_123"
    span.span_id = "span_456"
    span.parent_id = "span_789"

    # Mock span data (GenerationSpanData)
    data = MagicMock()
    data.__class__.__name__ = "GenerationSpanData"
    data.model = "gpt-4"
    data.usage = {"input_tokens": 10, "output_tokens": 5}
    data.input = "Hello"
    data.output = "World"

    span.span_data = data

    # Process span end
    processor.on_span_start(span)
    processor.on_span_end(span)

    # Verify TWO spans are enqueued:
    # 1. AgentSpan (parent)
    # 2. LLMRequest (child)
    assert span_queue.put_nowait.call_count == 2

    # Inspect calls
    calls = span_queue.put_nowait.call_args_list

    # First call should be AgentSpan
    agent_span = calls[0][0][0]
    assert agent_span.span_type == "generation"
    assert agent_span.trace_id == "123"

    # Second call should be LLMRequest wrapped in SpanContext
    span_ctx = calls[1][0][0]
    assert span_ctx.request.provider == "openai"
    assert span_ctx.request.messages == [{"role": "user", "content": "Hello"}]


def test_tool_span_processing(processor, span_queue):
    # Mock tool span
    span = MagicMock()
    span.trace_id = "trace_123"
    span.span_id = "span_999"

    # Mock span data (FunctionSpanData)
    data = MagicMock()
    data.__class__.__name__ = "FunctionSpanData"
    data.name = "get_weather"
    data.input = {"city": "Paris"}
    data.output = "Sunny"

    span.span_data = data

    # Process span end
    processor.on_span_start(span)
    processor.on_span_end(span)

    # Verify one AgentSpan enqueued
    assert span_queue.put_nowait.call_count == 1

    agent_span = span_queue.put_nowait.call_args[0][0]
    assert agent_span.span_type == "tool"
    assert agent_span.name == "get_weather"
    assert agent_span.input == {"city": "Paris"}


def test_span_type_detection(processor):
    # Helper to test type mapping
    def check_type(class_name, expected_type):
        data = MagicMock()
        data.__class__.__name__ = class_name
        assert processor._get_span_type(data) == expected_type

    check_type("GenerationSpanData", "generation")
    check_type("AgentSpanData", "agent")
    check_type("FunctionSpanData", "tool")
    check_type("HandoffSpanData", "agent")
    check_type("GuardrailSpanData", "guardrail")

    check_type("CustomSpanData", "custom")
