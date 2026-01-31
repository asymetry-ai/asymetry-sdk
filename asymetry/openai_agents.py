"""OpenAI Agents SDK integration for Asymetry observability.

This module provides a TracingProcessor implementation that captures
traces and spans from OpenAI Agents SDK and exports them to the
Asymetry backend.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Global state
_processor: "AsymetryTracingProcessor | None" = None
_instrumented = False


class AsymetryTracingProcessor:
    """
    Asymetry implementation of the OpenAI Agents SDK TracingProcessor.

    Captures traces and spans from OpenAI Agents and exports them to
    the Asymetry backend for observability.

    Usage:
        from asymetry.openai_agents import instrument_openai_agents

        # Option 1: Auto-register with OpenAI Agents SDK
        processor = instrument_openai_agents()

        # Option 2: Manual registration
        from agents import add_trace_processor
        processor = AsymetryTracingProcessor()
        add_trace_processor(processor)
    """

    def __init__(self, span_queue: Any = None):
        """
        Initialize the Asymetry tracing processor.

        Args:
            span_queue: Optional queue for collecting spans. If not provided,
                        will use the global exporter queue.
        """
        self._span_queue = span_queue
        self._active_traces: dict[str, dict[str, Any]] = {}
        self._active_spans: dict[str, dict[str, Any]] = {}

    def _get_queue(self) -> Any:
        """Get the span queue, falling back to global exporter if needed."""
        if self._span_queue is not None:
            return self._span_queue

        # Try to get queue from global exporter
        try:
            from .exporter import get_exporter

            exporter = get_exporter()
            if exporter is not None:
                return exporter.get_queue()
        except Exception:
            pass

        return None

    def _clean_id(self, id_value: str | None) -> str | None:
        """Strip 'trace_' or 'span_' prefix from IDs."""
        if id_value is None:
            return None
        if id_value.startswith("trace_"):
            return id_value[6:]  # Remove 'trace_' prefix
        if id_value.startswith("span_"):
            return id_value[5:]  # Remove 'span_' prefix
        return id_value

    def on_trace_start(self, trace: Any) -> None:
        """
        Called when a new trace starts.

        Args:
            trace: The Trace object from OpenAI Agents SDK
        """
        try:
            trace_id = self._clean_id(getattr(trace, "trace_id", None))
            if trace_id:
                self._active_traces[trace_id] = {
                    "trace_id": trace_id,
                    "name": getattr(trace, "name", "agent_trace"),
                    "start_time": time.time(),
                    "metadata": getattr(trace, "metadata", {}),
                    "group_id": getattr(trace, "group_id", None),
                }
                logger.debug(f"Trace started: {trace_id}")
        except Exception as e:
            logger.debug(f"Error on trace start: {e}")

    def on_trace_end(self, trace: Any) -> None:
        """
        Called when a trace ends.

        Args:
            trace: The Trace object from OpenAI Agents SDK
        """
        try:
            trace_id = self._clean_id(getattr(trace, "trace_id", None))
            if trace_id and trace_id in self._active_traces:
                trace_data = self._active_traces.pop(trace_id)
                end_time = time.time()
                duration_ms = (end_time - trace_data["start_time"]) * 1000
                logger.debug(f"Trace ended: {trace_id} (duration: {duration_ms:.2f}ms)")
        except Exception as e:
            logger.debug(f"Error on trace end: {e}")

    def on_span_start(self, span: Any) -> None:
        """
        Called when a new span starts.

        Args:
            span: The Span object from OpenAI Agents SDK
        """
        try:
            span_id = self._clean_id(getattr(span, "span_id", None))
            if span_id:
                self._active_spans[span_id] = {
                    "span_id": span_id,
                    "trace_id": self._clean_id(getattr(span, "trace_id", None)),
                    "parent_id": self._clean_id(getattr(span, "parent_id", None)),
                    "start_time": time.time(),
                }
                logger.debug(f"Span started: {span_id}")
        except Exception as e:
            logger.debug(f"Error on span start: {e}")

    def on_span_end(self, span: Any) -> None:
        """
        Called when a span ends.

        Args:
            span: The Span object from OpenAI Agents SDK
        """
        try:
            raw_span_id = getattr(span, "span_id", None)
            span_id = self._clean_id(raw_span_id)
            if not span_id:
                return

            # Get timing info
            start_time = time.time()
            # Try to find by cleaned ID first, then raw ID
            if span_id in self._active_spans:
                span_data = self._active_spans.pop(span_id)
                start_time = span_data.get("start_time", start_time)
            elif raw_span_id in self._active_spans:
                span_data = self._active_spans.pop(raw_span_id)
                start_time = span_data.get("start_time", start_time)

            end_time = time.time()

            # Get span data and convert to asymetry format
            data = getattr(span, "span_data", None)
            if data is None:
                return

            # Determine span type and convert accordingly
            span_type = self._get_span_type(data)

            if span_type == "generation":
                self._process_generation_span(span, data, start_time, end_time)
            else:
                self._process_trace_span(span, data, span_type, start_time, end_time)

        except Exception as e:
            logger.debug(f"Error on span end: {e}")

    def _get_span_type(self, data: Any) -> str:
        """Determine the span type from OpenAI Agents span data."""
        type_name = type(data).__name__

        type_mapping = {
            "GenerationSpanData": "generation",
            "AgentSpanData": "agent",
            "FunctionSpanData": "tool",
            "HandoffSpanData": "agent",
            "GuardrailSpanData": "guardrail",
            "ResponseSpanData": "generation",  # Treat as generation for LLM data
            "CustomSpanData": "custom",
            "MCPListToolsSpanData": "tool",
            "SpeechSpanData": "speech",
            "SpeechGroupSpanData": "speech",
            "TranscriptionSpanData": "transcription",
        }

        return type_mapping.get(type_name, "custom")

    def _process_generation_span(
        self, span: Any, data: Any, start_time: float, end_time: float
    ) -> None:
        """Process a GenerationSpanData or ResponseSpanData span.

        Creates BOTH:
        1. AgentSpan (the parent llm.request) -> goes to traces table via agent endpoint
        2. LLMRequest child span -> goes to llm_requests table via regular exporter
        """
        from .spans import AgentSpan, LLMRequest, TokenUsage, SpanContext

        try:
            type_name = type(data).__name__

            # Extract model information
            model = None
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0

            # Get input/output data
            input_data = getattr(data, "input", None)
            output_data = getattr(data, "output", None)

            # For ResponseSpanData, try to get response info
            if type_name == "ResponseSpanData":
                response = getattr(data, "response", None)
                if response:
                    model = getattr(response, "model", None)
                    usage = getattr(response, "usage", None)
                    if usage:
                        input_tokens = getattr(usage, "input_tokens", 0) or 0
                        output_tokens = getattr(usage, "output_tokens", 0) or 0
                        total_tokens = (
                            getattr(usage, "total_tokens", input_tokens + output_tokens) or 0
                        )
                    # Extract output from response
                    output_list = getattr(response, "output", None)
                    if output_list:
                        output_data = self._safe_serialize(output_list)

            # For GenerationSpanData
            elif type_name == "GenerationSpanData":
                model = getattr(data, "model", None)
                usage = getattr(data, "usage", None)
                if usage:
                    if isinstance(usage, dict):
                        input_tokens = usage.get("input_tokens", 0) or 0
                        output_tokens = usage.get("output_tokens", 0) or 0
                    else:
                        input_tokens = getattr(usage, "input_tokens", 0) or 0
                        output_tokens = getattr(usage, "output_tokens", 0) or 0
                    total_tokens = input_tokens + output_tokens

            # Serialize input/output for AgentSpan
            serialized_input = self._safe_serialize(input_data) if input_data else None
            serialized_output = self._safe_serialize(output_data) if output_data else None

            # Normalize input to messages format for LLMRequest
            normalized_messages = []
            if serialized_input:
                if isinstance(serialized_input, list):
                    normalized_messages = serialized_input
                elif isinstance(serialized_input, dict) and "messages" in serialized_input:
                    normalized_messages = serialized_input["messages"]
                else:
                    normalized_messages = [{"role": "user", "content": str(serialized_input)}]

            # Get error info
            error = getattr(span, "error", None)
            error_message = None
            status = "success"
            if error:
                status = "error"
                error_message = getattr(error, "message", str(error))

            # Get span IDs
            trace_id = self._clean_id(getattr(span, "trace_id", None)) or str(uuid.uuid4())
            span_id = self._clean_id(getattr(span, "span_id", None)) or str(uuid.uuid4())
            parent_span_id = self._clean_id(getattr(span, "parent_id", None))

            # Generate a new span_id for the LLMRequest child
            llm_request_span_id = str(uuid.uuid4())

            # 1. Create AgentSpan for the parent llm.request -> goes to traces
            agent_span = AgentSpan(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                span_type="generation",
                name="llm.request",
                start_time=start_time,
                end_time=end_time,
                status=status,
                error_message=error_message,
                input=serialized_input,
                output=serialized_output,
                attributes={"llm.stream": getattr(data, "stream", False)},
                model=model,
                provider="openai",
                usage={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
            )
            self._enqueue_agent_span(agent_span)

            # 2. Create LLMRequest child span -> goes to llm_requests
            llm_request = LLMRequest(
                provider="openai",
                model=model or "unknown",
                status=status,
                messages=normalized_messages,
                output=serialized_output,
                params={"stream": getattr(data, "stream", False)},
                finish_reason=None,
                trace_id=trace_id,
                parent_span_id=span_id,  # Parent is the AgentSpan we just created
                span_id=llm_request_span_id,
            )

            span_context = SpanContext(
                request=llm_request,
                start_time=start_time,
                end_time=end_time,
            )
            span_context.request.latency_ms = (end_time - start_time) * 1000

            span_context.tokens = TokenUsage(
                request_id=llm_request.request_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                estimated=False,
                estimation_method=None,
            )

            self._enqueue_span(span_context)
            logger.debug(f"Created AgentSpan + LLMRequest for {type_name}: model={model}")

        except Exception as e:
            logger.debug(f"Error processing generation span: {e}")

    def _process_trace_span(
        self, span: Any, data: Any, span_type: str, start_time: float, end_time: float
    ) -> None:
        """Process a non-generation span and convert to AgentSpan."""
        from .spans import AgentSpan

        try:
            # Get span name
            name = getattr(data, "name", None)
            if not name:
                # Try to infer name from span type
                type_name = type(data).__name__
                name = type_name.replace("SpanData", "").lower()

            # Get error info
            error = getattr(span, "error", None)
            error_message = None
            status = "success"
            if error:
                status = "error"
                error_message = getattr(error, "message", str(error))

            # Build attributes from span data
            attributes = self._extract_attributes(data)

            # Get input/output
            input_data = None
            output_data = None

            if hasattr(data, "input"):
                input_val = getattr(data, "input", None)
                if input_val is not None:
                    input_data = self._safe_serialize(input_val)

            if hasattr(data, "output"):
                output_val = getattr(data, "output", None)
                if output_val is not None:
                    output_data = self._safe_serialize(output_val)

            # Create AgentSpan
            agent_span = AgentSpan(
                trace_id=self._clean_id(getattr(span, "trace_id", None)) or str(uuid.uuid4()),
                span_id=self._clean_id(getattr(span, "span_id", None)) or str(uuid.uuid4()),
                parent_span_id=self._clean_id(getattr(span, "parent_id", None)),
                span_type=span_type,
                name=name,
                start_time=start_time,
                end_time=end_time,
                status=status,
                error_message=error_message,
                input=input_data,
                output=output_data,
                attributes=attributes,
            )

            # Enqueue span
            self._enqueue_agent_span(agent_span)
            logger.debug(f"Processed {span_type} span: {name}")

        except Exception as e:
            logger.debug(f"Error processing trace span: {e}")

    def _normalize_messages(self, messages: Any) -> list[dict[str, Any]]:
        """Normalize messages to a list of dicts."""
        result = []
        if not messages:
            return result

        for msg in messages:
            if isinstance(msg, dict):
                result.append(msg)
            elif hasattr(msg, "model_dump"):
                try:
                    result.append(msg.model_dump())
                except Exception:
                    result.append({"content": str(msg)})
            elif hasattr(msg, "role") and hasattr(msg, "content"):
                result.append(
                    {
                        "role": getattr(msg, "role", "user"),
                        "content": getattr(msg, "content", ""),
                    }
                )
            else:
                result.append({"content": str(msg)})

        return result

    def _normalize_output(self, output: Any) -> Any:
        """Normalize output to JSON-serializable format."""
        if output is None:
            return None

        if isinstance(output, (str, int, float, bool)):
            return output

        if isinstance(output, dict):
            return output

        if isinstance(output, list):
            return [self._normalize_output(item) for item in output]

        if hasattr(output, "model_dump"):
            try:
                return output.model_dump()
            except Exception:
                pass

        # Try to extract common fields
        if hasattr(output, "content"):
            return {"content": getattr(output, "content", None)}

        return str(output)

    def _extract_attributes(self, data: Any) -> dict[str, Any]:
        """Extract attributes from span data."""
        attributes = {}

        # Common fields to extract
        fields = [
            "name",
            "handoffs",
            "tools",
            "output_type",
            "triggered",
            "from_agent",
            "to_agent",
            "server",
            "result",
        ]

        for field in fields:
            if hasattr(data, field):
                val = getattr(data, field, None)
                if val is not None:
                    attributes[field] = self._safe_serialize(val)

        return attributes

    def _safe_serialize(self, value: Any) -> Any:
        """Safely serialize a value to JSON-compatible format."""
        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, (list, tuple)):
            return [self._safe_serialize(v) for v in value]

        if isinstance(value, dict):
            return {k: self._safe_serialize(v) for k, v in value.items()}

        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception:
                pass

        if hasattr(value, "__dict__"):
            try:
                return {
                    k: self._safe_serialize(v)
                    for k, v in value.__dict__.items()
                    if not k.startswith("_")
                }
            except Exception:
                pass

        return str(value)

    def _enqueue_span(self, span_context: Any) -> None:
        """Enqueue a SpanContext to the export queue."""
        queue = self._get_queue()
        if queue is not None:
            try:
                queue.put_nowait(span_context)
            except Exception as e:
                logger.debug(f"Failed to enqueue span: {e}")

    def _enqueue_trace_span(self, trace_span: Any) -> None:
        """Enqueue a TraceSpan to the export queue (legacy, for backward compatibility)."""
        queue = self._get_queue()
        if queue is not None:
            try:
                queue.put_nowait(trace_span)
            except Exception as e:
                logger.debug(f"Failed to enqueue trace span: {e}")

    def _enqueue_agent_span(self, agent_span: Any) -> None:
        """Enqueue an AgentSpan to the export queue."""
        queue = self._get_queue()
        if queue is not None:
            try:
                queue.put_nowait(agent_span)
            except Exception as e:
                logger.debug(f"Failed to enqueue agent span: {e}")

    def shutdown(self) -> None:
        """Clean up resources."""
        self._active_traces.clear()
        self._active_spans.clear()
        logger.debug("AsymetryTracingProcessor shutdown complete")

    def force_flush(self) -> None:
        """Force processing of any queued items."""
        # The queue-based export handles flushing automatically
        pass


def instrument_openai_agents(
    api_key: str | None = None,
    span_queue: Any = None,
) -> "AsymetryTracingProcessor":
    """
    Convenience function to set up Asymetry tracing for OpenAI Agents.

    This function creates an AsymetryTracingProcessor and registers it
    with the OpenAI Agents SDK.

    Args:
        api_key: Optional API key for Asymetry (uses env var if not provided)
        span_queue: Optional queue for collecting spans

    Returns:
        The configured AsymetryTracingProcessor instance

    Raises:
        ImportError: If openai-agents package is not installed

    Example:
        from asymetry.openai_agents import instrument_openai_agents

        # Initialize instrumentation
        processor = instrument_openai_agents()

        # Now use OpenAI Agents as normal - traces will be captured
        from agents import Agent, Runner
        agent = Agent(name="Assistant", instructions="Be helpful")
        result = await Runner.run(agent, input="Hello!")
    """
    global _processor, _instrumented

    if _instrumented and _processor is not None:
        logger.debug("OpenAI Agents already instrumented, returning existing processor")
        return _processor

    try:
        from agents import add_trace_processor
    except ImportError:
        raise ImportError(
            "openai-agents package is required for OpenAI Agents instrumentation. "
            "Install it with: pip install openai-agents"
        )

    # Create processor
    _processor = AsymetryTracingProcessor(span_queue=span_queue)

    # Register with OpenAI Agents SDK
    add_trace_processor(_processor)

    _instrumented = True
    logger.info("✓ Asymetry instrumentation enabled for OpenAI Agents SDK")

    # Auto-initialize observability if not already running
    # This ensures the background exporter is started and config is loaded
    try:
        from .main import init_observability, _initialized

        if not _initialized:
            # We pass enable_openai_agents=False because we are already handling it here manually
            # and we don't want to trigger a recursive call or redundant check
            init_observability(api_key=api_key, enable_openai_agents=False)
    except ImportError:
        # Should not happen within the package structure
        pass
    except Exception as e:
        logger.warning(f"Failed to auto-initialize Asymetry observability: {e}")

    return _processor


def uninstrument_openai_agents() -> None:
    """
    Remove OpenAI Agents instrumentation.

    Note: The OpenAI Agents SDK doesn't provide a way to remove processors,
    so this just clears the local state. The processor will still receive
    events but will be a no-op after shutdown.
    """
    global _processor, _instrumented

    if _processor is not None:
        _processor.shutdown()
        _processor = None

    _instrumented = False
    logger.info("OpenAI Agents instrumentation disabled")
