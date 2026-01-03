"""OpenTelemetry-based tracing for custom functions."""

import functools
import json
import logging
import time
import traceback
from typing import Any, Callable, Optional
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.trace import Status, StatusCode

from .spans import TraceSpan
from .config import get_config

from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter as _OTelSpanExporter
from opentelemetry.trace import StatusCode
from .spans import llmrequest_from_otel_span
from .exporter import get_exporter

logger = logging.getLogger(__name__)

# Global tracer
_tracer: Optional[trace.Tracer] = None
_tracer_provider: Optional[TracerProvider] = None
_span_queue: Any = None
_LLM_EXPORTER_INSTALLED = False


def set_trace_queue(queue: Any) -> None:
    """Set the queue for collecting trace spans."""
    global _span_queue
    _span_queue = queue


def init_tracing(service_name: str = "asymetry-ai-app") -> None:
    """
    Initialize OpenTelemetry tracing.

    Args:
        service_name: Name of the service for resource identification
    """
    global _tracer, _tracer_provider

    if _tracer is not None:
        logger.debug("Tracing already initialized")
        return

    try:
        # Create resource with service name
        resource = Resource(attributes={SERVICE_NAME: service_name})

        # Create tracer provider
        _tracer_provider = TracerProvider(resource=resource)

        # Add custom span processor (exports to our queue)
        span_processor = AsymetrySpanProcessor()
        _tracer_provider.add_span_processor(span_processor)

        # Set as global tracer provider
        trace.set_tracer_provider(_tracer_provider)

        _tracer_provider.add_span_processor(SimpleSpanProcessor(_OTelToLLMRequestExporter()))

        # Get tracer
        _tracer = trace.get_tracer(__name__)

        logger.info("✓ OpenTelemetry tracing initialized")

    except Exception as e:
        logger.error(f"Failed to initialize tracing: {e}", exc_info=True)


def shutdown_tracing() -> None:
    """Shutdown tracing and flush remaining spans."""
    global _tracer, _tracer_provider

    if _tracer_provider is not None:
        _tracer_provider.shutdown()
        _tracer_provider = None
        _tracer = None
        logger.info("✓ Tracing shutdown complete")


def observe(
    name: Optional[str] = None,
    kind: str = "internal",
    attributes: Optional[dict[str, Any]] = None,
    span_type: Optional[str] = None,
    capture_args: bool = True,
    capture_result: bool = True,
) -> Callable:
    """
    Decorator to trace function execution.

    Args:
        name: Span name (defaults to function name)
        kind: Span kind (internal, client, server, producer, consumer)
        attributes: Additional attributes to attach to span
        span_type: Type of span (e.g., "tool", "agent", "llm", "workflow")
        capture_args: Whether to capture function arguments
        capture_result: Whether to capture return value

    Example:
        ```python
        @trace(name="process_data", attributes={"version": "1.0"})
        def process_data(user_id: str, data: dict) -> dict:
            # Your code here
            return result

        @trace(capture_args=True, capture_result=True)
        async def async_operation(param: str) -> str:
            # Your async code here
            return result
        ```
    """

    def decorator(func: Callable) -> Callable:
        span_name = name or func.__qualname__

        # Merge span_type into attributes
        effective_attributes = attributes.copy() if attributes else {}
        if span_type:
            effective_attributes["span.type"] = span_type

        # Handle async functions
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await _trace_execution_async(
                    func, span_name, kind, effective_attributes, capture_args, capture_result, args, kwargs
                )

            return async_wrapper

        # Handle sync functions
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return _trace_execution_sync(
                func, span_name, kind, effective_attributes, capture_args, capture_result, args, kwargs
            )

        return sync_wrapper

    return decorator


def _trace_execution_sync(
    func: Callable,
    span_name: str,
    kind: str,
    attributes: Optional[dict[str, Any]],
    capture_args: bool,
    capture_result: bool,
    args: tuple,
    kwargs: dict,
) -> Any:
    """Execute sync function with tracing."""
    if _tracer is None:
        logger.warning("Tracer not initialized, executing without tracing")
        return func(*args, **kwargs)

    # Map kind string to SpanKind
    span_kind_map = {
        "internal": trace.SpanKind.INTERNAL,
        "client": trace.SpanKind.CLIENT,
        "server": trace.SpanKind.SERVER,
        "producer": trace.SpanKind.PRODUCER,
        "consumer": trace.SpanKind.CONSUMER,
    }
    span_kind = span_kind_map.get(kind.lower(), trace.SpanKind.INTERNAL)

    with _tracer.start_as_current_span(span_name, kind=span_kind) as span:
        start_time = time.time()

        # Add default attributes
        span.set_attribute("function.name", func.__name__)
        span.set_attribute("function.module", func.__module__)

        # Add custom attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        # Capture arguments if requested
        if capture_args:
            _capture_arguments(span, func, args, kwargs)

        try:
            # Execute function
            result = func(*args, **kwargs)

            # Capture result if requested
            if capture_result and result is not None:
                span.set_attribute("function.result", json.dumps(result))

            # Mark as successful
            span.set_status(Status(StatusCode.OK))

            return result

        except Exception as e:
            # Record exception
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)

            # Re-raise
            raise

        finally:
            # Record timing
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            span.set_attribute("function.duration_ms", duration_ms)


async def _trace_execution_async(
    func: Callable,
    span_name: str,
    kind: str,
    attributes: Optional[dict[str, Any]],
    capture_args: bool,
    capture_result: bool,
    args: tuple,
    kwargs: dict,
) -> Any:
    """Execute async function with tracing."""
    if _tracer is None:
        logger.warning("Tracer not initialized, executing without tracing")
        return await func(*args, **kwargs)

    # Map kind string to SpanKind
    span_kind_map = {
        "internal": trace.SpanKind.INTERNAL,
        "client": trace.SpanKind.CLIENT,
        "server": trace.SpanKind.SERVER,
        "producer": trace.SpanKind.PRODUCER,
        "consumer": trace.SpanKind.CONSUMER,
    }
    span_kind = span_kind_map.get(kind.lower(), trace.SpanKind.INTERNAL)

    with _tracer.start_as_current_span(span_name, kind=span_kind) as span:
        start_time = time.time()

        # Add default attributes
        span.set_attribute("function.name", func.__name__)
        span.set_attribute("function.module", func.__module__)

        # Add custom attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        # Capture arguments if requested
        if capture_args:
            _capture_arguments(span, func, args, kwargs)

        try:
            # Execute async function
            result = await func(*args, **kwargs)

            # Capture result if requested
            if capture_result and result is not None:
                span.set_attribute("function.result", json.dumps(result))

            # Mark as successful
            span.set_status(Status(StatusCode.OK))

            return result

        except Exception as e:
            # Record exception
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)

            # Re-raise
            raise

        finally:
            # Record timing
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            span.set_attribute("function.duration_ms", duration_ms)


@contextmanager
def trace_context(
    name: str,
    attributes: Optional[dict[str, Any]] = None,
    kind: str = "internal",
    span_type: Optional[str] = None,
):
    """
    Context manager for manual span creation.

    Example:
        ```python
        with trace_context("database_query", attributes={"table": "users"}):
            # Your code here
            result = db.query(...)
        ```
    """
    if _tracer is None:
        logger.warning("Tracer not initialized, executing without tracing")
        yield None
        return

    span_kind_map = {
        "internal": trace.SpanKind.INTERNAL,
        "client": trace.SpanKind.CLIENT,
        "server": trace.SpanKind.SERVER,
        "producer": trace.SpanKind.PRODUCER,
        "consumer": trace.SpanKind.CONSUMER,
    }
    span_kind = span_kind_map.get(kind.lower(), trace.SpanKind.INTERNAL)

    with _tracer.start_as_current_span(name, kind=span_kind) as span:
        # Add custom attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        if span_type:
            span.set_attribute("span.type", span_type)

        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def add_span_attribute(key: str, value: Any) -> None:
    """
    Add attribute to current active span.

    Example:
        ```python
        @trace()
        def my_function():
            add_span_attribute("user_id", "12345")
            add_span_attribute("action", "update")
        ```
    """
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.set_attribute(key, value)
    else:
        logger.warning("No active span to add attribute to")


def add_span_event(name: str, attributes: Optional[dict[str, Any]] = None) -> None:
    """
    Add event to current active span.

    Example:
        ```python
        @trace()
        def my_function():
            add_span_event("cache_hit", {"key": "user:123"})
        ```
    """
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.add_event(name, attributes=attributes or {})
    else:
        logger.warning("No active span to add event to")


class AsymetrySpanProcessor:
    """Custom span processor that exports to Asymetry queue."""

    def on_start(self, span: trace.Span, parent_context) -> None:
        """Called when a span starts."""
        pass

    def on_end(self, span: trace.Span) -> None:
        """Called when a span ends."""
        # ReadableSpan doesn't have is_recording, it's already ended
        # Just check if span exists and has required attributes
        if span is None:
            return

        # Convert OTel span to our internal format
        trace_span = self._convert_span(span)

        # Send to queue
        if _span_queue is not None:
            try:
                _span_queue.put_nowait(trace_span)
                logger.debug(f"Enqueued trace span: {span.name}")
            except Exception as e:
                logger.error(f"Failed to enqueue trace span: {e}")
        else:
            logger.warning("Trace queue not initialized")

    def shutdown(self) -> None:
        """Called during shutdown."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any buffered spans."""
        return True

    def _convert_span(self, span: trace.Span) -> "TraceSpan":
        """Convert OpenTelemetry span to Asymetry TraceSpan."""
        from .spans import TraceSpan

        # Extract timing
        start_time = span.start_time / 1e9  # Convert nanoseconds to seconds
        end_time = span.end_time / 1e9 if span.end_time else time.time()
        duration_ms = (end_time - start_time) * 1000

        # Extract attributes
        attributes = {}
        if hasattr(span, "attributes") and span.attributes:
            attributes = dict(span.attributes)

        # Extract input (function arguments) from attributes
        input_data = {}
        output_data = None

        for key, value in list(attributes.items()):
            if key.startswith("function.args."):
                # Extract argument name and value
                arg_name = key.replace("function.args.", "")
                input_data[arg_name] = json.loads(value)
                # Remove from attributes to avoid duplication
                del attributes[key]
            elif key == "function.result":
                # Extract output
                output_data = json.loads(value)
                # Remove from attributes to avoid duplication
                del attributes[key]

        # Extract status
        status = "success"
        error_message = None
        if hasattr(span, "status") and span.status:
            if span.status.status_code == StatusCode.ERROR:
                status = "error"
                error_message = span.status.description

        # Extract events
        events = []
        if hasattr(span, "events"):
            for event in span.events:
                events.append(
                    {
                        "name": event.name,
                        "timestamp": event.timestamp / 1e9,
                        "attributes": dict(event.attributes) if event.attributes else {},
                    }
                )

        return TraceSpan(
            trace_id=format(span.context.trace_id, "032x"),
            span_id=format(span.context.span_id, "016x"),
            parent_span_id=format(span.parent.span_id, "016x") if span.parent else None,
            name=span.name,
            kind=span.kind.name.lower(),
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            status=status,
            attributes=attributes,
            events=events,
            error_message=error_message,
            input=input_data if input_data else None,
            output=output_data,
        )


def _capture_arguments(span: trace.Span, func: Callable, args: tuple, kwargs: dict) -> None:
    """Capture function arguments as span attributes."""
    try:
        import inspect

        sig = inspect.signature(func)
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        for param_name, param_value in bound_args.arguments.items():
            # Skip self/cls parameters
            if param_name in ("self", "cls"):
                continue

            # Serialize and add as attribute
            serialized = json.dumps(param_value)
            span.set_attribute(f"function.args.{param_name}", serialized)

    except Exception as e:
        logger.debug(f"Failed to capture arguments: {e}")


def _serialize_value(value: Any, max_length: int = 200) -> str:
    """Serialize a value for span attributes."""
    try:
        # Handle common types
        if value is None:
            return "None"
        elif isinstance(value, (str, int, float, bool)):
            result = str(value)
        elif isinstance(value, (list, tuple)):
            result = f"[{len(value)} items]"
        elif isinstance(value, dict):
            result = f"{{...{len(value)} keys}}"
        else:
            result = f"<{type(value).__name__}>"

        # Truncate if too long
        if len(result) > max_length:
            result = result[:max_length] + "..."

        return result

    except Exception:
        return "<serialization failed>"


class _OTelToLLMRequestExporter(_OTelSpanExporter):
    """Projects OTel spans to LLMRequest and pushes them to the existing exporter queue."""

    def export(self, spans):
        try:
            q = get_exporter().get_queue()
        except Exception:
            return 1  # failure

        n = 0
        for s in spans:
            req = llmrequest_from_otel_span(s)
            if not req:
                continue
            try:
                # push the *projection* into your pipeline (the queue already exists in exporter.py)
                q.put_nowait(req)
                n += 1
            except Exception:
                # best-effort; continue
                continue
        return 0 if n else 0  # success even if none qualified


# Make asyncio available for async function detection
import asyncio
