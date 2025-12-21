"""Data models for LLM request spans and distributed traces."""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any
import uuid
from opentelemetry.trace import StatusCode as _OTelStatusCode


@dataclass
class LLMRequest:
    """Core LLM request span data."""

    # Unique identifiers
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Provider and model info
    provider: str = "openai"
    model: str = ""

    # Timing
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    latency_ms: float = 0.0

    # Status
    status: str = "success"  # success, error

    # User context
    route: str | None = None
    user_id: str | None = None

    # Messages sent
    messages: list[str] = field(default_factory=list)

    # Response received by LLM
    output: Any | None = None

    # Request parameters (stored as JSON)
    params: dict[str, Any] = field(default_factory=dict)

    # Response metadata
    finish_reason: str | None = None

    # Trace context (for correlation with distributed traces)
    trace_id: str | None = None  # Links to TraceSpan.trace_id
    parent_span_id: str | None = None  # Links to parent TraceSpan.span_id
    span_id: str | None = None  # This LLM call's own span ID

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API payload."""
        return asdict(self)


@dataclass
class TokenUsage:
    """Token usage data for a request."""

    request_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Estimation method
    estimated: bool = False
    estimation_method: str | None = None  # tiktoken, char_based

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API payload."""
        return asdict(self)


@dataclass
class LLMError:
    """Error data for failed requests."""

    request_id: str
    error_type: str  # e.g., "APIError", "RateLimitError", "Timeout"
    error_code: str | None = None  # e.g., "rate_limit_exceeded", "invalid_api_key"
    error_message: str = ""

    # Stack trace (optional, truncated)
    stack_trace: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API payload."""
        return asdict(self)


@dataclass
class SpanContext:
    """Context for tracking a span during execution."""

    request: LLMRequest
    tokens: TokenUsage | None = None
    error: LLMError | None = None

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0

    def finish(self, end_time: float) -> None:
        """Mark span as finished and calculate latency."""
        self.end_time = end_time
        self.request.latency_ms = (end_time - self.start_time) * 1000


@dataclass
class TraceSpan:
    """Distributed trace span data for custom function tracing."""

    # Unique identifiers
    trace_id: str  # Trace ID (shared across all spans in a trace)
    span_id: str  # Unique span ID
    parent_span_id: str | None = None  # Parent span ID (None for root spans)

    # Span metadata
    name: str = ""  # Span name (e.g., function name, operation name)
    kind: str = "internal"  # internal, client, server, producer, consumer

    # Timing
    start_time: float = 0.0  # Unix timestamp (seconds)
    end_time: float = 0.0  # Unix timestamp (seconds)
    duration_ms: float = 0.0  # Duration in milliseconds

    # Status
    status: str = "success"  # success, error
    error_message: str | None = None

    # Input/Output tracking
    input: dict[str, Any] | None = None  # Function input arguments
    output: Any | None = None  # Function return value

    # Attributes (custom key-value pairs)
    attributes: dict[str, Any] = field(default_factory=dict)

    # Events (timestamped log points within the span)
    events: list[dict[str, Any]] = field(default_factory=list)

    # Service/resource info
    service_name: str = "unknown"
    resource_attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API payload."""
        return asdict(self)

    def add_attribute(self, key: str, value: Any) -> None:
        """Add an attribute to the span."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to the span."""
        import time

        event = {"name": name, "timestamp": time.time(), "attributes": attributes or {}}
        self.events.append(event)


def llmrequest_from_otel_span(otel_span) -> "LLMRequest | None":
    """
    Build an LLMRequest projection from a finished OpenTelemetry span.
    We export only if the span 'looks like' an LLM call.
    """
    try:
        attrs = dict(getattr(otel_span, "attributes", {}) or {})
        name = getattr(otel_span, "name", "")
        if not (name == "llm.request" or "llm.provider" in attrs or "llm.model" in attrs):
            return None  # not an llm span

        sc = otel_span.get_span_context()
        status_code = getattr(otel_span.status, "status_code", _OTelStatusCode.UNSET)
        status = (
            "ok"
            if status_code == _OTelStatusCode.OK
            else ("error" if status_code == _OTelStatusCode.ERROR else "unset")
        )

        # OTel times are ns since epoch
        start_ts = getattr(otel_span, "start_time", None)
        end_ts = getattr(otel_span, "end_time", None)
        start_sec = (start_ts / 1e9) if start_ts else None
        end_sec = (end_ts / 1e9) if end_ts else None

        req = LLMRequest(
            # keep your existing defaults for request_id; use OTel IDs for trace/span
            trace_id=f"{sc.trace_id:032x}",
            span_id=f"{sc.span_id:016x}",
            parent_span_id=(
                f"{otel_span.parent.span_id:016x}" if getattr(otel_span, "parent", None) else None
            ),
            provider=attrs.get("llm.provider"),
            model=attrs.get("llm.model"),
            start_time=start_sec,
            end_time=end_sec,
            status=status,
            attributes=attrs,
        )
        # optional convenience fields if you carry tokens as attributes
        if "llm.tokens.input" in attrs or "llm.tokens.output" in attrs:
            try:
                from .spans import TokenUsage

                req.tokens = TokenUsage(
                    request_id=req.request_id,
                    input_tokens=int(attrs.get("llm.tokens.input", 0) or 0),
                    output_tokens=int(attrs.get("llm.tokens.output", 0) or 0),
                    total_tokens=int(attrs.get("llm.tokens.total", 0) or 0),
                )
            except Exception:
                pass
        return req
    except Exception:
        return None

