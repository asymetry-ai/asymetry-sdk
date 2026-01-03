"""Asymetry - LLM Observability SDK."""

from .version import __version__

# Core initialization
from .main import init_observability, shutdown_observability

# Tracing decorators and utilities
from .tracing import (
    observe,
    trace_context,
    add_span_attribute,
    add_span_event,
)

__all__ = [
    # Initialization
    "init_observability",
    "shutdown_observability",
    # Tracing
    "observe",
    "trace_context",
    "add_span_attribute",
    "add_span_event",
]

