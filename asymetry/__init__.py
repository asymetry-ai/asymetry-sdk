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
    VALID_SPAN_TYPES,
)

# OpenAI Agents SDK integration (optional)
try:
    from .openai_agents import AsymetryTracingProcessor, instrument_openai_agents

    _HAS_OPENAI_AGENTS = True
except ImportError:
    _HAS_OPENAI_AGENTS = False

__all__ = [
    # Initialization
    "init_observability",
    "shutdown_observability",
    # Tracing
    "observe",
    "trace_context",
    "add_span_attribute",
    "add_span_event",
    "VALID_SPAN_TYPES",
]

# Add optional exports if available
if _HAS_OPENAI_AGENTS:
    __all__.extend(
        [
            "AsymetryTracingProcessor",
            "instrument_openai_agents",
        ]
    )
