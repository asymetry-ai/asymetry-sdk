"""OpenAI and Anthropic instrumentation via monkey patching."""

import json
import logging
import time
import traceback
import random
from functools import wraps
from typing import Any, Callable

from .spans import LLMRequest, TokenUsage, LLMError, SpanContext
from .token_utils import extract_token_usage, extract_token_usage_anthropic
from .config import get_config

logger = logging.getLogger(__name__)

# Store original methods
_original_chat_create: Callable | None = None
_original_messages_create: Callable | None = None
_instrumented_openai = False
_instrumented_anthropic = False

# Queue for collecting spans (will be set by exporter)
_span_queue: Any = None


def set_span_queue(queue: Any) -> None:
    """Set the queue for collecting spans."""
    global _span_queue
    _span_queue = queue


def instrument_openai() -> None:
    """Monkey patch OpenAI client to capture telemetry."""
    global _original_chat_create, _instrumented_openai

    if _instrumented_openai:
        logger.debug("OpenAI already instrumented, skipping")
        return

    config = get_config()
    if not config.enabled:
        logger.info("Asymetry is disabled, skipping OpenAI instrumentation")
        return

    try:
        import openai
        from openai.resources.chat import completions

        # Store original method
        _original_chat_create = completions.Completions.create

        # Patch with instrumented version
        completions.Completions.create = _instrumented_chat_create

        _instrumented_openai = True
        logger.info("✓ Asymetry instrumentation enabled for OpenAI")

    except ImportError:
        logger.debug("OpenAI SDK not installed, skipping OpenAI instrumentation")
    except Exception as e:
        logger.error(f"Failed to instrument OpenAI: {e}")


def uninstrument_openai() -> None:
    """Remove OpenAI instrumentation (restore original methods)."""
    global _original_chat_create, _instrumented_openai

    if not _instrumented_openai:
        return

    try:
        import openai
        from openai.resources.chat import completions

        if _original_chat_create:
            completions.Completions.create = _original_chat_create
            _original_chat_create = None

        _instrumented_openai = False
        logger.info("OpenAI instrumentation removed")

    except Exception as e:
        logger.error(f"Failed to uninstrument OpenAI: {e}")


def instrument_anthropic() -> None:
    """Monkey patch Anthropic client to capture telemetry."""
    global _original_messages_create, _instrumented_anthropic

    if _instrumented_anthropic:
        logger.debug("Anthropic already instrumented, skipping")
        return

    config = get_config()
    if not config.enabled:
        logger.info("Asymetry is disabled, skipping Anthropic instrumentation")
        return

    try:
        import anthropic
        from anthropic.resources import messages

        # Store original method
        _original_messages_create = messages.Messages.create

        # Patch with instrumented version
        messages.Messages.create = _instrumented_messages_create

        _instrumented_anthropic = True
        logger.info("✓ Asymetry instrumentation enabled for Anthropic")

    except ImportError:
        logger.debug("Anthropic SDK not installed, skipping Anthropic instrumentation")
    except Exception as e:
        logger.error(f"Failed to instrument Anthropic: {e}")


def uninstrument_anthropic() -> None:
    """Remove Anthropic instrumentation (restore original methods)."""
    global _original_messages_create, _instrumented_anthropic

    if not _instrumented_anthropic:
        return

    try:
        import anthropic
        from anthropic.resources import messages

        if _original_messages_create:
            messages.Messages.create = _original_messages_create
            _original_messages_create = None

        _instrumented_anthropic = False
        logger.info("Anthropic instrumentation removed")

    except Exception as e:
        logger.error(f"Failed to uninstrument Anthropic: {e}")


def _get_active_trace_context() -> tuple[str | None, str | None, str | None]:
    """
    Get the active OpenTelemetry trace context.

    Returns:
        Tuple of (trace_id, parent_span_id, new_span_id)
    """
    trace_id = None
    parent_span_id = None
    llm_span_id = None

    try:
        from opentelemetry import trace as otel_trace

        # Get current active span
        current_span = otel_trace.get_current_span()

        # Check if span is valid and recording
        if current_span is not None:
            span_context = current_span.get_span_context()

            if span_context is not None and span_context.is_valid:
                # Extract trace ID and parent span ID
                trace_id = format(span_context.trace_id, "032x")
                parent_span_id = format(span_context.span_id, "016x")

                # Generate new span ID for this LLM call
                llm_span_id = format(random.getrandbits(64), "016x")

                logger.debug(
                    f"Captured trace context - trace_id: {trace_id[:8]}..., "
                    f"parent_span_id: {parent_span_id[:8]}..."
                )

    except ImportError:
        # OpenTelemetry not available
        logger.debug("OpenTelemetry not available, skipping trace context")
    except AttributeError as e:
        # Method doesn't exist
        logger.debug(f"Could not get span context: {e}")
    except Exception as e:
        # Any other error
        logger.debug(f"Error capturing trace context: {e}")

    return trace_id, parent_span_id, llm_span_id


# -------------------- PATCH: OTel auto-root + attributes --------------------


def _maybe_start_llm_span(provider: str, model: str, params: dict[str, Any]):
    """
    Ensure there's an active OTel span. If none, start a root llm.request span.
    Returns (span_or_None, context_manager_or_None, auto_rooted_bool, started_ns)
    """
    started_ns = time.time_ns()
    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.trace import SpanKind

        tracer = otel_trace.get_tracer(__name__)

        # Create a root LLM span transparently
        ctx_mgr = tracer.start_as_current_span(
            "llm.request",
            kind=SpanKind.CLIENT,
            attributes={
                "llm.auto_root": True,
                "llm.provider": provider,
                "llm.model": model,
                "llm.stream": bool(params.get("stream", False)),
            },
        )
        span = ctx_mgr.__enter__()
        # Attach common params as attributes (best-effort)
        for k in (
            "temperature",
            "top_p",
            "max_tokens",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "stop",
        ):
            if k in params:
                span.set_attribute(f"llm.{k}", params[k])
        return span, ctx_mgr, True, started_ns

    except Exception:
        # OTel not present or error; operate as no-op
        return None, None, False, started_ns


def _finalize_llm_span(ctx_mgr, span, started_ns, ok: bool, finish_reason: str | None = None):
    """Close auto-root span (if any) and set latency/status/finish_reason."""
    try:
        if span is not None:
            # latency
            dur_ms = (time.time_ns() - started_ns) / 1e6
            span.set_attribute("llm.latency_ms", int(dur_ms))
            if finish_reason is not None:
                span.set_attribute("llm.finish_reason", finish_reason)

            # status
            from opentelemetry.trace import Status, StatusCode

            span.set_status(Status(StatusCode.OK if ok else StatusCode.ERROR))
    except Exception:
        pass
    finally:
        if ctx_mgr is not None:
            try:
                ctx_mgr.__exit__(None if ok else Exception, None, None)
            except Exception:
                pass


def _serialize_openai_function_call(function_call: Any) -> dict[str, Any] | None:
    """
    Best-effort serialization of OpenAI function_call objects into plain dicts.
    Handles both legacy and v1 SDK structures.
    """
    if not function_call:
        return None

    # Some SDK objects expose model_dump()
    try:
        if hasattr(function_call, "model_dump"):
            data = function_call.model_dump()
            # Ensure we only keep JSON-serializable primitives
            return {
                "name": data.get("name"),
                "arguments": data.get("arguments"),
            }
    except Exception:
        pass

    # Fallback: use getattr
    try:
        return {
            "name": getattr(function_call, "name", None),
            "arguments": getattr(function_call, "arguments", None),
        }
    except Exception:
        return None


def _normalize_openai_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    """
    Convert OpenAI tool_calls objects to plain dicts so they are JSON serializable.
    """
    if not tool_calls:
        return []

    normalized: list[dict[str, Any]] = []

    for tc in tool_calls:
        try:
            # Some SDKs expose model_dump() on tool call objects
            if hasattr(tc, "model_dump"):
                data = tc.model_dump()
                # Standard OpenAI structure: {id, type, function: {name, arguments}}
                fn = data.get("function") or {}
                normalized.append(
                    {
                        "id": data.get("id"),
                        "type": data.get("type"),
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments"),
                    }
                )
                continue
        except Exception:
            # Fall through to getattr-based handling
            pass

        try:
            fn_obj = getattr(tc, "function", None)
            fn_name = getattr(fn_obj, "name", None) if fn_obj is not None else None
            fn_args = getattr(fn_obj, "arguments", None) if fn_obj is not None else None

            normalized.append(
                {
                    "id": getattr(tc, "id", None),
                    "type": getattr(tc, "type", None),
                    "name": fn_name or getattr(tc, "name", None),
                    "arguments": fn_args,
                }
            )
        except Exception:
            # Last-resort: just record type name so we don't break export
            normalized.append({"type": str(type(tc))})

    return normalized


def _normalize_messages_for_json(messages: list[Any]) -> list[dict[str, Any]]:
    """
    Ensure all messages are serializable dicts.
    Handles ChatCompletionMessage objects from OpenAI SDK.
    """
    if not messages:
        return []

    normalized = []
    for m in messages:
        if isinstance(m, dict):
            normalized.append(m)
            continue

        # Try model_dump() (Pydantic/OpenAI models)
        if hasattr(m, "model_dump"):
            try:
                # model_dump might return complex objects for tool_calls, so we need to process the result
                data = m.model_dump()
                
                # If tool_calls are present in the dumped data, ensure they are normalized
                if "tool_calls" in data and isinstance(data["tool_calls"], list):
                     # Check if items in tool_calls are dicts; if not, normalize them
                     if data["tool_calls"] and not isinstance(data["tool_calls"][0], dict):
                         data["tool_calls"] = _normalize_openai_tool_calls(getattr(m, "tool_calls", []))
                     # Also double-check for deeply nested objects just in case model_dump gave us mixed results
                     # We can just run _normalize_openai_tool_calls on the list anyway if we have access to original objects
                     # But if data["tool_calls"] are already dicts, we are fine.
                     
                     # Safety check: ensure everything in tool_calls IS a dict
                     safe_tool_calls = []
                     for tc in data["tool_calls"]:
                         if isinstance(tc, dict):
                             safe_tool_calls.append(tc)
                         elif hasattr(tc, "model_dump"):
                             safe_tool_calls.append(tc.model_dump())
                         else:
                             # Fallback to existing normalizer logic by pretending it's an object?
                             # Or use the _normalize_openai_tool_calls if we can get the original list?
                             # Best is to try to serialize THIS specific item
                             try:
                                 safe_tool_calls.append(
                                     {
                                         "id": getattr(tc, "id", None),
                                         "type": getattr(tc, "type", None),
                                         "function": {
                                             "name": getattr(tc.function, "name", None),
                                             "arguments": getattr(tc.function, "arguments", None)
                                         } if hasattr(tc, "function") else None
                                     }
                                 )
                             except Exception:
                                 safe_tool_calls.append({"type": str(type(tc))})
                     
                     data["tool_calls"] = safe_tool_calls
                
                normalized.append(data)
                continue
            except Exception:
                pass

        # Try to extract common fields manually if model_dump failed or didn't exist
        try:
            msg_dict = {}
            # Common fields
            for field in ["role", "content", "name"]:
                if hasattr(m, field):
                    val = getattr(m, field)
                    if val is not None:
                        msg_dict[field] = val

            # Complex fields
            if hasattr(m, "tool_calls"):
                tc = getattr(m, "tool_calls", None)
                if tc:
                    msg_dict["tool_calls"] = _normalize_openai_tool_calls(tc)
            
            if hasattr(m, "function_call"):
                fc = getattr(m, "function_call", None)
                if fc:
                    msg_dict["function_call"] = _serialize_openai_function_call(fc)

            if msg_dict:
                normalized.append(msg_dict)
                continue
        except Exception:
            pass

        # Fallback
        normalized.append({"_unserializable_type": str(type(m))})

    return normalized



# ---------------------------------------------------------------------------
# OpenAI Instrumentation
# ---------------------------------------------------------------------------


def _instrumented_chat_create(self, *args, **kwargs):
    """
    Instrumented version of openai.chat.completions.create().
    Wraps the original method to capture telemetry.
    """
    if not _original_chat_create:
        raise RuntimeError("Original OpenAI method not saved")

    # Start timing
    start_time = time.time()

    # Extract parameters
    model = kwargs.get("model", "unknown")
    messages = _normalize_messages_for_json(kwargs.get("messages", []))
    temperature = kwargs.get("temperature")
    max_tokens = kwargs.get("max_tokens")

    # Ensure OTel span + set attrs
    otel_span, otel_ctx, _auto_rooted, otel_started_ns = _maybe_start_llm_span(
        provider="openai",
        model=model,
        params=kwargs,
    )

    # Capture active trace context (if exists)
    trace_id, parent_span_id, llm_span_id = _get_active_trace_context()

    # Create request span
    request = LLMRequest(
        provider="openai",
        model=model,
        params={
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": kwargs.get("stream", False),
        },
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        span_id=llm_span_id,
    )

    span = SpanContext(request=request, start_time=start_time)

    try:
        # Call original method
        response = _original_chat_create(self, *args, **kwargs)

        # Capture request details
        request.messages = messages

        # Capture response
        finish_reason = None
        tool_calls_info = []
        if hasattr(response, "choices") and response.choices:
            normalized_output = []
            for choice in response.choices:
                message = getattr(choice, "message", None)
                if not message:
                    continue

                function_call = getattr(message, "function_call", None)
                tool_calls = getattr(message, "tool_calls", None)

                # Serialize function_call/tool_calls to JSON-safe structures
                serialized_function_call = _serialize_openai_function_call(function_call)
                serialized_tool_calls = _normalize_openai_tool_calls(tool_calls)

                # Normalize assistant output including function/tool calls
                normalized_output.append(
                    {
                        "role": "assistant",
                        "content": getattr(message, "content", None),
                        "function_call": serialized_function_call,
                        "tool_calls": serialized_tool_calls,
                    }
                )

                # Collect tool call telemetry (for tools / MCP tools)
                if serialized_tool_calls:
                    tool_calls_info.extend(
                        {"id": tc.get("id"), "type": tc.get("type"), "name": tc.get("name")}
                        for tc in serialized_tool_calls
                    )

            request.output = normalized_output

            # Extract finish reason
            finish_reason = response.choices[0].finish_reason
            request.finish_reason = finish_reason

        # Capture timing
        end_time = time.time()
        span.finish(end_time)

        # Extract token usage
        token_data = extract_token_usage(response, messages, model)
        span.tokens = TokenUsage(
            request_id=request.request_id,
            input_tokens=token_data["input_tokens"],
            output_tokens=token_data["output_tokens"],
            total_tokens=token_data["total_tokens"],
            estimated=token_data["estimated"],
            estimation_method=token_data["estimation_method"],
        )

        # Write tokens + tool metadata to OTel attrs
        try:
            if otel_span is not None:
                otel_span.set_attribute(
                    "llm.tokens.input", int(token_data.get("input_tokens", 0) or 0)
                )
                otel_span.set_attribute(
                    "llm.tokens.output", int(token_data.get("output_tokens", 0) or 0)
                )
                otel_span.set_attribute(
                    "llm.tokens.total", int(token_data.get("total_tokens", 0) or 0)
                )
                otel_span.set_attribute("function.args.messages", json.dumps(messages))
                otel_span.set_attribute("function.result", json.dumps(request.output))
                # Tool / MCP call observability on the LLM span
                if tool_calls_info:
                    otel_span.set_attribute("llm.tools.count", len(tool_calls_info))
                    otel_span.set_attribute(
                        "llm.tools.names",
                        [tc["name"] for tc in tool_calls_info if tc.get("name")],
                    )
        except Exception:
            pass

        # Mark as success
        request.status = "success"

        # Finalize OTel span
        _finalize_llm_span(
            otel_ctx, otel_span, otel_started_ns, ok=True, finish_reason=finish_reason
        )

        # Send to queue
        _enqueue_span(span)

        return response

    except Exception as e:
        # Capture error
        end_time = time.time()
        span.finish(end_time)

        request.status = "error"

        # Create error record
        error_type = type(e).__name__
        error_code = None
        error_message = str(e)

        # Extract OpenAI-specific error info
        if hasattr(e, "code"):
            error_code = e.code
        if hasattr(e, "message"):
            error_message = e.message

        span.error = LLMError(
            request_id=request.request_id,
            error_type=error_type,
            error_code=error_code,
            error_message=error_message,
            stack_trace=traceback.format_exc()[:1000],  # Truncate to 1000 chars
        )

        # Finalize OTel span (error)
        _finalize_llm_span(otel_ctx, otel_span, otel_started_ns, ok=False, finish_reason=None)

        # Send to queue
        _enqueue_span(span)

        # Re-raise the error
        raise


# ---------------------------------------------------------------------------
# Anthropic Instrumentation
# ---------------------------------------------------------------------------


def _instrumented_messages_create(self, *args, **kwargs):
    """
    Instrumented version of anthropic.messages.create().
    Wraps the original method to capture telemetry.
    """
    if not _original_messages_create:
        raise RuntimeError("Original Anthropic method not saved")

    # Start timing
    start_time = time.time()

    # Extract parameters
    model = kwargs.get("model", "unknown")
    messages = _normalize_messages_for_json(kwargs.get("messages", []))
    system = kwargs.get("system")  # Anthropic has separate system parameter
    temperature = kwargs.get("temperature")
    max_tokens = kwargs.get("max_tokens")

    # Ensure OTel span + set attrs
    otel_span, otel_ctx, _auto_rooted, otel_started_ns = _maybe_start_llm_span(
        provider="anthropic",
        model=model,
        params=kwargs,
    )

    # Capture active trace context (if exists)
    trace_id, parent_span_id, llm_span_id = _get_active_trace_context()

    # Create request span
    request = LLMRequest(
        provider="anthropic",
        model=model,
        params={
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system": system,
            "stream": kwargs.get("stream", False),
        },
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        span_id=llm_span_id,
    )

    span = SpanContext(request=request, start_time=start_time)

    try:
        # Call original method
        response = _original_messages_create(self, *args, **kwargs)

        # Capture request details (include system prompt if present)
        request.messages = messages
        if system:
            # Prepend system message for consistency with OpenAI format
            request.messages = [{"role": "system", "content": system}] + messages

        # Capture response - Anthropic returns content as list of blocks
        finish_reason = None
        tool_uses_info = []
        if hasattr(response, "content") and response.content:
            output_blocks = []
            for block in response.content:
                if not hasattr(block, "type"):
                    continue

                if block.type == "text" and hasattr(block, "text"):
                    output_blocks.append(
                        {
                            "role": "assistant",
                            "type": "text",
                            "content": block.text,
                        }
                    )
                elif block.type == "tool_use":
                    name = getattr(block, "name", None)
                    output_blocks.append(
                        {
                            "role": "assistant",
                            "type": "tool_use",
                            "id": getattr(block, "id", None),
                            "name": name,
                            "input": getattr(block, "input", None),
                        }
                    )
                    tool_uses_info.append(
                        {
                            "id": getattr(block, "id", None),
                            "name": name,
                        }
                    )

            request.output = output_blocks

        # Extract finish reason
        if hasattr(response, "stop_reason"):
            finish_reason = response.stop_reason
            request.finish_reason = finish_reason

        # Capture timing
        end_time = time.time()
        span.finish(end_time)

        # Extract token usage (Anthropic always provides this)
        token_data = extract_token_usage_anthropic(response, messages, model, system)
        span.tokens = TokenUsage(
            request_id=request.request_id,
            input_tokens=token_data["input_tokens"],
            output_tokens=token_data["output_tokens"],
            total_tokens=token_data["total_tokens"],
            estimated=token_data["estimated"],
            estimation_method=token_data["estimation_method"],
        )

        # Write tokens + tool metadata to OTel attrs
        try:
            if otel_span is not None:
                otel_span.set_attribute(
                    "llm.tokens.input", int(token_data.get("input_tokens", 0) or 0)
                )
                otel_span.set_attribute(
                    "llm.tokens.output", int(token_data.get("output_tokens", 0) or 0)
                )
                otel_span.set_attribute(
                    "llm.tokens.total", int(token_data.get("total_tokens", 0) or 0)
                )
                # Store messages and system prompt
                all_messages = messages.copy()
                if system:
                    all_messages.insert(0, {"role": "system", "content": system})
                otel_span.set_attribute("function.args.messages", json.dumps(all_messages))
                otel_span.set_attribute("function.result", json.dumps(request.output))
                # Tool / MCP call observability on the LLM span (Anthropic tool_use blocks)
                if tool_uses_info:
                    otel_span.set_attribute("llm.tools.count", len(tool_uses_info))
                    otel_span.set_attribute(
                        "llm.tools.names",
                        [tu["name"] for tu in tool_uses_info if tu.get("name")],
                    )
        except Exception:
            pass

        # Mark as success
        request.status = "success"

        # Finalize OTel span
        _finalize_llm_span(
            otel_ctx, otel_span, otel_started_ns, ok=True, finish_reason=finish_reason
        )

        # Send to queue
        _enqueue_span(span)

        return response

    except Exception as e:
        # Capture error
        end_time = time.time()
        span.finish(end_time)

        request.status = "error"

        # Create error record
        error_type = type(e).__name__
        error_code = None
        error_message = str(e)

        # Extract Anthropic-specific error info
        if hasattr(e, "status_code"):
            error_code = str(e.status_code)
        if hasattr(e, "message"):
            error_message = e.message

        span.error = LLMError(
            request_id=request.request_id,
            error_type=error_type,
            error_code=error_code,
            error_message=error_message,
            stack_trace=traceback.format_exc()[:1000],  # Truncate to 1000 chars
        )

        # Finalize OTel span (error)
        _finalize_llm_span(otel_ctx, otel_span, otel_started_ns, ok=False, finish_reason=None)

        # Send to queue
        _enqueue_span(span)

        # Re-raise the error
        raise


# ---------------------------------------------------------------------------
# Common Utilities
# ---------------------------------------------------------------------------


def _enqueue_span(span: SpanContext) -> None:
    """Send span to the async queue for export."""
    if _span_queue is None:
        logger.warning("Span queue not initialized, dropping span")
        return

    try:
        # Non-blocking put
        _span_queue.put_nowait(span)
    except Exception as e:
        logger.error(f"Failed to enqueue span: {e}")