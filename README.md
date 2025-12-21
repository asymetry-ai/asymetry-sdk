# Asymetry SDK

> Observability and tracing for OpenAI & Anthropic workloads – built for modern LLM Ops.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`asymetry` is a zero-code observability layer for LLM applications. It hooks into the official OpenAI and Anthropic SDKs, captures detailed telemetry (latency, token usage, params, errors), enriches it with OpenTelemetry context, and streams everything to the Asymetry backend via a resilient exporter.

---

## Capability Highlights (v0.1.0-alpha)

- **Automatic provider instrumentation** – monkey patches OpenAI `chat.completions.create` and Anthropic `messages.create` with one call to `init_observability()`.
- **Rich span payloads** – request/response payloads, finish reasons, structured token usage (real usage when available, otherwise `tiktoken`-powered estimates), error objects with stack traces, and trace correlations.
- **Custom tracing hooks** – decorators, context managers, and helpers (`observe`, `trace_context`, `add_span_attribute`, `add_span_event`) powered by OpenTelemetry for arbitrary Python functions.
- **Production-focused exporter** – lock-free queue, batching, retry/backoff, graceful shutdown, and queue backpressure so your app stays responsive.
- **OpenTelemetry bridge** – automatically starts `llm.request` spans if none exist, or enriches your active spans with token counts and finish reasons.

---

## Installation

```bash
# Poetry
poetry add asymetry

# Pip
pip install asymetry
```

Requires Python 3.13+ and an active Asymetry API key.

---

## Quick Start

1. **Export credentials**

```bash
export ASYMETRY_API_KEY="sk_test_..."   # required
export ASYMETRY_ENABLED=true              # optional (default true)
```

2. **Instrument your app**

```python
from asymetry import init_observability, observe
import openai, anthropic

init_observability()

openai_client = openai.OpenAI()
anthropic_client = anthropic.Anthropic()

# OpenAI – fully automatic tracking
response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize SOC2."}],
    temperature=0.2,
)

# Anthropic – same queue, no extra work
claude = anthropic_client.messages.create(
    model="claude-3-5-haiku-latest",
    max_tokens=200,
    messages=[{"role": "user", "content": "Summarize SOC2."}],
)

# Custom business logic
@observe(name="process_ticket", attributes={"tier": "gold"})
def process_ticket(ticket_id: str, body: str) -> str:
    ...
    return "ok"
```

3. **Optional explicit shutdown**

```python
from asymetry import shutdown_observability

shutdown_observability(timeout=10)  # flush remaining spans
```

---

## Configuration Surface

All knobs are environment variables (overrideable via `init_observability()`):

| Variable | Purpose | Default |
| --- | --- | --- |
| `ASYMETRY_API_KEY` | Auth token for ingest | **required** |
| `ASYMETRY_API_URL` | Ingest endpoint | `https://api.asymetry.co/v1/ingest` |
| `ASYMETRY_ENABLED` | Master switch | `true` |
| `ASYMETRY_BATCH_SIZE` | Export batch size | `100` |
| `ASYMETRY_FLUSH_INTERVAL` | Seconds between flushes | `5.0` |
| `ASYMETRY_QUEUE_MAX_SIZE` | In-memory queue bound | `10000` |
| `ASYMETRY_MAX_RETRIES` | Export retry attempts | `3` |
| `ASYMETRY_REQUEST_TIMEOUT` | HTTP timeout (s) | `10.0` |

Programmatic overrides:

```python
init_observability(
    api_key="sk_test...",
    enabled=True,
    log_level="DEBUG",
    enable_tracing=True,
)
```

---

## Telemetry Model

Every captured span is decomposed into three payload types and queued together:

- **LLM Requests** – provider, model, timestamp, latency, finish_reason, full prompt/messages, captured outputs, request params, OpenTelemetry identifiers.
- **Token Usage** – prompt/completion/total token counts; uses real values when provided by the SDK or falls back to `tiktoken` or character-based heuristics.
- **Errors** – Python exception type/code/message plus truncated stack traces.
- **Trace Spans** – when `observe`/`trace_context` are used, full OTel spans (attributes, events, IO metadata) are exported alongside LLM calls for end-to-end correlation.

---

## Architecture Snapshot

```
Your App (OpenAI / Anthropic / custom)
        ↓ monkey-patched clients & decorators
Asymetry Instrumentation (LLM + OTEL)
        ↓
Thread-safe queue (backpressure & drop-old)
        ↓
Background exporter thread
        ↓ async HTTP client (httpx)
Asymetry ingest API (batch + retry)
```

Key guarantees:
- Non-blocking: SDK calls enqueue telemetry without touching the network path.
- Resilient exporter: time or size-based flush, exponential backoff, graceful shutdown hook via `atexit`.
- Queue safety: bounded queue protects your app; oldest spans are dropped if pressure persists.

---

## Instrumentation Coverage

| Provider | API | Status | Notes |
| --- | --- | --- | --- |
| OpenAI | `chat.completions.create` (sync) | ✅ | Token usage extracted when OpenAI returns `usage`; otherwise estimated. |
| OpenAI | Streaming (`stream=True`) | ⚠️ Partial | See `examples/test_streaming.py` for current behavior; chunk-level telemetry still WIP. |
| Anthropic | `messages.create` (sync) | ✅ | Handles system prompt + tool use blocks; always captures usage. |
| Anthropic | Streaming | ⚠️ Partial | Events enumerated in tests; telemetry enrichment planned. |
| Custom code | `@observe`, `trace_context` | ✅ | OpenTelemetry spans exported via same queue. |

Roadmap items (tracked via GitHub issues): async client support, streaming deltas, additional providers, custom span ingestion, SDK-specific extras.

---

## Examples

The `examples/` directory doubles as runnable docs:

- `basic_usage.py` – OpenAI quick start and error capture.
- `advanced_usage.py` – custom configuration, workload simulation, batching demo.
- `multi-llm.py` – single init that tracks both OpenAI and Anthropic.
- `example_tracing.py` – custom tracing helpers (functions, contexts, events).
- `test_streaming.py` – current streaming behavior + gap analysis.

Run with Poetry:

```bash
poetry install
poetry run python examples/basic_usage.py
```

---

## Troubleshooting Cheatsheet

- **Missing API key** – set `ASYMETRY_API_KEY` or pass `api_key=` to `init_observability`.
- **OpenAI / Anthropic not found** – install the relevant SDK (`pip install openai anthropic`); instrumentation gracefully skips absent providers.
- **tiktoken warnings** – install `tiktoken` for precise usage numbers; otherwise we fallback to character estimates.
- **Need verbose logs** – pass `log_level="DEBUG"` to surface exporter + instrumentation details.
- **Lingering spans on exit** – call `shutdown_observability(timeout=10)` before terminating short-lived scripts/tests.

---

## Contributing & Local Dev

```bash
git clone https://github.com/asymetry-official/asymetry-sdk
cd asymetry
poetry install
poetry run pytest       # coming soon
poetry run python examples/basic_usage.py
```

We welcome PRs for new providers, better streaming support, and tracing improvements. Please discuss major changes via GitHub issues first.

---

## License & Links

- License: MIT (see `LICENSE`)
- Website: [asymetry.co](https://asymetry.co)
- Docs: [docs.asymetry.co](https://docs.asymetry.co)

Made with ❤️ by the Asymetry team.