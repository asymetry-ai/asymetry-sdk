"""Async exporter with queue, batching, and background worker."""

import asyncio
import logging
import queue
import threading
from typing import Any, Union

from .api.client import AsymetryAPIClient
from .config import get_config
from .spans import SpanContext, TraceSpan

logger = logging.getLogger(__name__)


class SpanExporter:
    """
    Async span exporter with batching and background worker.

    Features:
    - Non-blocking queue for span collection
    - Automatic batching with size/time triggers
    - Backpressure handling (drops old spans when queue full)
    - Background thread for async export
    - Graceful shutdown with flush
    - Supports both LLM spans and trace spans
    """

    def __init__(self):
        self.config = get_config()
        self.api_client = AsymetryAPIClient()

        # Synchronous queue (thread-safe, works with monkey-patching)
        # Can hold both SpanContext (LLM) and TraceSpan (custom functions)
        self._queue: queue.Queue[Union[SpanContext, TraceSpan]] = queue.Queue(
            maxsize=self.config.queue_max_size
        )

        # Background worker state
        self._worker_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

        # Batch accumulator for LLM data
        self._batch_requests: list[dict[str, Any]] = []
        self._batch_tokens: list[dict[str, Any]] = []
        self._batch_errors: list[dict[str, Any]] = []

        # Batch accumulator for trace data
        self._batch_traces: list[dict[str, Any]] = []

        self._last_flush_time = 0.0

        logger.info("Span exporter initialized")

    def start(self) -> None:
        """Start the background worker thread."""
        if self._worker_thread and self._worker_thread.is_alive():
            logger.warning("Exporter already running")
            return

        self._shutdown_event.clear()
        self._worker_thread = threading.Thread(
            target=self._run_worker,
            name="Asymetry-Exporter",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("✓ Background exporter started")

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the background worker and flush remaining spans.

        Args:
            timeout: Maximum time to wait for shutdown (seconds)
        """
        if not self._worker_thread or not self._worker_thread.is_alive():
            logger.info("Exporter not running")
            return

        logger.info("Stopping exporter, flushing remaining spans...")
        self._shutdown_event.set()

        # Wait for worker to finish
        self._worker_thread.join(timeout=timeout)

        if self._worker_thread.is_alive():
            logger.warning(f"Exporter did not stop within {timeout}s")
        else:
            logger.info("✓ Exporter stopped gracefully")

        # Close API client
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.api_client.close(), self._loop)

    def get_queue(self) -> queue.Queue[Union[SpanContext, TraceSpan]]:
        """Get the queue for enqueueing spans."""
        return self._queue

    def _run_worker(self) -> None:
        """Background worker that runs in a separate thread."""
        # Create new event loop for this thread
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._worker_loop())
        except Exception as e:
            logger.error(f"Worker crashed: {e}", exc_info=True)
        finally:
            self._loop.close()
            self._loop = None

    async def _worker_loop(self) -> None:
        """Main worker loop that processes spans."""
        import time

        self._last_flush_time = time.time()

        while not self._shutdown_event.is_set():
            try:
                # Check if we should flush based on time
                current_time = time.time()
                time_since_flush = current_time - self._last_flush_time

                should_flush_time = time_since_flush >= self.config.flush_interval and (
                    len(self._batch_requests) > 0 or len(self._batch_traces) > 0
                )

                # Try to get span from queue (with timeout)
                try:
                    span = self._queue.get(timeout=0.1)
                    self._add_span_to_batch(span)

                    # Check if batch is full (combined size)
                    total_items = len(self._batch_requests) + len(self._batch_traces)
                    should_flush_size = total_items >= self.config.batch_size

                    if should_flush_size:
                        await self._flush_batch()
                        self._last_flush_time = time.time()

                except queue.Empty:
                    # No spans available, check time-based flush
                    if should_flush_time:
                        await self._flush_batch()
                        self._last_flush_time = time.time()
                    continue

            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)

        # Shutdown: flush remaining spans
        logger.info("Flushing remaining spans before shutdown...")
        await self._flush_remaining()

    def _add_span_to_batch(self, span: Union[SpanContext, TraceSpan]) -> None:
        """Add span data to current batch."""
        if isinstance(span, SpanContext):
            # LLM span - add to LLM batches
            self._batch_requests.append(span.request.to_dict())

            if span.tokens:
                self._batch_tokens.append(span.tokens.to_dict())

            if span.error:
                self._batch_errors.append(span.error.to_dict())

        elif isinstance(span, TraceSpan):
            # Trace span - add to trace batch
            self._batch_traces.append(span.to_dict())

        else:
            logger.warning(f"Unknown span type: {type(span)}")

    async def _flush_batch(self) -> None:
        """Flush current batch to API."""
        if not self._batch_requests and not self._batch_traces:
            return

        batch_info = []
        if self._batch_requests:
            batch_info.append(f"{len(self._batch_requests)} LLM requests")
        if self._batch_traces:
            batch_info.append(f"{len(self._batch_traces)} traces")

        logger.debug(f"Flushing batch: {', '.join(batch_info)}")

        try:
            success = await self.api_client.send_batch_with_retry(
                requests=self._batch_requests,
                tokens=self._batch_tokens,
                errors=self._batch_errors,
                traces=self._batch_traces,
            )

            if success:
                logger.debug(f"✓ Batch exported successfully ({', '.join(batch_info)})")
            else:
                logger.error(f"✗ Failed to export batch ({', '.join(batch_info)})")

        except Exception as e:
            logger.error(f"Error flushing batch: {e}", exc_info=True)

        finally:
            # Clear batch regardless of success/failure
            self._batch_requests.clear()
            self._batch_tokens.clear()
            self._batch_errors.clear()
            self._batch_traces.clear()

    async def _flush_remaining(self) -> None:
        """Flush any remaining spans in queue and batch."""
        # Drain queue
        spans_drained = 0
        while True:
            try:
                span = self._queue.get_nowait()
                self._add_span_to_batch(span)
                spans_drained += 1
            except queue.Empty:
                break

        if spans_drained > 0:
            logger.info(f"Drained {spans_drained} spans from queue")

        # Flush final batch
        if self._batch_requests or self._batch_traces:
            await self._flush_batch()


# Global exporter instance
_exporter: SpanExporter | None = None


def get_exporter() -> SpanExporter:
    """Get or create the global exporter instance."""
    global _exporter
    if _exporter is None:
        _exporter = SpanExporter()
    return _exporter


def start_exporter() -> None:
    """Start the background exporter."""
    exporter = get_exporter()
    exporter.start()


def stop_exporter(timeout: float = 5.0) -> None:
    """Stop the background exporter."""
    global _exporter
    if _exporter:
        _exporter.stop(timeout=timeout)
        _exporter = None
