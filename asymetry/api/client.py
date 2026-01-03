"""API client for sending telemetry data to Asymetry backend."""

import asyncio
import logging
import httpx

from ..config import get_config
from ..version import __version__

logger = logging.getLogger(__name__)


class AsymetryAPIClient:
    """Async HTTP client for Asymetry backend API."""

    def __init__(self):
        self.config = get_config()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.config.request_timeout,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                headers={
                    "x-api-key": self.config.api_key,
                    "Content-Type": "application/json",
                    "User-Agent": f"asymetry-ai-sdk/{__version__}",
                },
            )
        return self._client

    async def send_batch(
        self,
        requests: list[dict[str, any]],
        tokens: list[dict[str, any]],
        errors: list[dict[str, any]],
        traces: list[dict[str, any]] | None = None,
    ) -> bool:
        """
        Send a batch of telemetry data to Asymetry API.

        Args:
            requests: List of LLM request records
            tokens: List of token usage records
            errors: List of error records
            traces: List of trace span records (optional)

        Returns:
            True if successful, False otherwise
        """
        if not self.config.enabled:
            logger.debug("Asymetry is disabled, skipping batch send")
            return True

        payload = {
            "requests": requests,
            "tokens": tokens,
            "errors": errors,
        }

        # Add traces if provided
        if traces:
            payload["traces"] = traces

        try:
            client = await self._get_client()

            # Debug logging
            logger.debug(
                f"Sending!! batch: {len(requests)} requests, "
                f"{len(tokens)} tokens, {len(errors)} errors, "
                f"{len(traces) if traces else 0} traces"
            )

            logger.debug(payload)

            response = await client.post(
                self.config.api_url,
                json=payload,
            )

            if response.status_code == 200:
                info_parts = [f"{len(requests)} requests"]
                if tokens:
                    info_parts.append(f"{len(tokens)} token records")
                if errors:
                    info_parts.append(f"{len(errors)} errors")
                if traces:
                    info_parts.append(f"{len(traces)} traces")

                logger.info(f"Successfully sent batch: {', '.join(info_parts)}")
                return True
            else:
                logger.error(
                    f"API request failed with status {response.status_code}: " f"{response.text}"
                )
                return False

        except httpx.TimeoutException:
            logger.error(f"API request timed out after {self.config.request_timeout}s")
            return False
        except httpx.RequestError as e:
            logger.error(f"API request error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending batch: {e}")
            return False

    async def send_batch_with_retry(
        self,
        requests: list[dict[str, any]],
        tokens: list[dict[str, any]],
        errors: list[dict[str, any]],
        traces: list[dict[str, any]] | None = None,
    ) -> bool:
        """
        Send batch with exponential backoff retry.

        Returns:
            True if successful, False after all retries exhausted
        """
        max_retries = self.config.max_retries

        for attempt in range(max_retries):
            success = await self.send_batch(requests, tokens, errors, traces)

            if success:
                return True

            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s...
                delay = 2**attempt
                logger.warning(
                    f"Batch send failed (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay}s..."
                )
                await asyncio.sleep(delay)

        logger.error(f"Batch send failed after {max_retries} attempts, dropping batch")
        return False

    async def close(self) -> None:
        """Close the HTTP client and clean up resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.debug("Asymetry API client closed")

