import pytest
from unittest.mock import MagicMock, patch
import sys


@pytest.fixture(autouse=True)
def mock_exporter():
    """
    Automatically mock the exporter module for all tests.
    This prevents any real telemetry from being sent.
    """
    with patch("asymetry.exporter.SpanExporter") as MockExporter:
        # Configure the mock to do nothing
        mock_instance = MockExporter.return_value
        mock_instance.start.return_value = None
        mock_instance.stop.return_value = None

        # Also mock the get_exporter function
        with patch("asymetry.exporter.get_exporter", return_value=mock_instance):
            yield mock_instance


@pytest.fixture(autouse=True)
def mock_network_calls():
    """
    Block all network calls in tests as a safety net.
    """
    with (
        patch("httpx.Client.post") as mock_post,
        patch("httpx.AsyncClient.post") as mock_async_post,
    ):
        mock_post.side_effect = Exception("Network calls are blocked in tests!")
        mock_async_post.side_effect = Exception("Network calls are blocked in tests!")
        yield (mock_post, mock_async_post)
