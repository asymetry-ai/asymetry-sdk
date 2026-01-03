"""Main initialization and shutdown functions for Asymetry SDK."""

import logging
import atexit

from .config import get_config
from .instrumentation import (
    instrument_openai,
    uninstrument_openai,
    instrument_anthropic,
    uninstrument_anthropic,
    set_span_queue,
)
from .exporter import start_exporter, stop_exporter, get_exporter
from .tracing import init_tracing, shutdown_tracing, set_trace_queue
from .version import __version__

logger = logging.getLogger(__name__)

# Track initialization state
_initialized = False


def init_observability(
    api_key: str | None = None,
    enabled: bool | None = None,
    log_level: str = "INFO",
    service_name: str = "asymetry-ai-app",
    enable_tracing: bool = True,
) -> None:
    """
    Initialize Asymetry observability.

    This function:
    1. Loads configuration from environment variables
    2. Validates API key
    3. Starts the background exporter
    4. Instruments the OpenAI and Anthropic SDKs (if installed)
    5. Initializes OpenTelemetry tracing (optional)
    6. Registers shutdown handler

    Args:
        api_key: Optional API key (overrides ASYMETRY_API_KEY env var)
        enabled: Optional enable flag (overrides ASYMETRY_ENABLED env var)
        log_level: Logging level for Asymetry (DEBUG, INFO, WARNING, ERROR)
        service_name: Service name for tracing (default: "asymetry-ai-app")
        enable_tracing: Enable OpenTelemetry tracing for custom functions

    Example:
        ```python
        from asymetry import init_observability, trace
        import openai
        import anthropic

        # Initialize Asymetry with tracing - automatically instruments both providers!
        init_observability(service_name="my-app")

        # Use OpenAI - automatically tracked!
        openai_client = openai.OpenAI()
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}]
        )

        # Use Anthropic - also automatically tracked!
        anthropic_client = anthropic.Anthropic()
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}]
        )

        # Trace custom functions
        @observe(name="process_user_input")
        def process_input(user_id: str, text: str):
            # Your code here
            return result
        ```

    Environment Variables:
        ASYMETRY_API_KEY: Your Asymetry API key (required)
        ASYMETRY_API_URL: API endpoint (default: https://api.asymetry.co/v1/ingest)
        ASYMETRY_ENABLED: Enable/disable observability (default: true)
        ASYMETRY_BATCH_SIZE: Records per batch (default: 100)
        ASYMETRY_FLUSH_INTERVAL: Seconds between flushes (default: 5)
        ASYMETRY_QUEUE_MAX_SIZE: Max queue size (default: 10000)
        ASYMETRY_MAX_RETRIES: Retry attempts per batch (default: 3)

    Note:
        Both OpenAI and Anthropic SDKs are instrumented if installed.
        If either SDK is not installed, it will be skipped gracefully.
    """
    global _initialized

    if _initialized:
        logger.warning("Asymetry already initialized, skipping")
        return

    # Setup logging
    _setup_logging(log_level)

    logger.info(f"🔍 Initializing Asymetry SDK v{__version__}")

    # Load and validate config
    config = get_config()

    # Override with function parameters
    if api_key is not None:
        config.api_key = api_key
    if enabled is not None:
        config.enabled = enabled

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise

    if not config.enabled:
        logger.info("Asymetry is disabled (ASYMETRY_ENABLED=false)")
        return

    logger.info(f"Config: {config}")

    try:
        # 1. Start background exporter
        start_exporter()

        # 2. Connect instrumentation to exporter queue
        exporter = get_exporter()
        set_span_queue(exporter.get_queue())

        # 3. Instrument LLM SDKs (gracefully handle missing SDKs)
        instrumented_providers = []

        # Try to instrument OpenAI
        try:
            instrument_openai()
            instrumented_providers.append("OpenAI")
        except Exception as e:
            logger.debug(f"Could not instrument OpenAI: {e}")

        # Try to instrument Anthropic
        try:
            instrument_anthropic()
            instrumented_providers.append("Anthropic")
        except Exception as e:
            logger.debug(f"Could not instrument Anthropic: {e}")

        # Log instrumentation status
        if instrumented_providers:
            providers_str = " and ".join(instrumented_providers)
            logger.info(f"📊 {providers_str} SDK(s) instrumented successfully")
        else:
            logger.warning(
                "⚠️  No LLM SDKs found to instrument. "
                "Install openai and/or anthropic packages to enable tracking."
            )

        # 4. Initialize OpenTelemetry tracing (if enabled)
        if enable_tracing:
            init_tracing(service_name=service_name)
            set_trace_queue(exporter.get_queue())
            logger.info("📊 Custom function tracing enabled")

        # 5. Register shutdown handler
        atexit.register(_cleanup_on_exit)

        _initialized = True
        logger.info("✅ Asymetry initialized successfully!")
        if instrumented_providers:
            logger.info(f"📊 {providers_str} requests will now be automatically tracked")

    except Exception as e:
        logger.error(f"Failed to initialize Asymetry: {e}", exc_info=True)
        raise


def shutdown_observability(timeout: float = 5.0) -> None:
    """
    Gracefully shutdown Asymetry observability.

    This function:
    1. Uninstruments the OpenAI and Anthropic SDKs
    2. Shuts down tracing
    3. Flushes remaining spans
    4. Stops the background exporter

    Args:
        timeout: Maximum time to wait for shutdown (seconds)

    Note:
        This is automatically called on program exit via atexit.
        You only need to call this manually if you want to shutdown
        before program exit.
    """
    global _initialized

    if not _initialized:
        return

    logger.info("🛑 Shutting down Asymetry...")

    try:
        # 1. Remove instrumentation
        uninstrument_openai()
        uninstrument_anthropic()

        # 2. Shutdown tracing
        shutdown_tracing()

        # 3. Stop exporter (flushes remaining spans)
        stop_exporter(timeout=timeout)

        _initialized = False
        logger.info("✅ Asymetry shutdown complete")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


def _cleanup_on_exit() -> None:
    """Cleanup handler called on program exit."""
    if _initialized:
        shutdown_observability()


def _setup_logging(level: str) -> None:
    """Setup logging for Asymetry SDK."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure logger for asymetry module
    asymetry_logger = logging.getLogger("asymetry")
    asymetry_logger.setLevel(numeric_level)

    # Only add handler if none exists
    if not asymetry_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [Asymetry] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        asymetry_logger.addHandler(handler)

    # Prevent propagation to root logger
    asymetry_logger.propagate = False
