"""Configuration module for Asymetry SDK."""

import os
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


class Config:
    """Asymetry SDK configuration loaded from environment variables."""

    def __init__(self):
        # Core settings
        self.api_key: str | None = os.getenv("ASYMETRY_API_KEY")
        self.api_url: str = os.getenv("ASYMETRY_API_URL", "https://api.asymetry.co/v1/ingest")
        self.enabled: bool = os.getenv("ASYMETRY_ENABLED", "true").lower() == "true"

        # Batch settings
        self.batch_size: int = int(os.getenv("ASYMETRY_BATCH_SIZE", "100"))
        self.flush_interval: float = float(os.getenv("ASYMETRY_FLUSH_INTERVAL", "5.0"))

        # Queue settings
        self.queue_max_size: int = int(os.getenv("ASYMETRY_QUEUE_MAX_SIZE", "10000"))
        self.max_retries: int = int(os.getenv("ASYMETRY_MAX_RETRIES", "3"))

        # Request timeout
        self.request_timeout: float = float(os.getenv("ASYMETRY_REQUEST_TIMEOUT", "10.0"))

    def validate(self) -> None:
        """Validate required configuration."""
        if self.enabled and not self.api_key:
            raise ValueError(
                "ASYMETRY_API_KEY is required when ASYMETRY_ENABLED=true. "
                "Get your API key from https://asymetry.co/dashboard"
            )

        if self.batch_size < 1:
            raise ValueError("ASYMETRY_BATCH_SIZE must be at least 1")

        if self.flush_interval < 0.1:
            raise ValueError("ASYMETRY_FLUSH_INTERVAL must be at least 0.1 seconds")

        if self.queue_max_size < 100:
            raise ValueError("ASYMETRY_QUEUE_MAX_SIZE must be at least 100")

    def __repr__(self) -> str:
        """String representation (hides API key)."""
        masked_key = f"{self.api_key[:8]}..." if self.api_key else "None"
        return (
            f"Config(enabled={self.enabled}, api_key={masked_key}, "
            f"api_url={self.api_url}, batch_size={self.batch_size})"
        )


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """Reset config (useful for testing)."""
    global _config
    _config = None

