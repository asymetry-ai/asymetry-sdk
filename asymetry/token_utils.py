"""Token estimation utilities using tiktoken."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Try to import tiktoken
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning(
        "tiktoken not available, falling back to character-based estimation. "
        "Install tiktoken for accurate token counts: pip install tiktoken"
    )


def estimate_tokens(text: str, model: str = "gpt-4") -> tuple[int, str]:
    """
    Estimate token count for text.

    Args:
        text: Input text to count tokens for
        model: Model name (for tiktoken encoding selection)

    Returns:
        Tuple of (token_count, estimation_method)
    """
    if TIKTOKEN_AVAILABLE:
        return _tiktoken_estimate(text, model)
    else:
        return _char_based_estimate(text)


def _tiktoken_estimate(text: str, model: str) -> tuple[int, str]:
    """Estimate tokens using tiktoken (accurate)."""
    try:
        # Get encoding for model
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        return len(tokens), "tiktoken"
    except KeyError:
        # Model not recognized, use cl100k_base (GPT-4 default)
        logger.debug(f"Model {model} not recognized, using cl100k_base encoding")
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return len(tokens), "tiktoken"
    except Exception as e:
        logger.warning(f"tiktoken estimation failed: {e}, falling back to char-based")
        return _char_based_estimate(text)


def _char_based_estimate(text: str) -> tuple[int, str]:
    """
    Fallback character-based estimation.
    Rule of thumb: ~4 characters per token for English text.
    """
    estimated_tokens = len(text) // 4
    return max(1, estimated_tokens), "char_based"


def estimate_messages_tokens(messages: list[dict[str, str]], model: str = "gpt-4") -> int:
    """
    Estimate tokens for a list of chat messages.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name

    Returns:
        Estimated total input tokens
    """
    total_tokens = 0

    # Add tokens for each message
    for message in messages:
        # Message overhead: role + formatting (~4 tokens per message)
        total_tokens += 4

        # Role tokens
        role = message.get("role", "")
        role_tokens, _ = estimate_tokens(role, model)
        total_tokens += role_tokens

        # Content tokens
        content = message.get("content", "")
        if isinstance(content, str):
            content_tokens, _ = estimate_tokens(content, model)
            total_tokens += content_tokens
        elif isinstance(content, list):
            # Handle multimodal content (text + images)
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_tokens, _ = estimate_tokens(item.get("text", ""), model)
                    total_tokens += text_tokens
                # Note: Image tokens would need special handling

    # Add overhead for message list formatting (~3 tokens)
    total_tokens += 3

    return total_tokens


def extract_token_usage(
    response: Any, messages: list[dict[str, str]], model: str
) -> dict[str, Any]:
    """
    Extract or estimate token usage from OpenAI response.

    Args:
        response: OpenAI API response object
        messages: Input messages (for estimation if needed)
        model: Model name

    Returns:
        Dict with token usage info
    """
    # Try to get actual usage from response
    if hasattr(response, "usage") and response.usage:
        return {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "estimated": False,
            "estimation_method": None,
        }

    # Fallback to estimation
    logger.debug("No usage data in response, estimating tokens")

    input_tokens = estimate_messages_tokens(messages, model)

    # Estimate output tokens from response
    output_tokens = 0
    if hasattr(response, "choices") and response.choices:
        for choice in response.choices:
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                content = choice.message.content or ""
                tokens, method = estimate_tokens(content, model)
                output_tokens += tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated": True,
        "estimation_method": "tiktoken" if TIKTOKEN_AVAILABLE else "char_based",
    }


def extract_token_usage_anthropic(
    response: Any, messages: list[dict[str, str]], model: str, system: str | None = None
) -> dict[str, Any]:
    """
    Extract token usage from Anthropic response.
    
    Anthropic always provides actual token usage in the response.usage object,
    so we don't need to estimate tokens like we do for OpenAI.

    Args:
        response: Anthropic API response object
        messages: Input messages (for fallback estimation if needed)
        model: Model name (Claude model)
        system: System prompt if provided (for fallback estimation)

    Returns:
        Dict with token usage info
    """
    # Anthropic responses always have usage data
    if hasattr(response, "usage") and response.usage:
        try:
            return {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                "estimated": False,
                "estimation_method": None,
            }
        except AttributeError as e:
            logger.warning(f"Error extracting Anthropic token usage: {e}")

    # Fallback to estimation (should rarely happen with Anthropic)
    logger.debug("No usage data in Anthropic response, estimating tokens")

    # Estimate input tokens from messages
    input_tokens = 0
    
    # Add system prompt tokens if present
    if system:
        tokens, _ = estimate_tokens(system, model)
        input_tokens += tokens
    
    # Add message tokens
    for message in messages:
        # Message overhead
        input_tokens += 4
        
        # Content tokens
        content = message.get("content", "")
        if isinstance(content, str):
            tokens, _ = estimate_tokens(content, model)
            input_tokens += tokens
        elif isinstance(content, list):
            # Handle multimodal content
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        tokens, _ = estimate_tokens(item.get("text", ""), model)
                        input_tokens += tokens
                    elif item.get("type") == "tool_use":
                        # Estimate tool use content
                        tool_input = str(item.get("input", ""))
                        tokens, _ = estimate_tokens(tool_input, model)
                        input_tokens += tokens

    # Estimate output tokens from response
    output_tokens = 0
    if hasattr(response, "content") and response.content:
        for block in response.content:
            if hasattr(block, "type"):
                if block.type == "text" and hasattr(block, "text"):
                    tokens, _ = estimate_tokens(block.text, model)
                    output_tokens += tokens
                elif block.type == "tool_use" and hasattr(block, "input"):
                    # Tool use block
                    tool_input = str(block.input)
                    tokens, _ = estimate_tokens(tool_input, model)
                    output_tokens += tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated": True,
        "estimation_method": "char_based",  # Claude uses different tokenizer than tiktoken
    }

