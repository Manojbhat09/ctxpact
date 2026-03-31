"""Token counting utilities — tiktoken with fallback to char estimation."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Try to load tiktoken; fall back to char-based estimation
_ENCODER = None
try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except (ImportError, Exception) as e:
    logger.warning(f"tiktoken unavailable, using char-based estimation: {e}")


def count_tokens(text: str) -> int:
    """Count tokens in a string."""
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    # Fallback: ~4 chars per token (conservative for English)
    return len(text) // 3


def count_message_tokens(message: dict[str, Any]) -> int:
    """Count tokens in a single OpenAI-format message.

    Accounts for role, content, tool calls, and message framing overhead.
    """
    tokens = 4  # Every message has framing overhead: <role> + content markers

    content = message.get("content", "")
    if isinstance(content, str):
        tokens += count_tokens(content)
    elif isinstance(content, list):
        # Multi-part content (text + images etc.)
        for part in content:
            if isinstance(part, dict):
                text = part.get("text", "")
                if text:
                    tokens += count_tokens(text)
                # Image tokens are harder to estimate — use a rough figure
                if part.get("type") == "image_url":
                    tokens += 85  # Low-detail image estimate
            elif isinstance(part, str):
                tokens += count_tokens(part)

    # Tool calls in assistant messages
    tool_calls = message.get("tool_calls", [])
    if tool_calls:
        for tc in tool_calls:
            tokens += count_tokens(tc.get("function", {}).get("name", ""))
            args = tc.get("function", {}).get("arguments", "")
            tokens += count_tokens(args if isinstance(args, str) else json.dumps(args))

    # Tool results
    if message.get("role") == "tool":
        tokens += 4  # tool_call_id framing

    return tokens


def count_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Count total tokens across all messages."""
    total = 3  # Conversation framing overhead
    for msg in messages:
        total += count_message_tokens(msg)
    return total
