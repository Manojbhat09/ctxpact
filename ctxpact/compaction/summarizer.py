"""Stage 2: LLM-based summarization — calls the model to compress context."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from ctxpact.compaction.prompts import (
    build_compaction_prompt,
    format_messages_for_summary,
)
from ctxpact.compaction.tokens import count_tokens

logger = logging.getLogger(__name__)


class Summarizer:
    """Sends compactible messages to an LLM for summarization."""

    def __init__(
        self,
        max_summary_tokens: int = 2000,
        code_line_limit: int = 50,
    ) -> None:
        self.max_summary_tokens = max_summary_tokens
        self.code_line_limit = code_line_limit

    async def summarize(
        self,
        messages: list[dict[str, Any]],
        backend_url: str,
        model: str,
        api_key: str = "dummy",
    ) -> dict[str, Any]:
        """Send messages to the LLM for summarization.

        Returns an OpenAI-format message dict with role=user and the summary as content.
        """
        conversation_text = format_messages_for_summary(messages)
        summary_messages = build_compaction_prompt(
            conversation_text=conversation_text,
            code_line_limit=self.code_line_limit,
        )

        # Call the LLM
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{backend_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": summary_messages,
                    "max_tokens": self.max_summary_tokens,
                    "temperature": 0.1,  # Low temp for factual summaries
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()

        # Extract the summary text
        summary_text = data["choices"][0]["message"]["content"]

        # Validate: ensure summary isn't too long
        summary_tokens = count_tokens(summary_text)
        if summary_tokens > self.max_summary_tokens * 1.5:
            logger.warning(
                f"Summary is {summary_tokens} tokens (limit {self.max_summary_tokens}). "
                "Consider increasing max_summary_tokens or adjusting the prompt."
            )

        logger.info(
            f"Summarization complete: {len(messages)} messages → "
            f"{summary_tokens} tokens summary"
        )

        # Return as a user message (same pattern as Anthropic SDK)
        return {
            "role": "user",
            "content": summary_text,
        }
