"""Async HTTP client for forwarding requests to LLM backends.

Supports both streaming (SSE) and non-streaming responses.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class BackendError(Exception):
    """Raised when the LLM backend returns an error response (4xx/5xx).

    This is distinct from connection errors — the backend is reachable but
    returned an error. For OOM-aware failover, we treat 5xx (especially 503)
    the same as connection failures for circuit breaker purposes.
    """

    def __init__(self, status_code: int, detail: str = "") -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")

    @property
    def is_server_error(self) -> bool:
        """True for 5xx errors (OOM, crash, overload)."""
        return self.status_code >= 500


# Errors that indicate backend is down or unhealthy.
# Includes both connection-level failures AND HTTP 5xx errors.
# This is critical for the OOM-aware failover differentiator:
# vLLM-MLX returns 503 on OOM — this MUST be treated as "backend down".
BACKEND_DOWN_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    ConnectionRefusedError,
    ConnectionResetError,
    OSError,
    BackendError,  # HTTP 5xx from the backend (OOM, crash, overload)
)


class LLMClient:
    """Async client for OpenAI-compatible LLM backends."""

    def __init__(
        self,
        timeout: float = 120.0,
        connect_timeout: float = 5.0,
        stream_timeout: float | None = None,
    ) -> None:
        self._timeout = httpx.Timeout(
            timeout=timeout,
            connect=connect_timeout,
        )
        if stream_timeout is None:
            self._stream_timeout = httpx.Timeout(
                timeout=None,
                connect=connect_timeout,
                read=None,
                write=None,
                pool=None,
            )
        else:
            self._stream_timeout = httpx.Timeout(
                timeout=stream_timeout,
                connect=connect_timeout,
            )

    async def chat_completion(
        self,
        url: str,
        model: str,
        messages: list[dict[str, Any]],
        api_key: str = "dummy",
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Non-streaming chat completion request."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            **kwargs,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            # Raise BackendError for 5xx so the circuit breaker can classify
            # OOM (503) correctly as a backend-down event
            if response.status_code >= 500:
                detail = ""
                try:
                    detail = response.text[:200]
                except Exception:
                    pass
                raise BackendError(response.status_code, detail)
            response.raise_for_status()
            return response.json()

    async def chat_completion_stream(
        self,
        url: str,
        model: str,
        messages: list[dict[str, Any]],
        api_key: str = "dummy",
        **kwargs: Any,
    ) -> tuple[httpx.AsyncClient, httpx.Response]:
        """Open a streaming connection eagerly (so connection errors are catchable).

        Returns (client, response) — caller must iterate response.aiter_lines()
        and close the client when done.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }

        client = httpx.AsyncClient(timeout=self._stream_timeout)
        try:
            # Use send() to open the connection eagerly
            request = client.build_request(
                "POST",
                f"{url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response = await client.send(request, stream=True)
            # Raise BackendError for 5xx (OOM) so router can failover
            if response.status_code >= 500:
                await response.aclose()
                raise BackendError(response.status_code, "streaming connection rejected")
            return client, response
        except Exception:
            await client.aclose()
            raise

    async def check_health(self, url: str, api_key: str = "dummy") -> bool:
        """Quick health check — returns True if backend responds with 2xx."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(
                    f"{url.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                return 200 <= response.status_code < 300
        except Exception:
            return False
