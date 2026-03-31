"""Provider router — selects backend with circuit breaker failover.

Priority-based routing:
  1. Try the highest-priority (lowest number) provider whose circuit is CLOSED
  2. If primary is OPEN, try the next available provider
  3. If all circuits are OPEN, try the primary anyway (last resort)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ctxpact.config import CircuitBreakerConfig, ProviderConfig
from ctxpact.routing.circuit_breaker import CircuitBreaker, CircuitState
from ctxpact.routing.client import BACKEND_DOWN_ERRORS, BackendError, LLMClient

logger = logging.getLogger(__name__)


class ProviderRouter:
    """Routes requests to available providers with failover."""

    def __init__(
        self,
        providers: list[ProviderConfig],
        circuit_config: CircuitBreakerConfig,
    ) -> None:
        # Sort by priority (lowest number = highest priority)
        self._providers = sorted(providers, key=lambda p: p.priority)
        self._breakers: dict[str, CircuitBreaker] = {}
        self._clients: dict[str, LLMClient] = {}

        for p in self._providers:
            self._clients[p.name] = LLMClient(
                timeout=p.timeout_seconds,
                connect_timeout=p.connect_timeout_seconds,
                stream_timeout=p.stream_timeout_seconds,
            )

            self._breakers[p.name] = CircuitBreaker(
                name=p.name,
                failure_threshold=circuit_config.failure_threshold,
                recovery_timeout_seconds=circuit_config.recovery_timeout_seconds,
                half_open_max_calls=circuit_config.half_open_max_calls,
            )

    @property
    def breakers(self) -> dict[str, CircuitBreaker]:
        return self._breakers

    def get_active_provider(self) -> ProviderConfig:
        """Get the highest-priority available provider."""
        for provider in self._providers:
            breaker = self._breakers[provider.name]
            if breaker.is_available:
                return provider

        # All circuits open — fall back to primary (best effort)
        logger.warning("All circuits open — attempting primary as last resort")
        return self._providers[0]

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], ProviderConfig]:
        """Route a non-streaming chat completion with failover.

        Returns (response_data, provider_used).
        """
        attempted: list[str] = []

        for provider in self._providers:
            breaker = self._breakers[provider.name]

            if not breaker.is_available and len(attempted) < len(self._providers) - 1:
                logger.debug(f"Skipping {provider.name} (circuit {breaker.state.value})")
                continue

            attempted.append(provider.name)

            try:
                if breaker.state == CircuitState.HALF_OPEN:
                    can_try = await breaker.attempt_half_open()
                    if not can_try:
                        continue

                response = await self._clients[provider.name].chat_completion(
                    url=provider.url,
                    model=provider.model,
                    messages=messages,
                    api_key=provider.api_key,
                    **kwargs,
                )
                await breaker.record_success()
                return response, provider

            except BACKEND_DOWN_ERRORS as e:
                error_msg = f"{type(e).__name__}: {e}"
                logger.warning(f"[{provider.name}] Backend error: {error_msg}")
                await breaker.record_failure(error_msg)
                continue

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                logger.error(f"[{provider.name}] Unexpected error: {error_msg}")
                await breaker.record_failure(error_msg)
                continue

        raise RuntimeError(
            f"All providers failed. Attempted: {attempted}. "
            "Check backend health and circuit breaker status."
        )

    async def chat_completion_stream(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[AsyncIterator[str], ProviderConfig]:
        """Route a streaming chat completion with failover.

        Returns (sse_line_iterator, provider_used).
        Connection errors are caught eagerly (before iteration starts).
        """
        attempted: list[str] = []

        for provider in self._providers:
            breaker = self._breakers[provider.name]

            if not breaker.is_available and len(attempted) < len(self._providers) - 1:
                continue

            attempted.append(provider.name)

            try:
                if breaker.state == CircuitState.HALF_OPEN:
                    can_try = await breaker.attempt_half_open()
                    if not can_try:
                        continue

                # Eagerly open connection — errors are caught here
                http_client, response = await self._clients[provider.name].chat_completion_stream(
                    url=provider.url,
                    model=provider.model,
                    messages=messages,
                    api_key=provider.api_key,
                    **kwargs,
                )
                # DO NOT record_success() here. The connection opened, but on a
                # 32k model the backend can OOM mid-stream. We must wait until
                # the stream completes successfully before telling the circuit
                # breaker the backend is healthy.

                # Wrap into an async iterator that tracks success/failure
                breaker_ref = breaker  # Capture for the closure
                provider_name = provider.name

                async def _stream_lines(
                    client: Any, resp: Any, cb: CircuitBreaker, pname: str
                ) -> AsyncIterator[str]:
                    stream_ok = False
                    saw_data = False
                    try:
                        async for line in resp.aiter_lines():
                            if line.startswith("data: "):
                                saw_data = True
                                yield line
                            elif line.strip() == "data: [DONE]":
                                saw_data = True
                                yield "data: [DONE]"
                                stream_ok = True
                                break
                    except asyncio.CancelledError:
                        # Client disconnected mid-stream; don't mark backend failure.
                        logger.info(f"[{pname}] Stream cancelled by client")
                        return
                    except httpx.ReadTimeout as e:
                        if saw_data:
                            # Backend is slow after producing data; don't mark as failure.
                            logger.warning(f"[{pname}] Stream timed out after data: {e}")
                            return
                        error_msg = f"Stream interrupted: {type(e).__name__}: {e}"
                        logger.warning(f"[{pname}] {error_msg}")
                        await cb.record_failure(error_msg)
                        raise
                    except Exception as e:
                        # Mid-stream failure (e.g. OOM during generation on 32k)
                        error_msg = f"Stream interrupted: {type(e).__name__}: {e}"
                        logger.warning(f"[{pname}] {error_msg}")
                        await cb.record_failure(error_msg)
                        raise
                    finally:
                        await resp.aclose()
                        await client.aclose()
                        if stream_ok:
                            # Stream completed with [DONE] — backend is truly healthy
                            await cb.record_success()

                return _stream_lines(http_client, response, breaker_ref, provider_name), provider

            except BACKEND_DOWN_ERRORS as e:
                error_msg = f"{type(e).__name__}: {e}"
                logger.warning(f"[{provider.name}] Stream error: {error_msg}")
                await breaker.record_failure(error_msg)
                continue

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                logger.error(f"[{provider.name}] Unexpected stream error: {error_msg}")
                await breaker.record_failure(error_msg)
                continue

        raise RuntimeError(
            f"All providers failed for streaming. Attempted: {attempted}."
        )

    def status(self) -> list[dict]:
        """Return status of all providers and their circuit breakers."""
        result = []
        for provider in self._providers:
            breaker = self._breakers[provider.name]
            result.append({
                "provider": provider.name,
                "url": provider.url,
                "model": provider.model,
                "priority": provider.priority,
                "max_context": provider.max_context,
                **breaker.status(),
            })
        return result
