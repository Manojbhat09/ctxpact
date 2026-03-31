"""Background health checker for LLM backends.

Periodically pings each provider's health endpoint to detect recovery
after OOM/crash events. Updates the circuit breaker state accordingly.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from ctxpact.config import ProviderConfig
    from ctxpact.routing.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class HealthChecker:
    """Background task that monitors provider health."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}  # type: ignore[type-arg]
        self._running = False

    async def start(
        self,
        providers: list[ProviderConfig],
        breakers: dict[str, CircuitBreaker],
    ) -> None:
        """Start health check loops for all providers."""
        self._running = True
        for provider in providers:
            breaker = breakers.get(provider.name)
            if breaker:
                task = asyncio.create_task(
                    self._check_loop(provider, breaker),
                    name=f"health-{provider.name}",
                )
                self._tasks[provider.name] = task
                logger.info(
                    f"Health checker started for {provider.name} "
                    f"(every {provider.health_check.interval_seconds}s)"
                )

    async def stop(self) -> None:
        """Stop all health check loops."""
        self._running = False
        for name, task in self._tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info(f"Health checker stopped for {name}")
        self._tasks.clear()

    async def _check_loop(
        self,
        provider: ProviderConfig,
        breaker: CircuitBreaker,
    ) -> None:
        """Periodically check a single provider's health."""
        hc = provider.health_check
        while self._running:
            try:
                await asyncio.sleep(hc.interval_seconds)
                await self._ping(provider, breaker)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Health check error for {provider.name}: {e}")

    async def _ping(
        self,
        provider: ProviderConfig,
        breaker: CircuitBreaker,
    ) -> None:
        """Send a single health check request."""
        hc = provider.health_check
        # Build health check URL — handle both absolute and relative endpoints
        base = provider.url.rstrip("/")
        endpoint = hc.endpoint
        if endpoint.startswith("/v1") and base.endswith("/v1"):
            # Avoid double /v1 — e.g. http://localhost:8080/v1 + /v1/models
            endpoint = endpoint[3:]  # Strip leading /v1
        url = f"{base}{endpoint}"

        try:
            async with httpx.AsyncClient(timeout=hc.timeout_seconds) as client:
                response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {provider.api_key}"},
                )
                if 200 <= response.status_code < 300:
                    # Truly healthy — backend is up and responding correctly.
                    # Record success to help the circuit breaker recover.
                    # We check for non-CLOSED state to avoid noisy logging,
                    # but always record the success so OPEN → HALF_OPEN → CLOSED
                    # transitions work correctly.
                    from ctxpact.routing.circuit_breaker import CircuitState
                    if breaker.state != CircuitState.CLOSED:
                        logger.info(
                            f"[{provider.name}] Health check passed (HTTP {response.status_code}) — "
                            "backend recovering"
                        )
                    await breaker.record_success()
                elif response.status_code >= 500:
                    # Server error (503 OOM, 500 crash) — backend is unhealthy
                    await breaker.record_failure(f"HTTP {response.status_code}")
                else:
                    # 3xx/4xx — backend is reachable but endpoint has issues
                    # (wrong API key, wrong path, etc). Don't touch the circuit
                    # — this isn't a health signal, it's a config problem.
                    logger.debug(
                        f"[{provider.name}] Health check got HTTP {response.status_code} "
                        f"(not a health signal, ignoring)"
                    )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            await breaker.record_failure(str(e))
        except Exception as e:
            await breaker.record_failure(f"Unexpected: {e}")
