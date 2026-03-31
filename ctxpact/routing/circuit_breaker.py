"""Circuit breaker pattern for provider failover.

States:
  CLOSED   → Normal operation, requests pass through
  OPEN     → Backend is down, requests fail-fast to fallback
  HALF_OPEN → Testing if backend recovered (allow limited requests)

Transitions:
  CLOSED → OPEN:      After `failure_threshold` consecutive failures
  OPEN → HALF_OPEN:   After `recovery_timeout_seconds`
  HALF_OPEN → CLOSED: On success
  HALF_OPEN → OPEN:   On failure
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Per-provider circuit breaker."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout_seconds: int = 30,
        half_open_max_calls: int = 1,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current state, with automatic OPEN → HALF_OPEN transition."""
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                return CircuitState.HALF_OPEN
        return self._state

    @property
    def is_available(self) -> bool:
        """Whether requests should be attempted through this breaker."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.half_open_max_calls
        return False  # OPEN

    async def record_success(self) -> None:
        """Record a successful request — may close the circuit."""
        async with self._lock:
            if self._state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
                logger.info(f"[{self.name}] Circuit CLOSED — backend recovered")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0

    async def record_failure(self, error: str = "") -> None:
        """Record a failed request — may open the circuit."""
        async with self._lock:
            error_msg = error.strip() or "(no error details)"
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._half_open_calls = 0
                logger.warning(
                    f"[{self.name}] Circuit OPEN (half-open test failed): {error_msg}"
                )
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"[{self.name}] Circuit OPEN after {self._failure_count} failures: {error_msg}"
                )

    async def attempt_half_open(self) -> bool:
        """Register a half-open test attempt. Returns False if limit reached."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    self._state = CircuitState.HALF_OPEN  # Explicitly set
                    return True
                return False
            return self._state == CircuitState.CLOSED

    def status(self) -> dict:
        """Return human-readable status dict."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "last_failure": self._last_failure_time,
            "is_available": self.is_available,
        }
