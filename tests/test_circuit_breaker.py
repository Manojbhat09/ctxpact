"""Tests for circuit breaker state machine."""

import asyncio
import time

import pytest

from ctxpact.routing.circuit_breaker import CircuitBreaker, CircuitState


@pytest.fixture
def breaker():
    return CircuitBreaker(
        name="test",
        failure_threshold=3,
        recovery_timeout_seconds=1,
        half_open_max_calls=1,
    )


class TestCircuitBreakerStates:
    @pytest.mark.asyncio
    async def test_starts_closed(self, breaker):
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_available

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self, breaker):
        for _ in range(3):
            await breaker.record_failure("test error")
        assert breaker.state == CircuitState.OPEN
        assert not breaker.is_available

    @pytest.mark.asyncio
    async def test_stays_closed_below_threshold(self, breaker):
        await breaker.record_failure("error 1")
        await breaker.record_failure("error 2")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_available

    @pytest.mark.asyncio
    async def test_success_resets_count(self, breaker):
        await breaker.record_failure("error 1")
        await breaker.record_failure("error 2")
        await breaker.record_success()
        await breaker.record_failure("error 3")
        # Should still be closed — success reset the counter
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_transitions_to_half_open(self, breaker):
        for _ in range(3):
            await breaker.record_failure("test")
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.is_available

    @pytest.mark.asyncio
    async def test_half_open_success_closes(self, breaker):
        for _ in range(3):
            await breaker.record_failure("test")
        await asyncio.sleep(1.1)
        assert breaker.state == CircuitState.HALF_OPEN

        await breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self, breaker):
        for _ in range(3):
            await breaker.record_failure("test")
        await asyncio.sleep(1.1)

        await breaker.record_failure("still broken")
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_status_dict(self, breaker):
        status = breaker.status()
        assert status["name"] == "test"
        assert status["state"] == "closed"
        assert status["is_available"] is True
