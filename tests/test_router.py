"""Tests for the core router logic."""

from unittest.mock import AsyncMock, patch

import pytest

from src.config import (
    AppConfig,
    CircuitBreakerConfig,
    HealthConfig,
    InferenceConfig,
    ModelConfig,
    RoutingConfig,
    set_config,
)
from src.health import get_health_tracker, reset_health_tracker
from src.models import CompletionRequest
from src.router import Router, RouterError, reset_router


@pytest.fixture(autouse=True)
def setup():
    set_config(AppConfig(
        models=[
            ModelConfig(name="primary", provider="mock", model_id="mock/a", priority=1, timeout_ms=5000),
            ModelConfig(name="secondary", provider="mock", model_id="mock/b", priority=2, timeout_ms=5000),
            ModelConfig(name="tertiary", provider="mock", model_id="mock/c", priority=3, timeout_ms=5000),
        ],
        routing=RoutingConfig(strategy="priority", sla_latency_ms=3000, max_fallback_attempts=3),
        health=HealthConfig(window_size=10),
        circuit_breaker=CircuitBreakerConfig(error_threshold=3, recovery_window_s=60),
        inference=InferenceConfig(mock=True),
    ))
    reset_health_tracker()
    reset_router()
    tracker = get_health_tracker()
    tracker.register_model("primary")
    tracker.register_model("secondary")
    tracker.register_model("tertiary")
    yield


def _mock_result(model_name: str, latency: float = 100.0):
    return {
        "result": f"Response from {model_name}",
        "input_tokens": 10,
        "output_tokens": 20,
        "latency_ms": latency,
        "model_used": model_name,
    }


@pytest.mark.asyncio
async def test_primary_model_succeeds():
    router = Router()
    with patch("src.router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_result("primary")
        resp = await router.route_request(CompletionRequest(prompt="Hello"))
    assert resp.model_used == "primary"
    assert resp.fallbacks_triggered == 0
    assert resp.attempt_number == 1


@pytest.mark.asyncio
async def test_fallback_on_primary_failure():
    router = Router()
    call_count = 0

    async def side_effect(model_cfg, prompt, timeout):
        nonlocal call_count
        call_count += 1
        if model_cfg.name == "primary":
            raise TimeoutError("primary timed out")
        return _mock_result(model_cfg.name)

    with patch("src.router.call_model", side_effect=side_effect):
        resp = await router.route_request(CompletionRequest(prompt="Hello"))
    assert resp.model_used == "secondary"
    assert resp.fallbacks_triggered == 1
    assert resp.attempt_number == 2


@pytest.mark.asyncio
async def test_all_models_fail():
    router = Router()

    async def fail(*args, **kwargs):
        raise Exception("model down")

    with patch("src.router.call_model", side_effect=fail):
        with pytest.raises(RouterError) as exc_info:
            await router.route_request(CompletionRequest(prompt="Hello"))
    assert "primary" in exc_info.value.models_attempted
    assert "secondary" in exc_info.value.models_attempted
    assert "tertiary" in exc_info.value.models_attempted


@pytest.mark.asyncio
async def test_sla_violation_flagged():
    router = Router()
    # Return a slow response
    with patch("src.router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_result("primary", latency=4000.0)
        # Patch time.perf_counter to simulate slow elapsed
        with patch("src.router.time.perf_counter", side_effect=[0.0, 4.0, 4.0]):
            resp = await router.route_request(CompletionRequest(prompt="Hello"))
    assert resp.sla_met is False


@pytest.mark.asyncio
async def test_circuit_breaker_skips_unhealthy_model():
    router = Router()
    tracker = get_health_tracker()
    # Trip circuit breaker on primary
    for _ in range(3):
        tracker.record_outcome("primary", 100.0, success=False)

    with patch("src.router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_result("secondary")
        resp = await router.route_request(CompletionRequest(prompt="Hello"))
    # Should skip primary and go straight to secondary
    assert resp.model_used == "secondary"


@pytest.mark.asyncio
async def test_disabled_model_skipped():
    router = Router()
    tracker = get_health_tracker()
    tracker.disable_model("primary")

    with patch("src.router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_result("secondary")
        resp = await router.route_request(CompletionRequest(prompt="Hello"))
    assert resp.model_used == "secondary"


@pytest.mark.asyncio
async def test_model_preference_respected():
    router = Router()
    with patch("src.router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_result("tertiary")
        resp = await router.route_request(CompletionRequest(prompt="Hello", model_preference="tertiary"))
    assert resp.model_used == "tertiary"


@pytest.mark.asyncio
async def test_round_robin_strategy():
    set_config(AppConfig(
        models=[
            ModelConfig(name="a", provider="mock", model_id="mock/a", priority=1, timeout_ms=5000),
            ModelConfig(name="b", provider="mock", model_id="mock/b", priority=2, timeout_ms=5000),
        ],
        routing=RoutingConfig(strategy="round-robin", sla_latency_ms=5000, max_fallback_attempts=2),
        health=HealthConfig(window_size=10),
        circuit_breaker=CircuitBreakerConfig(error_threshold=3, recovery_window_s=60),
        inference=InferenceConfig(mock=True),
    ))
    reset_health_tracker()
    tracker = get_health_tracker()
    tracker.register_model("a")
    tracker.register_model("b")

    router = Router()
    models_used = []

    async def return_result(model_cfg, prompt, timeout):
        return _mock_result(model_cfg.name)

    with patch("src.router.call_model", side_effect=return_result):
        for _ in range(4):
            resp = await router.route_request(CompletionRequest(prompt="Hi"))
            models_used.append(resp.model_used)

    # Should alternate between models
    assert "a" in models_used
    assert "b" in models_used


@pytest.mark.asyncio
async def test_no_models_available():
    router = Router()
    tracker = get_health_tracker()
    tracker.disable_model("primary")
    tracker.disable_model("secondary")
    tracker.disable_model("tertiary")

    with pytest.raises(RouterError) as exc_info:
        await router.route_request(CompletionRequest(prompt="Hello"))
    assert "No models available" in str(exc_info.value)


@pytest.mark.asyncio
async def test_stats_tracking():
    router = Router()
    with patch("src.router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = _mock_result("primary")
        await router.route_request(CompletionRequest(prompt="Hello"))
        await router.route_request(CompletionRequest(prompt="World"))

    stats = router.stats
    assert stats["total_requests"] == 2
    assert stats["successful"] == 2
    assert stats["failed"] == 0


@pytest.mark.asyncio
async def test_timeout_override():
    router = Router()

    async def check_timeout(model_cfg, prompt, timeout):
        assert timeout == 1000  # Override value
        return _mock_result(model_cfg.name)

    with patch("src.router.call_model", side_effect=check_timeout):
        await router.route_request(CompletionRequest(prompt="Hello", timeout_override_ms=1000))
