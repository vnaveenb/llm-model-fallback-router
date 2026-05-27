"""Tests for the API endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.api import app
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
from src.router import reset_router


@pytest.fixture(autouse=True)
def setup():
    set_config(AppConfig(
        models=[
            ModelConfig(name="primary", provider="mock", model_id="mock/a", priority=1, timeout_ms=5000),
            ModelConfig(name="secondary", provider="mock", model_id="mock/b", priority=2, timeout_ms=5000),
        ],
        routing=RoutingConfig(strategy="priority", sla_latency_ms=5000, max_fallback_attempts=2),
        health=HealthConfig(window_size=10),
        circuit_breaker=CircuitBreakerConfig(error_threshold=3, recovery_window_s=60),
        inference=InferenceConfig(mock=True),
        tracker=AppConfig().tracker.model_copy(update={"enabled": False}),
    ))
    reset_health_tracker()
    reset_router()
    tracker = get_health_tracker()
    tracker.register_model("primary")
    tracker.register_model("secondary")
    yield


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "models_healthy" in data


@pytest.mark.asyncio
async def test_list_models(client):
    resp = await client.get("/models")
    assert resp.status_code == 200
    models = resp.json()
    assert len(models) == 2
    assert models[0]["name"] == "primary"
    assert models[0]["state"] == "healthy"


@pytest.mark.asyncio
async def test_model_stats(client):
    resp = await client.get("/models/primary/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "primary"
    assert "stats" in data


@pytest.mark.asyncio
async def test_model_stats_not_found(client):
    resp = await client.get("/models/nonexistent/stats")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_disable_model(client):
    resp = await client.post("/models/primary/disable")
    assert resp.status_code == 200
    assert resp.json()["status"] == "disabled"

    # Verify it shows as disabled
    resp = await client.get("/models")
    models = resp.json()
    primary = next(m for m in models if m["name"] == "primary")
    assert primary["state"] == "disabled"


@pytest.mark.asyncio
async def test_enable_model(client):
    # Disable then re-enable
    await client.post("/models/primary/disable")
    resp = await client.post("/models/primary/enable")
    assert resp.status_code == 200
    assert resp.json()["status"] == "enabled"


@pytest.mark.asyncio
async def test_router_stats(client):
    resp = await client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_requests" in data
    assert "strategy" in data
    assert data["strategy"] == "priority"


@pytest.mark.asyncio
async def test_completions_endpoint(client):
    resp = await client.post("/completions", json={"prompt": "What is 2+2?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data
    assert "model_used" in data
    assert "latency_ms" in data


@pytest.mark.asyncio
async def test_completions_all_fail(client):
    # Disable all models
    await client.post("/models/primary/disable")
    await client.post("/models/secondary/disable")
    resp = await client.post("/completions", json={"prompt": "Hello"})
    assert resp.status_code == 503
    data = resp.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_disable_nonexistent_model(client):
    resp = await client.post("/models/ghost/disable")
    assert resp.status_code == 404
