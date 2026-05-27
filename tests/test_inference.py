"""Tests for the inference module."""

from unittest.mock import AsyncMock, patch

import pytest

from src.config import AppConfig, InferenceConfig, ModelConfig, set_config
from src.health import reset_health_tracker
from src.inference import call_model


@pytest.fixture(autouse=True)
def setup():
    set_config(AppConfig(inference=InferenceConfig(mock=True)))
    reset_health_tracker()
    yield


@pytest.mark.asyncio
async def test_mock_inference_returns_result():
    model = ModelConfig(name="test", provider="mock", model_id="mock/test", timeout_ms=10000)
    result = await call_model(model, "What is AI?")
    assert "result" in result
    assert "Mock response" in result["result"]
    assert result["input_tokens"] > 0
    assert result["output_tokens"] > 0
    assert result["latency_ms"] > 0


@pytest.mark.asyncio
async def test_mock_inference_respects_timeout():
    model = ModelConfig(name="test", provider="mock", model_id="mock/test", timeout_ms=50)
    # With very short timeout, mock may timeout (random 100ms-2s)
    # Run multiple times to catch at least one timeout
    timeouts = 0
    for _ in range(20):
        try:
            await call_model(model, "Hello")
        except TimeoutError:
            timeouts += 1
    # At least some should timeout with 50ms limit
    assert timeouts > 0


@pytest.mark.asyncio
async def test_timeout_override():
    model = ModelConfig(name="test", provider="mock", model_id="mock/test", timeout_ms=10000)
    # Override with very short timeout
    timeouts = 0
    for _ in range(20):
        try:
            await call_model(model, "Hello", timeout_ms=50)
        except TimeoutError:
            timeouts += 1
    assert timeouts > 0


@pytest.mark.asyncio
async def test_real_inference_called_when_not_mock():
    set_config(AppConfig(inference=InferenceConfig(mock=False)))
    model = ModelConfig(name="real", provider="gemini", model_id="gemini/gemini-2.5-flash", timeout_ms=5000)

    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = "Real response"
    mock_response.usage = AsyncMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20

    with patch("src.inference.litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        result = await call_model(model, "Hello")
    assert result["result"] == "Real response"
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 20


@pytest.mark.asyncio
async def test_model_used_field():
    model = ModelConfig(name="my-model", provider="mock", model_id="mock/x", timeout_ms=10000)
    result = await call_model(model, "Hello")
    assert result["model_used"] == "my-model"
