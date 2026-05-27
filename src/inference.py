"""LLM inference wrapper with mock/real modes and timeout enforcement."""

import asyncio
import random
import time

import litellm

from src.config import ModelConfig, get_config


async def call_model(
    model_cfg: ModelConfig,
    prompt: str,
    timeout_ms: int | None = None,
) -> dict:
    """Call a model with timeout enforcement. Returns result dict or raises on failure."""
    cfg = get_config()
    timeout = (timeout_ms or model_cfg.timeout_ms) / 1000.0

    if cfg.inference.mock or model_cfg.provider == "mock":
        return await _mock_inference(model_cfg, prompt, timeout)

    return await _real_inference(model_cfg, prompt, timeout, cfg.inference.temperature, cfg.inference.max_tokens)


async def _mock_inference(model_cfg: ModelConfig, prompt: str, timeout: float) -> dict:
    """Simulated inference with random latency."""
    # Simulate 100ms-2s latency
    latency = random.uniform(0.1, 2.0)
    if latency > timeout:
        await asyncio.sleep(timeout)
        raise TimeoutError(f"Model {model_cfg.name} timed out after {timeout}s")

    await asyncio.sleep(latency)

    input_tokens = len(prompt.split()) * 2
    output_tokens = random.randint(20, 100)

    return {
        "result": f"[Mock response from {model_cfg.name}] This is a simulated response to: {prompt[:50]}...",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency * 1000,
        "model_used": model_cfg.name,
    }


async def _real_inference(
    model_cfg: ModelConfig,
    prompt: str,
    timeout: float,
    temperature: float,
    max_tokens: int,
) -> dict:
    """Real LLM inference via LiteLLM with timeout."""
    start = time.perf_counter()

    try:
        response = await asyncio.wait_for(
            litellm.acompletion(
                model=model_cfg.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                metadata={
                    "project": "model-fallback-router",
                    "model_name": model_cfg.name,
                    "provider": model_cfg.provider,
                },
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise TimeoutError(f"Model {model_cfg.name} timed out after {timeout}s")

    elapsed_ms = (time.perf_counter() - start) * 1000
    usage = response.usage

    return {
        "result": response.choices[0].message.content,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "latency_ms": elapsed_ms,
        "model_used": model_cfg.name,
    }
