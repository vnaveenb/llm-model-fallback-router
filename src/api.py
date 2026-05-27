"""FastAPI application for Model Fallback Router."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import get_config, reload_config, set_config
from src.health import get_health_tracker, reset_health_tracker
from src.models import (
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
    HealthResponse,
    ModelStatus,
    RouterStats,
)
from src.router import RouterError, get_router, reset_router
from src.tracker import close_tracker, log_request, setup_tracker

_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _start_time
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = get_config()
    setup_tracker()

    # Register all configured models with health tracker
    tracker = get_health_tracker()
    for model in cfg.models:
        tracker.register_model(model.name)

    _start_time = time.time()
    logging.getLogger(__name__).info(
        "Model Fallback Router started — %d models, strategy=%s",
        len(cfg.models), cfg.routing.strategy,
    )
    yield
    close_tracker()


app = FastAPI(
    title="Model Fallback Router",
    description="Routes LLM requests across providers with automatic failover, SLA enforcement, and health tracking.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="src/static"), name="static")


# ── Inference Endpoint ──────────────────────────────────────────


@app.post("/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """Route a completion request through the model fallback chain."""
    cfg = get_config()
    router = get_router()

    try:
        response = await router.route_request(request)

        # Log to tracker
        model_cfg = next((m for m in cfg.models if m.name == response.model_used), None)
        log_request(
            model=response.model_used,
            provider=model_cfg.provider if model_cfg else "unknown",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            success=True,
            fallback_triggered=response.fallbacks_triggered > 0,
            sla_met=response.sla_met,
        )
        return response

    except RouterError as e:
        log_request(
            model="none",
            provider="none",
            success=False,
            error=str(e),
        )
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error="All models failed",
                detail=str(e),
                models_attempted=e.models_attempted,
            ).model_dump(),
        )


# ── Model Management ────────────────────────────────────────────


@app.get("/models", response_model=list[ModelStatus])
async def list_models():
    """List all configured models with health status."""
    cfg = get_config()
    tracker = get_health_tracker()
    result = []
    for m in cfg.models:
        result.append(ModelStatus(
            name=m.name,
            provider=m.provider,
            model_id=m.model_id,
            state=tracker.get_state(m.name),
            priority=m.priority,
            weight=m.weight,
            stats=tracker.get_stats(m.name),
        ))
    return result


@app.get("/models/{name}/stats")
async def model_stats(name: str):
    """Get detailed metrics for a specific model."""
    cfg = get_config()
    tracker = get_health_tracker()

    model = next((m for m in cfg.models if m.name == name), None)
    if not model:
        return JSONResponse(status_code=404, content={"error": f"Model '{name}' not found"})

    return ModelStatus(
        name=model.name,
        provider=model.provider,
        model_id=model.model_id,
        state=tracker.get_state(name),
        priority=model.priority,
        weight=model.weight,
        stats=tracker.get_stats(name),
    )


@app.post("/models/{name}/disable")
async def disable_model(name: str):
    """Manually disable a model from routing."""
    tracker = get_health_tracker()
    if tracker.disable_model(name):
        return {"status": "disabled", "model": name}
    return JSONResponse(status_code=404, content={"error": f"Model '{name}' not found"})


@app.post("/models/{name}/enable")
async def enable_model(name: str):
    """Re-enable a disabled model."""
    tracker = get_health_tracker()
    if tracker.enable_model(name):
        return {"status": "enabled", "model": name}
    return JSONResponse(status_code=404, content={"error": f"Model '{name}' not found"})


# ── Stats & Health ──────────────────────────────────────────────


@app.get("/stats", response_model=RouterStats)
async def router_stats():
    """Aggregate router statistics."""
    cfg = get_config()
    tracker = get_health_tracker()
    router = get_router()

    stats = router.stats
    healthy = sum(1 for m in cfg.models if tracker.is_available(m.name))
    unhealthy = len(cfg.models) - healthy

    return RouterStats(
        total_requests=stats["total_requests"],
        successful=stats["successful"],
        failed=stats["failed"],
        fallback_count=stats["fallback_count"],
        fallback_rate=stats["fallback_rate"],
        sla_violations=stats["sla_violations"],
        sla_violation_rate=stats["sla_violation_rate"],
        avg_latency_ms=stats["avg_latency_ms"],
        active_models=healthy,
        unhealthy_models=unhealthy,
        strategy=cfg.routing.strategy,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health check."""
    cfg = get_config()
    tracker = get_health_tracker()

    from src.models import ModelState

    healthy = sum(1 for m in cfg.models if tracker.get_state(m.name) == ModelState.HEALTHY)
    unhealthy = sum(1 for m in cfg.models if tracker.get_state(m.name) == ModelState.UNHEALTHY)
    disabled = sum(1 for m in cfg.models if tracker.get_state(m.name) == ModelState.DISABLED)

    return HealthResponse(
        status="ok" if healthy > 0 else "degraded",
        models_healthy=healthy,
        models_unhealthy=unhealthy,
        models_disabled=disabled,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


# ── Admin ───────────────────────────────────────────────────────


@app.post("/reload-config")
async def reload():
    """Hot-reload configuration from config.yaml."""
    cfg = reload_config()
    reset_health_tracker()
    reset_router()

    tracker = get_health_tracker()
    for model in cfg.models:
        tracker.register_model(model.name)

    return {
        "status": "reloaded",
        "models": len(cfg.models),
        "strategy": cfg.routing.strategy,
    }


# ── Dashboard ───────────────────────────────────────────────────


@app.get("/")
async def dashboard():
    """Serve the monitoring dashboard."""
    return FileResponse("src/static/index.html")
