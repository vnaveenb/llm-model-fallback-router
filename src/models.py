"""Request/response models for Model Fallback Router."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RoutingStrategy(str, Enum):
    PRIORITY = "priority"
    ROUND_ROBIN = "round-robin"
    LATENCY_WEIGHTED = "latency-weighted"


class ModelState(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DISABLED = "disabled"
    HALF_OPEN = "half-open"


# ── Request Models ──────────────────────────────────────────────


class CompletionRequest(BaseModel):
    prompt: str
    model_preference: str | None = None
    timeout_override_ms: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Response Models ─────────────────────────────────────────────


class CompletionResponse(BaseModel):
    result: str
    model_used: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    fallbacks_triggered: int = 0
    attempt_number: int = 1
    sla_met: bool = True


class ModelStatus(BaseModel):
    name: str
    provider: str
    model_id: str
    state: ModelState
    priority: int
    weight: int
    stats: "ModelStats | None" = None


class ModelStats(BaseModel):
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    error_rate: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    avg_latency_ms: float = 0.0


class RouterStats(BaseModel):
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    fallback_count: int = 0
    fallback_rate: float = 0.0
    sla_violations: int = 0
    sla_violation_rate: float = 0.0
    avg_latency_ms: float = 0.0
    active_models: int = 0
    unhealthy_models: int = 0
    strategy: str = "priority"


class HealthResponse(BaseModel):
    status: str
    models_healthy: int = 0
    models_unhealthy: int = 0
    models_disabled: int = 0
    uptime_seconds: float = 0.0


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    models_attempted: list[str] = Field(default_factory=list)
