"""Core routing engine with fallback, strategy selection, and SLA enforcement."""

import logging
import random
import time
from itertools import cycle

from src.config import ModelConfig, get_config
from src.health import get_health_tracker
from src.inference import call_model
from src.models import CompletionRequest, CompletionResponse, RoutingStrategy

logger = logging.getLogger(__name__)


class Router:
    """Routes inference requests across models with fallback and health-aware selection."""

    def __init__(self):
        self._round_robin_iter: cycle | None = None
        self._rr_models: list[str] = []
        self._total_requests: int = 0
        self._successful: int = 0
        self._failed: int = 0
        self._fallback_count: int = 0
        self._sla_violations: int = 0
        self._total_latency: float = 0.0

    def _get_models_by_strategy(self, strategy: str, preference: str | None = None) -> list[ModelConfig]:
        """Order models according to routing strategy."""
        cfg = get_config()
        tracker = get_health_tracker()

        # Filter to available models only
        available = [m for m in cfg.models if tracker.is_available(m.name)]

        if not available:
            return []

        # If a specific model is preferred and available, put it first
        if preference:
            preferred = [m for m in available if m.name == preference]
            others = [m for m in available if m.name != preference]
            if preferred:
                return preferred + self._sort_by_strategy(others, strategy)

        return self._sort_by_strategy(available, strategy)

    def _sort_by_strategy(self, models: list[ModelConfig], strategy: str) -> list[ModelConfig]:
        """Sort models according to strategy."""
        if strategy == RoutingStrategy.PRIORITY:
            return sorted(models, key=lambda m: m.priority)

        elif strategy == RoutingStrategy.ROUND_ROBIN:
            # Rotate through models - use priority as initial order
            names = [m.name for m in sorted(models, key=lambda m: m.priority)]
            if names != self._rr_models:
                self._rr_models = names
                self._round_robin_iter = cycle(names)
            if self._round_robin_iter:
                # Get next model in rotation and put it first
                next_name = next(self._round_robin_iter)
                ordered = [m for m in models if m.name == next_name]
                ordered += [m for m in models if m.name != next_name]
                return ordered
            return models

        elif strategy == RoutingStrategy.LATENCY_WEIGHTED:
            tracker = get_health_tracker()
            # Weight by inverse of average latency (faster = more likely first)
            weighted = []
            for m in models:
                stats = tracker.get_stats(m.name)
                # Default high latency for models with no data (explore them)
                avg = stats.avg_latency_ms if stats.total_requests > 0 else 1000.0
                weighted.append((m, avg))

            # Sort by latency ascending, with jitter to allow exploration
            weighted.sort(key=lambda x: x[1] + random.uniform(0, 100))
            return [m for m, _ in weighted]

        # Default: priority
        return sorted(models, key=lambda m: m.priority)

    async def route_request(self, request: CompletionRequest) -> CompletionResponse:
        """Route a request through models with fallback chain."""
        cfg = get_config()
        tracker = get_health_tracker()
        strategy = cfg.routing.strategy
        sla_ms = cfg.routing.sla_latency_ms

        models = self._get_models_by_strategy(strategy, request.model_preference)

        if not models:
            raise RouterError("No models available", models_attempted=[])

        max_attempts = min(cfg.routing.max_fallback_attempts, len(models))
        models_attempted: list[str] = []
        last_error: Exception | None = None

        self._total_requests += 1
        start_time = time.perf_counter()

        for attempt, model_cfg in enumerate(models[:max_attempts], 1):
            models_attempted.append(model_cfg.name)
            timeout = request.timeout_override_ms or model_cfg.timeout_ms

            try:
                result = await call_model(model_cfg, request.prompt, timeout)
                latency_ms = result["latency_ms"]

                # Record success
                tracker.record_outcome(model_cfg.name, latency_ms, success=True)

                total_latency = (time.perf_counter() - start_time) * 1000
                sla_met = total_latency <= sla_ms
                fallbacks = attempt - 1

                if not sla_met:
                    self._sla_violations += 1
                if fallbacks > 0:
                    self._fallback_count += 1

                self._successful += 1
                self._total_latency += total_latency

                return CompletionResponse(
                    result=result["result"],
                    model_used=result["model_used"],
                    latency_ms=round(total_latency, 2),
                    input_tokens=result["input_tokens"],
                    output_tokens=result["output_tokens"],
                    fallbacks_triggered=fallbacks,
                    attempt_number=attempt,
                    sla_met=sla_met,
                )

            except (TimeoutError, Exception) as e:
                last_error = e
                latency_ms = (time.perf_counter() - start_time) * 1000
                tracker.record_outcome(model_cfg.name, latency_ms, success=False)
                logger.warning(
                    "Model %s failed (attempt %d/%d): %s",
                    model_cfg.name, attempt, max_attempts, str(e)
                )
                continue

        # All models exhausted
        self._failed += 1
        raise RouterError(
            f"All models failed. Last error: {last_error}",
            models_attempted=models_attempted,
        )

    @property
    def stats(self) -> dict:
        """Return aggregate router statistics."""
        total = self._total_requests
        return {
            "total_requests": total,
            "successful": self._successful,
            "failed": self._failed,
            "fallback_count": self._fallback_count,
            "fallback_rate": self._fallback_count / total if total > 0 else 0.0,
            "sla_violations": self._sla_violations,
            "sla_violation_rate": self._sla_violations / total if total > 0 else 0.0,
            "avg_latency_ms": round(self._total_latency / self._successful, 2) if self._successful > 0 else 0.0,
        }


class RouterError(Exception):
    """Raised when routing fails after all fallback attempts."""

    def __init__(self, message: str, models_attempted: list[str]):
        super().__init__(message)
        self.models_attempted = models_attempted


# Module-level singleton
_router: Router | None = None


def get_router() -> Router:
    global _router
    if _router is None:
        _router = Router()
    return _router


def reset_router() -> None:
    global _router
    _router = None
