"""Health tracking and circuit breaker for model endpoints."""

import time
from collections import deque
from dataclasses import dataclass, field

from src.config import CircuitBreakerConfig, HealthConfig, get_config
from src.models import ModelState, ModelStats


@dataclass
class RequestOutcome:
    timestamp: float
    latency_ms: float
    success: bool


@dataclass
class CircuitBreakerState:
    failures: int = 0
    last_failure_time: float = 0.0
    state: ModelState = ModelState.HEALTHY
    half_open_successes: int = 0


@dataclass
class ModelHealth:
    name: str
    outcomes: deque = field(default_factory=lambda: deque(maxlen=50))
    circuit_breaker: CircuitBreakerState = field(default_factory=CircuitBreakerState)
    disabled: bool = False

    def __post_init__(self):
        cfg = get_config()
        self.outcomes = deque(maxlen=cfg.health.window_size)


class HealthTracker:
    """Tracks per-model health metrics and circuit breaker state."""

    def __init__(self, health_cfg: HealthConfig | None = None, cb_cfg: CircuitBreakerConfig | None = None):
        cfg = get_config()
        self.health_cfg = health_cfg or cfg.health
        self.cb_cfg = cb_cfg or cfg.circuit_breaker
        self._models: dict[str, ModelHealth] = {}

    def register_model(self, name: str) -> None:
        """Register a model for health tracking."""
        if name not in self._models:
            self._models[name] = ModelHealth(name=name)
            self._models[name].outcomes = deque(maxlen=self.health_cfg.window_size)

    def record_outcome(self, model_name: str, latency_ms: float, success: bool) -> None:
        """Record a request outcome for a model."""
        if model_name not in self._models:
            self.register_model(model_name)

        model = self._models[model_name]
        model.outcomes.append(RequestOutcome(
            timestamp=time.time(),
            latency_ms=latency_ms,
            success=success,
        ))

        cb = model.circuit_breaker
        if success:
            if cb.state == ModelState.HALF_OPEN:
                cb.half_open_successes += 1
                if cb.half_open_successes >= self.cb_cfg.half_open_requests:
                    cb.state = ModelState.HEALTHY
                    cb.failures = 0
                    cb.half_open_successes = 0
            elif cb.state == ModelState.HEALTHY:
                cb.failures = max(0, cb.failures - 1)
        else:
            cb.failures += 1
            cb.last_failure_time = time.time()
            if cb.failures >= self.cb_cfg.error_threshold:
                cb.state = ModelState.UNHEALTHY

    def is_available(self, model_name: str) -> bool:
        """Check if a model is available for routing."""
        if model_name not in self._models:
            return True

        model = self._models[model_name]
        if model.disabled:
            return False

        cb = model.circuit_breaker
        if cb.state == ModelState.HEALTHY:
            return True
        if cb.state == ModelState.HALF_OPEN:
            return True
        if cb.state == ModelState.UNHEALTHY:
            # Check if recovery window has elapsed
            elapsed = time.time() - cb.last_failure_time
            if elapsed >= self.cb_cfg.recovery_window_s:
                cb.state = ModelState.HALF_OPEN
                cb.half_open_successes = 0
                return True
            return False
        return False

    def get_state(self, model_name: str) -> ModelState:
        """Get the current state of a model."""
        if model_name not in self._models:
            return ModelState.HEALTHY
        model = self._models[model_name]
        if model.disabled:
            return ModelState.DISABLED
        # Re-check recovery window
        self.is_available(model_name)
        return model.circuit_breaker.state

    def get_stats(self, model_name: str) -> ModelStats:
        """Get metrics for a specific model."""
        if model_name not in self._models:
            return ModelStats()

        model = self._models[model_name]
        outcomes = list(model.outcomes)
        if not outcomes:
            return ModelStats()

        total = len(outcomes)
        successful = sum(1 for o in outcomes if o.success)
        failed = total - successful
        latencies = sorted(o.latency_ms for o in outcomes if o.success)

        return ModelStats(
            total_requests=total,
            successful=successful,
            failed=failed,
            error_rate=failed / total if total > 0 else 0.0,
            latency_p50=self._percentile(latencies, 50),
            latency_p95=self._percentile(latencies, 95),
            latency_p99=self._percentile(latencies, 99),
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        )

    def disable_model(self, model_name: str) -> bool:
        """Manually disable a model."""
        if model_name not in self._models:
            return False
        self._models[model_name].disabled = True
        return True

    def enable_model(self, model_name: str) -> bool:
        """Re-enable a disabled model and reset circuit breaker."""
        if model_name not in self._models:
            return False
        model = self._models[model_name]
        model.disabled = False
        model.circuit_breaker = CircuitBreakerState()
        return True

    def reset(self) -> None:
        """Reset all health state (used on config reload)."""
        self._models.clear()

    @staticmethod
    def _percentile(sorted_values: list[float], pct: int) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        idx = int(len(sorted_values) * pct / 100)
        idx = min(idx, len(sorted_values) - 1)
        return sorted_values[idx]


# Module-level singleton
_tracker: HealthTracker | None = None


def get_health_tracker() -> HealthTracker:
    global _tracker
    if _tracker is None:
        _tracker = HealthTracker()
    return _tracker


def reset_health_tracker() -> None:
    global _tracker
    _tracker = None
