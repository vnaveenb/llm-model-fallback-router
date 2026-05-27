"""Tests for the health tracker and circuit breaker."""

import time

import pytest

from src.config import AppConfig, CircuitBreakerConfig, HealthConfig, set_config
from src.health import HealthTracker, reset_health_tracker
from src.models import ModelState


@pytest.fixture(autouse=True)
def setup_config():
    set_config(AppConfig(
        health=HealthConfig(window_size=10, unhealthy_error_rate=0.5),
        circuit_breaker=CircuitBreakerConfig(error_threshold=3, recovery_window_s=2, half_open_requests=2),
    ))
    reset_health_tracker()
    yield


@pytest.fixture
def tracker():
    return HealthTracker()


def test_register_model(tracker):
    tracker.register_model("test-model")
    assert tracker.get_state("test-model") == ModelState.HEALTHY


def test_record_success(tracker):
    tracker.register_model("m1")
    tracker.record_outcome("m1", 100.0, success=True)
    stats = tracker.get_stats("m1")
    assert stats.total_requests == 1
    assert stats.successful == 1
    assert stats.error_rate == 0.0


def test_record_failure(tracker):
    tracker.register_model("m1")
    tracker.record_outcome("m1", 500.0, success=False)
    stats = tracker.get_stats("m1")
    assert stats.total_requests == 1
    assert stats.failed == 1
    assert stats.error_rate == 1.0


def test_circuit_breaker_opens_after_threshold(tracker):
    tracker.register_model("m1")
    # 3 failures should trip the circuit breaker
    for _ in range(3):
        tracker.record_outcome("m1", 100.0, success=False)
    assert tracker.get_state("m1") == ModelState.UNHEALTHY
    assert tracker.is_available("m1") is False


def test_circuit_breaker_recovery(tracker, monkeypatch):
    tracker.register_model("m1")
    for _ in range(3):
        tracker.record_outcome("m1", 100.0, success=False)
    assert tracker.is_available("m1") is False

    # Simulate time passing beyond recovery window
    future = time.time() + 3
    monkeypatch.setattr(time, "time", lambda: future)
    assert tracker.is_available("m1") is True
    assert tracker.get_state("m1") == ModelState.HALF_OPEN


def test_half_open_to_healthy(tracker, monkeypatch):
    tracker.register_model("m1")
    for _ in range(3):
        tracker.record_outcome("m1", 100.0, success=False)

    # Advance past recovery window
    future = time.time() + 3
    monkeypatch.setattr(time, "time", lambda: future)
    tracker.is_available("m1")  # Transitions to half-open

    # Two successes should close the circuit
    tracker.record_outcome("m1", 50.0, success=True)
    tracker.record_outcome("m1", 50.0, success=True)
    assert tracker.get_state("m1") == ModelState.HEALTHY


def test_percentile_calculation(tracker):
    tracker.register_model("m1")
    latencies = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for lat in latencies:
        tracker.record_outcome("m1", float(lat), success=True)

    stats = tracker.get_stats("m1")
    assert stats.latency_p50 == 600.0  # Index 5 of 10
    assert stats.latency_p95 == 1000.0
    assert stats.avg_latency_ms == 550.0


def test_disable_enable_model(tracker):
    tracker.register_model("m1")
    assert tracker.is_available("m1") is True

    tracker.disable_model("m1")
    assert tracker.get_state("m1") == ModelState.DISABLED
    assert tracker.is_available("m1") is False

    tracker.enable_model("m1")
    assert tracker.get_state("m1") == ModelState.HEALTHY
    assert tracker.is_available("m1") is True


def test_rolling_window_eviction(tracker):
    tracker.register_model("m1")
    # Window size is 10 — add 15 entries
    for i in range(15):
        tracker.record_outcome("m1", float(i * 100), success=True)

    stats = tracker.get_stats("m1")
    assert stats.total_requests == 10  # Only last 10 kept


def test_unregistered_model_defaults(tracker):
    assert tracker.get_state("unknown") == ModelState.HEALTHY
    assert tracker.is_available("unknown") is True
    stats = tracker.get_stats("unknown")
    assert stats.total_requests == 0
