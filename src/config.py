"""Configuration management for Model Fallback Router."""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8400
    reload: bool = False


class ModelConfig(BaseModel):
    name: str
    provider: str
    model_id: str
    priority: int = 1
    timeout_ms: int = 10000
    max_retries: int = 1
    weight: int = 50


class RoutingConfig(BaseModel):
    strategy: str = "priority"  # priority | round-robin | latency-weighted
    sla_latency_ms: int = 5000
    max_fallback_attempts: int = 3


class CircuitBreakerConfig(BaseModel):
    error_threshold: int = 5
    recovery_window_s: int = 60
    half_open_requests: int = 2


class HealthConfig(BaseModel):
    check_interval_s: int = 30
    window_size: int = 50
    unhealthy_error_rate: float = 0.5


class InferenceConfig(BaseModel):
    mock: bool = False
    temperature: float = 0.0
    max_tokens: int = 1024


class TrackerConfig(BaseModel):
    enabled: bool = True
    project_id: str = "model-fallback-router"
    db_path: str = "./data/usage.db"


class AppConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    models: list[ModelConfig] = []
    routing: RoutingConfig = RoutingConfig()
    circuit_breaker: CircuitBreakerConfig = CircuitBreakerConfig()
    health: HealthConfig = HealthConfig()
    inference: InferenceConfig = InferenceConfig()
    tracker: TrackerConfig = TrackerConfig()


_config: AppConfig | None = None


def get_config(path: str | None = None) -> AppConfig:
    """Load config from YAML file with environment variable overrides."""
    global _config
    if _config is not None:
        return _config

    if path is None:
        path = os.environ.get("CONFIG_PATH", "config.yaml")

    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    # Environment variable overrides
    if os.environ.get("SERVER_PORT"):
        data.setdefault("server", {})["port"] = int(os.environ["SERVER_PORT"])
    if os.environ.get("ROUTING_STRATEGY"):
        data.setdefault("routing", {})["strategy"] = os.environ["ROUTING_STRATEGY"]
    if os.environ.get("INFERENCE_MOCK"):
        data.setdefault("inference", {})["mock"] = os.environ["INFERENCE_MOCK"].lower() == "true"
    if os.environ.get("TRACKER_DB_PATH"):
        data.setdefault("tracker", {})["db_path"] = os.environ["TRACKER_DB_PATH"]

    _config = AppConfig(**data)
    return _config


def reload_config(path: str | None = None) -> AppConfig:
    """Force reload config from disk."""
    global _config
    _config = None
    return get_config(path)


def set_config(config: AppConfig) -> None:
    """Set config directly (used in tests)."""
    global _config
    _config = config
