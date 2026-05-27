# Model Fallback Router

> A production-grade LLM routing proxy with automatic failover, SLA enforcement, circuit breaker, and real-time dashboard. Works with any LiteLLM-supported provider (Gemini, OpenAI, Anthropic, etc).

![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![LiteLLM](https://img.shields.io/badge/LLM-LiteLLM_1.67+-purple)
![Tests](https://img.shields.io/badge/Tests-36%20passing-brightgreen)

---

## What This Does

Production LLM systems need robust routing and failover. This project demonstrates:

- **Automatic model fallback** — If the primary model fails or times out, requests are routed to the next available model.
- **Multiple routing strategies** — Priority, round-robin, or latency-weighted selection.
- **Circuit breaker** — Models are marked unhealthy after repeated failures and automatically recovered.
- **SLA enforcement** — Tracks and flags requests that exceed latency targets.
- **Per-model health metrics** — Rolling window of p50/p95/p99 latency, error rate, and request count.
- **Manual control** — Enable/disable models via API or dashboard.
- **Real-time dashboard** — HTML dashboard shows model health, stats, and routing distribution.
- **Cost tracking** — Logs every request to a shared SQLite DB for integration with the Token Cost Dashboard (Project 6).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run in mock mode (no API key needed)
INFERENCE_MOCK=true uvicorn src.api:app --host 0.0.0.0 --port 8400

# 3. Run with real API keys
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY, OPENAI_API_KEY, etc.
uvicorn src.api:app --host 0.0.0.0 --port 8400
```

---

## Docker

```bash
# Build and run (mock mode by default)
docker compose up --build

# To use real API keys, set them in .env or pass as environment variables
# Example: docker compose up -d
```

### Volumes

| Volume | Purpose |
|--------|---------|
| `mfr_tracker_data` | Stores SQLite usage DB for cost tracking (shared with Project 6) |

---

## How to Test

### Mock Mode (No API Key Needed)

By default, the router runs in mock mode:

```bash
INFERENCE_MOCK=true uvicorn src.api:app --host 0.0.0.0 --port 8400
# or with Docker
docker compose up --build
```

All completions return simulated responses with random latency (100ms–2s). This is ideal for demos, CI, and local development.

### Real API Key Mode

1. Copy `.env.example` to `.env` and add your provider keys:
   - `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.
2. Set `INFERENCE_MOCK=false` (or remove it)
3. Start the server:
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 8400
   # or
   docker compose up --build
   ```
4. Requests will now route to real LLMs via LiteLLM.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/completions` | Route inference request through fallback chain |
| `GET` | `/models` | List models with health status |
| `GET` | `/models/{name}/stats` | Detailed per-model metrics |
| `POST` | `/models/{name}/disable` | Manually disable a model |
| `POST` | `/models/{name}/enable` | Re-enable a model |
| `GET` | `/stats` | Aggregate router statistics |
| `GET` | `/health` | Service health check |
| `POST` | `/reload-config` | Hot-reload config.yaml |
| `GET` | `/` | Monitoring dashboard |

---

## Example Usage

```bash
# Send a completion request
curl -X POST http://localhost:8400/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain circuit breakers in distributed systems"}'
```

```json
{
  "result": "A circuit breaker is a design pattern...",
  "model_used": "gemini-flash",
  "latency_ms": 1234.56,
  "input_tokens": 12,
  "output_tokens": 87,
  "fallbacks_triggered": 0,
  "attempt_number": 1,
  "sla_met": true
}
```

---

## Dashboard

- Open [http://localhost:8410/](http://localhost:8410/) (or your mapped port)
- See real-time model health, stats, and routing distribution
- Enable/disable models and watch fallback in action

---

## Configuration

All settings are in [`config.yaml`](config.yaml):

- **models[]** — List of providers/models with priority, timeout, weight
- **routing.strategy** — `priority` | `round-robin` | `latency-weighted`
- **routing.sla_latency_ms** — SLA threshold for violation tracking
- **circuit_breaker** — Error threshold, recovery window, half-open probe count
- **health.window_size** — Rolling window size for metrics
- **inference.mock** — Enable mock mode (no API keys needed)
- **tracker** — SQLite logging for Project 6 integration

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | API key for Gemini models |
| `OPENAI_API_KEY` | API key for OpenAI models |
| `ANTHROPIC_API_KEY` | API key for Anthropic models |
| `TRACKER_DB_PATH` | SQLite database path |
| `ROUTING_STRATEGY` | Override config strategy |
| `INFERENCE_MOCK` | Force mock mode (`true`/`false`) |
| `SERVER_PORT` | Override server port |
| `CONFIG_PATH` | Path to config.yaml |

---

## Test Coverage

**36 tests** across 4 modules:

| Module | Tests | Coverage |
|--------|-------|----------|
| `test_router.py` | 12 | Fallback chain, SLA, circuit breaker, round-robin, stats |
| `test_health.py` | 10 | Health tracker, circuit breaker, rolling window |
| `test_api.py` | 10 | All endpoints, error handling, enable/disable |
| `test_inference.py` | 4 | Mock/real mode, timeout, token counting |

```bash
# Run tests (mock mode, no API key needed)
pytest tests/ -v

# Or inside Docker
docker run --rm model-fallback-router pytest tests/ -v
```

---

## Architecture

```mermaid
flowchart TD
    Client[Client / API] -->|POST /completions| Router
    Router -->|Strategy: priority / round-robin / latency| ModelA[Model A]
    Router --> ModelB[Model B]
    Router --> ModelC[Model C]
    ModelA & ModelB & ModelC --> Health[Health Tracker]
    Health --> TrackerDB[SQLite Tracker DB]
    TrackerDB --> Dashboard[Token Cost Dashboard (P6)]
```

---

## Design Decisions

- **No Redis** — Health state is in-memory (rolling window). Stateless proxy rebuilds metrics from live traffic on restart.
- **LiteLLM backend** — Unified interface to OpenAI, Gemini, Anthropic, local models. No per-provider code.
- **Circuit breaker** — Prevents cascading failures by stopping requests to unhealthy models, with automatic recovery probes.
- **Three strategies** — Each demonstrates a different production routing pattern (priority for reliability, round-robin for distribution, latency-weighted for performance).
- **Shared tracker volume** — Token Cost Dashboard (Project 6) picks up costs automatically via Docker volume.

---

## CI & Deployment

- **GitHub Actions**: `.github/workflows/ci.yml` runs tests, builds, and pushes Docker image to GHCR
- **Docker Compose**: Production-ready, non-root user, healthcheck, shared volume for cost tracking

---

## License

MIT — see [LICENSE](LICENSE)
