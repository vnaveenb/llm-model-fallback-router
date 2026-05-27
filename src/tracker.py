"""Cost/usage tracker — logs requests to SQLite for Project 6 dashboard integration."""

import logging
import sqlite3
import time
from pathlib import Path

from src.config import get_config

logger = logging.getLogger(__name__)

_conn: sqlite3.Connection | None = None


def setup_tracker() -> None:
    """Initialize the tracker database."""
    global _conn
    cfg = get_config()
    if not cfg.tracker.enabled:
        return

    db_path = Path(cfg.tracker.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    _conn = sqlite3.connect(str(db_path), check_same_thread=False)
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS usage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            project_id TEXT NOT NULL,
            model TEXT NOT NULL,
            provider TEXT NOT NULL,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            latency_ms REAL DEFAULT 0,
            success INTEGER DEFAULT 1,
            fallback_triggered INTEGER DEFAULT 0,
            sla_met INTEGER DEFAULT 1,
            error TEXT
        )
    """)
    _conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_log(timestamp)
    """)
    _conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_usage_project ON usage_log(project_id)
    """)
    _conn.commit()
    logger.info("Tracker initialized: %s", db_path)


def log_request(
    model: str,
    provider: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: float = 0.0,
    success: bool = True,
    fallback_triggered: bool = False,
    sla_met: bool = True,
    error: str | None = None,
) -> None:
    """Log a single request to the tracker database."""
    if _conn is None:
        return

    cfg = get_config()
    try:
        _conn.execute(
            """INSERT INTO usage_log
               (timestamp, project_id, model, provider, input_tokens, output_tokens,
                latency_ms, success, fallback_triggered, sla_met, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                cfg.tracker.project_id,
                model,
                provider,
                input_tokens,
                output_tokens,
                latency_ms,
                int(success),
                int(fallback_triggered),
                int(sla_met),
                error,
            ),
        )
        _conn.commit()
    except sqlite3.Error as e:
        logger.error("Tracker write failed: %s", e)


def close_tracker() -> None:
    """Close the tracker database connection."""
    global _conn
    if _conn:
        _conn.close()
        _conn = None
