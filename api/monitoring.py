"""Inference logging - Postgres if `FALLDET_DATABASE_URL` is set, CSV
otherwise.

This is still a monitoring STUB, not a production observability system:
no alerting, no thresholds, no dashboards - just a place to durably record
each prediction so scripts/plot_drift.py can later summarize it as a
rolling-mean-confidence plot. Postgres just makes that storage properly
concurrent-safe and queryable instead of an appended-to flat file, which
is the one part of "just a CSV" that was a genuine correctness problem
(multiple uvicorn workers writing to the same file with plain `open(...,
"a")` is a real race condition), not just a nice-to-have.

Falls back to CSV automatically when `FALLDET_DATABASE_URL` isn't set, so
local dev and tests never need a real database configured.
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

from api.schemas import PredictResponse

DEFAULT_LOG_PATH = Path("outputs/monitoring/inference_log.csv")
FIELDS = ["timestamp", "source", "predicted_class", "confidence", "latency_ms"]

SCHEMA = "falldet"
TABLE = "inference_log"

_pg_conn = None  # lazily-opened, reused across requests - not per-call


def _get_pg_connection(database_url: str):
    global _pg_conn
    import psycopg

    if _pg_conn is None or _pg_conn.closed:
        _pg_conn = psycopg.connect(database_url)
        with _pg_conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE} (
                    id BIGSERIAL PRIMARY KEY,
                    ts DOUBLE PRECISION NOT NULL,
                    source TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    latency_ms DOUBLE PRECISION NOT NULL
                )
                """
            )
        _pg_conn.commit()
    return _pg_conn


def _log_to_postgres(result: PredictResponse, source: str, database_url: str) -> None:
    conn = _get_pg_connection(database_url)
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {SCHEMA}.{TABLE} (ts, source, predicted_class, confidence, latency_ms) "
            "VALUES (%s, %s, %s, %s, %s)",
            (time.time(), source, result.predicted_class, result.confidence, result.latency_ms),
        )
    conn.commit()


def _log_to_csv(result: PredictResponse, source: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists()

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": time.time(),
                "source": source,
                "predicted_class": result.predicted_class,
                "confidence": result.confidence,
                "latency_ms": result.latency_ms,
            }
        )


def log_inference(result: PredictResponse, source: str, log_path: Path = DEFAULT_LOG_PATH) -> None:
    database_url = os.environ.get("FALLDET_DATABASE_URL")
    if database_url:
        _log_to_postgres(result, source, database_url)
    else:
        _log_to_csv(result, source, Path(log_path))


def read_recent_logs(limit: int = 500, csv_path: Path = DEFAULT_LOG_PATH) -> list[dict]:
    """Read the most recent inference log rows, from whichever backend is
    configured (Postgres if FALLDET_DATABASE_URL is set, else the CSV at
    csv_path). Returned oldest-first, matching what plot_drift.py expects.
    """
    database_url = os.environ.get("FALLDET_DATABASE_URL")
    if database_url:
        conn = _get_pg_connection(database_url)
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT ts, source, predicted_class, confidence, latency_ms "
                f"FROM {SCHEMA}.{TABLE} ORDER BY ts DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        rows.reverse()
        return [
            {"timestamp": r[0], "source": r[1], "predicted_class": r[2], "confidence": r[3], "latency_ms": r[4]}
            for r in rows
        ]

    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    import csv as _csv

    with open(csv_path, newline="") as f:
        rows = list(_csv.DictReader(f))
    for row in rows:
        row["timestamp"] = float(row["timestamp"])
        row["confidence"] = float(row["confidence"])
        row["latency_ms"] = float(row["latency_ms"])
    return rows[-limit:]
