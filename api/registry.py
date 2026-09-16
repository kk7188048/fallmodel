"""A minimal model registry backed by Postgres.

Stores metadata about trained checkpoints (a tag, the local file path, a
sha256 hash for integrity, and a few key config fields) in
`falldet.models`, and resolves "give me the model for tag X" at serving
startup - a real (if intentionally minimal) step up from a hardcoded file
path baked into an env var. This is "one active checkpoint per tag," not
a full versioning/promotion/rollback system - deploying model N means
running register_model.py again for that tag, not clicking "rollback."

Falls back to doing nothing when `FALLDET_DATABASE_URL` isn't set, same
pattern as api/monitoring.py - training/local dev never needs a database
configured, and api/main.py falls back to FALLDET_MODEL_PATH directly.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

SCHEMA = "falldet"
TABLE = "models"

_pg_conn = None


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
                    tag TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    window_size INTEGER,
                    stride INTEGER,
                    hidden_dim INTEGER,
                    registered_at DOUBLE PRECISION NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE
                )
                """
            )
        _pg_conn.commit()
    return _pg_conn


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def register_model(checkpoint_path: str | Path, tag: str, database_url: str | None = None) -> dict:
    """Register a checkpoint under `tag`, deactivating any previous model
    with the same tag so resolve_model_path(tag) always returns the
    latest one registered.
    """
    import torch

    database_url = database_url or os.environ["FALLDET_DATABASE_URL"]
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    file_hash = sha256_of_file(checkpoint_path)

    conn = _get_pg_connection(database_url)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {SCHEMA}.{TABLE} SET is_active = FALSE WHERE tag = %s", (tag,))
        cur.execute(
            f"""INSERT INTO {SCHEMA}.{TABLE}
                (tag, file_path, sha256, window_size, stride, hidden_dim, registered_at, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE) RETURNING id""",
            (
                tag,
                str(checkpoint_path.resolve()),
                file_hash,
                config["windowing"]["window_size"],
                config["windowing"]["stride"],
                config["model"]["hidden_dim"],
                time.time(),
            ),
        )
        model_id = cur.fetchone()[0]
    conn.commit()

    return {
        "id": model_id,
        "tag": tag,
        "file_path": str(checkpoint_path.resolve()),
        "sha256": file_hash,
        "window_size": config["windowing"]["window_size"],
        "stride": config["windowing"]["stride"],
        "hidden_dim": config["model"]["hidden_dim"],
    }


def resolve_model_path(tag: str, database_url: str | None = None) -> str | None:
    """Look up the currently active checkpoint path for `tag`. Returns
    None if no database is configured, or no active model is registered
    under that tag - callers (api/main.py) fall back to FALLDET_MODEL_PATH.
    """
    database_url = database_url or os.environ.get("FALLDET_DATABASE_URL")
    if not database_url:
        return None

    conn = _get_pg_connection(database_url)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT file_path FROM {SCHEMA}.{TABLE} WHERE tag = %s AND is_active = TRUE "
            f"ORDER BY registered_at DESC LIMIT 1",
            (tag,),
        )
        row = cur.fetchone()
    return row[0] if row else None


def list_models(database_url: str | None = None) -> list[dict]:
    database_url = database_url or os.environ["FALLDET_DATABASE_URL"]
    conn = _get_pg_connection(database_url)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id, tag, file_path, sha256, window_size, stride, hidden_dim, registered_at, is_active "
            f"FROM {SCHEMA}.{TABLE} ORDER BY registered_at DESC"
        )
        rows = cur.fetchall()
    cols = ["id", "tag", "file_path", "sha256", "window_size", "stride", "hidden_dim", "registered_at", "is_active"]
    return [dict(zip(cols, r)) for r in rows]
