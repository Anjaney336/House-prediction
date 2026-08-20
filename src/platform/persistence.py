from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.utils.config import ROOT_DIR


PLATFORM_DIR = ROOT_DIR / "data" / "platform"
DATABASE_PATH = PLATFORM_DIR / "platform.db"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    PLATFORM_DIR.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(DATABASE_PATH)
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA foreign_keys = ON")
    try:
        yield database
        database.commit()
    finally:
        database.close()


def initialize_database() -> None:
    with connection() as database:
        database.executescript(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                owner_type TEXT NOT NULL,
                name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_format TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                schema_contract TEXT NOT NULL,
                created_at TEXT NOT NULL,
                retention_until TEXT,
                UNIQUE(tenant_id, id)
            );
            CREATE INDEX IF NOT EXISTS idx_datasets_tenant ON datasets(tenant_id);
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                status TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 0,
                model_card TEXT NOT NULL,
                created_at TEXT NOT NULL,
                market TEXT,
                region TEXT,
                asset_type TEXT,
                property_type TEXT,
                transaction_type TEXT,
                model_scope TEXT NOT NULL DEFAULT 'platform',
                allow_region_fallback INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(dataset_id) REFERENCES datasets(id)
            );
            CREATE INDEX IF NOT EXISTS idx_models_tenant ON models(tenant_id);
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                estimate REAL NOT NULL,
                lower_bound REAL NOT NULL,
                upper_bound REAL NOT NULL,
                input_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(model_id) REFERENCES models(id)
            );
            CREATE TABLE IF NOT EXISTS leads (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                contact_json TEXT NOT NULL,
                consent INTEGER NOT NULL,
                retention_until TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(model_id) REFERENCES models(id)
            );
            CREATE TABLE IF NOT EXISTS training_jobs (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                status TEXT NOT NULL,
                detail TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY(dataset_id) REFERENCES datasets(id)
            );
            """
        )
        dataset_columns = {row["name"] for row in database.execute("PRAGMA table_info(datasets)").fetchall()}
        if "retention_until" not in dataset_columns:
            database.execute("ALTER TABLE datasets ADD COLUMN retention_until TEXT")
        model_columns = {row["name"] for row in database.execute("PRAGMA table_info(models)").fetchall()}
        migrations = {
            "market": "TEXT", "region": "TEXT", "asset_type": "TEXT", "property_type": "TEXT",
            "transaction_type": "TEXT", "model_scope": "TEXT NOT NULL DEFAULT 'platform'",
            "allow_region_fallback": "INTEGER NOT NULL DEFAULT 0",
        }
        for column, declaration in migrations.items():
            if column not in model_columns:
                database.execute(f"ALTER TABLE models ADD COLUMN {column} {declaration}")


def insert(table: str, values: dict[str, Any]) -> None:
    allowed = {"datasets", "models", "predictions", "leads", "training_jobs"}
    if table not in allowed:
        raise ValueError("Unsupported persistence table.")
    columns = list(values)
    placeholders = ", ".join("?" for _ in columns)
    with connection() as database:
        database.execute(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
            [json.dumps(value, default=str) if isinstance(value, (dict, list, tuple)) else value for value in values.values()],
        )


def update(table: str, record_id: str, tenant_id: str, values: dict[str, Any]) -> None:
    if table not in {"models", "training_jobs"}:
        raise ValueError("Unsupported persistence table.")
    assignments = ", ".join(f"{column} = ?" for column in values)
    parameters = [json.dumps(value, default=str) if isinstance(value, (dict, list, tuple)) else value for value in values.values()]
    with connection() as database:
        database.execute(
            f"UPDATE {table} SET {assignments} WHERE id = ? AND tenant_id = ?",
            parameters + [record_id, tenant_id],
        )


def get_one(table: str, record_id: str, tenant_id: str) -> dict[str, Any] | None:
    if table not in {"datasets", "models", "predictions", "training_jobs"}:
        raise ValueError("Unsupported persistence table.")
    with connection() as database:
        row = database.execute(
            f"SELECT * FROM {table} WHERE id = ? AND tenant_id = ?", (record_id, tenant_id)
        ).fetchone()
    return dict(row) if row else None


def list_records(table: str, tenant_id: str) -> list[dict[str, Any]]:
    if table not in {"datasets", "models", "training_jobs"}:
        raise ValueError("Unsupported persistence table.")
    with connection() as database:
        rows = database.execute(
            f"SELECT * FROM {table} WHERE tenant_id = ? ORDER BY created_at DESC", (tenant_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def purge_expired_leads(at: str | None = None) -> int:
    """Delete contact records whose disclosed retention period has elapsed."""
    cutoff = at or utc_now()
    with connection() as database:
        cursor = database.execute("DELETE FROM leads WHERE retention_until <= ?", (cutoff,))
        return int(cursor.rowcount)
