"""Transactional SQLite persistence for model state and prediction records."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .types import PredictionRecord

SCHEMA_VERSION = 1


class StateRepository:
    """Commit model parameters and changed ledger rows in one transaction."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        with self._connect() as database:
            self._create_schema(database)
            row = database.execute(
                "SELECT schema_version, sequence, algorithms_json FROM online_state WHERE id = 1"
            ).fetchone()
            if row is None:
                return None
            if int(row[0]) != SCHEMA_VERSION:
                raise ValueError(f"unsupported online research state schema in {self.path}")
            records = [
                json.loads(item[0])
                for item in database.execute(
                    "SELECT payload_json FROM prediction_ledger ORDER BY sequence"
                ).fetchall()
            ]
            return {
                "schema_version": int(row[0]),
                "sequence": int(row[1]),
                "algorithms": json.loads(row[2]),
                "records": records,
            }

    def save(
        self,
        *,
        sequence: int,
        algorithms: dict[str, Any],
        records: Sequence[PredictionRecord],
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as database:
            self._create_schema(database)
            database.execute("BEGIN IMMEDIATE")
            try:
                database.execute(
                    """
                    INSERT INTO online_state (id, schema_version, sequence, algorithms_json)
                    VALUES (1, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        schema_version = excluded.schema_version,
                        sequence = excluded.sequence,
                        algorithms_json = excluded.algorithms_json
                    """,
                    (
                        SCHEMA_VERSION,
                        int(sequence),
                        json.dumps(
                            algorithms, separators=(",", ":"), sort_keys=True, allow_nan=False
                        ),
                    ),
                )
                database.executemany(
                    """
                    INSERT INTO prediction_ledger
                        (prediction_id, sequence, algorithm, instrument, target_at, scored, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(prediction_id) DO UPDATE SET
                        scored = excluded.scored,
                        payload_json = excluded.payload_json
                    """,
                    [self._record_row(record) for record in records],
                )
                database.commit()
            except Exception:
                database.rollback()
                raise

    def _connect(self) -> sqlite3.Connection:
        database = sqlite3.connect(self.path, timeout=20.0)
        database.execute("PRAGMA synchronous = FULL")
        database.execute("PRAGMA foreign_keys = ON")
        return database

    @staticmethod
    def _create_schema(database: sqlite3.Connection) -> None:
        database.executescript(
            """
            CREATE TABLE IF NOT EXISTS online_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                schema_version INTEGER NOT NULL,
                sequence INTEGER NOT NULL,
                algorithms_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS prediction_ledger (
                prediction_id TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL UNIQUE,
                algorithm TEXT NOT NULL,
                instrument TEXT NOT NULL,
                target_at TEXT NOT NULL,
                scored INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS prediction_due
                ON prediction_ledger (instrument, scored, target_at);
            CREATE INDEX IF NOT EXISTS prediction_algorithm
                ON prediction_ledger (algorithm, sequence);
            """
        )

    @staticmethod
    def _record_row(record: PredictionRecord) -> tuple[Any, ...]:
        sequence_text = record.prediction_id.removeprefix("online-").split("-", 1)[0]
        return (
            record.prediction_id,
            int(sequence_text),
            record.algorithm,
            record.instrument,
            record.target_at.isoformat(),
            int(record.is_scored),
            json.dumps(
                record.as_dict(), separators=(",", ":"), sort_keys=True, allow_nan=False
            ),
        )


__all__ = ["SCHEMA_VERSION", "StateRepository"]
