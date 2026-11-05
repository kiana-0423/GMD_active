"""Persistent workflow state (SQLite, standard library only).

Every externally visible action is recorded here *before* it is performed (intent) and again
after it completes (result), inside transactions. After a crash the controller rebuilds its
position from this store and reconciles intents whose outcome is unknown, instead of repeating
work. Identifiers are deterministic, and uniqueness constraints make every insert idempotent.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from gmd_active_learning.core.exceptions import WorkflowError

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    cycle INTEGER NOT NULL DEFAULT 0,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    active_model_id TEXT,
    active_dataset_version TEXT,
    last_checkpoint TEXT,
    last_step INTEGER,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS transitions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    stage_from TEXT,
    stage_to TEXT NOT NULL,
    at TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS segments (
    run_id TEXT NOT NULL,
    segment_index INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    model_id TEXT,
    status TEXT NOT NULL,
    work_dir TEXT NOT NULL,
    start_step INTEGER,
    end_step INTEGER,
    checkpoint TEXT,
    detail TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, segment_index)
);
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    trigger_step INTEGER NOT NULL,
    complete INTEGER NOT NULL,
    state TEXT NOT NULL,
    path TEXT,
    summary TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    event_id TEXT,
    score REAL NOT NULL,
    state TEXT NOT NULL,
    reason TEXT,
    region_path TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    state TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 0,
    execution TEXT,
    validation TEXT,
    job TEXT,
    work_dir TEXT NOT NULL,
    failure_reasons TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS samples (
    sample_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL UNIQUE,
    candidate_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    path TEXT NOT NULL,
    split_group TEXT NOT NULL,
    dataset_version TEXT,
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS datasets (
    version TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    record TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    record TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS promotions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    previous_model_id TEXT,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS blobs (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
"""

TERMINAL_TASK_STATES = frozenset({"ingested", "invalid", "failed", "cancelled", "abandoned"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


@dataclass(slots=True)
class RunRecord:
    run_id: str
    config_hash: str
    cycle: int
    stage: str
    status: str
    active_model_id: str | None
    active_dataset_version: str | None
    last_checkpoint: str | None
    last_step: int | None
    metadata: dict[str, Any]


class WorkflowStore:
    """Thin transactional wrapper around one SQLite file per workflow directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, isolation_level=None, timeout=30.0)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.executescript(_SCHEMA)  # idempotent CREATE IF NOT EXISTS
        row = self._connection.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None:
            self._connection.execute("INSERT INTO meta(key, value) VALUES('schema_version', ?)", (str(SCHEMA_VERSION),))
        elif int(row["value"]) != SCHEMA_VERSION:
            raise WorkflowError(f"workflow store {self.path} has schema {row['value']}, expected {SCHEMA_VERSION}")

    def close(self) -> None:
        self._connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        cursor = self._connection.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        try:
            yield cursor
        except BaseException:
            cursor.execute("ROLLBACK")
            raise
        else:
            cursor.execute("COMMIT")

    # ------------------------------------------------------------------ runs and stages
    def ensure_run(self, run_id: str, *, config_hash: str, initial_model_id: str | None, stage: str) -> RunRecord:
        with self.transaction() as cursor:
            row = cursor.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                cursor.execute(
                    "INSERT INTO runs(run_id, created_at, config_hash, cycle, stage, status, active_model_id) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (run_id, utc_now(), config_hash, 0, stage, "active", initial_model_id),
                )
                cursor.execute(
                    "INSERT INTO transitions(run_id, cycle, stage_from, stage_to, at, detail) VALUES(?,?,?,?,?,?)",
                    (run_id, 0, None, stage, utc_now(), _dumps({"created": True})),
                )
            elif row["config_hash"] != config_hash:
                raise WorkflowError(
                    f"run {run_id} was created with config {row['config_hash']}, now {config_hash}; "
                    "resuming with a different configuration requires a new run id"
                )
        return self.run(run_id)

    def run(self, run_id: str) -> RunRecord:
        row = self._connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise WorkflowError(f"unknown run {run_id}")
        return RunRecord(
            run_id=row["run_id"],
            config_hash=row["config_hash"],
            cycle=int(row["cycle"]),
            stage=row["stage"],
            status=row["status"],
            active_model_id=row["active_model_id"],
            active_dataset_version=row["active_dataset_version"],
            last_checkpoint=row["last_checkpoint"],
            last_step=row["last_step"],
            metadata=json.loads(row["metadata"]),
        )

    def transition(self, run_id: str, stage: str, *, cycle: int | None = None, detail: dict[str, Any] | None = None, **fields: Any) -> None:
        allowed = {"status", "active_model_id", "active_dataset_version", "last_checkpoint", "last_step"}
        unknown = set(fields) - allowed
        if unknown:
            raise WorkflowError(f"cannot update run fields {sorted(unknown)}")
        with self.transaction() as cursor:
            row = cursor.execute("SELECT stage, cycle FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise WorkflowError(f"unknown run {run_id}")
            new_cycle = int(row["cycle"]) if cycle is None else int(cycle)
            assignments = ["stage=?", "cycle=?", *[f"{key}=?" for key in fields]]
            cursor.execute(
                f"UPDATE runs SET {', '.join(assignments)} WHERE run_id=?",  # noqa: S608 - keys are whitelisted
                (stage, new_cycle, *fields.values(), run_id),
            )
            cursor.execute(
                "INSERT INTO transitions(run_id, cycle, stage_from, stage_to, at, detail) VALUES(?,?,?,?,?,?)",
                (run_id, new_cycle, row["stage"], stage, utc_now(), _dumps(detail or {})),
            )

    def update_run(self, run_id: str, **fields: Any) -> None:
        run = self.run(run_id)
        self.transition(run_id, run.stage, cycle=run.cycle, detail={"update": sorted(fields)}, **fields)

    def transitions(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute("SELECT * FROM transitions WHERE run_id=? ORDER BY seq", (run_id,)).fetchall()
        return [{**dict(row), "detail": json.loads(row["detail"])} for row in rows]

    # ------------------------------------------------------------------ segments
    def upsert_segment(self, run_id: str, segment_index: int, **fields: Any) -> None:
        columns = {"cycle", "model_id", "status", "work_dir", "start_step", "end_step", "checkpoint", "detail"}
        if set(fields) - columns:
            raise WorkflowError(f"unknown segment fields {sorted(set(fields) - columns)}")
        if "detail" in fields:
            fields["detail"] = _dumps(fields["detail"])
        with self.transaction() as cursor:
            exists = cursor.execute(
                "SELECT 1 FROM segments WHERE run_id=? AND segment_index=?", (run_id, segment_index)
            ).fetchone()
            if exists is None:
                required = {"cycle", "status", "work_dir"}
                if required - set(fields):
                    raise WorkflowError(f"new segment needs {sorted(required)}")
                names = ["run_id", "segment_index", *fields]
                cursor.execute(
                    f"INSERT INTO segments({', '.join(names)}) VALUES({', '.join('?' * len(names))})",  # noqa: S608
                    (run_id, segment_index, *fields.values()),
                )
            elif fields:
                cursor.execute(
                    f"UPDATE segments SET {', '.join(f'{key}=?' for key in fields)} WHERE run_id=? AND segment_index=?",  # noqa: S608
                    (*fields.values(), run_id, segment_index),
                )

    def segments(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT * FROM segments WHERE run_id=? ORDER BY segment_index", (run_id,)
        ).fetchall()
        return [{**dict(row), "detail": json.loads(row["detail"])} for row in rows]

    # ------------------------------------------------------------------ events / candidates / tasks / samples
    def record_event(self, run_id: str, cycle: int, summary: dict[str, Any], *, path: str | None) -> bool:
        """Insert an event once; returns False if it was already recorded."""

        with self.transaction() as cursor:
            cursor.execute(
                "INSERT OR IGNORE INTO events(event_id, run_id, cycle, trigger_step, complete, state, path, summary) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (summary["event_id"], run_id, cycle, int(summary["trigger_step"]), int(bool(summary["complete"])), "captured", path, _dumps(summary)),
            )
            return cursor.rowcount == 1

    def set_event_state(self, event_id: str, state: str) -> None:
        with self.transaction() as cursor:
            cursor.execute("UPDATE events SET state=? WHERE event_id=?", (state, event_id))

    def events(self, run_id: str, *, state: str | None = None) -> list[dict[str, Any]]:
        query, params = "SELECT * FROM events WHERE run_id=?", [run_id]
        if state is not None:
            query, params = query + " AND state=?", [*params, state]
        rows = self._connection.execute(query + " ORDER BY trigger_step, event_id", params).fetchall()
        return [{**dict(row), "summary": json.loads(row["summary"])} for row in rows]

    def record_candidate(self, run_id: str, cycle: int, *, candidate_id: str, event_id: str | None, score: float, region_path: str, fingerprint: list[float]) -> bool:
        with self.transaction() as cursor:
            cursor.execute(
                "INSERT OR IGNORE INTO candidates(candidate_id, run_id, cycle, event_id, score, state, region_path, fingerprint, created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (candidate_id, run_id, cycle, event_id, float(score), "pending", region_path, _dumps(fingerprint), utc_now()),
            )
            return cursor.rowcount == 1

    def set_candidate_state(self, candidate_id: str, state: str, reason: str | None = None) -> None:
        with self.transaction() as cursor:
            cursor.execute("UPDATE candidates SET state=?, reason=? WHERE candidate_id=?", (state, reason, candidate_id))

    def candidates(self, run_id: str, *, state: str | None = None, cycle: int | None = None) -> list[dict[str, Any]]:
        query, params = "SELECT * FROM candidates WHERE run_id=?", [run_id]
        if state is not None:
            query, params = query + " AND state=?", [*params, state]
        if cycle is not None:
            query, params = query + " AND cycle=?", [*params, cycle]
        rows = self._connection.execute(query + " ORDER BY score DESC, candidate_id", params).fetchall()
        return [{**dict(row), "fingerprint": json.loads(row["fingerprint"])} for row in rows]

    def upsert_task(self, task_id: str, **fields: Any) -> None:
        columns = {"candidate_id", "run_id", "cycle", "state", "attempt", "execution", "validation", "job", "work_dir", "failure_reasons"}
        if set(fields) - columns:
            raise WorkflowError(f"unknown task fields {sorted(set(fields) - columns)}")
        for key in ("job", "failure_reasons"):
            if key in fields:
                fields[key] = _dumps(fields[key])
        fields["updated_at"] = utc_now()
        with self.transaction() as cursor:
            exists = cursor.execute("SELECT 1 FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if exists is None:
                names = ["task_id", *fields]
                cursor.execute(
                    f"INSERT INTO tasks({', '.join(names)}) VALUES({', '.join('?' * len(names))})",  # noqa: S608
                    (task_id, *fields.values()),
                )
            else:
                cursor.execute(
                    f"UPDATE tasks SET {', '.join(f'{key}=?' for key in fields)} WHERE task_id=?",  # noqa: S608
                    (*fields.values(), task_id),
                )

    def task(self, task_id: str) -> dict[str, Any] | None:
        row = self._connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return None if row is None else self._task_row(row)

    def tasks(self, run_id: str, *, states: set[str] | None = None, cycle: int | None = None) -> list[dict[str, Any]]:
        query, params = "SELECT * FROM tasks WHERE run_id=?", [run_id]
        if cycle is not None:
            query, params = query + " AND cycle=?", [*params, cycle]
        rows = [self._task_row(row) for row in self._connection.execute(query + " ORDER BY task_id", params).fetchall()]
        return rows if states is None else [row for row in rows if row["state"] in states]

    @staticmethod
    def _task_row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["job"] = None if data["job"] is None else json.loads(data["job"])
        data["failure_reasons"] = json.loads(data["failure_reasons"])
        return data

    def record_sample(self, *, sample_id: str, task_id: str, candidate_id: str, run_id: str, cycle: int, path: str, split_group: str) -> bool:
        """Insert a labeled sample once per task (UNIQUE task_id prevents duplicate labels)."""

        with self.transaction() as cursor:
            cursor.execute(
                "INSERT OR IGNORE INTO samples(sample_id, task_id, candidate_id, run_id, cycle, path, split_group, ingested_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (sample_id, task_id, candidate_id, run_id, cycle, path, split_group, utc_now()),
            )
            return cursor.rowcount == 1

    def samples(self, run_id: str, *, unpublished_only: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM samples WHERE run_id=?" + (" AND dataset_version IS NULL" if unpublished_only else "")
        return [dict(row) for row in self._connection.execute(query + " ORDER BY sample_id", (run_id,)).fetchall()]

    def mark_samples_published(self, sample_ids: list[str], version: str) -> None:
        with self.transaction() as cursor:
            cursor.executemany(
                "UPDATE samples SET dataset_version=? WHERE sample_id=? AND dataset_version IS NULL",
                [(version, sample_id) for sample_id in sample_ids],
            )

    # ------------------------------------------------------------------ datasets, models, promotions
    def record_dataset(self, run_id: str, cycle: int, record: dict[str, Any]) -> bool:
        with self.transaction() as cursor:
            cursor.execute(
                "INSERT OR IGNORE INTO datasets(version, run_id, cycle, sha256, record, created_at) VALUES(?,?,?,?,?,?)",
                (record["version"], run_id, cycle, record["sha256"], _dumps(record), utc_now()),
            )
            return cursor.rowcount == 1

    def dataset(self, version: str) -> dict[str, Any] | None:
        row = self._connection.execute("SELECT record FROM datasets WHERE version=?", (version,)).fetchone()
        return None if row is None else json.loads(row["record"])

    def record_model(self, record: dict[str, Any], *, state: str) -> bool:
        with self.transaction() as cursor:
            cursor.execute(
                "INSERT OR IGNORE INTO models(model_id, state, record, created_at) VALUES(?,?,?,?)",
                (record["model_id"], state, _dumps(record), utc_now()),
            )
            return cursor.rowcount == 1

    def set_model_state(self, model_id: str, state: str) -> None:
        with self.transaction() as cursor:
            cursor.execute("UPDATE models SET state=? WHERE model_id=?", (state, model_id))
            if cursor.rowcount != 1:
                raise WorkflowError(f"unknown model {model_id}")

    def model(self, model_id: str) -> dict[str, Any] | None:
        row = self._connection.execute("SELECT state, record FROM models WHERE model_id=?", (model_id,)).fetchone()
        return None if row is None else {"state": row["state"], **json.loads(row["record"])}

    def models(self) -> list[dict[str, Any]]:
        rows = self._connection.execute("SELECT state, record FROM models ORDER BY created_at, model_id").fetchall()
        return [{"state": row["state"], **json.loads(row["record"])} for row in rows]

    def record_promotion(self, run_id: str, *, model_id: str, previous_model_id: str | None, action: str, reason: str) -> None:
        with self.transaction() as cursor:
            cursor.execute(
                "INSERT INTO promotions(run_id, model_id, previous_model_id, action, reason, at) VALUES(?,?,?,?,?,?)",
                (run_id, model_id, previous_model_id, action, reason, utc_now()),
            )

    def promotions(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute("SELECT * FROM promotions WHERE run_id=? ORDER BY seq", (run_id,)).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------ opaque component state
    def put_blob(self, key: str, value: Any) -> None:
        with self.transaction() as cursor:
            cursor.execute(
                "INSERT INTO blobs(key, value, updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, _dumps(value), utc_now()),
            )

    def get_blob(self, key: str, default: Any = None) -> Any:
        row = self._connection.execute("SELECT value FROM blobs WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])
