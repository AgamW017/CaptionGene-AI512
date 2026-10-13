"""
SQLite store for WasteCap annotation records.

This is the ONLY module that touches the database.
All other modules (app pages, CLI, export) must go through this interface.

Schema:
  records:    one row per AnnotationRecord; full JSON blob + indexed columns
  candidates: denormalised for fast querying (also in the JSON blob)

The JSON blob is the source of truth; the indexed columns are projections
for efficient filtering and statistics. On every write the blob is updated
and the projections are re-derived from it.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from wastecap.schema import AnnotationRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = "data/db/wastecap.db"


def _db_path() -> Path:
    raw = os.environ.get("WASTECAP_DB_PATH", DEFAULT_DB_PATH)
    p = Path(raw)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


@contextmanager
def _connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a connection with WAL mode and foreign keys enabled."""
    path = db_path or _db_path()
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

_CREATE_RECORDS = """
CREATE TABLE IF NOT EXISTS records (
    image_id    TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    split       TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',
    width       INTEGER NOT NULL DEFAULT 0,
    height      INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    blob        TEXT NOT NULL  -- full AnnotationRecord JSON
);
"""

_CREATE_CANDIDATES = """
CREATE TABLE IF NOT EXISTS candidates (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id    TEXT NOT NULL REFERENCES records(image_id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    prompt_id   TEXT NOT NULL,
    clip_score  REAL,
    parse_failed INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_records_source ON records(source);",
    "CREATE INDEX IF NOT EXISTS idx_records_split  ON records(split);",
    "CREATE INDEX IF NOT EXISTS idx_records_status ON records(status);",
    "CREATE INDEX IF NOT EXISTS idx_cand_image     ON candidates(image_id);",
    "CREATE INDEX IF NOT EXISTS idx_cand_model     ON candidates(model);",
]


def init_db(db_path: Path | None = None) -> Path:
    """Create tables if they do not exist. Returns the DB path.

    Safe to call multiple times (idempotent).
    """
    path = db_path or _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as conn:
        conn.execute(_CREATE_RECORDS)
        conn.execute(_CREATE_CANDIDATES)
        for idx_sql in _CREATE_INDEXES:
            conn.execute(idx_sql)
    logger.info("DB initialised at %s", path)
    return path


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _sync_candidates(conn: sqlite3.Connection, record: AnnotationRecord) -> None:
    """Replace the candidates rows for this image from the blob."""
    conn.execute("DELETE FROM candidates WHERE image_id = ?", (record.image_id,))
    for cand in record.candidates:
        conn.execute(
            """
            INSERT INTO candidates
                (image_id, model, prompt_id, clip_score, parse_failed, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.image_id,
                cand.model,
                cand.prompt_id,
                cand.clip_score,
                int(cand.parse_failed),
                cand.created_at.isoformat(),
            ),
        )


def ingest_record(record: AnnotationRecord, db_path: Path | None = None) -> None:
    """Insert a new record. Raises IntegrityError if image_id already exists.

    Call update_record to modify an existing record.
    """
    blob = record.model_dump_json()
    now = _now_iso()
    with _connect(db_path or _db_path()) as conn:
        conn.execute(
            """
            INSERT INTO records (image_id, source, split, status, width, height,
                                 created_at, updated_at, blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.image_id,
                record.source.value,
                record.split.value if record.split else None,
                record.status.value,
                record.width,
                record.height,
                now,
                now,
                blob,
            ),
        )
        _sync_candidates(conn, record)
    logger.debug("Ingested record %s", record.image_id)


def update_record(record: AnnotationRecord, db_path: Path | None = None) -> None:
    """Update an existing record. Raises KeyError if image_id not found."""
    blob = record.model_dump_json()
    now = _now_iso()
    with _connect(db_path or _db_path()) as conn:
        cur = conn.execute(
            """
            UPDATE records
            SET source=?, split=?, status=?, width=?, height=?, updated_at=?, blob=?
            WHERE image_id=?
            """,
            (
                record.source.value,
                record.split.value if record.split else None,
                record.status.value,
                record.width,
                record.height,
                now,
                blob,
                record.image_id,
            ),
        )
        if cur.rowcount == 0:
            raise KeyError(f"Record {record.image_id!r} not found in DB. Use ingest_record first.")
        _sync_candidates(conn, record)
    logger.debug("Updated record %s", record.image_id)


def upsert_record(record: AnnotationRecord, db_path: Path | None = None) -> None:
    """Insert or update a record (idempotent)."""
    path = db_path or _db_path()
    try:
        ingest_record(record, path)
    except sqlite3.IntegrityError:
        update_record(record, path)


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


def get_record(image_id: str, db_path: Path | None = None) -> AnnotationRecord | None:
    """Return the AnnotationRecord for image_id, or None if not found."""
    with _connect(db_path or _db_path()) as conn:
        row = conn.execute("SELECT blob FROM records WHERE image_id = ?", (image_id,)).fetchone()
    if row is None:
        return None
    return AnnotationRecord.model_validate_json(row["blob"])


def list_records(
    *,
    source: str | None = None,
    split: str | None = None,
    status: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    db_path: Path | None = None,
) -> list[AnnotationRecord]:
    """Return a filtered list of AnnotationRecords.

    All filters are optional; omitting them returns all records.
    """
    clauses: list[str] = []
    params: list[str | int] = []

    if source is not None:
        clauses.append("source = ?")
        params.append(source)
    if split is not None:
        clauses.append("split = ?")
        params.append(split)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
    offset_clause = f"OFFSET {int(offset)}" if offset else ""

    sql = (
        f"SELECT blob FROM records {where} ORDER BY created_at DESC {limit_clause} {offset_clause}"
    )

    with _connect(db_path or _db_path()) as conn:
        rows = conn.execute(sql, params).fetchall()

    return [AnnotationRecord.model_validate_json(row["blob"]) for row in rows]


def count_records(
    *,
    source: str | None = None,
    split: str | None = None,
    status: str | None = None,
    db_path: Path | None = None,
) -> int:
    """Return the count of records matching the filters."""
    clauses: list[str] = []
    params: list[str] = []

    if source is not None:
        clauses.append("source = ?")
        params.append(source)
    if split is not None:
        clauses.append("split = ?")
        params.append(split)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT COUNT(*) FROM records {where}"

    with _connect(db_path or _db_path()) as conn:
        row = conn.execute(sql, params).fetchone()
    return row[0] if row else 0


def delete_record(image_id: str, db_path: Path | None = None) -> None:
    """Hard-delete a record. Cascades to candidates.

    Use sparingly — prefer setting status='rejected' instead.
    """
    with _connect(db_path or _db_path()) as conn:
        conn.execute("DELETE FROM records WHERE image_id = ?", (image_id,))
    logger.warning("Hard-deleted record %s", image_id)


# ---------------------------------------------------------------------------
# Statistics helpers (used by stats.py)
# ---------------------------------------------------------------------------


def status_counts(db_path: Path | None = None) -> dict[str, int]:
    """Return {status: count} for all records."""
    with _connect(db_path or _db_path()) as conn:
        rows = conn.execute("SELECT status, COUNT(*) AS n FROM records GROUP BY status").fetchall()
    return {row["status"]: row["n"] for row in rows}


def source_counts(db_path: Path | None = None) -> dict[str, int]:
    """Return {source: count} for all records."""
    with _connect(db_path or _db_path()) as conn:
        rows = conn.execute("SELECT source, COUNT(*) AS n FROM records GROUP BY source").fetchall()
    return {row["source"]: row["n"] for row in rows}


def candidate_model_counts(db_path: Path | None = None) -> dict[str, int]:
    """Return {model: count} for all candidates."""
    with _connect(db_path or _db_path()) as conn:
        rows = conn.execute("SELECT model, COUNT(*) AS n FROM candidates GROUP BY model").fetchall()
    return {row["model"]: row["n"] for row in rows}
