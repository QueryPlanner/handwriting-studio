"""Durable jobs and atomic claims. No upload index is exposed through the API."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


def now() -> str:
    return datetime.now(UTC).isoformat()


class QueueFull(ValueError):
    """Admission failed without changing the job."""


class Store:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.row_factory = sqlite3.Row
            with db:
                yield db
        finally:
            db.close()

    def initialize(self):
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
                    stage TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    options TEXT NOT NULL, result TEXT NOT NULL DEFAULT '{}',
                    error TEXT
                )
            """)

    def create(self, job_id: str, name: str, options: dict, max_pending: int):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','processing')"
            ).fetchone()[0]
            if count >= max_pending:
                raise QueueFull("The conversion queue is full. Try again later.")
            timestamp = now()
            db.execute(
                "INSERT INTO jobs (id,name,status,stage,created_at,updated_at,options) "
                "VALUES (?,?,'queued','analyze',?,?,?)",
                (job_id, name, timestamp, timestamp, json.dumps(options)),
            )

    @staticmethod
    def decode(row):
        if row is None:
            return None
        value = dict(row)
        value["options"] = json.loads(value["options"])
        value["result"] = json.loads(value["result"])
        return value

    def get(self, job_id: str):
        with self.connection() as db:
            return self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def recover(self):
        # A data-directory lock guarantees no other worker is using these claims.
        with self.connection() as db:
            db.execute(
                "UPDATE jobs SET status='queued', updated_at=? WHERE status='processing'", (now(),)
            )

    def claim(self):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at,id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE jobs SET status='processing',updated_at=? WHERE id=?", (now(), row["id"])
            )
            job = self.decode(row)
            job["status"] = "processing"
            return job

    def finish(self, job_id: str, status: str, stage: str, result: dict, error: str | None = None):
        with self.connection() as db:
            db.execute(
                "UPDATE jobs SET status=?,stage=?,result=?,error=?,updated_at=? WHERE id=?",
                (status, stage, json.dumps(result), error, now(), job_id),
            )

    def edit(self, job_id: str, glyphs: list[dict], ink_mode: str):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            job = self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())
            if job["status"] in ("queued", "processing"):
                raise ValueError("Wait for the current operation before editing crops.")
            options = job["options"] | {"ink_mode": ink_mode, "auto_generate": False}
            result = {
                "glyphs": glyphs,
                "mapping_errors": [],
                "ink_mode": ink_mode,
                "warnings": job["result"].get("warnings", []),
            }
            db.execute(
                "UPDATE jobs SET options=?,result=?,revision=revision+1,status='review_required',"
                "stage='generate',error=NULL,updated_at=? WHERE id=?",
                (json.dumps(options), json.dumps(result), now(), job_id),
            )

    def queue_generation(self, job_id: str, max_pending: int = 20):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            job = self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())
            if job["status"] not in ("review_required", "failed"):
                raise ValueError("Only reviewed or failed jobs can be generated.")
            if not job["result"].get("glyphs") or job["result"].get("mapping_errors"):
                raise ValueError("Correct the glyph mapping before generating the font.")
            count = db.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','processing')"
            ).fetchone()[0]
            if count >= max_pending:
                raise QueueFull("The conversion queue is full. Try again later.")
            db.execute(
                "UPDATE jobs SET status='queued',stage='generate',error=NULL,"
                "updated_at=? WHERE id=?",
                (now(), job_id),
            )
