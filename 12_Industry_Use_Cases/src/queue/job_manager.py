import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Optional


class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    job_type: str
    params: dict
    status: JobStatus
    progress: float
    result_path: Optional[str]
    error_message: Optional[str]
    created_at: datetime


class JobManager:
    def __init__(self, db_path: str = "data/jobs.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                params TEXT NOT NULL,
                status TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0.0,
                result_path TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

    def submit_job(self, job_type: str, params: dict) -> str:
        job_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO jobs (id, job_type, params, status, progress, result_path, error_message, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                job_type,
                json.dumps(params),
                JobStatus.QUEUED.value,
                0.0,
                None,
                None,
                created_at,
            ),
        )
        conn.commit()
        conn.close()

        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, job_type, params, status, progress, result_path, error_message, created_at
            FROM jobs WHERE id = ?
            """,
            (job_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return self._row_to_job(row)

    def update_progress(self, job_id: str, progress: float) -> None:
        clamped_progress = max(0.0, min(1.0, progress))

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE jobs SET progress = ? WHERE id = ?",
            (clamped_progress, job_id),
        )
        conn.commit()
        conn.close()

    def start_job(self, job_id: str) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE jobs SET status = ? WHERE id = ?",
            (JobStatus.RUNNING.value, job_id),
        )
        conn.commit()
        conn.close()

    def complete_job(self, job_id: str, result_path: str) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE jobs SET status = ?, progress = 1.0, result_path = ? WHERE id = ?",
            (JobStatus.COMPLETED.value, result_path, job_id),
        )
        conn.commit()
        conn.close()

    def fail_job(self, job_id: str, error_message: str) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE jobs SET status = ?, error_message = ? WHERE id = ?",
            (JobStatus.FAILED.value, error_message, job_id),
        )
        conn.commit()
        conn.close()

    def list_jobs(self, limit: int = 100) -> list[Job]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, job_type, params, status, progress, result_path, error_message, created_at
            FROM jobs ORDER BY created_at DESC LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_job(row) for row in rows]

    def _row_to_job(self, row: tuple) -> Job:
        return Job(
            id=row[0],
            job_type=row[1],
            params=json.loads(row[2]),
            status=JobStatus(row[3]),
            progress=row[4],
            result_path=row[5],
            error_message=row[6],
            created_at=datetime.fromisoformat(row[7]),
        )
