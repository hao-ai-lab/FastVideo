# SPDX-License-Identifier: Apache-2.0
"""
SQLite persistence for FastVideo UI.

Stores jobs and default settings. Uses Python's built-in sqlite3.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("fastvideo.ui.database")

# Default options schema - used for settings table defaults
DEFAULT_SETTINGS = {
    "default_model_id": "",
    "num_inference_steps": 50,
    "num_frames": 81,
    "height": 480,
    "width": 832,
    "guidance_scale": 5.0,
    "seed": 1024,
    "num_gpus": 1,
    "dit_cpu_offload": 0,
    "text_encoder_cpu_offload": 0,
    "use_fsdp_inference": 0,
}


def _get_db_path(data_dir: Path) -> Path:
    """Return path to SQLite database file."""
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "fastvideo_ui.db"


def init_db(db_path: Path) -> None:
    """Create database and tables if they do not exist."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at REAL NOT NULL,
                started_at REAL,
                finished_at REAL,
                error TEXT,
                output_path TEXT,
                log_file_path TEXT,
                num_inference_steps INTEGER NOT NULL DEFAULT 50,
                num_frames INTEGER NOT NULL DEFAULT 81,
                height INTEGER NOT NULL DEFAULT 480,
                width INTEGER NOT NULL DEFAULT 832,
                guidance_scale REAL NOT NULL DEFAULT 5.0,
                seed INTEGER NOT NULL DEFAULT 1024,
                num_gpus INTEGER NOT NULL DEFAULT 1,
                dit_cpu_offload INTEGER NOT NULL DEFAULT 0,
                text_encoder_cpu_offload INTEGER NOT NULL DEFAULT 0,
                use_fsdp_inference INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                default_model_id TEXT NOT NULL DEFAULT '',
                num_inference_steps INTEGER NOT NULL DEFAULT 50,
                num_frames INTEGER NOT NULL DEFAULT 81,
                height INTEGER NOT NULL DEFAULT 480,
                width INTEGER NOT NULL DEFAULT 832,
                guidance_scale REAL NOT NULL DEFAULT 5.0,
                seed INTEGER NOT NULL DEFAULT 1024,
                num_gpus INTEGER NOT NULL DEFAULT 1,
                dit_cpu_offload INTEGER NOT NULL DEFAULT 0,
                text_encoder_cpu_offload INTEGER NOT NULL DEFAULT 0,
                use_fsdp_inference INTEGER NOT NULL DEFAULT 0
            );

            INSERT OR IGNORE INTO settings (id) VALUES (1);
        """)
        conn.commit()
    finally:
        conn.close()


class Database:
    """Thread-safe SQLite access for jobs and settings."""

    def __init__(self, db_path: Path):
        self._path = db_path
        self._local = threading.local()
        init_db(db_path)

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        return self._conn().execute(sql, params)

    def _commit(self) -> None:
        self._conn().commit()

    # --- Jobs ---

    def insert_job(self, job: dict[str, Any]) -> None:
        """Insert a new job."""
        self._execute(
            """
            INSERT INTO jobs (
                id, model_id, prompt, status, created_at, started_at, finished_at,
                error, output_path, log_file_path, num_inference_steps, num_frames,
                height, width, guidance_scale, seed, num_gpus,
                dit_cpu_offload, text_encoder_cpu_offload, use_fsdp_inference
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job["id"],
                job["model_id"],
                job["prompt"],
                job["status"],
                job["created_at"],
                job.get("started_at"),
                job.get("finished_at"),
                job.get("error"),
                job.get("output_path"),
                job.get("log_file_path"),
                job.get("num_inference_steps", 50),
                job.get("num_frames", 81),
                job.get("height", 480),
                job.get("width", 832),
                job.get("guidance_scale", 5.0),
                job.get("seed", 1024),
                job.get("num_gpus", 1),
                1 if job.get("dit_cpu_offload") else 0,
                1 if job.get("text_encoder_cpu_offload") else 0,
                1 if job.get("use_fsdp_inference") else 0,
            ),
        )
        self._commit()

    def update_job(self, job_id: str, updates: dict[str, Any]) -> None:
        """Update job fields. Only provided keys are updated."""
        if not updates:
            return
        allowed = {
            "status", "started_at", "finished_at", "error",
            "output_path", "log_file_path",
        }
        cols = []
        vals = []
        for k, v in updates.items():
            if k in allowed:
                cols.append(f"{k} = ?")
                vals.append(v)
        if not cols:
            return
        vals.append(job_id)
        sql = f"UPDATE jobs SET {', '.join(cols)} WHERE id = ?"
        self._execute(sql, tuple(vals))
        self._commit()

    def delete_job(self, job_id: str) -> bool:
        """Delete a job. Returns True if a row was deleted."""
        cur = self._execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        self._commit()
        return cur.rowcount > 0

    def get_all_jobs(self) -> list[dict[str, Any]]:
        """Return all jobs, newest first."""
        cur = self._execute(
            "SELECT * FROM jobs ORDER BY created_at DESC"
        )
        return [_row_to_job(row) for row in cur.fetchall()]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get a single job by ID."""
        cur = self._execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cur.fetchone()
        return _row_to_job(row) if row else None

    # --- Settings ---

    def get_settings(self) -> dict[str, Any]:
        """Return current settings as a dict (camelCase keys for API)."""
        cur = self._execute("SELECT * FROM settings WHERE id = 1")
        row = cur.fetchone()
        if not row:
            return _default_settings_dict()
        return {
            "defaultModelId": row["default_model_id"] or "",
            "numInferenceSteps": int(row["num_inference_steps"]),
            "numFrames": int(row["num_frames"]),
            "height": int(row["height"]),
            "width": int(row["width"]),
            "guidanceScale": float(row["guidance_scale"]),
            "seed": int(row["seed"]),
            "numGpus": int(row["num_gpus"]),
            "ditCpuOffload": bool(row["dit_cpu_offload"]),
            "textEncoderCpuOffload": bool(row["text_encoder_cpu_offload"]),
            "useFsdpInference": bool(row["use_fsdp_inference"]),
        }

    def save_settings(self, settings: dict[str, Any]) -> None:
        """Save settings. Accepts camelCase keys from API."""
        # Map camelCase -> snake_case
        mapping = {
            "defaultModelId": "default_model_id",
            "numInferenceSteps": "num_inference_steps",
            "numFrames": "num_frames",
            "height": "height",
            "width": "width",
            "guidanceScale": "guidance_scale",
            "seed": "seed",
            "numGpus": "num_gpus",
            "ditCpuOffload": "dit_cpu_offload",
            "textEncoderCpuOffload": "text_encoder_cpu_offload",
            "useFsdpInference": "use_fsdp_inference",
        }
        updates = []
        params = []
        for api_key, db_col in mapping.items():
            if api_key in settings:
                v = settings[api_key]
                if isinstance(v, bool):
                    v = 1 if v else 0
                updates.append(f"{db_col} = ?")
                params.append(v)
        if not updates:
            return
        params.append(1)
        sql = f"UPDATE settings SET {', '.join(updates)} WHERE id = ?"
        self._execute(sql, tuple(params))
        self._commit()


def _row_to_job(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a DB row to job dict (snake_case, matching Job.to_dict)."""
    return {
        "id": row["id"],
        "model_id": row["model_id"],
        "prompt": row["prompt"],
        "status": row["status"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "error": row["error"],
        "output_path": row["output_path"],
        "log_file_path": row["log_file_path"],
        "num_inference_steps": row["num_inference_steps"],
        "num_frames": row["num_frames"],
        "height": row["height"],
        "width": row["width"],
        "guidance_scale": row["guidance_scale"],
        "seed": row["seed"],
        "num_gpus": row["num_gpus"],
        "dit_cpu_offload": bool(row["dit_cpu_offload"]),
        "text_encoder_cpu_offload": bool(row["text_encoder_cpu_offload"]),
        "use_fsdp_inference": bool(row["use_fsdp_inference"]),
        "progress": 0.0,
        "progress_msg": "",
        "phase": "initializing",
    }


def _default_settings_dict() -> dict[str, Any]:
    """Return default settings as API-style dict."""
    return {
        "defaultModelId": DEFAULT_SETTINGS["default_model_id"],
        "numInferenceSteps": DEFAULT_SETTINGS["num_inference_steps"],
        "numFrames": DEFAULT_SETTINGS["num_frames"],
        "height": DEFAULT_SETTINGS["height"],
        "width": DEFAULT_SETTINGS["width"],
        "guidanceScale": DEFAULT_SETTINGS["guidance_scale"],
        "seed": DEFAULT_SETTINGS["seed"],
        "numGpus": DEFAULT_SETTINGS["num_gpus"],
        "ditCpuOffload": bool(DEFAULT_SETTINGS["dit_cpu_offload"]),
        "textEncoderCpuOffload": bool(DEFAULT_SETTINGS["text_encoder_cpu_offload"]),
        "useFsdpInference": bool(DEFAULT_SETTINGS["use_fsdp_inference"]),
    }
