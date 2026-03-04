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
    "default_model_id": "",  # legacy; migrated to default_model_id_t2v
    "default_model_id_t2v": "",
    "default_model_id_i2v": "",
    "default_model_id_t2i": "",
    "num_inference_steps": 50,
    "num_frames": 81,
    "height": 480,
    "width": 832,
    "guidance_scale": 5.0,
    "guidance_rescale": 0.0,
    "fps": 24,
    "seed": 1024,
    "num_gpus": 1,
    "dit_cpu_offload": 0,
    "text_encoder_cpu_offload": 0,
    "vae_cpu_offload": 0,
    "image_encoder_cpu_offload": 0,
    "use_fsdp_inference": 0,
    "enable_torch_compile": 0,
    "vsa_sparsity": 0.0,
    "tp_size": -1,
    "sp_size": -1,
    "auto_start_job": 0,
}


def _get_db_path(data_dir: Path) -> Path:
    """Return path to SQLite database file."""
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "fastvideo_ui.db"


def _get_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return set of column names for a table."""
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, col: str, sql_type: str, default: str
) -> None:
    """Add a column to the table if it does not exist."""
    cols = _get_table_columns(conn, table)
    if col not in cols:
        conn.execute(
            f"ALTER TABLE {table} ADD COLUMN {col} {sql_type} DEFAULT {default}"
        )


def _migrate_db(conn: sqlite3.Connection) -> None:
    """Add new columns to existing tables for schema migrations."""
    # Jobs table
    _add_column_if_missing(
        conn, "jobs", "vae_cpu_offload", "INTEGER", "0"
    )
    _add_column_if_missing(
        conn, "jobs", "image_encoder_cpu_offload", "INTEGER", "0"
    )
    _add_column_if_missing(
        conn, "jobs", "enable_torch_compile", "INTEGER", "0"
    )
    _add_column_if_missing(
        conn, "jobs", "vsa_sparsity", "REAL", "0.0"
    )
    _add_column_if_missing(
        conn, "jobs", "tp_size", "INTEGER", "-1"
    )
    _add_column_if_missing(
        conn, "jobs", "sp_size", "INTEGER", "-1"
    )
    _add_column_if_missing(
        conn, "jobs", "negative_prompt", "TEXT", "''"
    )
    _add_column_if_missing(
        conn, "jobs", "guidance_rescale", "REAL", "0.0"
    )
    _add_column_if_missing(
        conn, "jobs", "fps", "INTEGER", "24"
    )
    _add_column_if_missing(
        conn, "jobs", "workload_type", "TEXT", "'t2v'"
    )
    _add_column_if_missing(
        conn, "jobs", "image_path", "TEXT", "''"
    )
    _add_column_if_missing(
        conn, "jobs", "job_type", "TEXT", "'inference'"
    )
    _add_column_if_missing(conn, "jobs", "data_path", "TEXT", "''")
    _add_column_if_missing(conn, "jobs", "max_train_steps", "INTEGER", "1000")
    _add_column_if_missing(conn, "jobs", "train_batch_size", "INTEGER", "1")
    _add_column_if_missing(conn, "jobs", "learning_rate", "REAL", "5e-5")
    _add_column_if_missing(conn, "jobs", "num_latent_t", "INTEGER", "20")
    _add_column_if_missing(conn, "jobs", "validation_dataset_file", "TEXT", "''")
    _add_column_if_missing(conn, "jobs", "lora_rank", "INTEGER", "32")
    # Settings table
    _add_column_if_missing(
        conn, "settings", "vae_cpu_offload", "INTEGER", "0"
    )
    _add_column_if_missing(
        conn, "settings", "image_encoder_cpu_offload", "INTEGER", "0"
    )
    _add_column_if_missing(
        conn, "settings", "enable_torch_compile", "INTEGER", "0"
    )
    _add_column_if_missing(
        conn, "settings", "vsa_sparsity", "REAL", "0.0"
    )
    _add_column_if_missing(
        conn, "settings", "tp_size", "INTEGER", "-1"
    )
    _add_column_if_missing(
        conn, "settings", "sp_size", "INTEGER", "-1"
    )
    _add_column_if_missing(
        conn, "settings", "guidance_rescale", "REAL", "0.0"
    )
    _add_column_if_missing(
        conn, "settings", "fps", "INTEGER", "24"
    )
    _add_column_if_missing(
        conn, "settings", "default_model_id_t2v", "TEXT", "''"
    )
    _add_column_if_missing(
        conn, "settings", "default_model_id_i2v", "TEXT", "''"
    )
    _add_column_if_missing(
        conn, "settings", "default_model_id_t2i", "TEXT", "''"
    )
    _add_column_if_missing(
        conn, "settings", "auto_start_job", "INTEGER", "0"
    )
    # Migrate legacy default_model_id to default_model_id_t2v
    if "default_model_id_t2v" in _get_table_columns(conn, "settings"):
        conn.execute(
            """
            UPDATE settings SET default_model_id_t2v = default_model_id
            WHERE id = 1 AND default_model_id != ''
            AND (default_model_id_t2v IS NULL OR default_model_id_t2v = '')
            """
        )


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
                vae_cpu_offload INTEGER NOT NULL DEFAULT 0,
                image_encoder_cpu_offload INTEGER NOT NULL DEFAULT 0,
                use_fsdp_inference INTEGER NOT NULL DEFAULT 0,
                enable_torch_compile INTEGER NOT NULL DEFAULT 0,
                vsa_sparsity REAL NOT NULL DEFAULT 0.0,
                tp_size INTEGER NOT NULL DEFAULT -1,
                sp_size INTEGER NOT NULL DEFAULT -1
            );

            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                default_model_id TEXT NOT NULL DEFAULT '',
                default_model_id_t2v TEXT NOT NULL DEFAULT '',
                default_model_id_i2v TEXT NOT NULL DEFAULT '',
                default_model_id_t2i TEXT NOT NULL DEFAULT '',
                num_inference_steps INTEGER NOT NULL DEFAULT 50,
                num_frames INTEGER NOT NULL DEFAULT 81,
                height INTEGER NOT NULL DEFAULT 480,
                width INTEGER NOT NULL DEFAULT 832,
                guidance_scale REAL NOT NULL DEFAULT 5.0,
                seed INTEGER NOT NULL DEFAULT 1024,
                num_gpus INTEGER NOT NULL DEFAULT 1,
                dit_cpu_offload INTEGER NOT NULL DEFAULT 0,
                text_encoder_cpu_offload INTEGER NOT NULL DEFAULT 0,
                vae_cpu_offload INTEGER NOT NULL DEFAULT 0,
                image_encoder_cpu_offload INTEGER NOT NULL DEFAULT 0,
                use_fsdp_inference INTEGER NOT NULL DEFAULT 0,
                enable_torch_compile INTEGER NOT NULL DEFAULT 0,
                vsa_sparsity REAL NOT NULL DEFAULT 0.0,
                tp_size INTEGER NOT NULL DEFAULT -1,
                sp_size INTEGER NOT NULL DEFAULT -1,
                auto_start_job INTEGER NOT NULL DEFAULT 0
            );

            INSERT OR IGNORE INTO settings (id) VALUES (1);

            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                raw_path TEXT NOT NULL,
                output_path TEXT,
                workload_type TEXT NOT NULL DEFAULT 't2v',
                model_path TEXT NOT NULL,
                dataset_type TEXT NOT NULL DEFAULT 'merged',
                status TEXT NOT NULL DEFAULT 'pending',
                error TEXT,
                created_at REAL NOT NULL,
                num_gpus INTEGER NOT NULL DEFAULT 1,
                log_file_path TEXT
            );
        """)
        conn.commit()
        _migrate_db(conn)
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
                id, model_id, prompt, workload_type, image_path, job_type, status,
                created_at, started_at, finished_at, error, output_path, log_file_path,
                num_inference_steps, num_frames, height, width, guidance_scale,
                guidance_rescale, fps, seed, num_gpus, dit_cpu_offload,
                text_encoder_cpu_offload, vae_cpu_offload, image_encoder_cpu_offload,
                use_fsdp_inference, enable_torch_compile, vsa_sparsity, tp_size,
                sp_size, negative_prompt,
                data_path, max_train_steps, train_batch_size, learning_rate,
                num_latent_t, validation_dataset_file, lora_rank
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job["id"],
                job["model_id"],
                job["prompt"],
                job.get("workload_type", "t2v"),
                job.get("image_path", ""),
                job.get("job_type", "inference"),
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
                job.get("guidance_rescale", 0.0),
                job.get("fps", 24),
                job.get("seed", 1024),
                job.get("num_gpus", 1),
                1 if job.get("dit_cpu_offload") else 0,
                1 if job.get("text_encoder_cpu_offload") else 0,
                1 if job.get("vae_cpu_offload") else 0,
                1 if job.get("image_encoder_cpu_offload") else 0,
                1 if job.get("use_fsdp_inference") else 0,
                1 if job.get("enable_torch_compile") else 0,
                job.get("vsa_sparsity", 0.0),
                job.get("tp_size", -1),
                job.get("sp_size", -1),
                job.get("negative_prompt", ""),
                job.get("data_path", ""),
                job.get("max_train_steps", 1000),
                job.get("train_batch_size", 1),
                job.get("learning_rate", 5e-5),
                job.get("num_latent_t", 20),
                job.get("validation_dataset_file", ""),
                job.get("lora_rank", 32),
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

    def get_all_jobs(
        self, job_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Return all jobs, newest first. Optionally filter by job_type."""
        if job_type:
            cur = self._execute(
                "SELECT * FROM jobs WHERE job_type = ? ORDER BY created_at DESC",
                (job_type,),
            )
        else:
            cur = self._execute(
                "SELECT * FROM jobs ORDER BY created_at DESC"
            )
        return [_row_to_job(row) for row in cur.fetchall()]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get a single job by ID."""
        cur = self._execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cur.fetchone()
        return _row_to_job(row) if row else None

    # --- Datasets ---

    def insert_dataset(self, dataset: dict[str, Any]) -> None:
        """Insert a new dataset."""
        self._execute(
            """
            INSERT INTO datasets (
                id, name, raw_path, output_path, workload_type, model_path,
                dataset_type, status, error, created_at, num_gpus, log_file_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset["id"],
                dataset["name"],
                dataset["raw_path"],
                dataset.get("output_path"),
                dataset.get("workload_type", "t2v"),
                dataset["model_path"],
                dataset.get("dataset_type", "merged"),
                dataset.get("status", "pending"),
                dataset.get("error"),
                dataset["created_at"],
                dataset.get("num_gpus", 1),
                dataset.get("log_file_path"),
            ),
        )
        self._commit()

    def update_dataset(
        self, dataset_id: str, updates: dict[str, Any]
    ) -> None:
        """Update dataset fields. Only provided keys are updated."""
        if not updates:
            return
        allowed = {
            "status", "error", "output_path", "log_file_path",
        }
        cols = []
        vals = []
        for k, v in updates.items():
            if k in allowed:
                cols.append(f"{k} = ?")
                vals.append(v)
        if not cols:
            return
        vals.append(dataset_id)
        sql = f"UPDATE datasets SET {', '.join(cols)} WHERE id = ?"
        self._execute(sql, tuple(vals))
        self._commit()

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset. Returns True if a row was deleted."""
        cur = self._execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        self._commit()
        return cur.rowcount > 0

    def get_all_datasets(
        self, status: str | None = None
    ) -> list[dict[str, Any]]:
        """Return all datasets, newest first. Optionally filter by status."""
        if status:
            cur = self._execute(
                "SELECT * FROM datasets WHERE status = ? ORDER BY created_at DESC",
                (status,),
            )
        else:
            cur = self._execute(
                "SELECT * FROM datasets ORDER BY created_at DESC"
            )
        return [_row_to_dataset(row) for row in cur.fetchall()]

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        """Get a single dataset by ID."""
        cur = self._execute(
            "SELECT * FROM datasets WHERE id = ?", (dataset_id,)
        )
        row = cur.fetchone()
        return _row_to_dataset(row) if row else None

    # --- Settings ---

    def get_settings(self) -> dict[str, Any]:
        """Return current settings as a dict (camelCase keys for API)."""
        cur = self._execute("SELECT * FROM settings WHERE id = 1")
        row = cur.fetchone()
        if not row:
            return _default_settings_dict()
        t2v = (
            (row["default_model_id_t2v"] or row["default_model_id"] or "")
            if "default_model_id_t2v" in row.keys()
            else (row["default_model_id"] or "")
        )
        i2v = (
            (row["default_model_id_i2v"] or "")
            if "default_model_id_i2v" in row.keys()
            else ""
        )
        t2i = (
            (row["default_model_id_t2i"] or "")
            if "default_model_id_t2i" in row.keys()
            else ""
        )
        result = {
            "defaultModelId": row["default_model_id"] or "",
            "defaultModelIdT2v": t2v,
            "defaultModelIdI2v": i2v,
            "defaultModelIdT2i": t2i,
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
        # New columns (may not exist in older DBs)
        for col, key, default in [
            ("vae_cpu_offload", "vaeCpuOffload", False),
            ("image_encoder_cpu_offload", "imageEncoderCpuOffload", False),
            ("enable_torch_compile", "enableTorchCompile", False),
            ("vsa_sparsity", "vsaSparsity", 0.0),
            ("tp_size", "tpSize", -1),
            ("sp_size", "spSize", -1),
            ("guidance_rescale", "guidanceRescale", 0.0),
            ("fps", "fps", 24),
            ("auto_start_job", "autoStartJob", False),
        ]:
            if col in row.keys():
                v = row[col]
                if isinstance(default, bool):
                    result[key] = bool(v)
                elif isinstance(default, float):
                    result[key] = float(v)
                else:
                    result[key] = int(v)
            else:
                result[key] = default
        return result

    def save_settings(self, settings: dict[str, Any]) -> None:
        """Save settings. Accepts camelCase keys from API."""
        # Map camelCase -> snake_case
        mapping = {
            "defaultModelId": "default_model_id",
            "defaultModelIdT2v": "default_model_id_t2v",
            "defaultModelIdI2v": "default_model_id_i2v",
            "defaultModelIdT2i": "default_model_id_t2i",
            "numInferenceSteps": "num_inference_steps",
            "numFrames": "num_frames",
            "height": "height",
            "width": "width",
            "guidanceScale": "guidance_scale",
            "guidanceRescale": "guidance_rescale",
            "fps": "fps",
            "seed": "seed",
            "numGpus": "num_gpus",
            "ditCpuOffload": "dit_cpu_offload",
            "textEncoderCpuOffload": "text_encoder_cpu_offload",
            "vaeCpuOffload": "vae_cpu_offload",
            "imageEncoderCpuOffload": "image_encoder_cpu_offload",
            "useFsdpInference": "use_fsdp_inference",
            "enableTorchCompile": "enable_torch_compile",
            "vsaSparsity": "vsa_sparsity",
            "tpSize": "tp_size",
            "spSize": "sp_size",
            "autoStartJob": "auto_start_job",
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


def _row_to_dataset(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a DB row to dataset dict."""
    return {
        "id": row["id"],
        "name": row["name"],
        "raw_path": row["raw_path"],
        "output_path": row["output_path"] or "",
        "workload_type": row["workload_type"] or "t2v",
        "model_path": row["model_path"],
        "dataset_type": row["dataset_type"] or "merged",
        "status": row["status"] or "pending",
        "error": row["error"],
        "created_at": row["created_at"],
        "num_gpus": row["num_gpus"] or 1,
        "log_file_path": row["log_file_path"] or "",
    }


def _row_to_job(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a DB row to job dict (snake_case, matching Job.to_dict)."""
    result = {
        "id": row["id"],
        "model_id": row["model_id"],
        "prompt": row["prompt"],
        "workload_type": (
            row["workload_type"] if "workload_type" in row.keys() else "t2v"
        ),
        "image_path": (
            (row["image_path"] or "") if "image_path" in row.keys() else ""
        ),
        "job_type": (
            row["job_type"] if "job_type" in row.keys() else "inference"
        ),
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
        "guidance_rescale": (
            float(row["guidance_rescale"])
            if "guidance_rescale" in row.keys()
            else 0.0
        ),
        "fps": int(row["fps"]) if "fps" in row.keys() else 24,
        "seed": row["seed"],
        "num_gpus": row["num_gpus"],
        "dit_cpu_offload": bool(row["dit_cpu_offload"]),
        "text_encoder_cpu_offload": bool(row["text_encoder_cpu_offload"]),
        "use_fsdp_inference": bool(row["use_fsdp_inference"]),
        "negative_prompt": (
            (row["negative_prompt"] or "") if "negative_prompt" in row.keys()
            else ""
        ),
        "progress": 0.0,
        "progress_msg": "",
        "phase": "initializing",
    }
    for col in ("vae_cpu_offload", "image_encoder_cpu_offload", "enable_torch_compile"):
        if col in row.keys():
            result[col] = bool(row[col])
    for col in ("vsa_sparsity",):
        if col in row.keys():
            result[col] = float(row[col])
    for col in ("tp_size", "sp_size"):
        if col in row.keys():
            result[col] = int(row[col])
    for col in (
        "data_path",
        "validation_dataset_file",
    ):
        if col in row.keys():
            result[col] = (row[col] or "") or ""
    for col in (
        "max_train_steps",
        "train_batch_size",
        "num_latent_t",
        "lora_rank",
    ):
        if col in row.keys():
            result[col] = int(row[col]) if row[col] is not None else (
                1000 if col == "max_train_steps" else
                1 if col == "train_batch_size" else
                20 if col == "num_latent_t" else 32
            )
    if "learning_rate" in row.keys():
        result["learning_rate"] = float(row["learning_rate"] or 5e-5)
    return result


def _default_settings_dict() -> dict[str, Any]:
    """Return default settings as API-style dict."""
    return {
        "defaultModelId": DEFAULT_SETTINGS["default_model_id"],
        "defaultModelIdT2v": DEFAULT_SETTINGS["default_model_id_t2v"],
        "defaultModelIdI2v": DEFAULT_SETTINGS["default_model_id_i2v"],
        "defaultModelIdT2i": DEFAULT_SETTINGS["default_model_id_t2i"],
        "numInferenceSteps": DEFAULT_SETTINGS["num_inference_steps"],
        "numFrames": DEFAULT_SETTINGS["num_frames"],
        "height": DEFAULT_SETTINGS["height"],
        "width": DEFAULT_SETTINGS["width"],
        "guidanceScale": DEFAULT_SETTINGS["guidance_scale"],
        "seed": DEFAULT_SETTINGS["seed"],
        "numGpus": DEFAULT_SETTINGS["num_gpus"],
        "ditCpuOffload": bool(DEFAULT_SETTINGS["dit_cpu_offload"]),
        "textEncoderCpuOffload": bool(DEFAULT_SETTINGS["text_encoder_cpu_offload"]),
        "vaeCpuOffload": bool(DEFAULT_SETTINGS["vae_cpu_offload"]),
        "imageEncoderCpuOffload": bool(DEFAULT_SETTINGS["image_encoder_cpu_offload"]),
        "useFsdpInference": bool(DEFAULT_SETTINGS["use_fsdp_inference"]),
        "enableTorchCompile": bool(DEFAULT_SETTINGS["enable_torch_compile"]),
        "vsaSparsity": float(DEFAULT_SETTINGS["vsa_sparsity"]),
        "tpSize": int(DEFAULT_SETTINGS["tp_size"]),
        "spSize": int(DEFAULT_SETTINGS["sp_size"]),
        "guidanceRescale": float(DEFAULT_SETTINGS["guidance_rescale"]),
        "fps": int(DEFAULT_SETTINGS["fps"]),
        "autoStartJob": bool(DEFAULT_SETTINGS["auto_start_job"]),
    }
