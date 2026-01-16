"""SQLite database for experiments."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .models import ExperimentStatus, generate_id


# Default database path
DEFAULT_DB_PATH = Path.home() / ".ipol_runner" / "experiments.db"


class Database:
    """SQLite database manager for experiments."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # Fall back to temp directory if home directory is not writable
            import tempfile
            fallback_dir = Path(tempfile.gettempdir()) / "ipol_runner"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = fallback_dir / "experiments.db"
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    method_name TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    execution_time REAL,
                    parameters TEXT,
                    inputs TEXT,
                    outputs TEXT,
                    primary_output TEXT,
                    error_message TEXT,
                    notes TEXT DEFAULT ''
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS uploads (
                    id TEXT PRIMARY KEY,
                    original_filename TEXT,
                    file_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS comparisons (
                    id TEXT PRIMARY KEY,
                    experiment_ids TEXT,
                    notes TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    def create_experiment(
        self,
        method_name: str,
        input_ids: List[str],
        parameters: Dict[str, Any]
    ) -> str:
        """Create a new experiment.

        Returns:
            Experiment ID
        """
        exp_id = generate_id()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO experiments (id, method_name, parameters, inputs, status)
                VALUES (?, ?, ?, ?, 'pending')
            """, (exp_id, method_name, json.dumps(parameters), json.dumps(input_ids)))
            conn.commit()
        return exp_id

    def update_experiment_status(
        self,
        exp_id: str,
        status: str,
        outputs: Optional[Dict[str, str]] = None,
        primary_output: Optional[str] = None,
        execution_time: Optional[float] = None,
        error_message: Optional[str] = None
    ):
        """Update experiment status and results."""
        with sqlite3.connect(self.db_path) as conn:
            updates = ["status = ?"]
            values = [status]

            if status in ("completed", "failed"):
                updates.append("completed_at = ?")
                values.append(datetime.now().isoformat())

            if outputs is not None:
                updates.append("outputs = ?")
                values.append(json.dumps(outputs))

            if primary_output is not None:
                updates.append("primary_output = ?")
                values.append(primary_output)

            if execution_time is not None:
                updates.append("execution_time = ?")
                values.append(execution_time)

            if error_message is not None:
                updates.append("error_message = ?")
                values.append(error_message)

            values.append(exp_id)
            conn.execute(
                f"UPDATE experiments SET {', '.join(updates)} WHERE id = ?",
                values
            )
            conn.commit()

    def get_experiment(self, exp_id: str) -> Optional[ExperimentStatus]:
        """Get experiment by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?",
                (exp_id,)
            ).fetchone()

            if not row:
                return None

            return ExperimentStatus(
                id=row["id"],
                method_name=row["method_name"],
                status=row["status"],
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
                completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                execution_time=row["execution_time"],
                parameters=json.loads(row["parameters"]) if row["parameters"] else {},
                inputs=json.loads(row["inputs"]) if row["inputs"] else [],
                outputs=json.loads(row["outputs"]) if row["outputs"] else {},
                primary_output=row["primary_output"],
                error_message=row["error_message"],
                notes=row["notes"] or ""
            )

    def list_experiments(
        self,
        method_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[ExperimentStatus]:
        """List experiments with optional filters."""
        query = "SELECT * FROM experiments"
        conditions = []
        values = []

        if method_name:
            conditions.append("method_name = ?")
            values.append(method_name)
        if status:
            conditions.append("status = ?")
            values.append(status)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ?"
        values.append(limit)

        experiments = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute(query, values):
                experiments.append(ExperimentStatus(
                    id=row["id"],
                    method_name=row["method_name"],
                    status=row["status"],
                    created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
                    completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                    execution_time=row["execution_time"],
                    parameters=json.loads(row["parameters"]) if row["parameters"] else {},
                    inputs=json.loads(row["inputs"]) if row["inputs"] else [],
                    outputs=json.loads(row["outputs"]) if row["outputs"] else {},
                    primary_output=row["primary_output"],
                    error_message=row["error_message"],
                    notes=row["notes"] or ""
                ))
        return experiments

    def update_experiment_notes(self, exp_id: str, notes: str):
        """Update experiment notes."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE experiments SET notes = ? WHERE id = ?",
                (notes, exp_id)
            )
            conn.commit()

    def save_upload(self, upload_id: str, filename: str, file_path: str):
        """Save upload record."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO uploads (id, original_filename, file_path)
                VALUES (?, ?, ?)
            """, (upload_id, filename, file_path))
            conn.commit()

    def get_upload(self, upload_id: str) -> Optional[Dict[str, str]]:
        """Get upload by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM uploads WHERE id = ?",
                (upload_id,)
            ).fetchone()

            if not row:
                return None

            return {
                "id": row["id"],
                "filename": row["original_filename"],
                "path": row["file_path"]
            }

    def delete_experiment(self, exp_id: str):
        """Delete an experiment."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
            conn.commit()
