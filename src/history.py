import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional


class HistoryRepository:
    def __init__(self, db_path: str = "history.db") -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    timestamp TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    top_label TEXT NOT NULL,
                    top_confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    top_predictions TEXT
                )
                """
            )
            self._ensure_columns(conn)
            self._ensure_indexes(conn)
            conn.commit()

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(prediction_history)").fetchall()
        }
        if "top_predictions" not in columns:
            conn.execute(
                "ALTER TABLE prediction_history ADD COLUMN top_predictions TEXT"
            )
        if "user_id" not in columns:
            conn.execute(
                "ALTER TABLE prediction_history ADD COLUMN user_id INTEGER"
            )

    def _ensure_indexes(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_prediction_history_timestamp
            ON prediction_history(timestamp)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_prediction_history_source
            ON prediction_history(source)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_prediction_history_top_label
            ON prediction_history(top_label)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_prediction_history_user_id
            ON prediction_history(user_id)
            """
        )

    def add(
        self,
        *,
        timestamp: str,
        filename: str,
        top_label: str,
        top_confidence: float,
        source: str,
        top_predictions: Optional[List[Dict[str, float]]] = None,
        user_id: Optional[int] = None,
    ) -> None:
        predictions_json = None
        if top_predictions is not None:
            predictions_json = json.dumps(top_predictions, ensure_ascii=True)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO prediction_history
                (user_id, timestamp, filename, top_label, top_confidence, source, top_predictions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    timestamp,
                    filename,
                    top_label,
                    top_confidence,
                    source,
                    predictions_json,
                ),
            )
            conn.commit()

    def list_recent(
        self,
        limit: int = 200,
        offset: int = 0,
        source: Optional[str] = None,
        label: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        include_predictions: bool = False,
        user_id: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        where_parts = []
        params: List[object] = []

        if source:
            where_parts.append("source = ?")
            params.append(source)
        if label:
            where_parts.append("top_label LIKE ?")
            params.append(f"%{label}%")
        if date_from:
            where_parts.append("timestamp >= ?")
            params.append(date_from)
        if date_to:
            where_parts.append("timestamp <= ?")
            params.append(date_to)
        if user_id is not None:
            where_parts.append("user_id = ?")
            params.append(user_id)

        where_clause = ""
        if where_parts:
            where_clause = "WHERE " + " AND ".join(where_parts)

        select_columns = "timestamp, filename, top_label, top_confidence, source"
        if include_predictions:
            select_columns += ", top_predictions"

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT {select_columns}
                FROM prediction_history
                {where_clause}
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()

        result = []
        for row in rows:
            item = {
                "timestamp": row[0],
                "filename": row[1],
                "top_label": row[2],
                "top_confidence": f"{row[3]:.2f}",
                "source": row[4],
            }
            if include_predictions:
                item["top_predictions"] = (
                    json.loads(row[5]) if row[5] else []
                )
            result.append(item)
        return result

    def count(
        self,
        source: Optional[str] = None,
        label: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> int:
        where_parts = []
        params: List[object] = []

        if source:
            where_parts.append("source = ?")
            params.append(source)
        if label:
            where_parts.append("top_label LIKE ?")
            params.append(f"%{label}%")
        if date_from:
            where_parts.append("timestamp >= ?")
            params.append(date_from)
        if date_to:
            where_parts.append("timestamp <= ?")
            params.append(date_to)
        if user_id is not None:
            where_parts.append("user_id = ?")
            params.append(user_id)

        where_clause = ""
        if where_parts:
            where_clause = "WHERE " + " AND ".join(where_parts)

        with self._connect() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM prediction_history {where_clause}",
                params,
            ).fetchone()
        return int(row[0]) if row else 0

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM prediction_history")
            conn.commit()
