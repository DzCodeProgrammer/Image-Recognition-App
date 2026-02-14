import sqlite3
from pathlib import Path
from typing import Dict, List


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
                    timestamp TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    top_label TEXT NOT NULL,
                    top_confidence REAL NOT NULL,
                    source TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def add(
        self,
        *,
        timestamp: str,
        filename: str,
        top_label: str,
        top_confidence: float,
        source: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO prediction_history
                (timestamp, filename, top_label, top_confidence, source)
                VALUES (?, ?, ?, ?, ?)
                """,
                (timestamp, filename, top_label, top_confidence, source),
            )
            conn.commit()

    def list_recent(self, limit: int = 200) -> List[Dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, filename, top_label, top_confidence, source
                FROM prediction_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            {
                "timestamp": row[0],
                "filename": row[1],
                "top_label": row[2],
                "top_confidence": f"{row[3]:.2f}",
                "source": row[4],
            }
            for row in rows
        ]

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM prediction_history")
            conn.commit()
