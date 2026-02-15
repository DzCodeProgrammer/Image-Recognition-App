from pathlib import Path
import sqlite3
import json

from src.history import HistoryRepository


def test_history_repository_insert_and_list(tmp_path: Path):
    db_path = tmp_path / "history.db"
    repo = HistoryRepository(str(db_path))

    repo.add(
        timestamp="2026-02-14T10:00:00",
        filename="sample.jpg",
        top_label="cat",
        top_confidence=88.55,
        source="test",
        top_predictions=[
            {"label": "cat", "confidence": 88.55},
            {"label": "dog", "confidence": 11.45},
        ],
    )

    rows = repo.list_recent(limit=10)
    assert len(rows) == 1
    assert rows[0]["filename"] == "sample.jpg"
    assert rows[0]["top_label"] == "cat"
    assert "top_predictions" not in rows[0]

    rows_with_predictions = repo.list_recent(limit=10, include_predictions=True)
    assert rows_with_predictions[0]["top_predictions"][0]["label"] == "cat"

    with sqlite3.connect(db_path) as conn:
        raw = conn.execute(
            "SELECT top_predictions FROM prediction_history LIMIT 1"
        ).fetchone()
    assert raw is not None
    parsed = json.loads(raw[0])
    assert parsed[0]["label"] == "cat"


def test_history_repository_clear(tmp_path: Path):
    db_path = tmp_path / "history.db"
    repo = HistoryRepository(str(db_path))

    repo.add(
        timestamp="2026-02-14T10:00:00",
        filename="sample.jpg",
        top_label="cat",
        top_confidence=88.55,
        source="test",
    )
    repo.clear()

    rows = repo.list_recent(limit=10)
    assert rows == []


def test_history_repository_filters_and_count(tmp_path: Path):
    db_path = tmp_path / "history.db"
    repo = HistoryRepository(str(db_path))

    repo.add(
        timestamp="2026-02-14T10:00:00",
        filename="sample1.jpg",
        top_label="cat",
        top_confidence=88.55,
        source="api",
        user_id=1,
    )
    repo.add(
        timestamp="2026-02-14T10:05:00",
        filename="sample2.jpg",
        top_label="dog",
        top_confidence=77.10,
        source="streamlit",
        user_id=2,
    )
    repo.add(
        timestamp="2026-02-14T10:10:00",
        filename="sample3.jpg",
        top_label="catfish",
        top_confidence=66.10,
        source="api",
        user_id=1,
    )

    rows = repo.list_recent(limit=10, source="api", label="cat")
    assert len(rows) == 2
    assert repo.count(source="api", label="cat") == 2

    paged = repo.list_recent(limit=1, offset=1)
    assert len(paged) == 1

    scoped = repo.list_recent(limit=10, user_id=1)
    assert len(scoped) == 2
    assert repo.count(user_id=2) == 1


def test_history_repository_creates_indexes(tmp_path: Path):
    db_path = tmp_path / "history.db"
    HistoryRepository(str(db_path))

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("PRAGMA index_list(prediction_history)").fetchall()

    index_names = {row[1] for row in rows}
    assert "idx_prediction_history_timestamp" in index_names
    assert "idx_prediction_history_source" in index_names
    assert "idx_prediction_history_top_label" in index_names
    assert "idx_prediction_history_user_id" in index_names
