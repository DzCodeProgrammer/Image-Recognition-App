from pathlib import Path

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
    )

    rows = repo.list_recent(limit=10)
    assert len(rows) == 1
    assert rows[0]["filename"] == "sample.jpg"
    assert rows[0]["top_label"] == "cat"


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
