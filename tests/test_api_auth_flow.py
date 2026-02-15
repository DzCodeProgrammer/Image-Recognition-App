import importlib

from fastapi.testclient import TestClient


def _client_with_temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "api_test.db"
    monkeypatch.setenv("APP_DB_PATH", str(db_path))
    import api as api_module

    importlib.reload(api_module)
    return TestClient(api_module.app)


def _register(client: TestClient, username: str, role: str = "user") -> dict:
    response = client.post(
        "/auth/register",
        json={
            "username": username,
            "password": "password123",
            "role": role,
        },
    )
    assert response.status_code == 200
    return response.json()


def test_protected_endpoint_requires_token(tmp_path, monkeypatch):
    client = _client_with_temp_db(tmp_path, monkeypatch)
    response = client.get("/history")
    assert response.status_code == 401


def test_admin_can_list_users(tmp_path, monkeypatch):
    client = _client_with_temp_db(tmp_path, monkeypatch)

    admin = _register(client, "admin01", role="admin")
    _register(client, "user01", role="user")

    response = client.get(
        "/admin/users?limit=10&offset=0",
        headers={"Authorization": f"Bearer {admin['access_token']}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] >= 2
    assert len(payload["rows"]) >= 2


def test_non_admin_cannot_list_users(tmp_path, monkeypatch):
    client = _client_with_temp_db(tmp_path, monkeypatch)
    user = _register(client, "user02", role="user")

    response = client.get(
        "/admin/users",
        headers={"Authorization": f"Bearer {user['access_token']}"},
    )
    assert response.status_code == 403


def test_history_scoped_to_logged_in_user(tmp_path, monkeypatch):
    client = _client_with_temp_db(tmp_path, monkeypatch)

    user_a = _register(client, "alice01", role="user")
    user_b = _register(client, "bob01", role="user")

    db_path = tmp_path / "api_test.db"
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO prediction_history
            (user_id, timestamp, filename, top_label, top_confidence, source, top_predictions)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_a["user"]["id"], "2026-02-15T10:00:00", "a.jpg", "cat", 90.0, "api", None),
        )
        conn.execute(
            """
            INSERT INTO prediction_history
            (user_id, timestamp, filename, top_label, top_confidence, source, top_predictions)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_b["user"]["id"], "2026-02-15T10:05:00", "b.jpg", "dog", 85.0, "api", None),
        )
        conn.commit()

    response = client.get(
        "/history",
        headers={"Authorization": f"Bearer {user_a['access_token']}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["rows"][0]["filename"] == "a.jpg"


def test_history_date_validation(tmp_path, monkeypatch):
    client = _client_with_temp_db(tmp_path, monkeypatch)
    user = _register(client, "dateuser01", role="user")
    headers = {"Authorization": f"Bearer {user['access_token']}"}

    invalid = client.get("/history?date_from=not-a-date", headers=headers)
    assert invalid.status_code == 400

    invalid_range = client.get(
        "/history?date_from=2026-02-16T00:00:00&date_to=2026-02-15T00:00:00",
        headers=headers,
    )
    assert invalid_range.status_code == 400


def test_refresh_rotates_token_and_old_refresh_is_rejected(tmp_path, monkeypatch):
    client = _client_with_temp_db(tmp_path, monkeypatch)
    user = _register(client, "rotate01", role="user")

    first_refresh = user["refresh_token"]
    refreshed = client.post("/auth/refresh", json={"refresh_token": first_refresh})
    assert refreshed.status_code == 200
    payload = refreshed.json()
    assert payload["access_token"]
    assert payload["refresh_token"]

    old_reuse = client.post("/auth/refresh", json={"refresh_token": first_refresh})
    assert old_reuse.status_code == 401


def test_logout_revokes_access_token(tmp_path, monkeypatch):
    client = _client_with_temp_db(tmp_path, monkeypatch)
    user = _register(client, "logout01", role="user")
    headers = {"Authorization": f"Bearer {user['access_token']}"}

    before = client.get("/history", headers=headers)
    assert before.status_code == 200

    logout = client.post(
        "/auth/logout",
        headers=headers,
        json={"refresh_token": user["refresh_token"]},
    )
    assert logout.status_code == 200

    after = client.get("/history", headers=headers)
    assert after.status_code == 401


def test_predict_url_image_with_mocked_media(tmp_path, monkeypatch):
    client = _client_with_temp_db(tmp_path, monkeypatch)
    user = _register(client, "urluser01", role="user")
    headers = {"Authorization": f"Bearer {user['access_token']}"}

    import api as api_module
    from src.classifier import Prediction
    from src.url_analyzer import URLAnalysisResult

    monkeypatch.setattr(
        api_module,
        "analyze_url",
        lambda **kwargs: URLAnalysisResult(
            url=kwargs["url"],
            final_url=kwargs["url"],
            content_type="image/png",
            content_kind="image",
            insight="ok",
            predictions=[Prediction(label="cat", confidence=88.0)],
            filtered_predictions=[Prediction(label="cat", confidence=88.0)],
            sampled_frames=None,
            total_frames=None,
            document=None,
        ),
    )

    response = client.post(
        "/predict/url",
        headers=headers,
        json={"url": "https://example.com/a.png", "media_type": "auto"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["content_kind"] == "image"
    assert payload["count"] == 1
    assert payload["predictions"][0]["label_raw"] == "cat"
