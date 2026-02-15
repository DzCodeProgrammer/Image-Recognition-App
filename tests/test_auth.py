from pathlib import Path

from src.auth import (
    AuthRepository,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_password_hash_and_verify():
    encoded = hash_password("secret123")
    assert verify_password("secret123", encoded)
    assert not verify_password("wrong", encoded)


def test_auth_repository_create_and_authenticate(tmp_path: Path):
    db_path = tmp_path / "history.db"
    repo = AuthRepository(str(db_path))

    user = repo.create_user("alice", "password123")
    assert user["username"] == "alice"
    assert user["role"] == "user"

    authenticated = repo.authenticate("alice", "password123")
    assert authenticated is not None
    assert authenticated["id"] == user["id"]

    rejected = repo.authenticate("alice", "wrong-password")
    assert rejected is None


def test_access_token_encode_decode():
    token = create_access_token(user_id=10, role="admin", expires_minutes=30)
    payload = decode_access_token(token)
    assert payload["sub"] == "10"
    assert payload["role"] == "admin"
    assert payload["typ"] == "access"


def test_update_role(tmp_path: Path):
    db_path = tmp_path / "history.db"
    repo = AuthRepository(str(db_path))
    user = repo.create_user("bob", "password123")

    updated = repo.update_role(user["id"], "admin")
    assert updated is not None
    assert updated["role"] == "admin"

    missing = repo.update_role(999999, "user")
    assert missing is None


def test_refresh_token_and_revocation(tmp_path: Path):
    db_path = tmp_path / "history.db"
    repo = AuthRepository(str(db_path))

    refresh = create_refresh_token(user_id=11, role="user")
    payload = decode_access_token(refresh)
    assert payload["typ"] == "refresh"

    jti = payload["jti"]
    assert not repo.is_token_revoked(jti)
    repo.revoke_token(jti)
    assert repo.is_token_revoked(jti)
