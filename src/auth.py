import hashlib
import hmac
import os
import secrets
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Optional, TypedDict

import jwt


class User(TypedDict):
    id: int
    username: str
    role: str
    created_at: str


class AuthRepository:
    def __init__(self, db_path: str = "history.db") -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS revoked_tokens (
                    jti TEXT PRIMARY KEY,
                    revoked_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def create_user(self, username: str, password: str, role: str = "user") -> User:
        normalized_username = username.strip().lower()
        if len(normalized_username) < 3:
            raise ValueError("username minimal 3 karakter")
        if len(password) < 6:
            raise ValueError("password minimal 6 karakter")
        if role not in {"user", "admin"}:
            raise ValueError("role tidak valid")

        password_hash = hash_password(password)
        created_at = datetime.now(UTC).isoformat(timespec="seconds")

        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO users (username, password_hash, role, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (normalized_username, password_hash, role, created_at),
                )
                conn.commit()
                user_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise ValueError("username sudah terdaftar") from exc

        return {
            "id": user_id,
            "username": normalized_username,
            "role": role,
            "created_at": created_at,
        }

    def authenticate(self, username: str, password: str) -> Optional[User]:
        normalized_username = username.strip().lower()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, username, password_hash, role, created_at
                FROM users
                WHERE username = ?
                """,
                (normalized_username,),
            ).fetchone()

        if not row:
            return None
        if not verify_password(password, row[2]):
            return None

        return {
            "id": int(row[0]),
            "username": row[1],
            "role": row[3],
            "created_at": row[4],
        }

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, username, role, created_at
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()

        if not row:
            return None
        return {
            "id": int(row[0]),
            "username": row[1],
            "role": row[2],
            "created_at": row[3],
        }

    def update_role(self, user_id: int, role: str) -> Optional[User]:
        if role not in {"user", "admin"}:
            raise ValueError("role tidak valid")

        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE users
                SET role = ?
                WHERE id = ?
                """,
                (role, user_id),
            )
            conn.commit()
            if cursor.rowcount == 0:
                return None

        return self.get_user_by_id(user_id)

    def list_users(self, limit: int = 50, offset: int = 0) -> list[User]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, username, role, created_at
                FROM users
                ORDER BY id ASC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

        return [
            {
                "id": int(row[0]),
                "username": row[1],
                "role": row[2],
                "created_at": row[3],
            }
            for row in rows
        ]

    def count_users(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM users").fetchone()
        return int(row[0]) if row else 0

    def revoke_token(self, jti: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO revoked_tokens (jti, revoked_at)
                VALUES (?, ?)
                """,
                (jti, datetime.now(UTC).isoformat(timespec="seconds")),
            )
            conn.commit()

    def is_token_revoked(self, jti: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM revoked_tokens WHERE jti = ? LIMIT 1",
                (jti,),
            ).fetchone()
        return row is not None


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        100_000,
    )
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded_hash: str) -> bool:
    try:
        salt_hex, digest_hex = encoded_hash.split("$", maxsplit=1)
    except ValueError:
        return False
    salt = bytes.fromhex(salt_hex)
    expected = bytes.fromhex(digest_hex)
    candidate = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        100_000,
    )
    return hmac.compare_digest(candidate, expected)


def _secret_key() -> str:
    return os.getenv("APP_SECRET_KEY", "dev-secret-change-this")


def _create_token(
    *,
    user_id: int,
    role: str,
    token_type: str,
    expires_minutes: int,
) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "role": role,
        "typ": token_type,
        "jti": str(uuid.uuid4()),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_minutes)).timestamp()),
    }
    return jwt.encode(payload, _secret_key(), algorithm="HS256")


def create_access_token(user_id: int, role: str, expires_minutes: int = 60) -> str:
    return _create_token(
        user_id=user_id,
        role=role,
        token_type="access",
        expires_minutes=expires_minutes,
    )


def create_refresh_token(user_id: int, role: str, expires_minutes: int = 60 * 24 * 7) -> str:
    return _create_token(
        user_id=user_id,
        role=role,
        token_type="refresh",
        expires_minutes=expires_minutes,
    )


def decode_access_token(token: str) -> dict:
    return jwt.decode(token, _secret_key(), algorithms=["HS256"])
