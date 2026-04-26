from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "database.db"
DEFAULT_HISTORY_LIMIT = 100


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                cgpa REAL NOT NULL,
                skills INTEGER NOT NULL,
                internship INTEGER NOT NULL,
                projects INTEGER NOT NULL,
                communication INTEGER NOT NULL,
                probability REAL NOT NULL,
                label TEXT NOT NULL,
                suggestions TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )
        conn.commit()


def create_user(username: str, password_hash: str) -> bool:
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash),
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def get_user_by_username(username: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def add_history_record(
    user_id: int,
    inputs: dict[str, Any],
    probability: float,
    label: str,
    suggestions: list[str],
) -> int:
    suggestions_text = " | ".join(suggestions)
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO history (
                user_id,
                cgpa,
                skills,
                internship,
                projects,
                communication,
                probability,
                label,
                suggestions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                float(inputs["cgpa"]),
                int(inputs["skills"]),
                int(inputs["internship"]),
                int(inputs["projects"]),
                int(inputs["communication"]),
                float(probability),
                label,
                suggestions_text,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_user_history(user_id: int, limit: int = DEFAULT_HISTORY_LIMIT) -> list[dict[str, Any]]:
    safe_limit = max(1, int(limit))
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, cgpa, skills, internship, projects, communication,
                   probability, label, suggestions, created_at
            FROM history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, safe_limit),
        ).fetchall()
    return [dict(row) for row in rows]


def get_history_record(record_id: int, user_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, user_id, cgpa, skills, internship, projects, communication,
                   probability, label, suggestions, created_at
            FROM history
            WHERE id = ? AND user_id = ?
            """,
            (record_id, user_id),
        ).fetchone()
    return dict(row) if row else None
