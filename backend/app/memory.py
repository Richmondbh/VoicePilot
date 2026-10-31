"""
Conversation history + long-term "remember this" memory, stored in SQLite.
Based on the standard-library sqlite3 tutorial:
https://docs.python.org/3/library/sqlite3.html#tutorial
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from . import config


@contextmanager
def _connect():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_message TEXT NOT NULL,
                intent TEXT,
                target TEXT,
                status TEXT,
                assistant_response TEXT
            );
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                content TEXT NOT NULL
            );
            """
        )


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ------------------------------------------------------- interactions ------
def add_interaction(user_message, intent, target, status, response) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO interactions (timestamp, user_message, intent, target, status, assistant_response)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (_now(), user_message, intent, target, status, response),
        )


def recent_interactions(limit: int = 20) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM interactions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]  # oldest first


def clear_interactions() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM interactions")


# ----------------------------------------------------------- memories ------
def add_memory(content: str) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO memories (timestamp, content) VALUES (?, ?)", (_now(), content)
        )
        return cur.lastrowid


def list_memories() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM memories ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def delete_memory(memory_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))


STOPWORDS = {"what", "did", "i", "you", "to", "about", "the", "a", "my", "me", "is",
             "tell", "remember", "ask", "asked", "do", "when", "where", "was", "of", "on"}


def search_memories(query: str) -> list[dict]:
    """Very simple keyword overlap search (used when no LLM is available)."""
    words = {w.strip("?.,!").lower() for w in query.split()} - STOPWORDS
    hits = []
    for m in list_memories():
        text = m["content"].lower()
        score = sum(1 for w in words if w and w in text)
        if score:
            hits.append((score, m))
    return [m for _, m in sorted(hits, key=lambda x: -x[0])]
