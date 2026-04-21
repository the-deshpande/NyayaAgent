from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ChatContext:
    """What the model sees from SQLite: rolling summary + last six messages (3 user + 3 assistant)."""

    summary: str
    messages: list[dict]  # each: {"role": "user"|"assistant", "content": str}


class ChatMemoryStore:
    """SQLite-backed session store: at most six chat rows; older content is merged into `summary`."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
                """
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)"
            )

    def ensure_session(self, session_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as c:
            c.execute(
                "INSERT OR IGNORE INTO sessions (session_id, summary, updated_at) VALUES (?, '', ?)",
                (session_id, now),
            )

    def get_context(self, session_id: str) -> ChatContext:
        self.ensure_session(session_id)
        with self._connect() as c:
            row = c.execute(
                "SELECT summary FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            summary = (row["summary"] if row else "") or ""
            rows = c.execute(
                """
                SELECT role, content FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        msgs = [{"role": r["role"], "content": r["content"]} for r in rows]
        return ChatContext(summary=summary, messages=msgs)

    def append_exchange(self, session_id: str, user_text: str, assistant_text: str) -> None:
        """Append one user message and one assistant message; roll summary when more than six rows exist."""

        self.ensure_session(session_id)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as c:
            c.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
                (session_id, user_text, now),
            )
            c.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, 'assistant', ?, ?)",
                (session_id, assistant_text, now),
            )
            rows = c.execute(
                "SELECT id, role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()

            if len(rows) <= 6:
                c.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )
                return

            # Evict oldest messages until six remain; merge evicted text into summary.
            evict_count = len(rows) - 6
            evicted = rows[:evict_count]

            evicted_text = "\n".join(f"{r['role']}: {r['content']}" for r in evicted)
            summary_row = c.execute(
                "SELECT summary FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            existing_summary = (summary_row["summary"] if summary_row else "") or ""

            new_summary = _roll_summary(existing_summary, evicted_text)

            for r in evicted:
                c.execute("DELETE FROM messages WHERE id = ?", (r["id"],))

            c.execute(
                "UPDATE sessions SET summary = ?, updated_at = ? WHERE session_id = ?",
                (new_summary, now, session_id),
            )


def _roll_summary(existing: str, evicted_block: str) -> str:
    """Merge evicted dialogue into the stored summary (LLM if keys present, else truncate concat)."""

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from nyaya_agent.llm import get_chat_model

        model = get_chat_model()
        sys = SystemMessage(
            content=(
                "You compress older chat turns into a concise factual summary for legal/compliance context. "
                "Preserve names, dates, statutes, and open questions. Max ~800 tokens. Bullet list OK."
            )
        )
        human = HumanMessage(
            content=(
                f"Previous summary (may be empty):\n{existing}\n\n"
                f"Older messages being removed from the live window (condense into the summary):\n{evicted_block}"
            )
        )
        out = model.invoke([sys, human])
        text = (out.content or "").strip()
        if text:
            return text
    except Exception:
        pass

    merged = (existing + "\n\n" + evicted_block).strip()
    return merged[:8000]
