# prompt_store.py
# ---------------
# Stable, single DB path: ~/.imagine_app/prompts.db  (persists across runs & refreshes)

from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Optional, List

APP_DIR = Path.home() / ".imagine_app"
APP_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = APP_DIR / "prompts.db"


def _connect(db_path: Optional[str | Path] = None) -> sqlite3.Connection:
    p = Path(db_path) if db_path else DB_FILE
    con = sqlite3.connect(p, check_same_thread=False, isolation_level=None)
    # Make writes safe & fast
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def init_db(db_path: Optional[str | Path] = None) -> None:
    with _connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompts_created_at ON prompts(created_at DESC)"
        )


def save_prompt(text: str, db_path: Optional[str | Path] = None) -> None:
    t = (text or "").strip()
    if not t:
        return
    with _connect(db_path) as con:
        con.execute("INSERT INTO prompts(text) VALUES (?)", (t,))


def get_recent_prompts(
    limit: int = 5, db_path: Optional[str | Path] = None
) -> List[str]:
    # Fetch a bit more and de-dup newest-first
    with _connect(db_path) as con:
        rows = con.execute(
            "SELECT text FROM prompts ORDER BY id DESC LIMIT 100"
        ).fetchall()
    out, seen = [], set()
    for (txt,) in rows:
        if txt not in seen:
            out.append(txt)
            seen.add(txt)
            if len(out) >= max(1, int(limit)):
                break
    return out
