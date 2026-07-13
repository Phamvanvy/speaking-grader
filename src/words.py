"""Từ đã lưu để luyện tập (per-user) + cache định nghĩa/ví dụ do LLM sinh.

Cùng triết lý src/history.py: "user" = uuid ẩn danh phía trình duyệt (không auth,
cách ly mềm theo user_id); mỗi hàm mở connection MỚI, WAL + busy_timeout,
transaction ngắn qua `with conn:`; schema versioning bằng PRAGMA user_version.
DB file RIÊNG (data/words.db) để không đụng versioning của history.db.

2 bảng:
- saved_words: từ user bookmark từ bảng lỗi phát âm (kèm IPA + snapshot phonemes
  + điểm), PK (user_id, word) — lưu lại từ đã có thì update điểm/thời điểm.
- word_info_cache: định nghĩa EN + ví dụ + nghĩa tiếng Việt do LLM sinh, key
  (word, lang) — mỗi từ chỉ tốn 1 call LLM, các lần sau đọc cache.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path

from .config import Config
from .history import validate_user_id  # noqa: F401 - re-export cho api.py

logger = logging.getLogger("toeic.words")

_SCHEMA_VERSION = 1

# Giữ tối đa N từ mỗi user (xoá từ lưu cũ nhất khi vượt) — chống phình vô hạn.
MAX_WORDS_PER_USER = 500

_DDL = """
CREATE TABLE IF NOT EXISTS saved_words (
  user_id           TEXT NOT NULL,
  word              TEXT NOT NULL,
  ipa               TEXT,
  phonemes_json     TEXT,
  accuracy          REAL,
  last_score        REAL,
  saved_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  last_practiced_at TEXT,
  PRIMARY KEY (user_id, word)
);
CREATE INDEX IF NOT EXISTS idx_saved_words_user_saved
  ON saved_words(user_id, saved_at DESC);

CREATE TABLE IF NOT EXISTS word_info_cache (
  word          TEXT NOT NULL,
  lang          TEXT NOT NULL,
  definition_en TEXT,
  example_en    TEXT,
  meaning       TEXT,
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  PRIMARY KEY (word, lang)
);
"""

# Từ hợp lệ để lưu/tra: chữ cái + nháy đơn/gạch nối (khớp token hoá reference).
_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'-]{0,39}$")


def validate_word(word: str) -> str:
    """Chuẩn hoá + validate từ (lowercase, strip). Raise ValueError nếu không hợp lệ."""
    w = (word or "").strip().lower()
    if not _WORD_RE.match(w):
        raise ValueError("word không hợp lệ (chỉ chữ cái, nháy đơn, gạch nối; ≤40 ký tự).")
    return w


# ── Kết nối / schema ─────────────────────────────────────────────────────


def _connect(cfg: Config) -> sqlite3.Connection:
    path = Path(cfg.words_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version < _SCHEMA_VERSION:
        with conn:
            conn.executescript(_DDL)
            conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
    return conn


# ── Saved words ──────────────────────────────────────────────────────────


def _row_to_entry(row: sqlite3.Row) -> dict:
    entry = dict(row)
    raw = entry.pop("phonemes_json", None)
    try:
        entry["phonemes"] = json.loads(raw) if raw else None
    except (TypeError, ValueError):
        entry["phonemes"] = None
    return entry


def list_words(cfg: Config, user_id: str) -> dict:
    """Toàn bộ từ đã lưu của user, mới lưu trước."""
    conn = _connect(cfg)
    try:
        rows = conn.execute(
            "SELECT * FROM saved_words WHERE user_id = ? ORDER BY saved_at DESC, word",
            (user_id,),
        ).fetchall()
        return {"words": [_row_to_entry(r) for r in rows], "total": len(rows)}
    finally:
        conn.close()


def upsert_word(
    cfg: Config,
    user_id: str,
    word: str,
    *,
    ipa: str | None = None,
    phonemes: object | None = None,
    accuracy: float | None = None,
    last_score: float | None = None,
) -> dict:
    """Lưu từ mới hoặc cập nhật từ đã có (điểm luyện gần nhất / snapshot mới hơn).

    COALESCE giữ dữ liệu cũ khi request không gửi lại field (vd cập nhật
    last_score từ popup luyện tập không cần gửi lại phonemes).
    """
    phonemes_json = (
        json.dumps(phonemes, ensure_ascii=False) if phonemes is not None else None
    )
    conn = _connect(cfg)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO saved_words
                  (user_id, word, ipa, phonemes_json, accuracy, last_score)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, word) DO UPDATE SET
                  ipa           = COALESCE(excluded.ipa, saved_words.ipa),
                  phonemes_json = COALESCE(excluded.phonemes_json, saved_words.phonemes_json),
                  accuracy      = COALESCE(excluded.accuracy, saved_words.accuracy),
                  last_score    = COALESCE(excluded.last_score, saved_words.last_score),
                  last_practiced_at = CASE
                    WHEN excluded.last_score IS NOT NULL
                    THEN strftime('%Y-%m-%dT%H:%M:%SZ','now')
                    ELSE saved_words.last_practiced_at
                  END
                """,
                (user_id, word, ipa, phonemes_json, accuracy, last_score),
            )
            # Quota: xoá từ lưu cũ nhất khi vượt trần.
            conn.execute(
                """
                DELETE FROM saved_words WHERE user_id = ? AND word NOT IN (
                  SELECT word FROM saved_words WHERE user_id = ?
                  ORDER BY saved_at DESC, word LIMIT ?
                )
                """,
                (user_id, user_id, MAX_WORDS_PER_USER),
            )
        row = conn.execute(
            "SELECT * FROM saved_words WHERE user_id = ? AND word = ?",
            (user_id, word),
        ).fetchone()
        return _row_to_entry(row)
    finally:
        conn.close()


def delete_word(cfg: Config, user_id: str, word: str) -> bool:
    conn = _connect(cfg)
    try:
        with conn:
            cur = conn.execute(
                "DELETE FROM saved_words WHERE user_id = ? AND word = ?",
                (user_id, word),
            )
        return cur.rowcount > 0
    finally:
        conn.close()


# ── Word info cache (định nghĩa/ví dụ LLM) ───────────────────────────────


def get_word_info(cfg: Config, word: str, lang: str) -> dict | None:
    conn = _connect(cfg)
    try:
        row = conn.execute(
            "SELECT * FROM word_info_cache WHERE word = ? AND lang = ?",
            (word, lang),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def put_word_info(
    cfg: Config, word: str, lang: str, definition_en: str, example_en: str, meaning: str
) -> None:
    conn = _connect(cfg)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO word_info_cache (word, lang, definition_en, example_en, meaning)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(word, lang) DO UPDATE SET
                  definition_en = excluded.definition_en,
                  example_en    = excluded.example_en,
                  meaning       = excluded.meaning
                """,
                (word, lang, definition_en, example_en, meaning),
            )
    finally:
        conn.close()
