"""Cache SQLite bền cho IPA tra theo yêu cầu (Cambridge-augmented).

Cùng triết lý src/words.py / src/history.py / src/auth.py: mỗi hàm mở connection
MỚI, WAL + busy_timeout, transaction ngắn qua `with conn:`; schema versioning bằng
PRAGMA user_version. DB file RIÊNG (data/ipa_cache.db) để không đụng versioning của
words.db.

CHỈ cache PHIÊN ÂM (uk_ipa/us_ipa) — KHÔNG cache audio (nặng; audio để cho /tts +
browser TTS lo). 1 bảng `ipa_cache` (key = từ đã chuẩn hoá):
- uk_ipa/us_ipa: Cambridge cho cả hai; CMUdict/eSpeak chỉ điền us_ipa (chuỗi hiển thị).
- source: nguồn của phiên âm ĐANG lưu ('cambridge' | 'cmudict' | 'override' | ...).
- cambridge_status: trạng thái RIÊNG của lần thử Cambridge, tách khỏi source để 1
  hàng vừa giữ đáp án CMUdict VỪA nhớ Cambridge đã thử hay chưa (cache warming +
  negative cache):
    NULL = chưa thử   1 = success   0 = not-found (404)   -1 = lỗi tạm thời (thử lại được)
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .config import Config

logger = logging.getLogger("toeic.ipa_cache")

_SCHEMA_VERSION = 2

# cambridge_status
CAMBRIDGE_UNTRIED: int | None = None
CAMBRIDGE_SUCCESS = 1
CAMBRIDGE_NOT_FOUND = 0
CAMBRIDGE_ERROR = -1

_DDL = """
CREATE TABLE IF NOT EXISTS ipa_cache (
  word             TEXT PRIMARY KEY,
  uk_ipa           TEXT,
  us_ipa           TEXT,
  uk_ipa_weak      TEXT,
  us_ipa_weak      TEXT,
  uk_ipa_alt       TEXT,
  us_ipa_alt       TEXT,
  source           TEXT,
  cambridge_status INTEGER,
  fetched_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
"""

# v1→v2: thêm 4 cột weak/alt (hàng cũ NULL — không cần rebuild/refetch; lần tra lại
# theo cascade sẽ điền dần). ADD COLUMN idempotent qua guard PRAGMA user_version.
_MIGRATE_V2 = (
    "ALTER TABLE ipa_cache ADD COLUMN uk_ipa_weak TEXT",
    "ALTER TABLE ipa_cache ADD COLUMN us_ipa_weak TEXT",
    "ALTER TABLE ipa_cache ADD COLUMN uk_ipa_alt  TEXT",
    "ALTER TABLE ipa_cache ADD COLUMN us_ipa_alt  TEXT",
)


@dataclass
class IPACacheRow:
    """1 hàng cache (giá trị thuần, không giữ connection)."""

    word: str
    uk_ipa: str | None = None
    us_ipa: str | None = None
    uk_ipa_weak: str | None = None
    us_ipa_weak: str | None = None
    uk_ipa_alt: str | None = None
    us_ipa_alt: str | None = None
    source: str | None = None
    cambridge_status: int | None = None


def _connect(cfg: Config) -> sqlite3.Connection:
    path = Path(cfg.ipa_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version < _SCHEMA_VERSION:
        with conn:
            conn.executescript(_DDL)  # fresh DB: bảng đã đủ cột; DB cũ: no-op
            # DB cũ (v1) đã có bảng thiếu cột weak/alt → ALTER bù (bỏ qua nếu đã có).
            existing = {r[1] for r in conn.execute("PRAGMA table_info(ipa_cache)")}
            for stmt in _MIGRATE_V2:
                col = stmt.split("ADD COLUMN")[1].split()[0]
                if col not in existing:
                    conn.execute(stmt)
            conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
    return conn


def get(cfg: Config, word: str) -> IPACacheRow | None:
    """Đọc hàng cache của `word` (đã chuẩn hoá) hoặc None nếu chưa có."""
    conn = _connect(cfg)
    try:
        row = conn.execute(
            "SELECT * FROM ipa_cache WHERE word = ?", (word,)
        ).fetchone()
        if not row:
            return None
        return IPACacheRow(
            word=row["word"],
            uk_ipa=row["uk_ipa"],
            us_ipa=row["us_ipa"],
            uk_ipa_weak=row["uk_ipa_weak"],
            us_ipa_weak=row["us_ipa_weak"],
            uk_ipa_alt=row["uk_ipa_alt"],
            us_ipa_alt=row["us_ipa_alt"],
            source=row["source"],
            cambridge_status=row["cambridge_status"],
        )
    finally:
        conn.close()


def put(cfg: Config, entry: IPACacheRow) -> None:
    """Upsert 1 hàng cache theo `word`. Ghi đè phiên âm + source + cambridge_status;
    fetched_at giữ lần đầu, updated_at cập nhật mỗi lần ghi."""
    conn = _connect(cfg)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO ipa_cache
                  (word, uk_ipa, us_ipa, uk_ipa_weak, us_ipa_weak,
                   uk_ipa_alt, us_ipa_alt, source, cambridge_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(word) DO UPDATE SET
                  uk_ipa           = excluded.uk_ipa,
                  us_ipa           = excluded.us_ipa,
                  uk_ipa_weak      = excluded.uk_ipa_weak,
                  us_ipa_weak      = excluded.us_ipa_weak,
                  uk_ipa_alt       = excluded.uk_ipa_alt,
                  us_ipa_alt       = excluded.us_ipa_alt,
                  source           = excluded.source,
                  cambridge_status = excluded.cambridge_status,
                  updated_at       = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                """,
                (
                    entry.word, entry.uk_ipa, entry.us_ipa,
                    entry.uk_ipa_weak, entry.us_ipa_weak,
                    entry.uk_ipa_alt, entry.us_ipa_alt,
                    entry.source, entry.cambridge_status,
                ),
            )
    finally:
        conn.close()


def set_cambridge_status(cfg: Config, word: str, status: int) -> None:
    """Cập nhật RIÊNG cambridge_status cho 1 hàng đã tồn tại (negative cache / lỗi),
    không đụng phiên âm CMUdict đã lưu. No-op nếu hàng chưa tồn tại."""
    conn = _connect(cfg)
    try:
        with conn:
            conn.execute(
                """
                UPDATE ipa_cache
                   SET cambridge_status = ?,
                       updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                 WHERE word = ?
                """,
                (status, word),
            )
    finally:
        conn.close()
