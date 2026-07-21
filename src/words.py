"""Từ đã lưu để luyện tập (per-user) + cache định nghĩa/ví dụ do LLM sinh.

Cùng triết lý src/history.py: "user" = uuid ẩn danh phía trình duyệt (không auth,
cách ly mềm theo user_id); mỗi hàm mở connection MỚI, WAL + busy_timeout,
transaction ngắn qua `with conn:`; schema versioning bằng PRAGMA user_version.
DB file RIÊNG (data/words.db) để không đụng versioning của history.db.

6 bảng:
- saved_words: từ user bookmark từ bảng lỗi phát âm (kèm IPA + snapshot phonemes
  + điểm), PK (user_id, word) — lưu lại từ đã có thì update điểm/thời điểm.
- word_info_cache: định nghĩa EN + ví dụ + nghĩa tiếng Việt do LLM sinh, key
  (word, lang) — mỗi từ chỉ tốn 1 call LLM, các lần sau đọc cache.
- phoneme_stats: thống kê per-user per-phoneme (ok/sub/del có trọng số) tổng hợp
  TĂNG DẦN từ history result_json — hồ sơ "âm yếu" cho gợi ý từ luyện tập
  (src/phoneme_profile.py). Cột đếm là REAL vì evidence được weight theo tuổi.
- phoneme_profile_state: con trỏ quét history per-user, composite
  (last_scan_at, last_scan_id) — tie-break theo id để record cùng giây không bị
  bỏ sót (nhất quán với retention tie-break của history.py).
- suggestion_cache: danh sách từ luyện tập do LLM chọn cho 1 phoneme, key
  (phoneme, lang), USER-AGNOSTIC (cá nhân hoá làm sau khi đọc cache). TTL +
  cache_version do src/word_suggest.py quyết định khi đọc.
- user_settings: KV per-user (PK (user_id, key)) cho tuỳ chọn client cần đồng bộ
  đa thiết bị — value là blob JSON opaque với server (vd key 'review_toast').
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

_SCHEMA_VERSION = 4

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

CREATE TABLE IF NOT EXISTS phoneme_stats (
  user_id      TEXT NOT NULL,
  symbol       TEXT NOT NULL,
  lang         TEXT NOT NULL DEFAULT 'en',
  attempts     REAL NOT NULL DEFAULT 0,
  ok           REAL NOT NULL DEFAULT 0,
  sub          REAL NOT NULL DEFAULT 0,
  del          REAL NOT NULL DEFAULT 0,
  err_weighted REAL NOT NULL DEFAULT 0,
  heard_json   TEXT,
  updated_at   TEXT,
  PRIMARY KEY (user_id, symbol, lang)
);

CREATE TABLE IF NOT EXISTS phoneme_profile_state (
  user_id      TEXT PRIMARY KEY,
  last_scan_at TEXT NOT NULL DEFAULT '',
  last_scan_id TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS suggestion_cache (
  phoneme       TEXT NOT NULL,
  lang          TEXT NOT NULL,
  cache_version INTEGER NOT NULL DEFAULT 1,
  words_json    TEXT NOT NULL,
  model         TEXT,
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  PRIMARY KEY (phoneme, lang)
);

CREATE TABLE IF NOT EXISTS user_settings (
  user_id    TEXT NOT NULL,
  key        TEXT NOT NULL,
  value      TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  PRIMARY KEY (user_id, key)
);
"""

# Migration v3→v4: thêm cột `lang` vào phoneme_stats (tách hồ sơ âm theo ngôn ngữ
# nói của kỳ thi — en cho TOEIC/IELTS, ko cho TOPIK). Dữ liệu cũ toàn bộ là tiếng
# Anh → backfill lang='en'. Rebuild bảng vì SQLite không ALTER được PRIMARY KEY.
_MIGRATE_PHONEME_STATS_V4 = """
ALTER TABLE phoneme_stats RENAME TO _phoneme_stats_v3;
CREATE TABLE phoneme_stats (
  user_id      TEXT NOT NULL,
  symbol       TEXT NOT NULL,
  lang         TEXT NOT NULL DEFAULT 'en',
  attempts     REAL NOT NULL DEFAULT 0,
  ok           REAL NOT NULL DEFAULT 0,
  sub          REAL NOT NULL DEFAULT 0,
  del          REAL NOT NULL DEFAULT 0,
  err_weighted REAL NOT NULL DEFAULT 0,
  heard_json   TEXT,
  updated_at   TEXT,
  PRIMARY KEY (user_id, symbol, lang)
);
INSERT INTO phoneme_stats
  (user_id, symbol, lang, attempts, ok, sub, del, err_weighted, heard_json, updated_at)
  SELECT user_id, symbol, 'en', attempts, ok, sub, del, err_weighted, heard_json, updated_at
  FROM _phoneme_stats_v3;
DROP TABLE _phoneme_stats_v3;
"""

# Giới hạn để bảng settings không bị lạm dụng làm kho dữ liệu tuỳ ý.
MAX_SETTING_VALUE_LEN = 4096

# Từ/cụm hợp lệ để lưu/tra: chữ cái + nháy đơn/gạch nối, thêm khoảng trắng để
# lưu được cụm gợi ý từ vựng ('borrow a book') — tối đa 4 từ (chặn nguyên câu).
_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z' -]{0,39}$")
_MAX_PHRASE_WORDS = 4


def validate_word(word: str) -> str:
    """Chuẩn hoá + validate từ/cụm (lowercase, gộp khoảng trắng). Raise ValueError nếu không hợp lệ."""
    w = " ".join((word or "").split()).lower()
    if not _WORD_RE.match(w) or len(w.split()) > _MAX_PHRASE_WORDS:
        raise ValueError(
            "word không hợp lệ (chỉ chữ cái, nháy đơn, gạch nối; ≤40 ký tự, cụm ≤4 từ)."
        )
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
            conn.executescript(_DDL)  # tạo bảng còn thiếu (IF NOT EXISTS)
            # v3→v4: phoneme_stats cũ chưa có cột lang → rebuild + backfill 'en'.
            cols = {r[1] for r in conn.execute("PRAGMA table_info(phoneme_stats)")}
            if "lang" not in cols:
                conn.executescript(_MIGRATE_PHONEME_STATS_V4)
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


def merge_user(cfg: Config, from_user_id: str, to_user_id: str) -> int:
    """Gộp dữ liệu từ user ẩn danh sang user tài khoản (dùng khi /auth/claim).

    - saved_words: chuyển từ CHƯA trùng sang tài khoản (UPDATE OR IGNORE bỏ qua từ
      đã có trong tài khoản do PK (user_id, word)), rồi xoá phần còn lại của user cũ.
    - phoneme_stats + phoneme_profile_state: XOÁ của CẢ HAI user (không cộng dồn
      thủ công) — hồ sơ âm sẽ tự dựng lại tăng dần từ history đã gộp ở lần gợi ý sau
      (src/phoneme_profile.py quét lại từ con trỏ rỗng). Đúng & đơn giản hơn.

    Trả số từ đã chuyển thành công.
    """
    from_user_id = validate_user_id(from_user_id)
    to_user_id = validate_user_id(to_user_id)
    if from_user_id == to_user_id:
        return 0
    conn = _connect(cfg)
    try:
        with conn:
            cur = conn.execute(
                "UPDATE OR IGNORE saved_words SET user_id = ? WHERE user_id = ?",
                (to_user_id, from_user_id),
            )
            moved = cur.rowcount
            conn.execute(
                "DELETE FROM saved_words WHERE user_id = ?", (from_user_id,)
            )
            conn.execute(
                "DELETE FROM phoneme_stats WHERE user_id IN (?, ?)",
                (from_user_id, to_user_id),
            )
            conn.execute(
                "DELETE FROM phoneme_profile_state WHERE user_id IN (?, ?)",
                (from_user_id, to_user_id),
            )
        return moved
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


# ── User settings (KV per-user, đồng bộ đa thiết bị) ─────────────────────


def get_setting(cfg: Config, user_id: str, key: str) -> str | None:
    """Đọc blob JSON (opaque) của 1 tuỳ chọn; None nếu chưa lưu."""
    conn = _connect(cfg)
    try:
        row = conn.execute(
            "SELECT value FROM user_settings WHERE user_id = ? AND key = ?",
            (user_id, key),
        ).fetchone()
        return row["value"] if row else None
    finally:
        conn.close()


def set_setting(cfg: Config, user_id: str, key: str, value: str) -> None:
    """Ghi đè blob JSON của 1 tuỳ chọn (upsert theo (user_id, key))."""
    if len(value) > MAX_SETTING_VALUE_LEN:
        raise ValueError(f"value quá dài (>{MAX_SETTING_VALUE_LEN} ký tự).")
    conn = _connect(cfg)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO user_settings (user_id, key, value, updated_at)
                VALUES (?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ','now'))
                ON CONFLICT(user_id, key) DO UPDATE SET
                    value = excluded.value, updated_at = excluded.updated_at
                """,
                (user_id, key, value),
            )
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


# ── Phoneme stats (hồ sơ âm yếu — src/phoneme_profile.py ghi/đọc) ─────────


def get_phoneme_stats(cfg: Config, user_id: str, lang: str = "en") -> dict[str, dict]:
    """Thống kê phoneme của user cho 1 NGÔN NGỮ: {symbol: {attempts, ok, sub, del,
    err_weighted, heard: {phone: weight}}}. Default 'en' → hành vi cũ (TOEIC/IELTS
    + tab Từ đã lưu) bit-for-bit; 'ko' cho khóa học TOPIK."""
    conn = _connect(cfg)
    try:
        rows = conn.execute(
            "SELECT * FROM phoneme_stats WHERE user_id = ? AND lang = ?",
            (user_id, lang),
        ).fetchall()
        stats: dict[str, dict] = {}
        for r in rows:
            entry = dict(r)
            raw = entry.pop("heard_json", None)
            try:
                entry["heard"] = json.loads(raw) if raw else {}
            except (TypeError, ValueError):
                entry["heard"] = {}
            stats[entry["symbol"]] = entry
        return stats
    finally:
        conn.close()


def get_profile_cursor(cfg: Config, user_id: str) -> tuple[str, str]:
    """Con trỏ quét history (last_scan_at, last_scan_id); ('', '') nếu chưa quét."""
    conn = _connect(cfg)
    try:
        row = conn.execute(
            "SELECT last_scan_at, last_scan_id FROM phoneme_profile_state WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return (row["last_scan_at"], row["last_scan_id"]) if row else ("", "")
    finally:
        conn.close()


def apply_phoneme_tallies(
    cfg: Config, user_id: str, tallies: dict[str, dict], new_cursor: tuple[str, str]
) -> bool:
    """Cộng dồn tallies vào phoneme_stats + tiến con trỏ quét, trong 1 transaction.

    Guard chống double-count khi 2 request đồng thời cùng quét 1 đoạn history:
    đọc LẠI con trỏ trong transaction — nếu đã >= new_cursor thì request kia đã
    apply xong đoạn này → bỏ qua. Trả True nếu có ghi.
    """
    conn = _connect(cfg)
    try:
        with conn:
            row = conn.execute(
                "SELECT last_scan_at, last_scan_id FROM phoneme_profile_state WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            current = (row["last_scan_at"], row["last_scan_id"]) if row else ("", "")
            if current >= new_cursor:
                return False
            now = "strftime('%Y-%m-%dT%H:%M:%SZ','now')"
            for key, t in tallies.items():
                # key: (symbol, lang) HOẶC symbol thuần → ('en'), giữ back-compat
                # với caller/test cũ truyền tallies keyed theo symbol.
                symbol, lang = key if isinstance(key, tuple) else (key, "en")
                old = conn.execute(
                    "SELECT heard_json FROM phoneme_stats WHERE user_id = ? AND symbol = ? AND lang = ?",
                    (user_id, symbol, lang),
                ).fetchone()
                heard: dict[str, float] = {}
                if old and old["heard_json"]:
                    try:
                        heard = json.loads(old["heard_json"])
                    except (TypeError, ValueError):
                        heard = {}
                for phone, w in (t.get("heard") or {}).items():
                    heard[phone] = heard.get(phone, 0) + w
                conn.execute(
                    f"""
                    INSERT INTO phoneme_stats
                      (user_id, symbol, lang, attempts, ok, sub, del, err_weighted, heard_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, {now})
                    ON CONFLICT(user_id, symbol, lang) DO UPDATE SET
                      attempts     = attempts + excluded.attempts,
                      ok           = ok + excluded.ok,
                      sub          = sub + excluded.sub,
                      del          = del + excluded.del,
                      err_weighted = err_weighted + excluded.err_weighted,
                      heard_json   = excluded.heard_json,
                      updated_at   = excluded.updated_at
                    """,
                    (
                        user_id, symbol, lang,
                        t.get("attempts", 0), t.get("ok", 0), t.get("sub", 0),
                        t.get("del", 0), t.get("err_weighted", 0),
                        json.dumps(heard, ensure_ascii=False) if heard else None,
                    ),
                )
            conn.execute(
                """
                INSERT INTO phoneme_profile_state (user_id, last_scan_at, last_scan_id)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                  last_scan_at = excluded.last_scan_at,
                  last_scan_id = excluded.last_scan_id
                """,
                (user_id, new_cursor[0], new_cursor[1]),
            )
        return True
    finally:
        conn.close()


# ── Suggestion cache (từ luyện tập do LLM chọn per-phoneme) ───────────────


def get_suggestion_cache(cfg: Config, phoneme: str, lang: str) -> dict | None:
    """Entry cache thô {words, model, cache_version, created_at} hoặc None.
    Validity (TTL, version) do caller (src/word_suggest.py) quyết định."""
    conn = _connect(cfg)
    try:
        row = conn.execute(
            "SELECT * FROM suggestion_cache WHERE phoneme = ? AND lang = ?",
            (phoneme, lang),
        ).fetchone()
        if not row:
            return None
        entry = dict(row)
        try:
            entry["words"] = json.loads(entry.pop("words_json"))
        except (TypeError, ValueError):
            return None
        return entry
    finally:
        conn.close()


def put_suggestion_cache(
    cfg: Config, phoneme: str, lang: str, words_list: list[dict],
    model: str | None, cache_version: int,
) -> None:
    conn = _connect(cfg)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO suggestion_cache (phoneme, lang, cache_version, words_json, model, created_at)
                VALUES (?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ','now'))
                ON CONFLICT(phoneme, lang) DO UPDATE SET
                  cache_version = excluded.cache_version,
                  words_json    = excluded.words_json,
                  model         = excluded.model,
                  created_at    = excluded.created_at
                """,
                (phoneme, lang, cache_version,
                 json.dumps(words_list, ensure_ascii=False), model),
            )
    finally:
        conn.close()
