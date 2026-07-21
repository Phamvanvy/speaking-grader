"""Lưu trữ khóa học cá nhân hóa (per-user) — DB file RIÊNG data/course.db.

Cùng triết lý src/words.py: "user" = uuid ẩn danh phía trình duyệt (cách ly mềm
theo user_id); mỗi hàm mở connection MỚI, WAL + busy_timeout, transaction ngắn
qua `with conn:`; schema versioning bằng PRAGMA user_version. DB RIÊNG để không
đụng versioning của history.db/words.db.

5 bảng — chia làm 2 nhóm:
- Mastery (tổng hợp TĂNG DẦN từ history result_json — mirror phoneme_stats):
  - criterion_stats: điểm ĐÃ CHUẨN HÓA 0-1 per (user, exam, criterion), cột đếm
    REAL vì evidence weight theo tuổi (recency).
  - qtype_stats: như trên nhưng per (user, exam, question_type) — điểm tổng
    chuẩn hóa 0-1.
  - course_scan_state: con trỏ quét history per-user (composite last_scan_at,
    last_scan_id) — tie-break theo id để record cùng giây không sót (nhất quán
    với phoneme_profile_state của words.py).
- Tiến độ BỀN (KHÔNG tính từ history — phải sống sót retention capping):
  - lesson_progress: trạng thái từng lesson (in_progress|done) + best_score.
    not_started = KHÔNG có hàng (bảng nhỏ, status suy ra khi đọc).
  - course_activity: streak/hoạt động per-user.

Consumer: src/course/profile.py (mastery) + src/course/generate.py (build_course)
+ src/course/__init__.py (mark_lesson_complete).
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from ..config import Config
from ..history import validate_user_id

logger = logging.getLogger("toeic.course.store")

_SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS criterion_stats (
  user_id    TEXT NOT NULL,
  exam       TEXT NOT NULL,
  criterion  TEXT NOT NULL,
  attempts   REAL NOT NULL DEFAULT 0,
  score_sum  REAL NOT NULL DEFAULT 0,   -- Σ điểm ĐÃ chuẩn hóa 0-1 (weighted)
  updated_at TEXT,
  PRIMARY KEY (user_id, exam, criterion)
);

CREATE TABLE IF NOT EXISTS qtype_stats (
  user_id       TEXT NOT NULL,
  exam          TEXT NOT NULL,
  question_type TEXT NOT NULL,
  attempts      REAL NOT NULL DEFAULT 0,
  score_sum     REAL NOT NULL DEFAULT 0,   -- Σ overall ĐÃ chuẩn hóa 0-1 (weighted)
  updated_at    TEXT,
  PRIMARY KEY (user_id, exam, question_type)
);

CREATE TABLE IF NOT EXISTS course_scan_state (
  user_id      TEXT PRIMARY KEY,
  last_scan_at TEXT NOT NULL DEFAULT '',
  last_scan_id TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS lesson_progress (
  user_id      TEXT NOT NULL,
  lesson_id    TEXT NOT NULL,
  status       TEXT NOT NULL DEFAULT 'in_progress',   -- in_progress | done
  best_score   REAL,                                  -- best NORMALIZED (0-1)
  attempts     INTEGER NOT NULL DEFAULT 0,
  completed_at TEXT,
  updated_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  PRIMARY KEY (user_id, lesson_id)
);
CREATE INDEX IF NOT EXISTS idx_lesson_progress_user
  ON lesson_progress(user_id);

CREATE TABLE IF NOT EXISTS course_activity (
  user_id         TEXT PRIMARY KEY,
  streak_days     INTEGER NOT NULL DEFAULT 0,
  longest_streak  INTEGER NOT NULL DEFAULT 0,
  last_active_day TEXT,                                -- 'YYYY-MM-DD' UTC
  total_completed INTEGER NOT NULL DEFAULT 0,
  updated_at      TEXT
);

-- Cache nội dung bài (chỉ dạng-câu: SampleAnswer do LLM sinh) — USER-AGNOSTIC,
-- key (lesson_id, lang). TTL + version do content.py quyết định khi đọc. Phát âm
-- reuse words.suggestion_cache; rubric tính từ history mỗi lần (không cache).
CREATE TABLE IF NOT EXISTS lesson_content_cache (
  lesson_id     TEXT NOT NULL,
  lang          TEXT NOT NULL,
  cache_version INTEGER NOT NULL DEFAULT 1,
  content_json  TEXT NOT NULL,
  model         TEXT,
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  PRIMARY KEY (lesson_id, lang)
);
"""


# ── Kết nối / schema ─────────────────────────────────────────────────────


def _connect(cfg: Config) -> sqlite3.Connection:
    path = Path(cfg.course_db_path)
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


# ── Mastery: cursor + tally (mirror words.get_profile_cursor/apply_phoneme_tallies) ──


def get_scan_cursor(cfg: Config, user_id: str) -> tuple[str, str]:
    """Con trỏ quét history (last_scan_at, last_scan_id); ('', '') nếu chưa quét."""
    conn = _connect(cfg)
    try:
        row = conn.execute(
            "SELECT last_scan_at, last_scan_id FROM course_scan_state WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        return (row["last_scan_at"], row["last_scan_id"]) if row else ("", "")
    finally:
        conn.close()


def get_mastery_stats(cfg: Config, user_id: str, exam: str) -> dict:
    """Thống kê mastery của user cho 1 kỳ thi.

    Trả {'criteria': {crit: {attempts, score_sum}}, 'question_types': {qt: {...}}}.
    """
    conn = _connect(cfg)
    try:
        crit = {
            r["criterion"]: {"attempts": r["attempts"], "score_sum": r["score_sum"]}
            for r in conn.execute(
                "SELECT criterion, attempts, score_sum FROM criterion_stats "
                "WHERE user_id = ? AND exam = ?",
                (user_id, exam),
            ).fetchall()
        }
        qt = {
            r["question_type"]: {"attempts": r["attempts"], "score_sum": r["score_sum"]}
            for r in conn.execute(
                "SELECT question_type, attempts, score_sum FROM qtype_stats "
                "WHERE user_id = ? AND exam = ?",
                (user_id, exam),
            ).fetchall()
        }
        return {"criteria": crit, "question_types": qt}
    finally:
        conn.close()


def apply_mastery_tallies(
    cfg: Config,
    user_id: str,
    crit_tallies: dict[tuple[str, str], dict],
    qt_tallies: dict[tuple[str, str], dict],
    new_cursor: tuple[str, str],
) -> bool:
    """Cộng dồn tallies vào criterion_stats/qtype_stats + tiến con trỏ, trong 1
    transaction.

    tallies key theo (exam, criterion) / (exam, question_type); value có
    {'attempts': float, 'score_sum': float} (điểm đã chuẩn hóa 0-1 & weighted).

    Guard chống double-count khi 2 request đồng thời cùng quét 1 đoạn history:
    đọc LẠI con trỏ trong transaction — nếu đã >= new_cursor thì request kia đã
    apply xong đoạn này → bỏ qua. Trả True nếu có ghi. (Mirror
    words.apply_phoneme_tallies.)
    """
    conn = _connect(cfg)
    try:
        with conn:
            row = conn.execute(
                "SELECT last_scan_at, last_scan_id FROM course_scan_state WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            current = (row["last_scan_at"], row["last_scan_id"]) if row else ("", "")
            if current >= new_cursor:
                return False
            now = "strftime('%Y-%m-%dT%H:%M:%SZ','now')"
            for (exam, criterion), t in crit_tallies.items():
                conn.execute(
                    f"""
                    INSERT INTO criterion_stats
                      (user_id, exam, criterion, attempts, score_sum, updated_at)
                    VALUES (?, ?, ?, ?, ?, {now})
                    ON CONFLICT(user_id, exam, criterion) DO UPDATE SET
                      attempts   = attempts + excluded.attempts,
                      score_sum  = score_sum + excluded.score_sum,
                      updated_at = excluded.updated_at
                    """,
                    (user_id, exam, criterion, t["attempts"], t["score_sum"]),
                )
            for (exam, qtype), t in qt_tallies.items():
                conn.execute(
                    f"""
                    INSERT INTO qtype_stats
                      (user_id, exam, question_type, attempts, score_sum, updated_at)
                    VALUES (?, ?, ?, ?, ?, {now})
                    ON CONFLICT(user_id, exam, question_type) DO UPDATE SET
                      attempts   = attempts + excluded.attempts,
                      score_sum  = score_sum + excluded.score_sum,
                      updated_at = excluded.updated_at
                    """,
                    (user_id, exam, qtype, t["attempts"], t["score_sum"]),
                )
            conn.execute(
                f"""
                INSERT INTO course_scan_state (user_id, last_scan_at, last_scan_id)
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


# ── Tiến độ lesson (BỀN) ─────────────────────────────────────────────────


def get_progress(cfg: Config, user_id: str) -> dict[str, dict]:
    """{lesson_id: {status, best_score, attempts, completed_at}} của user."""
    conn = _connect(cfg)
    try:
        rows = conn.execute(
            "SELECT lesson_id, status, best_score, attempts, completed_at "
            "FROM lesson_progress WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        return {
            r["lesson_id"]: {
                "status": r["status"],
                "best_score": r["best_score"],
                "attempts": r["attempts"],
                "completed_at": r["completed_at"],
            }
            for r in rows
        }
    finally:
        conn.close()


def upsert_lesson_progress(
    cfg: Config,
    user_id: str,
    lesson_id: str,
    *,
    status: str,
    score: float | None,
) -> dict:
    """Ghi tiến độ 1 lesson (upsert theo (user_id, lesson_id)).

    - attempts += 1 mỗi lần gọi.
    - best_score giữ max giữa cũ và score mới (COALESCE để lần đầu không NULL).
    - status: 'done' ghi đè & set completed_at; 'in_progress' KHÔNG hạ 'done'
      đã có (đã hoàn thành thì vẫn hoàn thành, chỉ cập nhật best_score/attempts).
    """
    conn = _connect(cfg)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO lesson_progress
                  (user_id, lesson_id, status, best_score, attempts, completed_at, updated_at)
                VALUES (
                  ?, ?, ?, ?, 1,
                  CASE WHEN ?='done' THEN strftime('%Y-%m-%dT%H:%M:%SZ','now') END,
                  strftime('%Y-%m-%dT%H:%M:%SZ','now')
                )
                ON CONFLICT(user_id, lesson_id) DO UPDATE SET
                  status = CASE
                    WHEN lesson_progress.status='done' OR excluded.status='done'
                    THEN 'done' ELSE excluded.status END,
                  best_score = MAX(
                    COALESCE(lesson_progress.best_score, excluded.best_score),
                    COALESCE(excluded.best_score, lesson_progress.best_score)
                  ),
                  attempts = lesson_progress.attempts + 1,
                  completed_at = CASE
                    WHEN lesson_progress.completed_at IS NOT NULL
                      THEN lesson_progress.completed_at
                    WHEN excluded.status='done'
                      THEN strftime('%Y-%m-%dT%H:%M:%SZ','now')
                    ELSE NULL END,
                  updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                """,
                (user_id, lesson_id, status, score, status),
            )
            row = conn.execute(
                "SELECT lesson_id, status, best_score, attempts, completed_at "
                "FROM lesson_progress WHERE user_id = ? AND lesson_id = ?",
                (user_id, lesson_id),
            ).fetchone()
        return dict(row)
    finally:
        conn.close()


# ── Streak / hoạt động ───────────────────────────────────────────────────


def get_activity(cfg: Config, user_id: str) -> dict:
    """Streak/hoạt động của user; giá trị 0/None nếu chưa có."""
    conn = _connect(cfg)
    try:
        row = conn.execute(
            "SELECT streak_days, longest_streak, last_active_day, total_completed "
            "FROM course_activity WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            return {
                "streak_days": 0, "longest_streak": 0,
                "last_active_day": None, "total_completed": 0,
            }
        return dict(row)
    finally:
        conn.close()


def bump_streak(cfg: Config, user_id: str) -> dict:
    """Cập nhật streak khi user hoàn thành 1 lesson HÔM NAY (UTC).

    - last_active_day == hôm nay → chỉ tăng total_completed (streak giữ nguyên).
    - == hôm qua → streak_days += 1.
    - còn lại (gap hoặc lần đầu) → streak_days = 1.
    longest_streak = max(longest_streak, streak_days).
    """
    today = datetime.now(timezone.utc).date()
    conn = _connect(cfg)
    try:
        with conn:
            row = conn.execute(
                "SELECT streak_days, longest_streak, last_active_day, total_completed "
                "FROM course_activity WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row:
                prev_day = _parse_day(row["last_active_day"])
                if prev_day == today:
                    streak = row["streak_days"]
                elif prev_day == today - timedelta(days=1):
                    streak = row["streak_days"] + 1
                else:
                    streak = 1
                longest = max(row["longest_streak"], streak)
                total = row["total_completed"] + 1
            else:
                streak, longest, total = 1, 1, 1
            conn.execute(
                """
                INSERT INTO course_activity
                  (user_id, streak_days, longest_streak, last_active_day,
                   total_completed, updated_at)
                VALUES (?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ','now'))
                ON CONFLICT(user_id) DO UPDATE SET
                  streak_days = excluded.streak_days,
                  longest_streak = excluded.longest_streak,
                  last_active_day = excluded.last_active_day,
                  total_completed = excluded.total_completed,
                  updated_at = excluded.updated_at
                """,
                (user_id, streak, longest, today.isoformat(), total),
            )
        return {
            "streak_days": streak, "longest_streak": longest,
            "last_active_day": today.isoformat(), "total_completed": total,
        }
    finally:
        conn.close()


def _parse_day(day: str | None) -> date | None:
    try:
        return date.fromisoformat(day) if day else None
    except (TypeError, ValueError):
        return None


# ── Cache nội dung bài dạng-câu (LLM) ────────────────────────────────────


def get_lesson_content_cache(cfg: Config, lesson_id: str, lang: str) -> dict | None:
    """Entry cache {content, model, cache_version, created_at}; None nếu chưa có."""
    import json

    conn = _connect(cfg)
    try:
        row = conn.execute(
            "SELECT content_json, model, cache_version, created_at "
            "FROM lesson_content_cache WHERE lesson_id = ? AND lang = ?",
            (lesson_id, lang),
        ).fetchone()
        if not row:
            return None
        try:
            content = json.loads(row["content_json"])
        except (TypeError, ValueError):
            return None
        return {
            "content": content,
            "model": row["model"],
            "cache_version": row["cache_version"],
            "created_at": row["created_at"],
        }
    finally:
        conn.close()


def put_lesson_content_cache(
    cfg: Config, lesson_id: str, lang: str, content: dict, model: str | None,
    cache_version: int,
) -> None:
    """Ghi/ghi đè cache nội dung bài (upsert theo (lesson_id, lang))."""
    import json

    conn = _connect(cfg)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO lesson_content_cache
                  (lesson_id, lang, cache_version, content_json, model, created_at)
                VALUES (?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ','now'))
                ON CONFLICT(lesson_id, lang) DO UPDATE SET
                  cache_version = excluded.cache_version,
                  content_json = excluded.content_json,
                  model = excluded.model,
                  created_at = excluded.created_at
                """,
                (lesson_id, lang, cache_version, json.dumps(content, ensure_ascii=False), model),
            )
    finally:
        conn.close()


# ── Gộp tài khoản (dùng khi /auth/claim) ─────────────────────────────────


def merge_user(cfg: Config, from_user_id: str, to_user_id: str) -> int:
    """Gộp dữ liệu khóa học từ user ẩn danh sang tài khoản (khi /auth/claim).

    - lesson_progress + course_activity: chuyển sang tài khoản. lesson_progress
      dùng UPDATE OR IGNORE (bỏ qua lesson trùng do PK); course_activity chỉ
      chuyển nếu tài khoản CHƯA có (INSERT OR IGNORE style qua kiểm tra) — không
      cộng gộp streak thủ công. Xoá phần còn lại của user cũ.
    - criterion_stats/qtype_stats/course_scan_state: XOÁ của CẢ HAI user — mastery
      tự dựng lại tăng dần từ history đã gộp ở lần refresh sau (con trỏ rỗng).
      Đúng & đơn giản hơn cộng dồn thủ công (mirror words.merge_user).

    Trả số lesson_progress đã chuyển thành công.
    """
    from_user_id = validate_user_id(from_user_id)
    to_user_id = validate_user_id(to_user_id)
    if from_user_id == to_user_id:
        return 0
    conn = _connect(cfg)
    try:
        with conn:
            cur = conn.execute(
                "UPDATE OR IGNORE lesson_progress SET user_id = ? WHERE user_id = ?",
                (to_user_id, from_user_id),
            )
            moved = cur.rowcount
            conn.execute(
                "DELETE FROM lesson_progress WHERE user_id = ?", (from_user_id,)
            )
            # course_activity: giữ của tài khoản nếu đã có, else chuyển của anon.
            has_acct = conn.execute(
                "SELECT 1 FROM course_activity WHERE user_id = ?", (to_user_id,)
            ).fetchone()
            if has_acct:
                conn.execute(
                    "DELETE FROM course_activity WHERE user_id = ?", (from_user_id,)
                )
            else:
                conn.execute(
                    "UPDATE OR IGNORE course_activity SET user_id = ? WHERE user_id = ?",
                    (to_user_id, from_user_id),
                )
            # Mastery + con trỏ: xoá cả hai để rebuild từ history đã gộp.
            for table in ("criterion_stats", "qtype_stats", "course_scan_state"):
                conn.execute(
                    f"DELETE FROM {table} WHERE user_id IN (?, ?)",
                    (from_user_id, to_user_id),
                )
        return moved
    finally:
        conn.close()
