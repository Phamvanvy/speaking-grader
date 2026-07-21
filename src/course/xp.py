"""Hệ thưởng XP / level / huy hiệu (gamification) cho khóa học.

Lớp CỘNG THÊM hoàn toàn ngoài đường chấm điểm — chỉ đọc điểm ĐÃ chuẩn hóa (0-1)
rồi quy ra XP. Cùng course.db với store.py (mỗi hàm mở connection mới, WAL,
transaction ngắn `with conn:`), tái dùng bảng `user_xp` / `user_badges` /
`xp_daily` do store._DDL tạo (schema v2).

Ranh giới tin cậy: **backend tự tính XP** từ `score` — client KHÔNG BAO GIỜ gửi
số XP (chỉ gửi event + score). Chống farm: quota XP-practice TỔNG theo ngày
(`xp_daily`, một hạn mức chung, không tách source).

Level là hàm thuần của tổng XP (đường cong tam giác) — backend là nguồn sự thật.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config

logger = logging.getLogger("toeic.course.xp")

# ── Hằng số cân bằng game ────────────────────────────────────────────────
# Ngưỡng XP để LÊN level L (floor của level L) = LEVEL_BASE * L*(L-1)/2 → mỗi
# level rộng thêm LEVEL_BASE XP so với level trước (2→100, 3→300, 4→600, 5→1000).
LEVEL_BASE = 100
# Trần XP practice cấp trong 1 ngày (UTC) — chống luyện lặp để cày XP.
DAILY_PRACTICE_CAP = 200
# XP mỗi lần luyện 1 từ = round(score * WORD_XP_MAX), tối thiểu WORD_XP_MIN.
WORD_XP_MAX = 20
WORD_XP_MIN = 2
# XP hoàn thành 1 lesson (chỉ first-transition) = LESSON_XP_BASE + round(score*BONUS).
LESSON_XP_BASE = 50
LESSON_XP_SCORE_BONUS = 50
# Ngưỡng "đạt mastery" của 1 TỪ để tính huy hiệu words_* (last_score >= ngưỡng).
WORD_MASTERY_MIN = 0.8
WORD_PERFECT_MIN = 0.999


# ── Level curve (thuần, không I/O) ───────────────────────────────────────


def xp_to_level(xp: int) -> dict:
    """Quy tổng XP → thông tin level cho thanh tiến độ trong-cấp.

    Trả {level, xp, level_floor, level_ceil, into_level, span}. Backend là nguồn
    sự thật của level; frontend chỉ vẽ.
    """
    xp = max(0, int(xp))
    # level = L lớn nhất sao cho LEVEL_BASE*L*(L-1)/2 <= xp.
    level = int((1 + math.sqrt(1 + 8 * xp / LEVEL_BASE)) / 2)
    level = max(1, level)
    floor = LEVEL_BASE * level * (level - 1) // 2
    ceil = LEVEL_BASE * level * (level + 1) // 2
    return {
        "level": level,
        "xp": xp,
        "level_floor": floor,
        "level_ceil": ceil,
        "into_level": xp - floor,
        "span": ceil - floor,
    }


# ── Kết nối (course.db — schema do store._connect dựng) ──────────────────


def _connect(cfg: Config) -> sqlite3.Connection:
    # Bảng XP nằm trong course.db; import store để đảm bảo schema v2 đã dựng
    # (store._connect chạy _DDL đầy đủ + bump user_version).
    from . import store

    conn = store._connect(cfg)  # đã PRAGMA WAL + tạo bảng (IF NOT EXISTS)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _read_xp(conn: sqlite3.Connection, user_id: str) -> tuple[int, int]:
    row = conn.execute(
        "SELECT xp, coins FROM user_xp WHERE user_id = ?", (user_id,)
    ).fetchone()
    return (int(row["xp"]), int(row["coins"])) if row else (0, 0)


def _list_badges(conn: sqlite3.Connection, user_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT badge_id, earned_at FROM user_badges WHERE user_id = ? "
        "ORDER BY earned_at",
        (user_id,),
    ).fetchall()
    return [{"id": r["badge_id"], "earned_at": r["earned_at"]} for r in rows]


# ── Đọc trạng thái ───────────────────────────────────────────────────────


def get_xp_state(cfg: Config, user_id: str) -> dict:
    """Trạng thái XP đầy đủ: {xp, level, level_floor, level_ceil, into_level,
    span, coins, badges:[{id, earned_at}]}."""
    conn = _connect(cfg)
    try:
        xp, coins = _read_xp(conn, user_id)
        badges = _list_badges(conn, user_id)
    finally:
        conn.close()
    state = xp_to_level(xp)
    state["coins"] = coins
    state["badges"] = badges
    return state


# ── Huy hiệu ─────────────────────────────────────────────────────────────
# (badge_id, hàm điều kiện trên ctx). Idempotent: đã có trong user_badges thì
# bỏ qua (PK). Tiêu chí tính từ dữ liệu ĐÃ CÓ — không đếm sự kiện practice.
_BADGE_RULES: list[tuple[str, "callable"]] = [
    ("first_lesson", lambda c: c["lessons_done"] >= 1),
    ("streak_3", lambda c: c["streak"] >= 3),
    ("streak_7", lambda c: c["streak"] >= 7),
    ("streak_30", lambda c: c["streak"] >= 30),
    # words_* = số TỪ DUY NHẤT đạt mastery (không phải số lần luyện).
    ("words_10", lambda c: c["words_mastered"] >= 10),
    ("words_50", lambda c: c["words_mastered"] >= 50),
    ("words_100", lambda c: c["words_mastered"] >= 100),
    ("perfect_10", lambda c: c["words_perfect"] >= 10),
    ("level_5", lambda c: c["level"] >= 5),
    ("level_10", lambda c: c["level"] >= 10),
]


def _badge_context(cfg: Config, user_id: str) -> dict:
    """Gom số liệu đánh giá huy hiệu từ các store (progress/streak/xp/words)."""
    from .. import words
    from . import store

    xp, _coins = 0, 0
    conn = _connect(cfg)
    try:
        xp, _coins = _read_xp(conn, user_id)
    finally:
        conn.close()
    progress = store.get_progress(cfg, user_id)
    lessons_done = sum(1 for p in progress.values() if p.get("status") == "done")
    activity = store.get_activity(cfg, user_id)
    # longest_streak (max từng đạt) để huy hiệu không "mất" khi streak rơi.
    streak = max(
        int(activity.get("streak_days") or 0),
        int(activity.get("longest_streak") or 0),
    )
    words_mastered = words.count_words_at_mastery(cfg, user_id, WORD_MASTERY_MIN)
    words_perfect = words.count_words_at_mastery(cfg, user_id, WORD_PERFECT_MIN)
    return {
        "level": xp_to_level(xp)["level"],
        "lessons_done": lessons_done,
        "streak": streak,
        "words_mastered": words_mastered,
        "words_perfect": words_perfect,
    }


def check_and_award_badges(cfg: Config, user_id: str) -> list[str]:
    """Đánh giá & cấp các huy hiệu MỚI đạt (idempotent). Trả list badge_id vừa cấp."""
    ctx = _badge_context(cfg, user_id)
    earned_ids = {
        bid
        for bid, cond in _BADGE_RULES
        if _safe(cond, ctx)
    }
    if not earned_ids:
        return []
    conn = _connect(cfg)
    try:
        have = {
            r["badge_id"]
            for r in conn.execute(
                "SELECT badge_id FROM user_badges WHERE user_id = ?", (user_id,)
            ).fetchall()
        }
        new_ids = sorted(earned_ids - have)
        if new_ids:
            with conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO user_badges (user_id, badge_id, earned_at) "
                    "VALUES (?, ?, ?)",
                    [(user_id, bid, _now()) for bid in new_ids],
                )
        return new_ids
    finally:
        conn.close()


def _safe(cond, ctx) -> bool:
    try:
        return bool(cond(ctx))
    except Exception:  # noqa: BLE001 — badge không được làm hỏng luồng award
        return False


# ── Cấp XP ───────────────────────────────────────────────────────────────


def _award_xp(cfg: Config, user_id: str, amount: int) -> tuple[int, int]:
    """Cộng `amount` XP (>=0) vào user_xp, trả (xp_trước, xp_sau)."""
    amount = max(0, int(amount))
    conn = _connect(cfg)
    try:
        with conn:
            before, _coins = _read_xp(conn, user_id)
            if amount:
                conn.execute(
                    """
                    INSERT INTO user_xp (user_id, xp, coins, updated_at)
                    VALUES (?, ?, 0, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                      xp = user_xp.xp + excluded.xp,
                      updated_at = excluded.updated_at
                    """,
                    (user_id, amount, _now()),
                )
            after = before + amount
        return before, after
    finally:
        conn.close()


def award_practice_xp(cfg: Config, user_id: str, score: float) -> dict:
    """XP cho 1 lần luyện từ (event 'word_practice'). Backend tự tính từ score,
    áp QUOTA NGÀY TỔNG (không bypass được bằng nhiều event). Trả state mới +
    {awarded, leveled_up, new_badges}."""
    try:
        score = max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        score = 0.0
    computed = max(WORD_XP_MIN, round(score * WORD_XP_MAX))
    today = _today()
    conn = _connect(cfg)
    try:
        with conn:
            before, _coins = _read_xp(conn, user_id)
            used_row = conn.execute(
                "SELECT practice_xp FROM xp_daily WHERE user_id = ? AND day = ?",
                (user_id, today),
            ).fetchone()
            used = int(used_row["practice_xp"]) if used_row else 0
            remaining = max(0, DAILY_PRACTICE_CAP - used)
            grant = min(computed, remaining)
            if grant > 0:
                conn.execute(
                    """
                    INSERT INTO user_xp (user_id, xp, coins, updated_at)
                    VALUES (?, ?, 0, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                      xp = user_xp.xp + excluded.xp, updated_at = excluded.updated_at
                    """,
                    (user_id, grant, _now()),
                )
                conn.execute(
                    """
                    INSERT INTO xp_daily (user_id, day, practice_xp)
                    VALUES (?, ?, ?)
                    ON CONFLICT(user_id, day) DO UPDATE SET
                      practice_xp = xp_daily.practice_xp + excluded.practice_xp
                    """,
                    (user_id, today, grant),
                )
            after = before + grant
    finally:
        conn.close()
    new_badges = check_and_award_badges(cfg, user_id)
    return _award_result(cfg, user_id, before, after, grant, new_badges)


def award_lesson_xp(
    cfg: Config, user_id: str, score: float, streak_days: int = 0
) -> dict:
    """XP hoàn thành 1 lesson (KÊNH RIÊNG, không chịu quota practice). Chỉ nên gọi
    ở first-transition → done (guard ở mark_lesson_complete). Trả state mới +
    {awarded, leveled_up, new_badges}."""
    try:
        score = max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        score = 0.0
    amount = LESSON_XP_BASE + round(score * LESSON_XP_SCORE_BONUS)
    # Thưởng streak nhỏ (cap 7 ngày) để khuyến khích học đều mà không cày được.
    amount += min(max(0, int(streak_days)), 7) * 2
    before, after = _award_xp(cfg, user_id, amount)
    new_badges = check_and_award_badges(cfg, user_id)
    return _award_result(cfg, user_id, before, after, amount, new_badges)


def _award_result(
    cfg: Config,
    user_id: str,
    before: int,
    after: int,
    awarded: int,
    new_badges: list[str],
) -> dict:
    state = get_xp_state(cfg, user_id)
    state["awarded"] = awarded
    state["leveled_up"] = xp_to_level(after)["level"] > xp_to_level(before)["level"]
    state["new_badges"] = new_badges
    return state


# ── Gộp khi /auth/claim (chạy trong transaction của store.merge_user) ────


def merge_user_xp(
    conn: sqlite3.Connection, from_user_id: str, to_user_id: str
) -> None:
    """Gộp XP/huy hiệu/quota-ngày từ user ẩn danh sang tài khoản, DÙNG CHUNG
    connection/transaction của store.merge_user.

    - user_xp: cộng dồn xp + coins.
    - user_badges: union (INSERT OR IGNORE) — giữ earned_at sớm nhất theo PK.
    - xp_daily: cộng dồn practice_xp theo `day` rồi CLAMP về DAILY_PRACTICE_CAP để
      việc gộp KHÔNG reset/bypass quota ngày.
    Sau đó xoá dữ liệu user cũ.
    """
    if from_user_id == to_user_id:
        return
    conn.execute(
        """
        INSERT INTO user_xp (user_id, xp, coins, updated_at)
        SELECT ?, xp, coins, ? FROM user_xp WHERE user_id = ?
        ON CONFLICT(user_id) DO UPDATE SET
          xp = user_xp.xp + excluded.xp,
          coins = user_xp.coins + excluded.coins,
          updated_at = excluded.updated_at
        """,
        (to_user_id, _now(), from_user_id),
    )
    conn.execute("DELETE FROM user_xp WHERE user_id = ?", (from_user_id,))
    conn.execute(
        "INSERT OR IGNORE INTO user_badges (user_id, badge_id, earned_at) "
        "SELECT ?, badge_id, earned_at FROM user_badges WHERE user_id = ?",
        (to_user_id, from_user_id),
    )
    conn.execute("DELETE FROM user_badges WHERE user_id = ?", (from_user_id,))
    conn.execute(
        f"""
        INSERT INTO xp_daily (user_id, day, practice_xp)
        SELECT ?, day, practice_xp FROM xp_daily WHERE user_id = ?
        ON CONFLICT(user_id, day) DO UPDATE SET
          practice_xp = MIN({DAILY_PRACTICE_CAP},
                            xp_daily.practice_xp + excluded.practice_xp)
        """,
        (to_user_id, from_user_id),
    )
    conn.execute("DELETE FROM xp_daily WHERE user_id = ?", (from_user_id,))
