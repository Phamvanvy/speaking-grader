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
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..config import Config

logger = logging.getLogger("toeic.course.xp")

# ── Hằng số cân bằng game ────────────────────────────────────────────────
# Ngưỡng XP để LÊN level L (floor của level L) = LEVEL_BASE * L*(L-1)/2 → mỗi
# level rộng thêm LEVEL_BASE XP so với level trước (2→100, 3→300, 4→600, 5→1000).
LEVEL_BASE = 100
# Trần XP practice cấp trong 1 ngày (UTC) — chống luyện lặp để cày XP.
DAILY_PRACTICE_CAP = 200
# Nhiệm vụ ngày: luyện đủ DAILY_GOAL từ (đếm KỂ CẢ khi đã kịch trần XP) → thưởng
# DAILY_GOAL_COINS xu, MỘT LẦN/ngày (cột goal_coins_awarded chống cấp lại).
DAILY_GOAL = 5
DAILY_GOAL_COINS = 15
# XP mỗi lần luyện 1 từ = round(score * WORD_XP_MAX), tối thiểu WORD_XP_MIN.
WORD_XP_MAX = 20
WORD_XP_MIN = 2
# XP hoàn thành 1 lesson (chỉ first-transition) = LESSON_XP_BASE + round(score*BONUS).
LESSON_XP_BASE = 50
LESSON_XP_SCORE_BONUS = 50
# XP+xu thưởng khi hoàn thành Boss/Quest (Phase 3) — KÊNH RIÊNG, KHÔNG dính quota
# practice, cấp MỘT LẦN (caller đảm bảo idempotent qua bảng state). Cao hơn lesson
# vì là thử thách tổng hợp; kèm xu (nguồn xu mới ngoài nhiệm vụ ngày).
BONUS_XP_BASE = 60
BONUS_XP_SCORE_BONUS = 40
BONUS_COINS = 20
# Thử thách tuần: mục tiêu XP-practice trong cửa sổ 7 ngày (chỉ để tạo động lực +
# vẽ thanh tiến độ trên bảng xếp hạng; KHÔNG cấp huy hiệu riêng — badge tính từ dữ
# liệu tích lũy). Trần practice 200/ngày → tối đa 1400/tuần, khó cày.
WEEKLY_XP_GOAL = 300
LEADERBOARD_WINDOW_DAYS = 7
# Ngưỡng "đạt mastery" của 1 TỪ để tính huy hiệu words_* (last_score >= ngưỡng).
WORD_MASTERY_MIN = 0.8
WORD_PERFECT_MIN = 0.999

# ── Cửa hàng cosmetic (Phase 4 game hóa) ─────────────────────────────────
# Catalog TĨNH — nguồn sự thật của GIÁ nằm ở backend (client KHÔNG gửi giá, RB#5).
# Item thuần HIỂN THỊ (theme thanh XP, màu ngọn lửa streak); mua bằng `coins`
# (nguồn xu = thưởng mốc nhiệm vụ ngày, xem DAILY_GOAL_COINS — đã có cap 1/ngày).
# `slot`: tối đa 1 item được TRANG BỊ mỗi slot; ánh xạ id→style nằm ở frontend
# (features/gamify/cosmetics.ts) — backend chỉ giữ id/slot/giá/nhãn.
SHOP_ITEMS: list[dict] = [
    # slot 'xp_theme' — màu thanh XP.
    {"id": "xp_ocean", "slot": "xp_theme", "price": 40, "icon": "🌊", "label": "Đại dương", "desc": "Thanh XP xanh biển"},
    {"id": "xp_forest", "slot": "xp_theme", "price": 40, "icon": "🌲", "label": "Rừng xanh", "desc": "Thanh XP xanh lá"},
    {"id": "xp_sunset", "slot": "xp_theme", "price": 60, "icon": "🌅", "label": "Hoàng hôn", "desc": "Thanh XP hồng cam"},
    {"id": "xp_royal", "slot": "xp_theme", "price": 80, "icon": "👑", "label": "Hoàng gia", "desc": "Thanh XP tím ánh kim"},
    # slot 'streak_flame' — màu ngọn lửa streak.
    {"id": "flame_azure", "slot": "streak_flame", "price": 50, "icon": "💙", "label": "Lửa lam", "desc": "Ngọn lửa xanh dương"},
    {"id": "flame_violet", "slot": "streak_flame", "price": 50, "icon": "💜", "label": "Lửa tím", "desc": "Ngọn lửa tím"},
    {"id": "flame_emerald", "slot": "streak_flame", "price": 70, "icon": "💚", "label": "Lửa ngọc", "desc": "Ngọn lửa xanh ngọc"},
]
_SHOP_BY_ID: dict[str, dict] = {it["id"]: it for it in SHOP_ITEMS}


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


def _read_daily(conn: sqlite3.Connection, user_id: str) -> dict:
    """Tiến độ nhiệm vụ HÔM NAY (UTC): {count, goal, coins_reward, done}."""
    row = conn.execute(
        "SELECT practice_count FROM xp_daily WHERE user_id = ? AND day = ?",
        (user_id, _today()),
    ).fetchone()
    count = int(row["practice_count"]) if row else 0
    return {
        "count": count,
        "goal": DAILY_GOAL,
        "coins_reward": DAILY_GOAL_COINS,
        "done": count >= DAILY_GOAL,
    }


def _read_inventory(conn: sqlite3.Connection, user_id: str) -> list[dict]:
    """Item đã mua của user: [{item_id, equipped}]. Bỏ item không còn trong catalog."""
    rows = conn.execute(
        "SELECT item_id, equipped FROM user_inventory WHERE user_id = ?", (user_id,)
    ).fetchall()
    return [
        {"item_id": r["item_id"], "equipped": bool(r["equipped"])}
        for r in rows
        if r["item_id"] in _SHOP_BY_ID
    ]


def _equipped_map(inventory: list[dict]) -> dict[str, str]:
    """{slot: item_id} cho các item ĐANG trang bị (frontend áp cosmetic từ đây)."""
    out: dict[str, str] = {}
    for inv in inventory:
        if inv["equipped"]:
            out[_SHOP_BY_ID[inv["item_id"]]["slot"]] = inv["item_id"]
    return out


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
        daily = _read_daily(conn, user_id)
        inventory = _read_inventory(conn, user_id)
    finally:
        conn.close()
    state = xp_to_level(xp)
    state["coins"] = coins
    state["badges"] = badges
    state["daily"] = daily
    # Cosmetic đang trang bị {slot: item_id} — frontend áp theme thanh XP / màu lửa.
    state["cosmetics"] = _equipped_map(inventory)
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
    # Boss cuối chặng (Phase 3A) — số Boss ĐÃ hạ (bảng unit_boss).
    ("boss_1", lambda c: c["bosses_beaten"] >= 1),
    ("boss_5", lambda c: c["bosses_beaten"] >= 5),
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
    bosses_beaten = len(store.get_boss_states(cfg, user_id))
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
        "bosses_beaten": bosses_beaten,
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
    daily_goal_hit = False
    conn = _connect(cfg)
    try:
        with conn:
            before, _coins = _read_xp(conn, user_id)
            row = conn.execute(
                "SELECT practice_xp, practice_count, goal_coins_awarded "
                "FROM xp_daily WHERE user_id = ? AND day = ?",
                (user_id, today),
            ).fetchone()
            used = int(row["practice_xp"]) if row else 0
            count = int(row["practice_count"]) if row else 0
            goal_awarded = int(row["goal_coins_awarded"]) if row else 0
            remaining = max(0, DAILY_PRACTICE_CAP - used)
            grant = min(computed, remaining)
            # practice_count LUÔN +1 (nhiệm vụ ngày đếm số từ luyện, độc lập trần XP).
            new_count = count + 1
            # Xu thưởng mốc ngày: cấp MỘT LẦN khi lần đầu chạm DAILY_GOAL trong ngày.
            coins_reward = 0
            if not goal_awarded and new_count >= DAILY_GOAL:
                coins_reward = DAILY_GOAL_COINS
                goal_awarded = 1
                daily_goal_hit = True
            if grant > 0 or coins_reward > 0:
                conn.execute(
                    """
                    INSERT INTO user_xp (user_id, xp, coins, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                      xp = user_xp.xp + excluded.xp,
                      coins = user_xp.coins + excluded.coins,
                      updated_at = excluded.updated_at
                    """,
                    (user_id, grant, coins_reward, _now()),
                )
            conn.execute(
                """
                INSERT INTO xp_daily
                  (user_id, day, practice_xp, practice_count, goal_coins_awarded)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(user_id, day) DO UPDATE SET
                  practice_xp = xp_daily.practice_xp + excluded.practice_xp,
                  practice_count = xp_daily.practice_count + 1,
                  goal_coins_awarded =
                    MAX(xp_daily.goal_coins_awarded, excluded.goal_coins_awarded)
                """,
                (user_id, today, grant, goal_awarded),
            )
            after = before + grant
    finally:
        conn.close()
    new_badges = check_and_award_badges(cfg, user_id)
    result = _award_result(cfg, user_id, before, after, grant, new_badges)
    result["daily_goal_hit"] = daily_goal_hit
    return result


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


def award_bonus_xp(cfg: Config, user_id: str, kind: str, score: float) -> dict:
    """XP + xu cho hoàn thành Boss/Quest (kind: 'boss'|'roleplay'|'story') — Phase 3.

    KÊNH RIÊNG, cấp MỘT LẦN: caller PHẢI đảm bảo idempotent (chỉ gọi khi
    first-transition, vd `store.mark_boss_beaten` trả True). KHÔNG dùng quota
    practice / `word_recall` → không mở kênh farm. Backend tự tính từ score;
    client KHÔNG gửi số XP. Trả state mới + {awarded, leveled_up, new_badges}.
    """
    try:
        score = max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        score = 0.0
    amount = BONUS_XP_BASE + round(score * BONUS_XP_SCORE_BONUS)
    conn = _connect(cfg)
    try:
        with conn:
            before, _coins = _read_xp(conn, user_id)
            conn.execute(
                """
                INSERT INTO user_xp (user_id, xp, coins, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                  xp = user_xp.xp + excluded.xp,
                  coins = user_xp.coins + excluded.coins,
                  updated_at = excluded.updated_at
                """,
                (user_id, amount, BONUS_COINS, _now()),
            )
            after = before + amount
    finally:
        conn.close()
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


# ── Cửa hàng cosmetic ────────────────────────────────────────────────────


def _shop_items_view(coins: int, inventory: list[dict]) -> list[dict]:
    """Catalog + cờ owned/equipped theo inventory của user (giữ thứ tự SHOP_ITEMS)."""
    owned = {inv["item_id"] for inv in inventory}
    equipped = {inv["item_id"] for inv in inventory if inv["equipped"]}
    out: list[dict] = []
    for it in SHOP_ITEMS:
        row = dict(it)
        row["owned"] = it["id"] in owned
        row["equipped"] = it["id"] in equipped
        row["affordable"] = row["owned"] or coins >= it["price"]
        out.append(row)
    return out


def get_shop_state(cfg: Config, user_id: str) -> dict:
    """Trạng thái cửa hàng: {coins, items:[...owned/equipped/affordable], cosmetics}."""
    conn = _connect(cfg)
    try:
        _xp, coins = _read_xp(conn, user_id)
        inventory = _read_inventory(conn, user_id)
    finally:
        conn.close()
    return {
        "coins": coins,
        "items": _shop_items_view(coins, inventory),
        "cosmetics": _equipped_map(inventory),
    }


def buy_item(cfg: Config, user_id: str, item_id: str) -> dict:
    """Mua 1 item cosmetic bằng xu. GIÁ lấy từ catalog backend (client không gửi
    giá — RB#5). Trừ coins + ghi inventory trong 1 transaction; raise ValueError nếu
    item lạ / đã sở hữu / không đủ xu. Trả trạng thái cửa hàng mới."""
    item = _SHOP_BY_ID.get((item_id or "").strip())
    if item is None:
        raise ValueError(f"Không có vật phẩm '{item_id}'.")
    price = int(item["price"])
    conn = _connect(cfg)
    try:
        with conn:
            row = conn.execute(
                "SELECT coins FROM user_xp WHERE user_id = ?", (user_id,)
            ).fetchone()
            coins = int(row["coins"]) if row else 0
            already = conn.execute(
                "SELECT 1 FROM user_inventory WHERE user_id = ? AND item_id = ?",
                (user_id, item["id"]),
            ).fetchone()
            if already:
                raise ValueError("Đã sở hữu vật phẩm này.")
            if coins < price:
                raise ValueError(f"Không đủ xu (cần {price}, có {coins}).")
            conn.execute(
                """
                INSERT INTO user_xp (user_id, xp, coins, updated_at)
                VALUES (?, 0, 0, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                  coins = user_xp.coins - ?,
                  updated_at = excluded.updated_at
                """,
                (user_id, _now(), price),
            )
            conn.execute(
                "INSERT INTO user_inventory (user_id, item_id, equipped, acquired_at) "
                "VALUES (?, ?, 0, ?)",
                (user_id, item["id"], _now()),
            )
    finally:
        conn.close()
    return get_shop_state(cfg, user_id)


def equip_item(cfg: Config, user_id: str, item_id: str, equipped: bool) -> dict:
    """Trang bị / tháo 1 item ĐÃ SỞ HỮU. Trang bị → tự tháo các item khác CÙNG slot
    (tối đa 1/slot) trong 1 transaction. raise ValueError nếu chưa sở hữu. Trả
    trạng thái cửa hàng mới."""
    item = _SHOP_BY_ID.get((item_id or "").strip())
    if item is None:
        raise ValueError(f"Không có vật phẩm '{item_id}'.")
    slot = item["slot"]
    same_slot = [it["id"] for it in SHOP_ITEMS if it["slot"] == slot]
    conn = _connect(cfg)
    try:
        with conn:
            owned = conn.execute(
                "SELECT 1 FROM user_inventory WHERE user_id = ? AND item_id = ?",
                (user_id, item["id"]),
            ).fetchone()
            if not owned:
                raise ValueError("Chưa sở hữu vật phẩm này.")
            if equipped:
                # Tháo mọi item cùng slot trước (giữ bất biến tối đa 1/slot).
                placeholders = ",".join("?" for _ in same_slot)
                conn.execute(
                    f"UPDATE user_inventory SET equipped = 0 "
                    f"WHERE user_id = ? AND item_id IN ({placeholders})",
                    (user_id, *same_slot),
                )
            conn.execute(
                "UPDATE user_inventory SET equipped = ? "
                "WHERE user_id = ? AND item_id = ?",
                (1 if equipped else 0, user_id, item["id"]),
            )
    finally:
        conn.close()
    return get_shop_state(cfg, user_id)


# ── Bảng xếp hạng tuần (opt-in) ──────────────────────────────────────────


def _week_start() -> str:
    """Ngày đầu cửa sổ xếp hạng (hôm nay − 6, UTC) dạng 'YYYY-MM-DD'."""
    d = datetime.now(timezone.utc).date() - timedelta(days=LEADERBOARD_WINDOW_DAYS - 1)
    return d.isoformat()


def set_leaderboard_optin(cfg: Config, user_id: str, opt_in: bool) -> None:
    """Bật/tắt việc user xuất hiện trên bảng xếp hạng (mặc định tắt — riêng tư).
    Chỉ nên gọi cho tài khoản đăng nhập (gate account ở api.py)."""
    conn = _connect(cfg)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO user_xp (user_id, xp, coins, leaderboard_opt_in, updated_at)
                VALUES (?, 0, 0, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                  leaderboard_opt_in = excluded.leaderboard_opt_in,
                  updated_at = excluded.updated_at
                """,
                (user_id, 1 if opt_in else 0, _now()),
            )
    finally:
        conn.close()


def get_leaderboard_optin(cfg: Config, user_id: str) -> bool:
    conn = _connect(cfg)
    try:
        row = conn.execute(
            "SELECT leaderboard_opt_in FROM user_xp WHERE user_id = ?", (user_id,)
        ).fetchone()
        return bool(row["leaderboard_opt_in"]) if row else False
    finally:
        conn.close()


def weekly_leaderboard_rows(cfg: Config) -> list[dict]:
    """[{user_id, total_xp, weekly_xp}] cho MỌI user đã opt-in (kể cả tuần này 0 XP).
    weekly_xp = Σ practice_xp trong cửa sổ 7 ngày. Việc lọc "chỉ tài khoản" + gắn
    username do lớp trên làm (xp.py không biết khái niệm account)."""
    start = _week_start()
    conn = _connect(cfg)
    try:
        rows = conn.execute(
            """
            SELECT u.user_id AS user_id, u.xp AS total_xp,
                   COALESCE(SUM(CASE WHEN d.day >= ? THEN d.practice_xp ELSE 0 END), 0)
                     AS weekly_xp
            FROM user_xp u
            LEFT JOIN xp_daily d ON d.user_id = u.user_id
            WHERE u.leaderboard_opt_in = 1
            GROUP BY u.user_id, u.xp
            """,
            (start,),
        ).fetchall()
        return [
            {
                "user_id": r["user_id"],
                "total_xp": int(r["total_xp"]),
                "weekly_xp": int(r["weekly_xp"]),
            }
            for r in rows
        ]
    finally:
        conn.close()


# ── Gộp khi /auth/claim (chạy trong transaction của store.merge_user) ────


def _dedupe_equipped(conn: sqlite3.Connection, user_id: str) -> None:
    """Ép bất biến "tối đa 1 item trang bị mỗi slot" cho user (dùng sau merge, khi
    union inventory có thể để 2 item cùng slot cùng equipped). Giữ item ACQUIRED
    SỚM NHẤT của mỗi slot, tháo phần còn lại — tất định."""
    rows = conn.execute(
        "SELECT item_id FROM user_inventory WHERE user_id = ? AND equipped = 1 "
        "ORDER BY acquired_at, item_id",
        (user_id,),
    ).fetchall()
    seen: set[str] = set()
    for r in rows:
        item = _SHOP_BY_ID.get(r["item_id"])
        if item is None:
            continue
        slot = item["slot"]
        if slot in seen:
            conn.execute(
                "UPDATE user_inventory SET equipped = 0 WHERE user_id = ? AND item_id = ?",
                (user_id, r["item_id"]),
            )
        else:
            seen.add(slot)


def merge_user_xp(
    conn: sqlite3.Connection, from_user_id: str, to_user_id: str
) -> None:
    """Gộp XP/huy hiệu/quota-ngày từ user ẩn danh sang tài khoản, DÙNG CHUNG
    connection/transaction của store.merge_user.

    - user_xp: cộng dồn xp + coins.
    - user_badges: union (INSERT OR IGNORE) — giữ earned_at sớm nhất theo PK.
    - user_inventory: union (INSERT OR IGNORE) — giữ item đã sở hữu của cả hai;
      trạng thái `equipped` giữ theo tài khoản đích (không đụng nếu đã có item đó).
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
        "INSERT OR IGNORE INTO user_inventory (user_id, item_id, equipped, acquired_at) "
        "SELECT ?, item_id, equipped, acquired_at FROM user_inventory WHERE user_id = ?",
        (to_user_id, from_user_id),
    )
    conn.execute("DELETE FROM user_inventory WHERE user_id = ?", (from_user_id,))
    _dedupe_equipped(conn, to_user_id)  # union có thể để 2 item cùng slot equipped
    conn.execute(
        f"""
        INSERT INTO xp_daily
          (user_id, day, practice_xp, practice_count, goal_coins_awarded)
        SELECT ?, day, practice_xp, practice_count, goal_coins_awarded
          FROM xp_daily WHERE user_id = ?
        ON CONFLICT(user_id, day) DO UPDATE SET
          practice_xp = MIN({DAILY_PRACTICE_CAP},
                            xp_daily.practice_xp + excluded.practice_xp),
          practice_count = xp_daily.practice_count + excluded.practice_count,
          goal_coins_awarded =
            MAX(xp_daily.goal_coins_awarded, excluded.goal_coins_awarded)
        """,
        (to_user_id, from_user_id),
    )
    conn.execute("DELETE FROM xp_daily WHERE user_id = ?", (from_user_id,))
