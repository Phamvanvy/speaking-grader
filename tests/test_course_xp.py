"""Test hệ XP/level/huy hiệu khóa học (src/course/xp.py) — lớp gamification.

Khóa chặt 5 ràng buộc chống-farm/tin-cậy đã chốt:
  RB#1 badge words = số TỪ DUY NHẤT đạt mastery (không đếm số lần luyện).
  RB#2 quota XP-practice TỔNG theo ngày (đa event vẫn ≤ cap).
  RB#3 mark_lesson_complete award đúng 1 lần ở first-transition → done.
  RB#4 merge_user gộp xp_daily (clamp cap) + union badge + cộng xp.
  RB#5 award chỉ nhận event+score; backend tự tính XP; event lạ → lỗi.
Cùng cờ COURSE_XP_ENABLED tắt → no-op. Không LLM, không server.
"""

from __future__ import annotations

import dataclasses
from datetime import date, timedelta

import pytest

import src.words as words
from src.config import load_config
from src.course import (
    award_practice_xp,
    buy_shop_item,
    equip_shop_item,
    get_leaderboard,
    get_shop,
    get_xp,
    mark_lesson_complete,
    merge_user,
    set_leaderboard_optin,
    store,
    xp,
)


def _names(*account_ids: str):
    """Fake resolve_usernames: chỉ các id được coi là 'tài khoản' mới có username."""
    accounts = set(account_ids)
    return lambda ids: {u: f"user_{u}" for u in ids if u in accounts}


@pytest.fixture()
def cfg(tmp_path):
    return dataclasses.replace(
        load_config(),
        anthropic_api_key=None,
        course_db_path=str(tmp_path / "course.db"),
        words_db_path=str(tmp_path / "words.db"),
        history_db_path=str(tmp_path / "history.db"),
        history_audio_dir=str(tmp_path / "history_audio"),
        course_xp_enabled=True,
    )


# ── Level curve ──────────────────────────────────────────────────────────


def test_level_curve_boundaries():
    assert xp.xp_to_level(0)["level"] == 1
    assert xp.xp_to_level(99)["level"] == 1
    assert xp.xp_to_level(100)["level"] == 2   # floor level 2 = 100
    assert xp.xp_to_level(299)["level"] == 2
    assert xp.xp_to_level(300)["level"] == 3   # floor level 3 = 300
    d = xp.xp_to_level(150)
    assert d["level"] == 2
    assert d["level_floor"] == 100 and d["level_ceil"] == 300
    assert d["into_level"] == 50 and d["span"] == 200


# ── RB#2: quota practice TỔNG theo ngày ──────────────────────────────────


def test_practice_daily_cap_total(cfg):
    u = "u-cap"
    last = None
    for _ in range(50):  # score 1.0 → 20 XP mỗi lần, thừa sức vượt cap 200
        last = award_practice_xp(cfg, u, "word_practice", 1.0)
    assert last["xp"] == xp.DAILY_PRACTICE_CAP  # đúng bằng cap, không hơn
    # gọi thêm nữa vẫn không tăng (đã cạn quota ngày)
    again = award_practice_xp(cfg, u, "word_practice", 1.0)
    assert again["xp"] == xp.DAILY_PRACTICE_CAP
    assert again["awarded"] == 0


def test_practice_backend_computes_xp_from_score(cfg):
    # RB#5: điểm cao được nhiều XP hơn điểm thấp — client không quyết định số XP.
    lo = award_practice_xp(cfg, "u-lo", "word_practice", 0.1)
    hi = award_practice_xp(cfg, "u-hi", "word_practice", 1.0)
    assert hi["awarded"] > lo["awarded"]
    assert lo["awarded"] >= xp.WORD_XP_MIN


def test_award_rejects_unknown_event(cfg):
    with pytest.raises(ValueError):
        award_practice_xp(cfg, "u1", "hack_event", 1.0)


def test_word_recall_event_accepted_and_shares_quota(cfg):
    # Mini-game không nói (word_recall) cấp XP như word_practice và CHỊU CHUNG
    # trần XP ngày — không mở kênh XP thoát cap (Phase 3).
    r = award_practice_xp(cfg, "u-recall", "word_recall", 1.0)
    assert r["enabled"] is True
    assert r["awarded"] == xp.WORD_XP_MAX  # score 1.0 → XP tối đa

    # Trộn word_practice + word_recall vẫn ≤ cap ngày TỔNG.
    u = "u-mixed"
    for i in range(50):
        award_practice_xp(cfg, u, "word_recall" if i % 2 else "word_practice", 1.0)
    assert get_xp(cfg, u)["xp"] == xp.DAILY_PRACTICE_CAP


# ── RB#3: mark_lesson_complete award đúng 1 lần (first-transition) ────────


def test_lesson_award_only_on_first_transition(cfg):
    u = "u-lesson"
    lid = "toeic.pron.th_family"
    # Lần đầu đạt ngưỡng → done + có XP.
    r1 = mark_lesson_complete(cfg, u, lid, 0.9, "toeic")
    assert r1["done"] is True
    assert "xp" in r1 and r1["xp"]["awarded"] > 0
    xp_after_first = r1["xp"]["xp"]
    # Luyện lại lesson ĐÃ done → KHÔNG award thêm (không có khóa 'xp').
    r2 = mark_lesson_complete(cfg, u, lid, 0.95, "toeic")
    assert r2["done"] is True
    assert "xp" not in r2
    assert get_xp(cfg, u)["xp"] == xp_after_first


def test_lesson_no_award_when_not_done(cfg):
    u = "u-nd"
    r = mark_lesson_complete(cfg, u, "toeic.pron.th_family", 0.5, "toeic")
    assert r["done"] is False
    assert "xp" not in r
    assert get_xp(cfg, u)["xp"] == 0
    assert xp.get_xp_state(cfg, u)["badges"] == []


def test_first_lesson_badge_on_completion(cfg):
    u = "u-badge"
    r = mark_lesson_complete(cfg, u, "toeic.pron.th_family", 0.9, "toeic")
    assert "first_lesson" in r.get("new_badges", [])


# ── RB#1: badge words theo TỪ DUY NHẤT đạt mastery ───────────────────────


def test_words_badge_counts_distinct_mastered_words(cfg):
    u = "u-words"
    # Luyện CÙNG một từ nhiều lần → chỉ 1 từ đạt mastery → KHÔNG đủ badge words_10.
    for _ in range(20):
        words.upsert_word(cfg, u, "practice", last_score=0.95)
    assert xp.check_and_award_badges(cfg, u) == [] or "words_10" not in xp.check_and_award_badges(cfg, u)
    assert words.count_words_at_mastery(cfg, u, xp.WORD_MASTERY_MIN) == 1
    # Thêm đủ 10 TỪ KHÁC NHAU đạt mastery → mới có badge.
    for i in range(10):
        words.upsert_word(cfg, u, f"distinct{i}", last_score=0.9)
    new = xp.check_and_award_badges(cfg, u)
    assert "words_10" in new
    # Idempotent: lần sau không cấp lại.
    assert "words_10" not in xp.check_and_award_badges(cfg, u)


def test_low_score_words_not_counted(cfg):
    u = "u-low"
    for i in range(15):
        words.upsert_word(cfg, u, f"weak{i}", last_score=0.5)  # dưới ngưỡng mastery
    assert words.count_words_at_mastery(cfg, u, xp.WORD_MASTERY_MIN) == 0
    assert "words_10" not in xp.check_and_award_badges(cfg, u)


# ── RB#4: merge_user gộp XP/badge/xp_daily (clamp cap) ───────────────────


def test_merge_user_sums_xp_and_unions_badges(cfg):
    anon, acct = "anon-1", "acct-1"
    award_practice_xp(cfg, anon, "word_practice", 1.0)   # anon có ít XP + badge?
    for i in range(10):
        words.upsert_word(cfg, anon, f"w{i}", last_score=0.9)
    xp.check_and_award_badges(cfg, anon)  # anon có words_10
    award_practice_xp(cfg, acct, "word_practice", 0.5)
    xp_anon = get_xp(cfg, anon)["xp"]
    xp_acct = get_xp(cfg, acct)["xp"]

    merge_user(cfg, anon, acct)

    merged = get_xp(cfg, acct)
    assert merged["xp"] == xp_anon + xp_acct
    assert "words_10" in [b["id"] for b in merged["badges"]]
    # user cũ sạch
    assert get_xp(cfg, anon)["xp"] == 0


def test_merge_user_clamps_daily_cap(cfg):
    anon, acct = "anon-2", "acct-2"
    # Cả hai đều đã cày sát/đầy cap trong CÙNG ngày.
    for _ in range(50):
        award_practice_xp(cfg, anon, "word_practice", 1.0)
    for _ in range(50):
        award_practice_xp(cfg, acct, "word_practice", 1.0)
    assert get_xp(cfg, anon)["xp"] == xp.DAILY_PRACTICE_CAP
    assert get_xp(cfg, acct)["xp"] == xp.DAILY_PRACTICE_CAP

    merge_user(cfg, anon, acct)

    # Tổng XP giữ nguyên (2×cap) NHƯNG quota ngày sau gộp phải bị clamp về cap →
    # award thêm hôm nay KHÔNG được cộng (đã đầy quota).
    conn = store._connect(cfg)
    try:
        today = xp._today()
        row = conn.execute(
            "SELECT practice_xp FROM xp_daily WHERE user_id = ? AND day = ?",
            (acct, today),
        ).fetchone()
    finally:
        conn.close()
    assert row["practice_xp"] == xp.DAILY_PRACTICE_CAP  # clamp, không phải 2×cap
    before = get_xp(cfg, acct)["xp"]
    after = award_practice_xp(cfg, acct, "word_practice", 1.0)
    assert after["awarded"] == 0 and after["xp"] == before


# ── Phase 2: nhiệm vụ ngày + xu ──────────────────────────────────────────


def test_daily_goal_progress_and_coins_once(cfg):
    u = "u-daily"
    # Chưa luyện → count 0, chưa xong, chưa có xu.
    st0 = get_xp(cfg, u)
    assert st0["daily"] == {
        "count": 0,
        "goal": xp.DAILY_GOAL,
        "coins_reward": xp.DAILY_GOAL_COINS,
        "done": False,
    }
    assert st0["coins"] == 0
    hit_events = []
    for i in range(xp.DAILY_GOAL):
        r = award_practice_xp(cfg, u, "word_practice", 0.9)
        assert r["daily"]["count"] == i + 1
        hit_events.append(r["daily_goal_hit"])
    # daily_goal_hit CHỈ True đúng lần chạm mốc (lần thứ DAILY_GOAL).
    assert hit_events == [False] * (xp.DAILY_GOAL - 1) + [True]
    st = get_xp(cfg, u)
    assert st["daily"]["done"] is True
    assert st["coins"] == xp.DAILY_GOAL_COINS
    # Luyện thêm KHÔNG cấp lại xu (idempotent 1/ngày) và không hit lần nữa.
    r_extra = award_practice_xp(cfg, u, "word_practice", 0.9)
    assert r_extra["daily_goal_hit"] is False
    assert r_extra["daily"]["count"] == xp.DAILY_GOAL + 1
    assert get_xp(cfg, u)["coins"] == xp.DAILY_GOAL_COINS


def test_daily_count_increments_past_xp_cap(cfg):
    # Nhiệm vụ ngày đếm SỐ TỪ luyện, độc lập trần XP: kịch cap XP vẫn tăng count.
    u = "u-daily-cap"
    for _ in range(50):
        award_practice_xp(cfg, u, "word_practice", 1.0)
    st = get_xp(cfg, u)
    assert st["xp"] == xp.DAILY_PRACTICE_CAP  # XP đụng trần
    assert st["daily"]["count"] == 50          # count vẫn đếm đủ
    assert st["coins"] == xp.DAILY_GOAL_COINS   # xu mốc vẫn cấp đúng 1 lần


def test_merge_user_sums_daily_count_and_coins(cfg):
    anon, acct = "anon-d", "acct-d"
    for _ in range(3):
        award_practice_xp(cfg, anon, "word_practice", 0.8)  # 3 từ, chưa đủ goal
    for _ in range(3):
        award_practice_xp(cfg, acct, "word_practice", 0.8)  # 3 từ, chưa đủ goal
    assert get_xp(cfg, anon)["coins"] == 0 and get_xp(cfg, acct)["coins"] == 0

    merge_user(cfg, anon, acct)

    merged = get_xp(cfg, acct)
    # count cộng dồn (3+3=6 ≥ goal) và user cũ sạch.
    assert merged["daily"]["count"] == 6
    assert merged["daily"]["done"] is True
    assert get_xp(cfg, anon)["daily"]["count"] == 0


# ── Phase 4: cửa hàng cosmetic (buy/equip/merge) ─────────────────────────


def _grant_coins(cfg, user_id: str, coins: int) -> None:
    """Nạp xu trực tiếp cho test (xu thật chỉ từ mốc nhiệm vụ ngày, cap 1/ngày)."""
    conn = store._connect(cfg)
    try:
        with conn:
            conn.execute(
                "INSERT INTO user_xp (user_id, xp, coins, updated_at) VALUES (?, 0, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET coins = user_xp.coins + excluded.coins",
                (user_id, coins, xp._now()),
            )
    finally:
        conn.close()


def test_shop_buy_deducts_coins_and_grants_item(cfg):
    u = "u-shop"
    item = xp.SHOP_ITEMS[0]
    _grant_coins(cfg, u, item["price"] + 5)
    bought = xp.buy_item(cfg, u, item["id"])
    assert bought["coins"] == 5  # trừ ĐÚNG giá catalog (client không gửi giá — RB#5)
    row = next(i for i in bought["items"] if i["id"] == item["id"])
    assert row["owned"] is True and row["equipped"] is False


def test_shop_buy_insufficient_coins_rejected(cfg):
    u = "u-poor"
    item = xp.SHOP_ITEMS[-1]
    _grant_coins(cfg, u, item["price"] - 1)
    with pytest.raises(ValueError):
        xp.buy_item(cfg, u, item["id"])
    st = xp.get_shop_state(cfg, u)
    assert st["coins"] == item["price"] - 1        # không trừ xu
    assert all(not i["owned"] for i in st["items"])  # không cấp item


def test_shop_buy_duplicate_rejected(cfg):
    u = "u-dup"
    item = xp.SHOP_ITEMS[0]
    _grant_coins(cfg, u, item["price"] * 2)
    xp.buy_item(cfg, u, item["id"])
    with pytest.raises(ValueError):
        xp.buy_item(cfg, u, item["id"])
    assert xp.get_shop_state(cfg, u)["coins"] == item["price"]  # chỉ trừ 1 lần


def test_shop_equip_one_per_slot(cfg):
    u = "u-equip"
    themes = [it for it in xp.SHOP_ITEMS if it["slot"] == "xp_theme"]
    a, b = themes[0], themes[1]
    _grant_coins(cfg, u, a["price"] + b["price"])
    xp.buy_item(cfg, u, a["id"])
    xp.buy_item(cfg, u, b["id"])
    xp.equip_item(cfg, u, a["id"], True)
    assert xp.get_xp_state(cfg, u)["cosmetics"]["xp_theme"] == a["id"]
    # Trang bị b cùng slot → a tự tháo (bất biến tối đa 1/slot).
    xp.equip_item(cfg, u, b["id"], True)
    assert xp.get_xp_state(cfg, u)["cosmetics"]["xp_theme"] == b["id"]
    # Tháo b → slot trống.
    xp.equip_item(cfg, u, b["id"], False)
    assert "xp_theme" not in xp.get_xp_state(cfg, u)["cosmetics"]


def test_shop_equip_unowned_rejected(cfg):
    with pytest.raises(ValueError):
        xp.equip_item(cfg, "u-noown", xp.SHOP_ITEMS[0]["id"], True)


def test_shop_buy_unknown_item_rejected(cfg):
    _grant_coins(cfg, "u-x", 999)
    with pytest.raises(ValueError):
        xp.buy_item(cfg, "u-x", "no_such_item")


def test_merge_user_unions_inventory(cfg):
    anon, acct = "anon-inv", "acct-inv"
    a = next(it for it in xp.SHOP_ITEMS if it["slot"] == "xp_theme")
    b = next(it for it in xp.SHOP_ITEMS if it["slot"] == "streak_flame")
    _grant_coins(cfg, anon, a["price"])
    _grant_coins(cfg, acct, b["price"])
    xp.buy_item(cfg, anon, a["id"])
    xp.equip_item(cfg, anon, a["id"], True)
    xp.buy_item(cfg, acct, b["id"])

    merge_user(cfg, anon, acct)

    owned = {i["id"] for i in xp.get_shop_state(cfg, acct)["items"] if i["owned"]}
    assert a["id"] in owned and b["id"] in owned          # union đủ cả hai
    assert xp.get_xp_state(cfg, acct)["cosmetics"]["xp_theme"] == a["id"]  # giữ equipped
    assert all(not i["owned"] for i in xp.get_shop_state(cfg, anon)["items"])  # user cũ sạch


def test_merge_user_dedupes_same_slot_equipped(cfg):
    # Cả anon lẫn acct đều trang bị item KHÁC NHAU cùng slot 'xp_theme' trước khi
    # gộp → sau merge chỉ còn ĐÚNG 1 item equipped ở slot đó (bất biến 1/slot).
    anon, acct = "anon-slot", "acct-slot"
    themes = [it for it in xp.SHOP_ITEMS if it["slot"] == "xp_theme"]
    a, b = themes[0], themes[1]
    _grant_coins(cfg, anon, a["price"])
    _grant_coins(cfg, acct, b["price"])
    xp.buy_item(cfg, anon, a["id"])
    xp.equip_item(cfg, anon, a["id"], True)
    xp.buy_item(cfg, acct, b["id"])
    xp.equip_item(cfg, acct, b["id"], True)

    merge_user(cfg, anon, acct)

    st = xp.get_shop_state(cfg, acct)
    equipped_theme = [i["id"] for i in st["items"] if i["equipped"] and i["slot"] == "xp_theme"]
    assert len(equipped_theme) == 1  # đúng 1 item/slot, không phải 2


# ── Phase 5: bảng xếp hạng tuần (opt-in, chỉ tài khoản) ──────────────────


def test_leaderboard_optin_defaults_off_and_toggles(cfg):
    u = "acct-lb"
    assert get_leaderboard(cfg, u, _names(u))["opted_in"] is False  # mặc định riêng tư
    set_leaderboard_optin(cfg, u, True)
    assert get_leaderboard(cfg, u, _names(u))["opted_in"] is True
    set_leaderboard_optin(cfg, u, False)
    assert get_leaderboard(cfg, u, _names(u))["opted_in"] is False


def test_leaderboard_ranks_only_opted_in_accounts(cfg):
    for u in ("a", "b", "c"):
        set_leaderboard_optin(cfg, u, True)
    award_practice_xp(cfg, "a", "word_practice", 1.0)          # a: 20 XP tuần
    for _ in range(3):
        award_practice_xp(cfg, "b", "word_practice", 1.0)      # b: 60 XP tuần
    award_practice_xp(cfg, "c", "word_practice", 1.0)          # c opt-in NHƯNG ẩn danh
    award_practice_xp(cfg, "d", "word_practice", 1.0)          # d KHÔNG opt-in

    lb = get_leaderboard(cfg, "a", _names("a", "b"))           # chỉ a,b là tài khoản
    names = [e["username"] for e in lb["entries"]]
    assert names == ["user_b", "user_a"]                       # b nhiều XP → hạng 1
    assert [e["rank"] for e in lb["entries"]] == [1, 2]
    assert lb["me"]["username"] == "user_a" and lb["me"]["is_me"] is True
    # c (ẩn danh) và d (opt-out) không xuất hiện.
    assert "user_c" not in names and "user_d" not in names


def test_leaderboard_weekly_window_excludes_old_xp(cfg):
    u = "oldie"
    set_leaderboard_optin(cfg, u, True)
    old_day = (date.today() - timedelta(days=10)).isoformat()  # ngoài cửa sổ 7 ngày
    conn = store._connect(cfg)
    try:
        with conn:
            conn.execute(
                "INSERT INTO xp_daily (user_id, day, practice_xp, practice_count, "
                "goal_coins_awarded) VALUES (?, ?, 999, 0, 0)",
                (u, old_day),
            )
    finally:
        conn.close()
    lb = get_leaderboard(cfg, u, _names(u))
    assert lb["me"]["weekly_xp"] == 0        # XP 10 ngày trước KHÔNG tính vào tuần
    assert lb["goal"] == xp.WEEKLY_XP_GOAL


def test_leaderboard_anon_viewer_has_no_rank(cfg):
    # Ẩn danh xem được bảng nhưng không có hạng (me=None) dù có opt-in row.
    set_leaderboard_optin(cfg, "anon-view", True)
    award_practice_xp(cfg, "anon-view", "word_practice", 1.0)
    lb = get_leaderboard(cfg, "anon-view", _names())  # không id nào là tài khoản
    assert lb["entries"] == [] and lb["me"] is None


def test_leaderboard_flag_off_is_noop(cfg):
    off = dataclasses.replace(cfg, course_xp_enabled=False)
    assert get_leaderboard(off, "u1", _names("u1")) == {"enabled": False}
    assert set_leaderboard_optin(off, "u1", True) == {"enabled": False}


# ── Cờ COURSE_XP_ENABLED tắt → no-op ─────────────────────────────────────


def test_flag_off_is_noop(cfg):
    off = dataclasses.replace(cfg, course_xp_enabled=False)
    assert get_xp(off, "u1") == {"enabled": False}
    assert award_practice_xp(off, "u1", "word_practice", 1.0) == {"enabled": False}
    # mark_lesson_complete vẫn chạy nhưng KHÔNG kèm xp.
    r = mark_lesson_complete(off, "u1", "toeic.pron.th_family", 0.9, "toeic")
    assert r["done"] is True and "xp" not in r


def test_shop_flag_off_is_noop(cfg):
    off = dataclasses.replace(cfg, course_xp_enabled=False)
    item_id = xp.SHOP_ITEMS[0]["id"]
    assert get_shop(off, "u1") == {"enabled": False}
    assert buy_shop_item(off, "u1", item_id) == {"enabled": False}
    assert equip_shop_item(off, "u1", item_id, True) == {"enabled": False}
