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

import pytest

import src.words as words
from src.config import load_config
from src.course import (
    award_practice_xp,
    get_xp,
    mark_lesson_complete,
    merge_user,
    store,
    xp,
)


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


# ── Cờ COURSE_XP_ENABLED tắt → no-op ─────────────────────────────────────


def test_flag_off_is_noop(cfg):
    off = dataclasses.replace(cfg, course_xp_enabled=False)
    assert get_xp(off, "u1") == {"enabled": False}
    assert award_practice_xp(off, "u1", "word_practice", 1.0) == {"enabled": False}
    # mark_lesson_complete vẫn chạy nhưng KHÔNG kèm xp.
    r = mark_lesson_complete(off, "u1", "toeic.pron.th_family", 0.9, "toeic")
    assert r["done"] is True and "xp" not in r
