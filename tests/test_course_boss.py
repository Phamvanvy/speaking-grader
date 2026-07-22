"""Test Boss cuối chặng (Phase 3A) — src/course boss gate + award MỘT LẦN + bonus.

Kiểm tra offline (không LLM, không server):
- build_course gắn node boss mỗi unit; status theo tiến độ lesson.
- Gate server-side: chưa hoàn thành hết lesson trong unit → get_unit_boss_content +
  complete_unit_boss raise PermissionError (403).
- complete_unit_boss: clamp + ngưỡng; đạt → ghi unit_boss + award_bonus_xp MỘT LẦN
  (gọi lại KHÔNG cấp thêm XP); dưới ngưỡng → không hạ.
- BONUS-only: Boss KHÔNG đụng lesson_progress/mastery → progress.done/total giữ nguyên.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.config import load_config
from src.course import (
    complete_unit_boss,
    generate,
    get_unit_boss_content,
    mark_lesson_complete,
    store,
)
from src.course.generate import BOSS_DONE_THRESHOLD
from src.course.syllabus import get_unit

_EXAM = "toeic"
_UNIT_ID = "toeic.pron"


@pytest.fixture()
def cfg(tmp_path):
    return dataclasses.replace(
        load_config(),
        anthropic_api_key=None,
        course_xp_enabled=True,
        course_db_path=str(tmp_path / "course.db"),
        words_db_path=str(tmp_path / "words.db"),
        history_db_path=str(tmp_path / "history.db"),
        history_audio_dir=str(tmp_path / "history_audio"),
    )


def _finish_all_lessons(cfg, user_id, unit_id):
    """Đánh dấu MỌI lesson trong unit là done (mở khóa Boss)."""
    unit = get_unit(unit_id)
    for ls in unit.lessons:
        mark_lesson_complete(cfg, user_id, ls.id, 1.0, ls.exam)


def _build_view(cfg, user_id):
    return generate.build_course(
        _EXAM, {}, [], store.get_progress(cfg, user_id), store.get_activity(cfg, user_id),
        store.get_boss_states(cfg, user_id),
    )


def test_build_course_has_boss_node_locked_initially(cfg):
    view = _build_view(cfg, "u1")
    for unit in view["units"]:
        assert "boss" in unit
        assert unit["boss"]["id"] == f"{unit['id']}.boss"
        assert unit["boss"]["status"] == "locked"  # chưa done lesson nào


def test_boss_locked_gate_raises(cfg):
    # Chưa hoàn thành lesson → cả content lẫn complete đều bị chặn (403 ở API).
    with pytest.raises(PermissionError):
        get_unit_boss_content(cfg, cfg, "u1", _UNIT_ID, "vi")
    with pytest.raises(PermissionError):
        complete_unit_boss(cfg, "u1", _UNIT_ID, 1.0)


def test_boss_available_after_all_lessons_done(cfg):
    _finish_all_lessons(cfg, "u1", _UNIT_ID)
    view = _build_view(cfg, "u1")
    boss = next(u["boss"] for u in view["units"] if u["id"] == _UNIT_ID)
    assert boss["status"] == "available"


def test_complete_awards_once_and_is_idempotent(cfg):
    _finish_all_lessons(cfg, "u1", _UNIT_ID)

    r1 = complete_unit_boss(cfg, "u1", _UNIT_ID, 0.9)
    assert r1["done"] is True
    assert r1["best_score"] == pytest.approx(0.9)
    assert "xp" in r1 and r1["xp"]["awarded"] > 0
    assert "boss_1" in (r1.get("new_badges") or [])
    xp_after_first = r1["xp"]["xp"]

    # Hạ lại → KHÔNG cấp thêm XP (idempotent qua unit_boss first-transition).
    r2 = complete_unit_boss(cfg, "u1", _UNIT_ID, 0.95)
    assert r2["done"] is True
    assert "xp" not in r2  # không first-transition → không award
    assert store.get_boss_states(cfg, "u1")[f"{_UNIT_ID}.boss"]["best_score"] == pytest.approx(0.95)
    # Tổng XP không tăng do hạ lại.
    from src.course import xp as _xp

    assert _xp.get_xp_state(cfg, "u1")["xp"] == xp_after_first


def test_below_threshold_does_not_beat(cfg):
    _finish_all_lessons(cfg, "u1", _UNIT_ID)
    low = max(0.0, BOSS_DONE_THRESHOLD - 0.1)
    r = complete_unit_boss(cfg, "u1", _UNIT_ID, low)
    assert r["done"] is False
    assert store.get_boss_states(cfg, "u1") == {}  # chưa hạ → không ghi state


def test_boss_does_not_touch_lesson_progress_totals(cfg):
    _finish_all_lessons(cfg, "u1", _UNIT_ID)
    before = _build_view(cfg, "u1")["progress"]
    complete_unit_boss(cfg, "u1", _UNIT_ID, 1.0)
    after = _build_view(cfg, "u1")["progress"]
    # Boss là BONUS: done/total của khóa học KHÔNG đổi khi hạ Boss.
    assert after["done"] == before["done"]
    assert after["total"] == before["total"]
