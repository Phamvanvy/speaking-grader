"""Test Quest nhập vai (Phase 3B) — src/course quests: build/cache + complete gate.

Kiểm tra offline (không LLM thật, không server):
- build_roleplay: cache-first (LLM chỉ gọi 1 lần), guard nội dung không hợp lệ → None.
- list_quests: liệt kê topic curated + trạng thái cleared từ quest_clears (bonus).
- complete_quest: clamp + ngưỡng; đạt → ghi quest_clears + award_bonus_xp MỘT LẦN
  (gọi lại KHÔNG cấp thêm XP); dưới ngưỡng → không clear; kind sai → ValueError.
- BONUS-only: Quest KHÔNG đụng lesson_progress/mastery → progress.done/total giữ nguyên.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.config import load_config
from src.course import (
    complete_quest,
    generate,
    get_roleplay_quest,
    list_quests,
    quests as _quests,
    store,
)
from src.course import get_story_quest
from src.course.generate import QUEST_DONE_THRESHOLD
from src.schema import RolePlayScript, RolePlayTurn, StoryQuest, StorySegment

_EXAM = "toeic"


@pytest.fixture()
def cfg(tmp_path):
    return dataclasses.replace(
        load_config(),
        anthropic_api_key=None,  # LLM thật không bao giờ được gọi
        course_xp_enabled=True,
        course_db_path=str(tmp_path / "course.db"),
        words_db_path=str(tmp_path / "words.db"),
        history_db_path=str(tmp_path / "history.db"),
        history_audio_dir=str(tmp_path / "history_audio"),
    )


def _fake_script() -> RolePlayScript:
    return RolePlayScript(
        scenario="Bạn đang nhận phòng khách sạn.",
        role_user="khách du lịch",
        role_npc="lễ tân",
        turns=[
            RolePlayTurn(npc="Good evening! Do you have a reservation?",
                         expected_user="Yes, I booked a room under Pham.", hint="Xác nhận đã đặt phòng."),
            RolePlayTurn(npc="Great, here is your key.",
                         expected_user="Thank you very much.", hint="Cảm ơn."),
        ],
    )


def _fake_story() -> StoryQuest:
    return StoryQuest(
        title="A busy morning",
        segments=[
            StorySegment(text="Anna woke up late on a rainy Monday morning."),
            StorySegment(text="She rushed to catch the crowded city bus."),
            StorySegment(text="Luckily, she arrived at the office just in time."),
        ],
    )


def _topic_slug() -> str:
    return _quests.list_roleplay_topics(_EXAM)[0].slug


def _story_slug() -> str:
    return _quests.list_story_topics(_EXAM)[0].slug


# ── build_roleplay: cache + guard ────────────────────────────────────────


def test_build_roleplay_generates_then_caches(cfg, monkeypatch):
    calls = {"n": 0}

    def fake_suggest(config, exam, setting):
        calls["n"] += 1
        return _fake_script()

    monkeypatch.setattr("src.suggest.suggest_roleplay", fake_suggest)
    slug = _topic_slug()

    first = _quests.build_roleplay(cfg, cfg, _EXAM, slug, "vi")
    assert first is not None
    assert len(first["turns"]) == 2
    assert first["turns"][0]["expected_user"]
    # Lần 2 phải lấy TỪ CACHE (không gọi LLM lần nữa).
    second = _quests.build_roleplay(cfg, cfg, _EXAM, slug, "vi")
    assert second == first
    assert calls["n"] == 1


def test_build_roleplay_unknown_topic_is_none(cfg):
    assert _quests.build_roleplay(cfg, cfg, _EXAM, "no_such_topic", "vi") is None


def test_build_roleplay_invalid_script_is_none(cfg, monkeypatch):
    def fake_suggest(config, exam, setting):
        return RolePlayScript(scenario="x", role_user="a", role_npc="b", turns=[])

    monkeypatch.setattr("src.suggest.suggest_roleplay", fake_suggest)
    assert _quests.build_roleplay(cfg, cfg, _EXAM, _topic_slug(), "vi") is None


def test_build_roleplay_llm_error_is_none(cfg, monkeypatch):
    def boom(config, exam, setting):
        raise RuntimeError("LLM down")

    monkeypatch.setattr("src.suggest.suggest_roleplay", boom)
    assert _quests.build_roleplay(cfg, cfg, _EXAM, _topic_slug(), "vi") is None


# ── build_story: cache + guard ───────────────────────────────────────────


def test_build_story_generates_then_caches(cfg, monkeypatch):
    calls = {"n": 0}

    def fake_suggest(config, exam, setting):
        calls["n"] += 1
        return _fake_story()

    monkeypatch.setattr("src.suggest.suggest_story", fake_suggest)
    slug = _story_slug()

    first = _quests.build_story(cfg, cfg, _EXAM, slug, "vi")
    assert first is not None
    assert len(first["segments"]) == 3
    assert first["title"] == "A busy morning"
    second = _quests.build_story(cfg, cfg, _EXAM, slug, "vi")
    assert second == first
    assert calls["n"] == 1  # lần 2 lấy từ cache


def test_build_story_too_few_segments_is_none(cfg, monkeypatch):
    def fake_suggest(config, exam, setting):
        return StoryQuest(title="x", segments=[StorySegment(text="one two three four")])

    monkeypatch.setattr("src.suggest.suggest_story", fake_suggest)
    assert _quests.build_story(cfg, cfg, _EXAM, _story_slug(), "vi") is None


def test_build_story_short_segment_is_none(cfg, monkeypatch):
    def fake_suggest(config, exam, setting):
        return StoryQuest(
            title="x",
            segments=[
                StorySegment(text="A full sentence here."),
                StorySegment(text="Too short."),  # < 4 từ
                StorySegment(text="Another full sentence here now."),
            ],
        )

    monkeypatch.setattr("src.suggest.suggest_story", fake_suggest)
    assert _quests.build_story(cfg, cfg, _EXAM, _story_slug(), "vi") is None


def test_get_story_quest_shape(cfg, monkeypatch):
    monkeypatch.setattr("src.suggest.suggest_story", lambda c, e, s: _fake_story())
    slug = _story_slug()
    view = get_story_quest(cfg, cfg, "u1", _EXAM, slug, "vi")
    assert view is not None
    assert view["kind"] == "story"
    assert view["quest_id"] == _quests.story_quest_id(_EXAM, slug)
    assert view["cleared"] is False
    assert len(view["segments"]) == 3


def test_complete_story_awards_and_counts_quest_badge(cfg):
    # Story clear cũng tính vào badge quest_1 (quest_clears chung 2 loại).
    qid = _quests.story_quest_id(_EXAM, _story_slug())
    r = complete_quest(cfg, "u1", qid, "story", 0.9)
    assert r["done"] is True
    assert r["xp"]["awarded"] > 0
    assert "quest_1" in (r.get("new_badges") or [])


# ── list_quests ──────────────────────────────────────────────────────────


def test_list_quests_lists_topics_uncleared(cfg):
    view = list_quests(cfg, "u1", _EXAM)
    assert view["exam"] == _EXAM
    expected = len(_quests.list_roleplay_topics(_EXAM)) + len(_quests.list_story_topics(_EXAM))
    assert len(view["quests"]) == expected
    assert all(not q["cleared"] for q in view["quests"])
    kinds = {q["kind"] for q in view["quests"]}
    assert kinds == {"roleplay", "story"}


def test_list_quests_unsupported_exam_empty(cfg):
    view = list_quests(cfg, "u1", "topik")
    assert view["quests"] == []


# ── complete_quest: award MỘT LẦN + gate ─────────────────────────────────


def _quest_id() -> str:
    return _quests.roleplay_quest_id(_EXAM, _topic_slug())


def test_complete_awards_once_and_is_idempotent(cfg):
    qid = _quest_id()
    r1 = complete_quest(cfg, "u1", qid, "roleplay", 0.9)
    assert r1["done"] is True
    assert r1["xp"]["awarded"] > 0
    assert "quest_1" in (r1.get("new_badges") or [])

    # Hoàn thành lại → KHÔNG cấp thêm XP (idempotent qua quest_clears first-transition).
    r2 = complete_quest(cfg, "u1", qid, "roleplay", 0.95)
    assert r2["done"] is True
    assert "xp" not in r2
    assert store.get_quest_clears(cfg, "u1")[qid]["best_score"] == pytest.approx(0.95)


def test_below_threshold_does_not_clear(cfg):
    low = max(0.0, QUEST_DONE_THRESHOLD - 0.1)
    r = complete_quest(cfg, "u1", _quest_id(), "roleplay", low)
    assert r["done"] is False
    assert "xp" not in r
    assert store.get_quest_clears(cfg, "u1") == {}


def test_complete_clamps_score(cfg):
    r = complete_quest(cfg, "u1", _quest_id(), "roleplay", 5.0)
    assert r["done"] is True
    assert r["score"] == pytest.approx(1.0)


def test_complete_bad_kind_raises(cfg):
    with pytest.raises(ValueError):
        complete_quest(cfg, "u1", _quest_id(), "not_a_kind", 0.9)


def test_quest_does_not_touch_lesson_progress(cfg):
    before = store.get_progress(cfg, "u1")
    complete_quest(cfg, "u1", _quest_id(), "roleplay", 1.0)
    after = store.get_progress(cfg, "u1")
    assert before == after  # Quest là BONUS: không đụng tiến độ lesson


# ── get_roleplay_quest: view-model + trạng thái ──────────────────────────


def test_get_roleplay_quest_shape(cfg, monkeypatch):
    monkeypatch.setattr("src.suggest.suggest_roleplay", lambda c, e, s: _fake_script())
    slug = _topic_slug()
    view = get_roleplay_quest(cfg, cfg, "u1", _EXAM, slug, "vi")
    assert view is not None
    assert view["kind"] == "roleplay"
    assert view["threshold"] == pytest.approx(QUEST_DONE_THRESHOLD)
    assert view["quest_id"] == _quests.roleplay_quest_id(_EXAM, slug)
    assert view["cleared"] is False
    assert len(view["turns"]) == 2
