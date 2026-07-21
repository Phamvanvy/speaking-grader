"""Test offline Phase 0 khóa học cá nhân hóa (src/course/).

Kiểm tra: tổng hợp mastery từ history result_json (chuẩn hóa 0-1 + TÁCH theo kỳ
thi), con trỏ quét idempotent + guard chống double-count, tiến độ lesson bền,
streak (liên tiếp/gap), build_course cá nhân hóa (cold-start + ưu tiên âm yếu +
mở khóa tuần tự trong Unit + badge focus), ngưỡng done theo dimension, và gộp tài
khoản. Không gọi LLM, không cần server.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src import history, word_suggest
from src.config import load_config
from src.course import (
    content,
    generate,
    get_lesson_content,
    mark_lesson_complete,
    merge_user,
    profile,
    store,
)
from src.course.syllabus import SUPPORTED_EXAMS, all_lessons


@pytest.fixture()
def cfg(tmp_path):
    return dataclasses.replace(
        load_config(),
        anthropic_api_key=None,
        course_db_path=str(tmp_path / "course.db"),
        words_db_path=str(tmp_path / "words.db"),
        history_db_path=str(tmp_path / "history.db"),
        history_audio_dir=str(tmp_path / "history_audio"),
    )


def _recent(days_ago: int = 1) -> str:
    """Timestamp gần đây (recency weight = 1.0 dù test chạy lúc nào)."""
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _result(exam, question_type, criteria, overall):
    """result_json tối giản (khớp report.build_output): exam/question_type/scores."""
    field = {
        "toeic": "estimated_toeic_score",
        "ielts": "estimated_ielts_band",
        "topik": "estimated_topik_score",
    }[exam]
    return {
        "exam": exam,
        "question_type": question_type,
        "scores": {
            "criteria": [{"criterion": k, "score": v} for k, v in criteria],
            field: overall,
        },
    }


def _insert(cfg, user_id, created_at, rec_id, result):
    conn = history._connect(cfg)
    try:
        with conn:
            conn.execute(
                "INSERT INTO history_records (id, user_id, kind, result_json, created_at)"
                " VALUES (?, ?, 'single', ?, ?)",
                (rec_id, user_id, json.dumps(result), created_at),
            )
    finally:
        conn.close()


# ── Tổng hợp mastery ──────────────────────────────────────────────────────


def test_norm_criteria_scales_per_exam():
    toeic = _result("toeic", "read_aloud", [("pronunciation", 1.5)], 120)
    assert profile._norm_criteria(toeic, "toeic") == [("pronunciation", 0.5)]
    ielts = _result("ielts", "part1_interview", [("pronunciation", 4.5)], 6.0)
    assert profile._norm_criteria(ielts, "ielts") == [("pronunciation", 0.5)]


def test_norm_overall_scales_per_exam():
    assert profile._norm_overall(
        _result("toeic", "read_aloud", [], 100), "toeic"
    ) == pytest.approx(0.5)
    assert profile._norm_overall(
        _result("ielts", "part1_interview", [], 9), "ielts"
    ) == pytest.approx(1.0)


def test_refresh_mastery_normalized_and_exam_separated(cfg):
    uid = "u1"
    _insert(cfg, uid, _recent(2), "aaa",
            _result("toeic", "read_aloud",
                    [("pronunciation", 1.5), ("intonation_stress", 3.0)], 120))
    _insert(cfg, uid, _recent(1), "bbb",
            _result("ielts", "part1_interview", [("pronunciation", 9.0)], 9.0))
    profile.refresh_mastery(cfg, uid)

    toeic = profile.get_mastery(cfg, uid, "toeic")
    assert toeic["criteria"]["pronunciation"]["mastery"] == pytest.approx(0.5)
    assert toeic["criteria"]["intonation_stress"]["mastery"] == pytest.approx(1.0)
    assert toeic["question_types"]["read_aloud"]["mastery"] == pytest.approx(0.6)

    ielts = profile.get_mastery(cfg, uid, "ielts")
    # TÁCH theo kỳ thi: pronunciation IELTS (9/9=1.0) KHÔNG trộn với TOEIC (0.5).
    assert ielts["criteria"]["pronunciation"]["mastery"] == pytest.approx(1.0)
    assert "intonation_stress" not in ielts["criteria"]


def test_refresh_mastery_cursor_idempotent(cfg):
    uid = "u1"
    _insert(cfg, uid, _recent(1), "aaa",
            _result("toeic", "read_aloud", [("pronunciation", 1.5)], 120))
    profile.refresh_mastery(cfg, uid)
    profile.refresh_mastery(cfg, uid)  # quét lại → no-op
    stats = store.get_mastery_stats(cfg, uid, "toeic")
    assert stats["criteria"]["pronunciation"]["attempts"] == pytest.approx(1.0)


def test_min_attempts_weakness_none(cfg):
    uid = "u1"
    _insert(cfg, uid, _recent(1), "aaa",
            _result("toeic", "read_aloud", [("pronunciation", 0.0)], 40))
    profile.refresh_mastery(cfg, uid)
    m = profile.get_mastery(cfg, uid, "toeic")
    # 1 attempt < _MIN_ATTEMPTS(2) → weakness None dù mastery tính được.
    assert m["criteria"]["pronunciation"]["mastery"] == pytest.approx(0.0)
    assert m["criteria"]["pronunciation"]["weakness"] is None


def test_apply_mastery_tallies_concurrent_guard(cfg):
    uid = "u1"
    crit = {("toeic", "grammar"): {"attempts": 1.0, "score_sum": 0.5}}
    cur = ("2026-07-01T00:00:01Z", "aaa")
    assert store.apply_mastery_tallies(cfg, uid, crit, {}, cur)
    # Request thứ 2 cùng đoạn (cursor không tiến) → guard bỏ qua.
    assert not store.apply_mastery_tallies(cfg, uid, crit, {}, cur)
    assert store.get_mastery_stats(cfg, uid, "toeic")["criteria"]["grammar"][
        "attempts"
    ] == pytest.approx(1.0)


def test_unsupported_exam_not_tallied(cfg):
    uid = "u1"
    _insert(cfg, uid, _recent(1), "aaa",
            {"exam": "toefl", "question_type": "speaking_1",
             "scores": {"criteria": [{"criterion": "delivery", "score": 3}],
                        "estimated_toefl_score": 25}})
    profile.refresh_mastery(cfg, uid)
    # toefl chưa có khóa học → không tally (CRITERION_MAX không chứa toefl).
    assert store.get_mastery_stats(cfg, uid, "toefl") == {"criteria": {}, "question_types": {}}


# ── Tiến độ + streak ──────────────────────────────────────────────────────


def test_upsert_lesson_progress_best_and_done(cfg):
    uid = "u1"
    lid = "toeic.rubric.grammar"
    store.upsert_lesson_progress(cfg, uid, lid, status="in_progress", score=0.4)
    store.upsert_lesson_progress(cfg, uid, lid, status="done", score=0.8)
    # in_progress điểm thấp hơn KHÔNG hạ done, best_score giữ max.
    store.upsert_lesson_progress(cfg, uid, lid, status="in_progress", score=0.2)
    prog = store.get_progress(cfg, uid)[lid]
    assert prog["status"] == "done"
    assert prog["best_score"] == pytest.approx(0.8)
    assert prog["attempts"] == 3
    assert prog["completed_at"] is not None


def _set_activity(cfg, uid, last_day, streak, longest, total):
    conn = sqlite3.connect(cfg.course_db_path)
    with conn:
        conn.execute(
            "INSERT INTO course_activity (user_id, streak_days, longest_streak,"
            " last_active_day, total_completed) VALUES (?, ?, ?, ?, ?)",
            (uid, streak, longest, last_day, total),
        )
    conn.close()


def test_bump_streak_consecutive_gap_sameday(cfg):
    today = datetime.now(timezone.utc).date()
    # Lần đầu (chưa có row) → streak 1.
    assert store.bump_streak(cfg, "new")["streak_days"] == 1

    # Hôm qua → +1.
    _set_activity(cfg, "u_consec", (today - timedelta(days=1)).isoformat(), 3, 5, 3)
    r = store.bump_streak(cfg, "u_consec")
    assert r["streak_days"] == 4 and r["longest_streak"] == 5 and r["total_completed"] == 4

    # Cách 3 ngày → reset về 1.
    _set_activity(cfg, "u_gap", (today - timedelta(days=3)).isoformat(), 9, 9, 9)
    assert store.bump_streak(cfg, "u_gap")["streak_days"] == 1

    # Cùng ngày → streak giữ nguyên, total +1.
    _set_activity(cfg, "u_same", today.isoformat(), 4, 7, 10)
    r = store.bump_streak(cfg, "u_same")
    assert r["streak_days"] == 4 and r["total_completed"] == 11


# ── build_course (hàm thuần) ──────────────────────────────────────────────


def _empty_mastery():
    return {"criteria": {}, "question_types": {}}


def test_build_course_cold_start(cfg):
    # Fresh user: mastery rỗng + fallback weak phonemes → vẫn ra giáo trình đủ.
    fallback_weak = [
        {"symbol": s, "error_rate": None, "fallback": True}
        for s in ("θ", "ð", "s", "z")
    ]
    course = generate.build_course("toeic", _empty_mastery(), fallback_weak, {}, {})
    assert course["exam"] == "toeic"
    total = sum(len(u["lessons"]) for u in course["units"])
    assert total == len(all_lessons("toeic")) == course["progress"]["total"]
    assert course["progress"]["done"] == 0
    # Mỗi Unit: đúng 1 lesson available (đầu chuỗi), phần còn lại locked.
    for unit in course["units"]:
        statuses = [ls["status"] for ls in unit["lessons"]]
        assert statuses.count("available") == 1
        assert statuses[0] == "available"  # đã xếp yếu-trước, lesson đầu mở
        assert set(statuses[1:]) <= {"locked"}


def test_build_course_prioritizes_weak_pronunciation():
    # /θ/ yếu nặng (group th_family) → unit phát âm lên đầu, lesson th_family focus.
    weak = [{"symbol": "θ", "error_rate": 0.9, "fallback": False}]
    course = generate.build_course("toeic", _empty_mastery(), weak, {}, {})
    assert course["units"][0]["dimension"] == "pronunciation"
    first = course["units"][0]["lessons"][0]
    assert first["target"] == "th_family" and first["status"] == "available"
    assert first["focus"] is True


def test_build_course_rubric_weakness_orders():
    mastery = {
        "criteria": {"grammar": {"mastery": 0.2, "attempts": 5, "weakness": 0.8}},
        "question_types": {},
    }
    course = generate.build_course("toeic", mastery, [], {}, {})
    rubric_unit = next(u for u in course["units"] if u["dimension"] == "rubric")
    # grammar yếu nhất → đứng đầu unit rubric + available.
    assert rubric_unit["lessons"][0]["target"] == "grammar"
    assert rubric_unit["lessons"][0]["status"] == "available"


def test_build_course_progress_overlay_unlocks_next():
    # th_family done → lesson kế trong unit phát âm mở khóa.
    weak = [
        {"symbol": "θ", "error_rate": 0.9, "fallback": False},
        {"symbol": "s", "error_rate": 0.5, "fallback": False},
    ]
    progress = {"toeic.pron.th_family": {"status": "done", "best_score": 0.9,
                                         "attempts": 1, "completed_at": "x"}}
    course = generate.build_course("toeic", _empty_mastery(), weak, progress, {})
    pron = course["units"][0]
    assert pron["lessons"][0]["target"] == "th_family"
    assert pron["lessons"][0]["status"] == "done"
    assert pron["lessons"][1]["status"] == "available"  # mở khóa sau done
    assert course["progress"]["done"] == 1


# ── mark_lesson_complete (ngưỡng theo dimension) ──────────────────────────


def test_mark_complete_pronunciation_threshold(cfg):
    uid = "u1"
    # Ngưỡng phát âm 0.80: 0.75 chưa đạt.
    r = mark_lesson_complete(cfg, uid, "toeic.pron.th_family", 0.75, "toeic")
    assert r["done"] is False and r["progress"]["status"] == "in_progress"
    # 0.85 đạt → done + streak bump.
    r = mark_lesson_complete(cfg, uid, "toeic.pron.th_family", 0.85, "toeic")
    assert r["done"] is True and r["progress"]["status"] == "done"
    assert r["streak"]["streak_days"] == 1


def test_mark_complete_rubric_threshold(cfg):
    uid = "u1"
    # Ngưỡng rubric 0.67: 0.70 đạt (khác phát âm 0.80).
    r = mark_lesson_complete(cfg, uid, "toeic.rubric.grammar", 0.70, "toeic")
    assert r["done"] is True


def test_mark_complete_unknown_lesson_raises(cfg):
    with pytest.raises(ValueError):
        mark_lesson_complete(cfg, "u1", "toeic.pron.nope", 0.9, "toeic")


# ── Gộp tài khoản ─────────────────────────────────────────────────────────


def test_merge_user_moves_progress_wipes_mastery(cfg):
    anon, acct = "a" * 36, "b" * 36
    store.upsert_lesson_progress(cfg, anon, "toeic.rubric.grammar",
                                 status="done", score=0.8)
    store.apply_mastery_tallies(
        cfg, anon, {("toeic", "grammar"): {"attempts": 3.0, "score_sum": 1.0}},
        {}, ("2026-07-01T00:00:01Z", "aaa"),
    )
    moved = merge_user(cfg, anon, acct)
    assert moved == 1
    assert "toeic.rubric.grammar" in store.get_progress(cfg, acct)
    assert store.get_progress(cfg, anon) == {}
    # Mastery của cả hai bị wipe → rebuild từ history đã gộp ở refresh sau.
    assert store.get_mastery_stats(cfg, anon, "toeic")["criteria"] == {}
    assert store.get_mastery_stats(cfg, acct, "toeic")["criteria"] == {}
    assert store.get_scan_cursor(cfg, acct) == ("", "")


def test_supported_exams():
    assert SUPPORTED_EXAMS == ("toeic", "ielts", "topik")


# ── TOPIK (Phase 3a: rubric + dạng câu, không phát âm) ───────────────────


def test_topik_mastery_normalized_by_five(cfg):
    uid = "u1"
    # TOPIK criterion 0-5: delivery 4/5 = 0.8; overall 150/200 = 0.75.
    _insert(cfg, uid, _recent(1), "aaa",
            _result("topik", "q1_answer_question",
                    [("delivery", 4.0), ("content_task", 2.5)], 150))
    profile.refresh_mastery(cfg, uid)
    m = profile.get_mastery(cfg, uid, "topik")
    assert m["criteria"]["delivery"]["mastery"] == pytest.approx(0.8)
    assert m["criteria"]["content_task"]["mastery"] == pytest.approx(0.5)
    assert m["question_types"]["q1_answer_question"]["mastery"] == pytest.approx(0.75)


def test_build_course_topik_has_no_pronunciation():
    course = generate.build_course("topik", _empty_mastery(), [], {}, {})
    dims = [u["dimension"] for u in course["units"]]
    assert "pronunciation" not in dims
    assert set(dims) == {"rubric", "question_type"}
    assert course["progress"]["total"] == len(all_lessons("topik")) == 10


def test_topik_mark_complete_rubric_threshold(cfg):
    r = mark_lesson_complete(cfg, "u1", "topik.rubric.delivery", 0.7, "topik")
    assert r["done"] is True  # rubric threshold 0.67


# ── Nội dung bài (content.py) ─────────────────────────────────────────────


@pytest.fixture()
def small_index(monkeypatch):
    """Index nhỏ deterministic (mirror test_word_suggest) để không đọc wordlist thật."""
    wl = ["the", "think", "this", "that", "cat", "see", "zoo", "ship",
          "chair", "job", "van", "sit", "month", "three", "bath", "thin"]
    monkeypatch.setattr(word_suggest, "_load_wordlist", lambda: wl)
    monkeypatch.setattr(word_suggest, "_index", None)
    yield wl
    word_suggest._index = None


def test_lesson_content_pronunciation(cfg, small_index, monkeypatch):
    # cached_suggestions: dùng fallback tần suất (không LLM) → deterministic.
    monkeypatch.setattr(
        word_suggest, "cached_suggestions",
        lambda c, cfg2, sym, lang, **k: (
            [{"word": w, "reason": None} for w in word_suggest.candidates_for(sym)[:5]],
            False,
        ),
    )
    out = get_lesson_content(cfg, cfg, "u1", "toeic.pron.th_family", "vi")
    assert out["dimension"] == "pronunciation"
    assert out["phonemes"]  # normalized θ/ð
    assert out["words"] and all(w["ipa"] for w in out["words"])
    assert out["done_threshold"] == pytest.approx(0.80)


def test_lesson_content_pronunciation_excludes_saved(cfg, small_index, monkeypatch):
    from src import words as words_mod
    words_mod.upsert_word(cfg, "u1", "think")
    monkeypatch.setattr(
        word_suggest, "cached_suggestions",
        lambda c, cfg2, sym, lang, **k: ([{"word": "think", "reason": None}], False),
    )
    out = get_lesson_content(cfg, cfg, "u1", "toeic.pron.th_family", "vi")
    assert "think" not in [w["word"] for w in out["words"]]


def test_lesson_content_rubric_aggregates_history(cfg):
    uid = "u1"
    r = _result("toeic", "describe_picture",
                [("grammar", 2.0)], 140)
    r["scores"]["criteria"][0]["suggestions"] = ["Dùng đúng thì hiện tại tiếp diễn"]
    _insert(cfg, uid, _recent(1), "aaa", r)
    out = get_lesson_content(cfg, cfg, uid, "toeic.rubric.grammar", "vi")
    assert out["dimension"] == "rubric"
    assert "Dùng đúng thì hiện tại tiếp diễn" in out["learner_suggestions"]
    assert out["tips"]  # tips tĩnh vẫn có
    assert out["done_threshold"] == pytest.approx(0.67)


def test_lesson_content_qtype_uses_and_caches_sample(cfg, monkeypatch):
    calls = []

    class _Ans:
        answer = "A model answer."
        outline = ["intro", "body"]
        highlights = ["strong collocation"]
        target_band = "200"

    def fake_suggest(config, qt, **k):
        calls.append(qt.key)
        return _Ans()

    monkeypatch.setattr(content, "suggest_answer", fake_suggest)
    out1 = get_lesson_content(cfg, cfg, "u1", "toeic.qtype.read_aloud", "vi")
    out2 = get_lesson_content(cfg, cfg, "u1", "toeic.qtype.read_aloud", "vi")
    assert out1["sample_answer"]["answer"] == "A model answer."
    assert out1["scale_description"] and out1["guidance"]
    # Lần 2 đọc cache → không gọi LLM lại.
    assert calls == ["read_aloud"]
    assert out2["sample_answer"]["answer"] == "A model answer."


def test_lesson_content_unknown_raises(cfg):
    with pytest.raises(ValueError):
        get_lesson_content(cfg, cfg, "u1", "toeic.pron.nope", "vi")


# ── Tự động hoàn thành từ mastery (Phase 2) ──────────────────────────────


def test_auto_completions_selects_qualified_rubric_qtype():
    mastery = {
        "criteria": {
            "grammar": {"mastery": 0.9, "attempts": 3, "weakness": 0.1},   # đạt
            "vocabulary": {"mastery": 0.5, "attempts": 5, "weakness": 0.5},  # < 0.67
            "cohesion": {"mastery": 0.9, "attempts": 2, "weakness": None},   # < 3 mẫu
        },
        "question_types": {
            "read_aloud": {"mastery": 0.8, "attempts": 4, "weakness": 0.2},  # đạt
        },
    }
    got = dict(generate.auto_completions("toeic", mastery))
    assert "toeic.rubric.grammar" in got
    assert "toeic.qtype.read_aloud" in got
    assert "toeic.rubric.vocabulary" not in got
    assert "toeic.rubric.cohesion" not in got
    # Phát âm KHÔNG bao giờ tự hoàn thành.
    assert not any(k.startswith("toeic.pron.") for k in got)


def test_auto_complete_lesson_idempotent_no_attempt_bump(cfg):
    uid = "u1"
    lid = "toeic.rubric.grammar"
    assert store.auto_complete_lesson(cfg, uid, lid, 0.9) is True
    # Lần 2 → đã done, không ghi lại.
    assert store.auto_complete_lesson(cfg, uid, lid, 0.95) is False
    prog = store.get_progress(cfg, uid)[lid]
    assert prog["status"] == "done"
    assert prog["attempts"] == 0  # auto KHÔNG tăng attempts
    assert prog["best_score"] == pytest.approx(0.9)  # đã done → lần 2 no-op


def test_auto_complete_does_not_bump_streak(cfg):
    uid = "u1"
    store.auto_complete_lesson(cfg, uid, "toeic.rubric.grammar", 0.9)
    # Streak chỉ cho hành động luyện chủ động (mark_lesson_complete), không phải auto.
    assert store.get_activity(cfg, uid)["streak_days"] == 0


def test_get_course_auto_completes_from_real_grades(cfg):
    uid = "u1"
    # 3 bài chấm THẬT grammar mạnh (2.7/3 = 0.9) → tự done lesson grammar.
    for i in range(3):
        _insert(cfg, uid, _recent(i + 1), f"r{i}",
                _result("toeic", "respond_questions", [("grammar", 2.7)], 160))
    from src.course import get_course

    course = get_course(cfg, uid, "toeic")
    grammar = [
        ls for u in course["units"] for ls in u["lessons"]
        if ls["id"] == "toeic.rubric.grammar"
    ][0]
    assert grammar["status"] == "done"
    assert course["progress"]["done"] >= 1
