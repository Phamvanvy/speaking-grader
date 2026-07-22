"""Khóa học cá nhân hóa (course) — điểm vào cho api.py.

Delegation mỏng như src/words.py: `from . import course` rồi `course.get_course()`.
Tầng dưới:
- store.py: DB course.db (mastery aggregate + tiến độ/streak bền + merge_user).
- profile.py: tổng hợp mastery từ history (mirror phoneme_profile).
- syllabus.py: giáo trình tĩnh Unit → Lesson (TOEIC+IELTS).
- generate.py: build_course cá nhân hóa (thứ tự/mở khóa/priority) + DONE_THRESHOLD.

Phase 0 (bản này): get_course + refresh + mark_lesson_complete + merge_user.
Nội dung bài (get_lesson_content) là Phase 1.
"""

from __future__ import annotations

import logging

from ..config import Config
from ..phoneme_profile import get_weak_phonemes
from ..rubrics.base import exam_language
from . import boss as _boss
from . import content as _content
from . import generate
from . import practice as _practice
from . import profile, store
from . import quests as _quests
from . import xp as _xp
from .generate import BOSS_DONE_THRESHOLD, DONE_THRESHOLD, QUEST_DONE_THRESHOLD
from .syllabus import SUPPORTED_EXAMS, Unit, get_lesson, get_unit

logger = logging.getLogger("toeic.course")

__all__ = [
    "SUPPORTED_EXAMS",
    "get_course",
    "get_lesson_content",
    "build_practice_task",
    "score_practice",
    "lesson_exam",
    "refresh",
    "mark_lesson_complete",
    "get_unit_boss_content",
    "complete_unit_boss",
    "list_quests",
    "get_roleplay_quest",
    "complete_quest",
    "merge_user",
    "get_xp",
    "award_practice_xp",
    "get_shop",
    "buy_shop_item",
    "equip_shop_item",
    "get_leaderboard",
    "set_leaderboard_optin",
]


def _validate_exam(exam: str) -> str:
    exam = (exam or "").strip().lower()
    if exam not in SUPPORTED_EXAMS:
        raise ValueError(
            f"Kỳ thi '{exam}' chưa có khóa học. Hợp lệ: {sorted(SUPPORTED_EXAMS)}."
        )
    return exam


def get_course(cfg: Config, user_id: str, exam: str = "toeic") -> dict:
    """Giáo trình cá nhân hóa + tiến độ + streak cho 1 kỳ thi.

    Refresh mastery (quét history mới) trước, rồi dựng view model. get_weak_phonemes
    tự refresh hồ sơ âm (phoneme_profile) nên mảng phát âm cũng cập nhật.
    """
    exam = _validate_exam(exam)
    profile.refresh_mastery(cfg, user_id)
    mastery = profile.get_mastery(cfg, user_id, exam)
    # top_k phủ HẾT FALLBACK_WEAK_PHONEMES (10) để cold-start mọi nhóm âm đều có
    # tín hiệu → các Unit hòa priority và về đúng thứ tự syllabus (phát âm trước),
    # thay vì short_vowels bị cắt thành no-signal kéo mean phát âm xuống.
    # Âm yếu theo NGÔN NGỮ nói của kỳ thi (TOEIC/IELTS→en, TOPIK→ko) — hồ sơ âm
    # đã tách lang ở phoneme_stats nên khóa TOPIK chỉ lấy âm yếu tiếng Hàn.
    weak, _source = get_weak_phonemes(
        cfg, user_id, top_k=10, lang=exam_language(exam)
    )
    # Khép vòng "khóa học theo kết quả test": lesson rubric/qtype mà bài chấm THẬT
    # đã cho thấy thành thạo (đủ mẫu + đạt ngưỡng) tự đánh dấu done — không cần
    # luyện lại thủ công. Không bump streak (không phải hành động luyện chủ động).
    for lid, score in generate.auto_completions(exam, mastery):
        store.auto_complete_lesson(cfg, user_id, lid, score)
    progress = store.get_progress(cfg, user_id)
    activity = store.get_activity(cfg, user_id)
    boss_states = store.get_boss_states(cfg, user_id)
    return generate.build_course(exam, mastery, weak, progress, activity, boss_states)


def get_lesson_content(
    cfg: Config, config: Config, user_id: str, lesson_id: str, lang: str
) -> dict:
    """Nội dung 1 lesson (từ luyện / tips / bài mẫu) — xem course/content.py."""
    return _content.get_lesson_content(cfg, config, user_id, lesson_id, lang)


def build_practice_task(
    cfg: Config, config: Config, user_id: str, lesson_id: str, lang: str
) -> dict | None:
    """Đề luyện task-context cho lesson rubric/dạng câu (None nếu không chấm được
    chỉ từ text, vd dạng cần ảnh). user_id giữ trong chữ ký cho nhất quán dù đề
    hiện user-agnostic (cache theo lesson_id, lang). Xem course/practice.py."""
    lesson = get_lesson(lesson_id)
    if lesson is None:
        raise ValueError(f"Không có lesson '{lesson_id}'.")
    return _practice.build_practice(cfg, config, lesson, lang)


def score_practice(lesson_id: str, result: dict) -> float | None:
    """Điểm practice chuẩn hóa 0-1 của 1 lesson từ output chấm; None nếu thiếu."""
    lesson = get_lesson(lesson_id)
    if lesson is None:
        raise ValueError(f"Không có lesson '{lesson_id}'.")
    return profile.practice_score(
        result,
        lesson.exam,
        lesson.dimension,
        lesson.target if lesson.dimension == "rubric" else None,
    )


def lesson_exam(lesson_id: str) -> str:
    """Kỳ thi của lesson (để endpoint chấm chọn pipeline đúng)."""
    lesson = get_lesson(lesson_id)
    if lesson is None:
        raise ValueError(f"Không có lesson '{lesson_id}'.")
    return lesson.exam


def refresh(cfg: Config, user_id: str) -> dict:
    """Ép quét lại history vào mastery (GET /course cũng tự refresh)."""
    profile.refresh_mastery(cfg, user_id)
    return {"refreshed": True}


def mark_lesson_complete(
    cfg: Config, user_id: str, lesson_id: str, score: float, exam: str = "toeic"
) -> dict:
    """Ghi kết quả luyện 1 lesson. score đã CHUẨN HÓA 0-1.

    Đạt ngưỡng theo dimension → status 'done' + bump streak; chưa đạt →
    'in_progress' (vẫn lưu best_score/attempts). Trả {lesson, done, progress, streak}.
    """
    lesson = get_lesson(lesson_id)
    if lesson is None:
        raise ValueError(f"Không có lesson '{lesson_id}'.")
    try:
        score = float(score)
    except (TypeError, ValueError) as e:
        raise ValueError("score phải là số 0-1.") from e
    score = max(0.0, min(1.0, score))
    threshold = DONE_THRESHOLD.get(lesson.dimension, 0.7)
    done = score >= threshold
    prog = store.upsert_lesson_progress(
        cfg, user_id, lesson_id, status="done" if done else "in_progress", score=score
    )
    streak = store.bump_streak(cfg, user_id) if done else store.get_activity(cfg, user_id)
    result = {"lesson_id": lesson_id, "done": done, "progress": prog, "streak": streak}
    # XP/huy hiệu CHỈ award khi CHUYỂN trạng thái lần đầu sang done (first-transition)
    # → luyện thêm lesson đã done không farm được (RB#3). Streak giữ nguyên hành vi
    # cũ (bump mỗi lần done) vì đã idempotent 1/ngày và phản ánh "học hôm nay".
    if prog.get("just_completed") and cfg.course_xp_enabled:
        xp_state = _xp.award_lesson_xp(
            cfg, user_id, score, int(streak.get("streak_days", 0) or 0)
        )
        result["xp"] = xp_state
        result["new_badges"] = xp_state.get("new_badges", [])
    return result


# ── Boss cuối chặng (Phase 3A) — BONUS, tách khỏi mastery/hoàn thành lesson ──


def _all_lessons_done(unit: Unit, progress: dict[str, dict]) -> bool:
    """True khi MỌI lesson trong unit đã done (điều kiện mở khóa + gate Boss)."""
    return all(
        (progress.get(ls.id) or {}).get("status") == "done" for ls in unit.lessons
    )


def get_unit_boss_content(
    cfg: Config, config: Config, user_id: str, unit_id: str, lang: str
) -> dict:
    """Nội dung Boss của 1 chặng (đoạn đọc-to tổng hợp). Raise ValueError nếu unit
    sai; PermissionError nếu chưa mở khóa (còn lesson chưa done)."""
    unit = get_unit(unit_id)
    if unit is None:
        raise ValueError(f"Không có chặng '{unit_id}'.")
    if not _all_lessons_done(unit, store.get_progress(cfg, user_id)):
        raise PermissionError("Chưa mở khóa Boss — hoàn thành tất cả bài trong chặng.")
    content = _boss.build_boss(cfg, config, user_id, unit, lang)
    beaten = store.get_boss_states(cfg, user_id).get(f"{unit_id}.boss")
    return {
        "boss_id": f"{unit_id}.boss",
        "unit_id": unit_id,
        "title": f"Boss: {unit.title}",
        "exam": unit.exam,
        "dimension": unit.dimension,
        "threshold": BOSS_DONE_THRESHOLD,
        "reference_text": content["reference_text"],
        "words": content["words"],
        "best_score": (beaten or {}).get("best_score"),
        "done": beaten is not None,
    }


def complete_unit_boss(
    cfg: Config, user_id: str, unit_id: str, score: float
) -> dict:
    """Ghi kết quả hạ Boss. score đã CHUẨN HÓA 0-1 (client chấm qua /grade).

    P0 chống gian lận (server-side): (1) gate lại all-lessons-done → PermissionError;
    (2) clamp score; (3) chỉ mark beaten + award khi score >= BOSS_DONE_THRESHOLD.
    XP/huy hiệu qua kênh RIÊNG award_bonus_xp, MỘT LẦN (mark_boss_beaten trả True ở
    lần đầu). KHÔNG đụng lesson_progress/mastery/streak. Trả {done, best_score, xp?}.
    """
    unit = get_unit(unit_id)
    if unit is None:
        raise ValueError(f"Không có chặng '{unit_id}'.")
    try:
        score = float(score)
    except (TypeError, ValueError) as e:
        raise ValueError("score phải là số 0-1.") from e
    score = max(0.0, min(1.0, score))
    if not _all_lessons_done(unit, store.get_progress(cfg, user_id)):
        raise PermissionError("Chưa mở khóa Boss — hoàn thành tất cả bài trong chặng.")
    boss_id = f"{unit_id}.boss"
    passed = score >= BOSS_DONE_THRESHOLD
    result: dict = {"boss_id": boss_id, "done": passed, "score": score}
    if passed:
        first = store.mark_boss_beaten(cfg, user_id, boss_id, score)
        beaten = store.get_boss_states(cfg, user_id).get(boss_id)
        result["best_score"] = (beaten or {}).get("best_score")
        if first and cfg.course_xp_enabled:
            xp_state = _xp.award_bonus_xp(cfg, user_id, "boss", score)
            result["xp"] = xp_state
            result["new_badges"] = xp_state.get("new_badges", [])
    return result


# ── Quest nhập vai / truyện (Phase 3B/3C) — BONUS, tách khỏi mastery/lesson ──

_QUEST_KINDS = {"roleplay", "story"}


def list_quests(cfg: Config, user_id: str, exam: str) -> dict:
    """Danh sách Quest của kỳ thi (hiện: Role-play) + trạng thái đã hoàn thành.

    Bonus-only: chỉ đọc quest_clears (KHÔNG đụng mastery/progress). Trả
    {exam, quests:[{quest_id, kind, topic, title, cleared, best_score}]}. Chưa hỗ
    trợ kỳ thi (vd topik) → quests rỗng (frontend ẩn khu vực Quest)."""
    exam = _validate_exam(exam)
    clears = store.get_quest_clears(cfg, user_id)
    quests: list[dict] = []
    for topic in _quests.list_roleplay_topics(exam):
        qid = _quests.roleplay_quest_id(exam, topic.slug)
        c = clears.get(qid)
        quests.append(
            {
                "quest_id": qid,
                "kind": "roleplay",
                "topic": topic.slug,
                "title": topic.title,
                "cleared": c is not None,
                "best_score": (c or {}).get("best_score"),
            }
        )
    return {"exam": exam, "quests": quests}


def get_roleplay_quest(
    cfg: Config, config: Config, user_id: str, exam: str, topic: str, lang: str
) -> dict | None:
    """Kịch bản Role-play của (exam, topic) + trạng thái. None nếu không dựng được
    (chủ đề sai / LLM lỗi) → endpoint trả null để frontend ẩn quest, KHÔNG chặn.

    Có thể chạm LLM (build_roleplay) → caller nên chạy trong threadpool."""
    exam = _validate_exam(exam)
    content = _quests.build_roleplay(cfg, config, exam, topic, lang)
    if content is None:
        return None
    qid = _quests.roleplay_quest_id(exam, topic)
    cleared = store.get_quest_clears(cfg, user_id).get(qid)
    return {
        "quest_id": qid,
        "kind": "roleplay",
        "exam": exam,
        "topic": topic,
        "threshold": QUEST_DONE_THRESHOLD,
        "scenario": content["scenario"],
        "role_user": content["role_user"],
        "role_npc": content["role_npc"],
        "turns": content["turns"],
        "best_score": (cleared or {}).get("best_score"),
        "cleared": cleared is not None,
    }


def complete_quest(
    cfg: Config, user_id: str, quest_id: str, kind: str, score: float
) -> dict:
    """Ghi kết quả hoàn thành 1 Quest (Role-play/Story). score CHUẨN HÓA 0-1
    (client chấm phát âm trung bình các lượt qua /grade).

    P0 chống gian lận (server-side): (1) clamp score; (2) chỉ mark cleared + award
    khi score >= QUEST_DONE_THRESHOLD. XP/huy hiệu qua kênh RIÊNG award_bonus_xp,
    MỘT LẦN (mark_quest_cleared trả True lần đầu). KHÔNG đụng lesson_progress/
    mastery/streak (Quest là bonus). Trả {done, best_score, xp?, new_badges?}."""
    kind = (kind or "").strip().lower()
    if kind not in _QUEST_KINDS:
        raise ValueError(f"kind không hợp lệ: {kind!r} (hợp lệ: {sorted(_QUEST_KINDS)}).")
    if not (quest_id or "").strip():
        raise ValueError("Thiếu quest_id.")
    try:
        score = float(score)
    except (TypeError, ValueError) as e:
        raise ValueError("score phải là số 0-1.") from e
    score = max(0.0, min(1.0, score))
    passed = score >= QUEST_DONE_THRESHOLD
    result: dict = {"quest_id": quest_id, "kind": kind, "done": passed, "score": score}
    if passed:
        first = store.mark_quest_cleared(cfg, user_id, quest_id, score)
        cleared = store.get_quest_clears(cfg, user_id).get(quest_id)
        result["best_score"] = (cleared or {}).get("best_score")
        if first and cfg.course_xp_enabled:
            xp_state = _xp.award_bonus_xp(cfg, user_id, kind, score)
            result["xp"] = xp_state
            result["new_badges"] = xp_state.get("new_badges", [])
    return result


def get_xp(cfg: Config, user_id: str) -> dict:
    """Trạng thái XP/level/huy hiệu + streak (no-op nếu tắt cờ COURSE_XP_ENABLED).

    Kèm streak để tab "Từ đã lưu" hiện ngọn lửa mà không phải gọi /course/state
    (vốn refresh mastery, nặng hơn)."""
    if not cfg.course_xp_enabled:
        return {"enabled": False}
    state = _xp.get_xp_state(cfg, user_id)
    state["enabled"] = True
    state["streak"] = store.get_activity(cfg, user_id)
    return state


# Các sự kiện luyện được chấp nhận. TẤT CẢ dùng CHUNG hạn mức XP ngày
# (DAILY_PRACTICE_CAP) — thêm dạng bài (mini-game) KHÔNG mở kênh XP thoát cap.
#   word_practice  — luyện nói 1 từ (chấm phoneme).
#   word_recall    — mini-game không nói (nghe & chọn / nghĩa → nhớ từ); client
#                    gửi score nhị phân đúng=1.0 / sai=0.0, backend tự quy XP.
_PRACTICE_EVENTS = frozenset({"word_practice", "word_recall"})


def award_practice_xp(cfg: Config, user_id: str, event: str, score: float) -> dict:
    """Cấp XP cho 1 sự kiện luyện (word_practice / word_recall). Backend TỰ tính XP
    từ score (client không gửi XP — RB#5); quota ngày TỔNG chống farm (dùng chung
    cho MỌI event). No-op nếu tắt cờ."""
    if not cfg.course_xp_enabled:
        return {"enabled": False}
    if (event or "").strip() not in _PRACTICE_EVENTS:
        valid = ", ".join(sorted(_PRACTICE_EVENTS))
        raise ValueError(f"event không hợp lệ: '{event}'. Hợp lệ: {valid}.")
    state = _xp.award_practice_xp(cfg, user_id, score)
    state["enabled"] = True
    return state


# ── Cửa hàng cosmetic (Phase 4 game hóa) ─────────────────────────────────


def get_shop(cfg: Config, user_id: str) -> dict:
    """Danh mục cửa hàng + xu + item đã sở hữu/trang bị (no-op nếu tắt cờ)."""
    if not cfg.course_xp_enabled:
        return {"enabled": False}
    state = _xp.get_shop_state(cfg, user_id)
    state["enabled"] = True
    return state


def buy_shop_item(cfg: Config, user_id: str, item_id: str) -> dict:
    """Mua 1 item cosmetic bằng xu (backend giữ giá — RB#5). No-op nếu tắt cờ."""
    if not cfg.course_xp_enabled:
        return {"enabled": False}
    state = _xp.buy_item(cfg, user_id, item_id)
    state["enabled"] = True
    return state


def equip_shop_item(cfg: Config, user_id: str, item_id: str, equipped: bool) -> dict:
    """Trang bị / tháo 1 item đã sở hữu (tối đa 1/slot). No-op nếu tắt cờ."""
    if not cfg.course_xp_enabled:
        return {"enabled": False}
    state = _xp.equip_item(cfg, user_id, item_id, equipped)
    state["enabled"] = True
    return state


# ── Bảng xếp hạng tuần (opt-in, chỉ tài khoản — Phase 5) ─────────────────


def set_leaderboard_optin(cfg: Config, user_id: str, opt_in: bool) -> dict:
    """Bật/tắt xuất hiện trên bảng xếp hạng. Gate account do api.py (chỉ tài khoản
    đăng nhập được bật). No-op nếu tắt cờ."""
    if not cfg.course_xp_enabled:
        return {"enabled": False}
    _xp.set_leaderboard_optin(cfg, user_id, opt_in)
    return {"enabled": True, "opted_in": bool(opt_in)}


def get_leaderboard(
    cfg: Config, user_id: str, resolve_usernames, limit: int = 50
) -> dict:
    """Bảng xếp hạng theo XP-practice TUẦN (7 ngày), CHỈ tài khoản đã opt-in.

    `resolve_usernames(ids) -> {id: username}` được api.py tiêm vào (giữ course
    độc lập với auth). Người không có username (ẩn danh lỡ opt-in) bị loại. Trả
    top `limit` + hạng của chính user (kể cả ngoài top). No-op nếu tắt cờ.
    """
    if not cfg.course_xp_enabled:
        return {"enabled": False}
    rows = _xp.weekly_leaderboard_rows(cfg)
    names = resolve_usernames([r["user_id"] for r in rows]) or {}
    ranked = [
        {**r, "username": names[r["user_id"]]}
        for r in rows
        if r["user_id"] in names  # chỉ tài khoản có username
    ]
    # Hạng: XP tuần giảm dần, hòa thì theo tên (tất định).
    ranked.sort(key=lambda x: (-x["weekly_xp"], x["username"].lower()))
    entries: list[dict] = []
    me: dict | None = None
    for i, r in enumerate(ranked):
        entry = {
            "rank": i + 1,
            "username": r["username"],
            "weekly_xp": r["weekly_xp"],
            "level": _xp.xp_to_level(r["total_xp"])["level"],
            "is_me": r["user_id"] == user_id,
        }
        if entry["is_me"]:
            me = entry
        if i < limit:
            entries.append(entry)
    return {
        "enabled": True,
        "week_start": _xp._week_start(),
        "goal": _xp.WEEKLY_XP_GOAL,
        "opted_in": _xp.get_leaderboard_optin(cfg, user_id),
        "entries": entries,
        "me": me,  # None nếu chưa opt-in / không phải tài khoản
    }


def merge_user(cfg: Config, from_user_id: str, to_user_id: str) -> int:
    """Gộp dữ liệu khóa học khi /auth/claim (xem store.merge_user)."""
    return store.merge_user(cfg, from_user_id, to_user_id)
