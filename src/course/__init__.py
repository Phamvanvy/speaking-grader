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
from . import content as _content
from . import generate, profile, store
from .generate import DONE_THRESHOLD
from .syllabus import SUPPORTED_EXAMS, get_lesson

logger = logging.getLogger("toeic.course")

__all__ = [
    "SUPPORTED_EXAMS",
    "get_course",
    "get_lesson_content",
    "refresh",
    "mark_lesson_complete",
    "merge_user",
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
    return generate.build_course(exam, mastery, weak, progress, activity)


def get_lesson_content(
    cfg: Config, config: Config, user_id: str, lesson_id: str, lang: str
) -> dict:
    """Nội dung 1 lesson (từ luyện / tips / bài mẫu) — xem course/content.py."""
    return _content.get_lesson_content(cfg, config, user_id, lesson_id, lang)


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
    return {"lesson_id": lesson_id, "done": done, "progress": prog, "streak": streak}


def merge_user(cfg: Config, from_user_id: str, to_user_id: str) -> int:
    """Gộp dữ liệu khóa học khi /auth/claim (xem store.merge_user)."""
    return store.merge_user(cfg, from_user_id, to_user_id)
