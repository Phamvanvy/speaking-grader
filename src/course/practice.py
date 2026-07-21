"""Đề luyện tập (task-context) cho lesson rubric/dạng câu — để CHẤM THẬT (P2).

Trước P2, lesson rubric & question_type chỉ có nút "Đã học xong" thủ công (chưa
đủ dữ liệu test để tự chấm). P2 sinh MỘT đề luyện có ngữ cảnh (task-context) cho
mỗi lesson như vậy: học viên trả lời bằng lời nói → chấm qua pipeline đầy đủ
(question_type + prompt/reference/provided_info) → điểm chuẩn hóa 0-1 → hoàn thành.

Chỉ chấm được các dạng câu KHÔNG cần ảnh (required_inputs không có "image"). Dạng
cần ảnh (describe_picture) → build_practice trả None → frontend giữ nút thủ công.

- Lesson dạng câu (question_type): dùng chính dạng câu đó.
- Lesson tiêu chí (rubric): chọn 1 dạng câu ĐẠI DIỆN có chứa tiêu chí đó (ưu tiên
  read_aloud cho phát âm/ngữ điệu — bằng chứng khách quan; else đề mở giàu ngữ
  cảnh), rồi trích điểm tiêu chí target ở endpoint chấm.

Đề sinh 1 lần qua LLM (suggest_practice_task) rồi cache USER-AGNOSTIC theo
(lesson_id, lang) trong course.db (dùng lesson_content_cache với id tổng hợp
'<lesson_id>#practice' để độc lập version với sample_answer).

Điểm vào: build_practice(cfg, config, lesson, lang) — dùng bởi content.py +
endpoint POST /course/lesson/{id}/grade.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..config import Config
from ..rubrics import EXAM_REGISTRIES, resolve_question_type
from ..rubrics.base import QuestionType
from ..suggest import suggest_practice_task
from . import store
from .syllabus import Lesson

logger = logging.getLogger("toeic.course.practice")

_PRACTICE_CACHE_VERSION = 1
_PRACTICE_TTL_SECONDS = 30 * 86400

# Tiêu chí "delivery" (phát âm/ngữ điệu) → ưu tiên read_aloud: có script tham
# chiếu nên chấm khách quan (WER/coverage), không phụ thuộc nội dung tự do.
_DELIVERY_CRITERIA = {"pronunciation", "intonation_stress", "delivery"}


def _is_text_gradable(qt: QuestionType) -> bool:
    """True nếu dạng câu chấm được CHỈ từ đề dạng text (không cần ảnh)."""
    return "image" not in qt.required_inputs


def _qtype_for_criterion(exam: str, criterion: str) -> QuestionType | None:
    """Dạng câu đại diện (text-gradable) chứa `criterion`; None nếu không có."""
    reg = EXAM_REGISTRIES.get(exam, {})
    cands = [
        qt
        for qt in reg.values()
        if _is_text_gradable(qt) and any(c.key == criterion for c in qt.criteria)
    ]
    if not cands:
        return None
    if criterion in _DELIVERY_CRITERIA:
        for qt in cands:
            if qt.key == "read_aloud":
                return qt
    # Đề mở giàu ngữ cảnh (nhiều tiêu chí nhất) chấm nội dung tốt hơn; tie-break
    # theo key cho tất định.
    return sorted(cands, key=lambda qt: (-len(qt.criteria), qt.key))[0]


def qtype_for_lesson(lesson: Lesson) -> QuestionType | None:
    """Dạng câu dùng để chấm practice của `lesson`; None nếu không chấm từ text."""
    if lesson.dimension == "question_type":
        qt = resolve_question_type(lesson.target, lesson.exam)
        return qt if _is_text_gradable(qt) else None
    if lesson.dimension == "rubric":
        return _qtype_for_criterion(lesson.exam, lesson.target)
    return None  # pronunciation dùng đường chấm phoneme riêng


def _cache_age_seconds(created_at: str | None) -> float:
    try:
        ts = datetime.strptime(created_at or "", "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return float("inf")
    return (datetime.now(timezone.utc) - ts).total_seconds()


def _cache_id(lesson_id: str) -> str:
    return f"{lesson_id}#practice"


def _task_valid(qt: QuestionType, content: dict) -> bool:
    """Đề hợp lệ khi field CỐT LÕI của dạng câu có nội dung (LLM có thể bỏ trống)."""
    if qt.key == "read_aloud":
        return bool((content.get("reference") or "").strip())
    return bool((content.get("prompt") or "").strip())


def _get_or_generate_task(
    cfg: Config, config: Config, lesson: Lesson, qt: QuestionType, lang: str
) -> dict | None:
    """Đề luyện (dict prompt/reference/provided_info) — cache-first; None nếu lỗi."""
    cid = _cache_id(lesson.id)
    entry = store.get_lesson_content_cache(cfg, cid, lang)
    if (
        entry
        and entry.get("cache_version") == _PRACTICE_CACHE_VERSION
        and _cache_age_seconds(entry.get("created_at")) < _PRACTICE_TTL_SECONDS
        and _task_valid(qt, entry["content"])
    ):
        return entry["content"]

    try:
        task = suggest_practice_task(config, qt)
    except Exception:  # noqa: BLE001 - LLM lỗi không chặn nội dung bài
        logger.exception("Lỗi sinh đề luyện cho %s (bỏ qua)", lesson.id)
        return None

    content = {
        "prompt": task.prompt or "",
        "reference": task.reference or "",
        "provided_info": task.provided_info or "",
    }
    if not _task_valid(qt, content):
        logger.warning("Đề luyện %s thiếu field cốt lõi — bỏ qua", lesson.id)
        return None

    try:
        model = config.local_model if config.is_local else config.model
        store.put_lesson_content_cache(
            cfg, cid, lang, content, model, _PRACTICE_CACHE_VERSION
        )
    except Exception:  # noqa: BLE001
        logger.exception("Lỗi ghi cache đề luyện (bỏ qua)")
    return content


def build_practice(
    cfg: Config, config: Config, lesson: Lesson, lang: str
) -> dict | None:
    """Đề luyện đầy đủ cho lesson rubric/dạng câu; None nếu không chấm được từ text.

    Trả {'question_type', 'prompt', 'reference', 'provided_info'} và với lesson
    rubric kèm 'target_criterion' (tiêu chí cần trích điểm). Frontend gửi lại các
    field này về POST /course/lesson/{id}/grade.
    """
    qt = qtype_for_lesson(lesson)
    if qt is None:
        return None
    content = _get_or_generate_task(cfg, config, lesson, qt, lang)
    if content is None:
        return None
    out = {
        "question_type": qt.key,
        "prompt": content.get("prompt", ""),
        "reference": content.get("reference", ""),
        "provided_info": content.get("provided_info", ""),
    }
    if lesson.dimension == "rubric":
        out["target_criterion"] = lesson.target
    return out
