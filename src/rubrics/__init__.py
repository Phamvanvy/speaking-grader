"""Rubrics package — phân giải dạng câu (QuestionType) theo kỳ thi.

Điểm vào DUY NHẤT cho api/cli/main: `resolve_question_type(key, exam)`. Mỗi kỳ
thi có registry riêng (toeic.py, ielts.py); ở đây gộp lại theo `Exam` và chặn
truy vấn cross-exam (vd hỏi 'read_aloud' với exam='ielts' sẽ báo lỗi).
"""

from __future__ import annotations

from .base import (
    Criterion,
    Exam,
    EXAM_LANGUAGE,
    EXAM_SCORE,
    QuestionType,
    exam_language,
    exam_score_field,
    exam_score_max,
)
from .ielts import IELTS_QUESTION_TYPES
from .toeic import TOEIC_QUESTION_TYPES
from .topik import TOPIK_QUESTION_TYPES

# Keyed bằng giá trị enum (không hardcode chuỗi) để Exam là nguồn hằng duy nhất.
EXAM_REGISTRIES: dict[str, dict[str, QuestionType]] = {
    Exam.TOEIC.value: TOEIC_QUESTION_TYPES,
    Exam.IELTS.value: IELTS_QUESTION_TYPES,
    Exam.TOPIK.value: TOPIK_QUESTION_TYPES,
}


def _registry_for(exam: str) -> dict[str, QuestionType]:
    if exam not in EXAM_REGISTRIES:
        raise KeyError(
            f"Không biết kỳ thi '{exam}'. Hợp lệ: {sorted(EXAM_REGISTRIES)}"
        )
    return EXAM_REGISTRIES[exam]


def resolve_question_type(key: str, exam: str = Exam.TOEIC.value) -> QuestionType:
    """Trả về QuestionType của `key` TRONG kỳ thi `exam`.

    Raise KeyError nếu exam không hợp lệ, hoặc key không thuộc registry của exam
    đó (chặn cross-exam — không lặng lẽ trả nhầm dạng câu của kỳ thi khác).
    """
    registry = _registry_for(exam)
    if key not in registry:
        raise KeyError(
            f"Dạng câu '{key}' không thuộc kỳ thi '{exam}'. "
            f"Hợp lệ: {sorted(registry)}"
        )
    return registry[key]


def list_question_types(exam: str = Exam.TOEIC.value) -> list[str]:
    """Danh sách key dạng câu của một kỳ thi (raise nếu exam không hợp lệ)."""
    return sorted(_registry_for(exam))


__all__ = [
    "Criterion",
    "Exam",
    "EXAM_LANGUAGE",
    "EXAM_SCORE",
    "QuestionType",
    "EXAM_REGISTRIES",
    "exam_language",
    "exam_score_field",
    "exam_score_max",
    "resolve_question_type",
    "list_question_types",
]
