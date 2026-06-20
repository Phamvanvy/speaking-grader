"""Nạp ngân hàng câu hỏi từ data/questions/{exam}.json (toeic / ielts)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .rubrics.base import Exam

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "questions"


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    prompt: str
    reference_script: str | None = None
    expected_duration_sec: float | None = None
    # Đường dẫn ảnh đề bài (tương đối từ project root) — dùng cho Describe Picture.
    image_path: str | None = None
    # Tài liệu cho sẵn (text) — dùng cho Respond with info (Q8-10) / IELTS Part 2 cue card.
    provided_info: str | None = None


def _data_file(exam: str) -> Path:
    return _DATA_DIR / f"{exam}.json"


def _load_all(exam: str = Exam.TOEIC.value) -> dict[str, Question]:
    path = _data_file(exam)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy ngân hàng câu hỏi cho kỳ thi '{exam}': {path}"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions: dict[str, Question] = {}
    for item in raw:
        q = Question(
            id=item["id"],
            type=item["type"],
            prompt=item.get("prompt", ""),
            reference_script=item.get("reference_script"),
            expected_duration_sec=item.get("expected_duration_sec"),
            image_path=item.get("image_path"),
            provided_info=item.get("provided_info"),
        )
        questions[q.id] = q
    return questions


def get_question(question_id: str, exam: str = Exam.TOEIC.value) -> Question:
    questions = _load_all(exam)
    if question_id not in questions:
        raise KeyError(
            f"Không tìm thấy câu hỏi '{question_id}' trong kỳ thi '{exam}'. "
            f"Có sẵn: {sorted(questions)}"
        )
    return questions[question_id]
