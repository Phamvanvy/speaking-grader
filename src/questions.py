"""Nạp ngân hàng câu hỏi từ data/questions/{exam}/{set_id}.json (toeic / ielts).

Mỗi kỳ thi có NHIỀU bộ đề (set) để người dùng chọn trước khi thi (vd
"TOEIC Speaking Practice Test 1/2/3"), thay vì chỉ 1 bộ câu hỏi cố định.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .rubrics.base import Exam

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "questions"
DEFAULT_SET = "set1"


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


def _exam_dir(exam: str) -> Path:
    return _DATA_DIR / exam


def _set_file(exam: str, set_id: str) -> Path:
    return _exam_dir(exam) / f"{set_id}.json"


def list_sets(exam: str) -> list[dict]:
    """Danh sách bộ đề có sẵn cho 1 kỳ thi: [{"id", "title"}, ...] sắp theo id."""
    d = _exam_dir(exam)
    if not d.is_dir():
        raise FileNotFoundError(f"Không tìm thấy kỳ thi '{exam}': {d}")
    sets = []
    for path in sorted(d.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        sets.append({"id": path.stem, "title": raw.get("title") or path.stem})
    return sets


def _load_set(exam: str, set_id: str = DEFAULT_SET) -> tuple[str, dict[str, Question]]:
    """Nạp 1 bộ đề → (title, {question_id: Question})."""
    path = _set_file(exam, set_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy bộ đề '{set_id}' cho kỳ thi '{exam}': {path}"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions: dict[str, Question] = {}
    for item in raw.get("questions", []):
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
    return raw.get("title") or set_id, questions


def _load_all(exam: str = Exam.TOEIC.value, set_id: str = DEFAULT_SET) -> dict[str, Question]:
    _, questions = _load_set(exam, set_id)
    return questions


def get_question(
    question_id: str, exam: str = Exam.TOEIC.value, set_id: str = DEFAULT_SET
) -> Question:
    questions = _load_all(exam, set_id)
    if question_id not in questions:
        raise KeyError(
            f"Không tìm thấy câu hỏi '{question_id}' trong bộ đề '{set_id}' của kỳ thi "
            f"'{exam}'. Có sẵn: {sorted(questions)}"
        )
    return questions[question_id]
