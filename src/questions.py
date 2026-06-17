"""Nạp ngân hàng câu hỏi từ data/questions/toeic.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_DATA_FILE = (
    Path(__file__).resolve().parent.parent / "data" / "questions" / "toeic.json"
)


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    prompt: str
    reference_script: str | None = None
    expected_duration_sec: float | None = None


def _load_all(path: Path = _DATA_FILE) -> dict[str, Question]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy ngân hàng câu hỏi: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions: dict[str, Question] = {}
    for item in raw:
        q = Question(
            id=item["id"],
            type=item["type"],
            prompt=item.get("prompt", ""),
            reference_script=item.get("reference_script"),
            expected_duration_sec=item.get("expected_duration_sec"),
        )
        questions[q.id] = q
    return questions


def get_question(question_id: str) -> Question:
    questions = _load_all()
    if question_id not in questions:
        raise KeyError(
            f"Không tìm thấy câu hỏi '{question_id}'. "
            f"Có sẵn: {sorted(questions)}"
        )
    return questions[question_id]
