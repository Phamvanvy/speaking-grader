"""Mô hình "đề thi đầy đủ" — một bộ câu hỏi có thứ tự cho luồng *Thi cả đề*.

Khác `src/questions.py` (ngân hàng câu hỏi tĩnh, 1 câu/lần): ở đây một `ExamPaper`
là TRỌN một bài thi (TOEIC Q1–11 hoặc IELTS Part 1–3) mà cá nhân làm tuần tự rồi
chấm gộp. Dùng cho `/exam/import` (bóc tách từ tài liệu) và `/exam/grade`.

THIẾT KẾ:
- `ExamQuestion.sequence` là NGUỒN CHÂN LÝ cho thứ tự câu — KHÔNG suy ra từ vị trí
  trong list (UI cho phép reorder/thêm/xoá). Mọi nơi chuẩn hoá bằng sort theo sequence.
- Ảnh (Describe Picture) đi kèm dạng base64 ngay trong câu (`image_b64`), client giữ
  và gửi lại lúc chấm → server KHÔNG lưu file ảnh (tránh tích rác, giữ stateless).
- Tách khỏi `questions.Question` (frozen, dùng nơi khác) để không tăng blast radius.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _clean_opt(value: str | None) -> str | None:
    """Chuẩn hoá field text optional: '' / 'null' / khoảng trắng → None."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "null":
        return None
    return s


@dataclass
class ExamQuestion:
    id: str
    sequence: int
    type: str
    prompt: str = ""
    reference_script: str | None = None
    provided_info: str | None = None
    expected_duration_sec: float | None = None
    # Ảnh đề bài (Describe Picture) dạng base64 — KHÔNG lưu ra đĩa; client giữ và
    # resubmit khi chấm. media_type vd "image/png".
    image_b64: str | None = None
    image_media_type: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sequence": self.sequence,
            "type": self.type,
            "prompt": self.prompt or "",
            "reference_script": self.reference_script,
            "provided_info": self.provided_info,
            "expected_duration_sec": self.expected_duration_sec,
            "image_b64": self.image_b64,
            "image_media_type": self.image_media_type,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExamQuestion":
        dur = d.get("expected_duration_sec")
        return cls(
            id=str(d.get("id") or ""),
            sequence=int(d.get("sequence") or 0),
            type=str(d.get("type") or ""),
            prompt=str(d.get("prompt") or ""),
            reference_script=_clean_opt(d.get("reference_script")),
            provided_info=_clean_opt(d.get("provided_info")),
            expected_duration_sec=float(dur) if dur not in (None, "") else None,
            image_b64=_clean_opt(d.get("image_b64")),
            image_media_type=_clean_opt(d.get("image_media_type")),
        )


@dataclass
class ExamPaper:
    exam: str
    title: str
    questions: list[ExamQuestion] = field(default_factory=list)

    def ordered(self) -> list[ExamQuestion]:
        """Câu hỏi đã sort theo sequence (thứ tự làm bài tường minh)."""
        return sorted(self.questions, key=lambda q: q.sequence)

    def to_dict(self) -> dict:
        return {
            "exam": self.exam,
            "title": self.title,
            "questions": [q.to_dict() for q in self.ordered()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExamPaper":
        return cls(
            exam=str(d.get("exam") or "toeic").strip().lower(),
            title=str(d.get("title") or ""),
            questions=[ExamQuestion.from_dict(q) for q in (d.get("questions") or [])],
        )
