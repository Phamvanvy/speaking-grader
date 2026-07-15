"""Kiểu dữ liệu rubric dùng chung cho mọi kỳ thi (exam-agnostic).

Tách khỏi `toeic.py` để các rubric khác (ielts.py, và sau này TOEFL/VSTEP) cùng
import từ một nguồn trung lập, thay vì phụ thuộc ngược vào module của một kỳ thi.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Exam(str, Enum):
    """Mã kỳ thi — nguồn hằng + validate duy nhất (diệt typo chuỗi).

    Kế thừa `str` nên giá trị enum vẫn là chuỗi: domain model giữ `exam: str`
    (xem QuestionType.exam), chỉ dùng `Exam.X.value` làm hằng số/khoá registry.
    """

    TOEIC = "toeic"
    IELTS = "ielts"
    TOPIK = "topik"


# Ngôn ngữ NÓI của kỳ thi — quyết định ASR language + G2P + bộ rule phát âm.
# Khác với feedback_lang (ngôn ngữ NHẬN XÉT, config.py): một bài TOPIK có thể
# nói tiếng Hàn nhưng nhận xét bằng tiếng Việt.
EXAM_LANGUAGE: dict[str, str] = {
    Exam.TOEIC.value: "en",
    Exam.IELTS.value: "en",
    Exam.TOPIK.value: "ko",
}


def exam_language(exam: str) -> str:
    """Ngôn ngữ nói của kỳ thi; exam lạ → "en" (an toàn, khớp hành vi cũ)."""
    return EXAM_LANGUAGE.get(exam, "en")


# (field điểm tổng trong scores dict, max điểm tổng) theo kỳ thi — nguồn duy nhất,
# thay cho các hardcode `9 if exam == "ielts" else 200` rải ở api/core/scoring.
EXAM_SCORE: dict[str, tuple[str, int]] = {
    Exam.TOEIC.value: ("estimated_toeic_score", 200),
    Exam.IELTS.value: ("estimated_ielts_band", 9),
    # TOPIK 말하기: 0-200, level 1-6. Field vào schema + compute ở M3; trước đó
    # scores.get(field) trả None → overall None (an toàn, không nhận nhầm điểm TOEIC).
    Exam.TOPIK.value: ("estimated_topik_score", 200),
}


def exam_score_field(exam: str) -> str:
    """Field điểm tổng của kỳ thi; exam lạ → field TOEIC (khớp hành vi cũ)."""
    return EXAM_SCORE.get(exam, EXAM_SCORE[Exam.TOEIC.value])[0]


def exam_score_max(exam: str) -> int:
    """Max điểm tổng của kỳ thi; exam lạ → max TOEIC (khớp hành vi cũ)."""
    return EXAM_SCORE.get(exam, EXAM_SCORE[Exam.TOEIC.value])[1]


@dataclass(frozen=True)
class Criterion:
    key: str
    label: str
    description: str


@dataclass(frozen=True)
class QuestionType:
    key: str
    label: str
    criteria: list[Criterion]
    scale_description: str
    # Hướng dẫn riêng cho dạng câu này (đưa vào system prompt)
    guidance: str = ""
    uses_reference_script: bool = False
    # Dạng câu có tài liệu cho sẵn (Q8-10 / IELTS Part 2 cue card): thí sinh phải
    # trả lời dựa trên provided_info (text) và/hoặc ảnh đính kèm.
    uses_provided_info: bool = False
    # Kỳ thi mà dạng câu này thuộc về (quyết định công thức tính điểm tổng +
    # văn phong system prompt). Giữ kiểu str (= Exam.X.value) cho đồng nhất với
    # các field chuỗi khác và để serialize JSON không cần xử lý enum.
    exam: str = Exam.TOEIC.value
    # Inputs LIÊN QUAN dạng câu này → điều khiển hiển thị ô nhập trên UI.
    # Giá trị hợp lệ: "prompt", "reference", "image". (Frontend chỉ dùng để
    # ẩn/hiện; KHÔNG phải nguồn quyết định chấm điểm.)
    display_inputs: tuple[str, ...] = ("prompt",)
    # CHỈ CẦN một trong các input này có mặt là coi như "có đề bài" → chấm đầy đủ.
    # Thiếu hết → chỉ chấm phát âm (xem QuestionType.has_task_context + core.py).
    # Đây là NGUỒN CHÂN LÝ DUY NHẤT cho quyết định pronunciation-only.
    # Giá trị hợp lệ: "prompt", "reference", "image", "provided_info".
    required_inputs: tuple[str, ...] = ("prompt",)

    def has_task_context(
        self,
        *,
        prompt: str | None = None,
        reference: str | None = None,
        image: bool = False,
        provided_info: str | None = None,
    ) -> bool:
        """True nếu có đủ "đề bài" để chấm nội dung (không chỉ phát âm).

        Đủ khi ít nhất MỘT required_input có mặt. Text được strip để " " không
        bị tính là có đề.
        """
        present: set[str] = set()
        if prompt and prompt.strip():
            present.add("prompt")
        if reference and reference.strip():
            present.add("reference")
        if image:
            present.add("image")
        if provided_info and provided_info.strip():
            present.add("provided_info")
        return bool(set(self.required_inputs) & present)
