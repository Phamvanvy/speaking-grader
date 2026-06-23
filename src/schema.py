"""Pydantic models cho kết quả chấm điểm (structured output của Claude).

Claude trả JSON đúng schema này qua client.messages.parse(...).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CompletionLevel(str, Enum):
    very_low = "very_low"
    low = "low"
    medium = "medium"
    high = "high"


class LexicalCorrection(BaseModel):
    """Một lỗi dùng từ cụ thể + cách sửa, cho tiêu chí lexical_resource/vocabulary.

    `said` PHẢI là chuỗi con xuất hiện đúng trong transcript (được validate lại
    sau khi model trả lời — xem _drop_invalid_corrections trong scoring.py).
    """
    said: str = Field(
        description="Cụm từ thí sinh đã nói (trích NGUYÊN VĂN từ transcript)"
    )
    suggested: str = Field(description="Từ/cụm từ đúng nên dùng thay thế")
    reason: str | None = Field(
        default=None, description="Lý do ngắn gọn vì sao nên sửa"
    )
    example: str = Field(
        description="Một câu ví dụ tự nhiên dùng từ/cụm từ được đề xuất"
    )


class CriterionScore(BaseModel):
    criterion: str = Field(
        description="Tên tiêu chí, vd 'pronunciation', 'intonation_stress'"
    )
    score: float = Field(
        description="Điểm tiêu chí: thang 0-3 cho TOEIC, band 0-9 cho IELTS"
    )
    justification: str = Field(description="Lý do chấm, dựa trên số liệu + transcript")
    suggestions: list[str] = Field(
        default_factory=list,
        description=(
            "BẮT BUỘC cho MỌI tiêu chí: 2-4 gợi ý cải thiện cụ thể, hành động "
            "được (mỗi phần tử là một câu). KHÔNG để trống — luôn nêu được ít "
            "nhất vài điều thí sinh có thể luyện để lên điểm. Mỗi gợi ý phải bám "
            "vào điểm yếu/bằng chứng đã nêu trong justification, không nói chung chung."
        ),
    )
    corrections: list[LexicalCorrection] = Field(
        default_factory=list,
        description=(
            "Sửa lỗi dùng từ cụ thể (said → suggested + example). Chỉ điền cho "
            "tiêu chí lexical_resource (IELTS) / vocabulary (TOEIC); để rỗng cho "
            "các tiêu chí khác."
        ),
    )


class SpeakingResult(BaseModel):
    question_type: str
    # task_completion là tiêu chí hạng nhất: trả lời đúng/đủ yêu cầu hay không.
    task_completion: CompletionLevel
    content_relevance: CompletionLevel
    criteria: list[CriterionScore]
    # Điểm tổng theo từng thang đo — KHÔNG do LLM sinh, được TÍNH TỰ ĐỘNG trong
    # scoring.py từ điểm tiêu chí + task_completion + content_relevance, để cùng
    # một bộ điểm luôn ra cùng một số (loại bỏ dao động do model tự "bốc" số).
    # Chỉ MỘT field được set tuỳ kỳ thi (qt.exam); field còn lại để None. Cả hai
    # optional (default None) nên không bắt buộc trong schema gửi cho model; giá
    # trị model trả (nếu có) sẽ bị ghi đè.
    #
    # TOEIC Speaking: thang 0-200 (báo theo bước 10).
    estimated_toeic_score: int | None = Field(
        default=None,
        ge=0,
        le=200,
        description=(
            "Điểm TOEIC Speaking (0-200) — TÍNH TỰ ĐỘNG từ điểm tiêu chí, model "
            "KHÔNG cần điền. None nếu kỳ thi không phải TOEIC."
        ),
    )
    # IELTS Speaking: band 0-9 (bước 0.5) — trung bình 4 tiêu chí làm tròn 0.5.
    estimated_ielts_band: float | None = Field(
        default=None,
        ge=0,
        le=9,
        description=(
            "Band IELTS Speaking (0-9, bước 0.5) — TÍNH TỰ ĐỘNG từ band từng "
            "tiêu chí, model KHÔNG cần điền. None nếu kỳ thi không phải IELTS."
        ),
    )
    # Giải thích logic chấm: tiêu chí nào mạnh/yếu và mức độ hoàn thành tổng thể.
    # KHÔNG nêu một con số 0-200 cụ thể (số tổng do code tính). Phải viết bằng
    # ngôn ngữ nhận xét được cấu hình.
    score_rationale: str = Field(
        description=(
            "Lập luận từng bước: tiêu chí nào kéo chất lượng lên/xuống, mức "
            "task_completion / content_relevance, và vì sao bài ở mức này. "
            "KHÔNG cần nêu con số tổng 0-200 — số đó được tính tự động."
        )
    )
    summary_feedback: str
