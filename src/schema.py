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


class CriterionScore(BaseModel):
    criterion: str = Field(
        description="Tên tiêu chí, vd 'pronunciation', 'intonation_stress'"
    )
    score: float = Field(description="Điểm theo thang TOEIC cho tiêu chí (0-3)")
    justification: str = Field(description="Lý do chấm, dựa trên số liệu + transcript")
    suggestions: list[str] = Field(
        default_factory=list, description="Gợi ý cải thiện cụ thể"
    )


class SpeakingResult(BaseModel):
    question_type: str
    # task_completion là tiêu chí hạng nhất: trả lời đúng/đủ yêu cầu hay không.
    task_completion: CompletionLevel
    content_relevance: CompletionLevel
    criteria: list[CriterionScore]
    # TOEIC Speaking dùng thang 0-200, KHÔNG dùng Band như IELTS.
    # QUAN TRỌNG: trường này KHÔNG do LLM sinh nữa — nó được TÍNH TỰ ĐỘNG trong
    # code (scoring._compute_toeic_score) từ điểm tiêu chí 0-3 + task_completion
    # + content_relevance, để cùng một bộ điểm tiêu chí luôn ra cùng một số (loại
    # bỏ dao động do model tự "bốc" số). Có default=0 nên không bắt buộc trong
    # schema gửi cho model; giá trị model trả (nếu có) sẽ bị ghi đè.
    estimated_toeic_score: int = Field(
        default=0,
        ge=0,
        le=200,
        description=(
            "Điểm TOEIC Speaking (0-200) — TÍNH TỰ ĐỘNG từ điểm tiêu chí, model "
            "KHÔNG cần điền."
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
