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
    estimated_toeic_score: int = Field(
        ge=0, le=200, description="Điểm TOEIC Speaking ước tính (0-200)"
    )
    # Giải thích logic vì sao ra đúng số điểm tổng: nối điểm từng tiêu chí +
    # số liệu khách quan + cờ gating -> khoảng điểm 0-200. Phải viết bằng
    # ngôn ngữ nhận xét được cấu hình.
    score_rationale: str = Field(
        description=(
            "Lập luận từng bước dẫn tới estimated_toeic_score: tiêu chí nào kéo "
            "điểm lên/xuống và vì sao rơi vào khoảng điểm này chứ không phải cao "
            "hay thấp hơn."
        )
    )
    summary_feedback: str
