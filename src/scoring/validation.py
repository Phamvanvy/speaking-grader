"""Validate/clean output LLM — bắt 'hợp lệ schema nhưng rác' mà Pydantic không chặn.

_validate_result gắn cờ output hỏng (thiếu tiêu chí / suggestions lẫn key / text cụt);
_drop_invalid_corrections loại correction mà `said` không có thật trong transcript.
"""

from __future__ import annotations

import logging

from ..rubrics.base import QuestionType
from ..schema import SpeakingResult

logger = logging.getLogger("toeic.scoring")

# Ký tự mở ngoặc ở cuối chuỗi → dấu hiệu text bị cắt giữa chừng (JSON degenerate).
_DANGLING_OPEN = ("(", "[", "{", "（", "［", "｛")


def _is_truncated(text: str) -> bool:
    """True nếu chuỗi rỗng hoặc kết thúc bằng dấu mở ngoặc (bị cắt giữa chừng)."""
    s = (text or "").strip()
    return not s or s.endswith(_DANGLING_OPEN)


def _norm_for_match(s: str) -> str:
    """Chuẩn hoá để so khớp substring khoan dung: lower + gộp khoảng trắng."""
    return " ".join(s.lower().split())


def _drop_invalid_corrections(result: SpeakingResult, transcript: str) -> None:
    """Bỏ các LexicalCorrection mà `said` không có trong transcript (mutate result).

    LLM vẫn có thể paraphrase `said` dù prompt cấm → mọi correction phải truy
    ngược được về điều thí sinh thực sự nói. So khớp khoan dung (case-insensitive,
    gộp khoảng trắng) để tránh loại nhầm vì khác hoa/thường hay spacing.
    """
    haystack = _norm_for_match(transcript)
    dropped = 0
    for c in result.criteria:
        if not c.corrections:
            continue
        kept = [
            corr for corr in c.corrections
            if corr.said and _norm_for_match(corr.said) in haystack
        ]
        dropped += len(c.corrections) - len(kept)
        c.corrections = kept
    if dropped:
        logger.info(
            "Đã loại %d correction có `said` không khớp transcript (LLM paraphrase).",
            dropped,
        )


def _validate_result(result: SpeakingResult, qt: QuestionType) -> list[str]:
    """Bắt output 'hợp lệ schema nhưng rác' mà Pydantic không chặn được.

    Trả về danh sách mô tả lỗi (rỗng nếu OK). Chỉ gắn cờ 3 dạng hỏng đã quan
    sát thực tế: thiếu tiêu chí bắt buộc, suggestions điền nhầm tên key tiêu chí,
    và text bị cắt/rỗng. KHÔNG bắt suggestions rỗng — model trả thiếu suggestions
    vẫn là output hợp lệ.
    """
    problems: list[str] = []
    required = {c.key for c in qt.criteria}

    present = {c.criterion for c in result.criteria}
    missing = required - present
    if missing:
        problems.append(f"thiếu tiêu chí bắt buộc: {sorted(missing)}")

    for c in result.criteria:
        polluted = [s for s in c.suggestions if s in required]
        if polluted:
            problems.append(
                f"suggestions của '{c.criterion}' chứa tên tiêu chí: {polluted}"
            )
        if _is_truncated(c.justification):
            problems.append(f"justification của '{c.criterion}' bị cắt/rỗng")

    if _is_truncated(result.score_rationale):
        problems.append("score_rationale bị cắt/rỗng")
    if _is_truncated(result.summary_feedback):
        problems.append("summary_feedback bị cắt/rỗng")

    return problems
