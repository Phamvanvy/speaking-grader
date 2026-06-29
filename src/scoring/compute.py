"""Quy đổi điểm tiêu chí → điểm tổng TẤT ĐỊNH (TOEIC 0-200 / IELTS band 0-9).

Tách khỏi LLM call: cùng một bộ điểm tiêu chí + mức hoàn thành luôn ra cùng một số
(loại dao động do model tự bốc số trong prose).
"""

from __future__ import annotations

import math

from ..schema import CompletionLevel, SpeakingResult

# --- Quy đổi điểm tiêu chí (0-3) → điểm TOEIC Speaking (0-200) -----------------
# Vì sao có khối này: trước đây estimated_toeic_score do LLM TỰ CHỌN trong prose
# ("rơi vào khoảng 80-90 → 85"), nên cùng một bộ điểm tiêu chí mỗi lần lại ra số
# khác (85 vs 75) — nhất là với model local nhỏ/nén. Giờ LLM chỉ chấm 0-3 mỗi
# tiêu chí + mức hoàn thành; con số 0-200 được TÍNH bằng công thức dưới đây nên
# CỐ ĐỊNH với cùng input. Hằng số để lộ ở module-level cho dễ tinh chỉnh.
#
# Mốc neo theo thang proficiency ETS: điểm tiêu chí 2/3 ("đạt, vài lỗi") ~ level
# 5 (~110đ), 3/3 ~ level 7-8 (~190đ). Nội suy tuyến tính cho điểm float.
_CRIT_ANCHORS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (1.0, 60.0),
    (2.0, 110.0),
    (3.0, 190.0),
)

# task_completion / content_relevance dưới mức 'high' nhân phạt vào điểm tổng
# (mắt xích yếu nhất quyết định). Đảm bảo bài làm dở/lạc đề không được điểm cao
# dù phát âm tốt — khớp yêu cầu gating trong system prompt.
_LEVEL_PENALTY: dict[CompletionLevel, float] = {
    CompletionLevel.very_low: 0.35,
    CompletionLevel.low: 0.60,
    CompletionLevel.medium: 0.85,
    CompletionLevel.high: 1.0,
}


def _interp_crit_points(score: float) -> float:
    """Nội suy điểm tiêu chí (0-3) → điểm thành phần (0-190) theo _CRIT_ANCHORS."""
    s = max(0.0, min(3.0, score))
    for (x0, y0), (x1, y1) in zip(_CRIT_ANCHORS, _CRIT_ANCHORS[1:]):
        if s <= x1:
            return y0 + (y1 - y0) * (s - x0) / (x1 - x0)
    return _CRIT_ANCHORS[-1][1]


def _compute_toeic_score(result: SpeakingResult) -> int:
    """Tính estimated_toeic_score (0-200) TẤT ĐỊNH từ điểm tiêu chí + mức hoàn thành.

    Cùng một bộ (điểm tiêu chí, task_completion, content_relevance) luôn cho cùng
    một số → loại bỏ dao động do LLM tự bốc số. Làm tròn về bội số của 10 (thang
    TOEIC Speaking báo theo bước 10).
    """
    if not result.criteria:
        return 0
    base = sum(_interp_crit_points(c.score) for c in result.criteria) / len(
        result.criteria
    )
    penalty = min(
        _LEVEL_PENALTY.get(result.task_completion, 1.0),
        _LEVEL_PENALTY.get(result.content_relevance, 1.0),
    )
    raw = base * penalty
    return max(0, min(200, int(round(raw / 10.0) * 10)))


# --- Quy đổi điểm tiêu chí (band 0-9) → overall band IELTS (0-9) ---------------
# IELTS Speaking: LLM chấm mỗi tiêu chí trên band 0-9; overall = TRUNG BÌNH 4 tiêu
# chí, làm tròn về 0.5 gần nhất (đúng cách giám khảo IELTS tổng hợp). Tính trong
# code (không để LLM bốc) nên cùng bộ band tiêu chí luôn ra cùng một overall.

# Trần overall band khi task_completion / content_relevance thấp — GUARDRAIL NỘI
# BỘ (không phải công thức IELTS official) chống "nói mượt nhưng lạc đề/quá ngắn".
# Đặt nới tay: chỉ thực sự cắn khi completion very_low/low; medium ~6.5 để bài
# tốt nhưng hơi ngắn không bị tụt quá đáng.
_IELTS_LEVEL_CAP: dict[CompletionLevel, float] = {
    CompletionLevel.very_low: 3.0,
    CompletionLevel.low: 4.5,
    CompletionLevel.medium: 6.5,
    CompletionLevel.high: 9.0,
}


def _round_half(x: float) -> float:
    """Làm tròn về bội 0.5 theo quy tắc IELTS (round-half-UP).

    KHÔNG dùng round() built-in (banker's rounding: round(6.25*2)/2 = 6.0 — sai).
    Làm sạch sai số nhị phân (round(x, 4)) TRƯỚC khi floor để tránh 6.75 lưu thành
    13.4999… → floor lệch về 6.5. Cận: 6.124→6.0, 6.25→6.5, 6.74→6.5, 6.75→7.0.
    """
    clean = round(x, 4)
    return math.floor(clean * 2 + 0.5) / 2


def _compute_ielts_band(result: SpeakingResult) -> float:
    """Tính estimated_ielts_band (0-9, bước 0.5) TẤT ĐỊNH từ band tiêu chí.

    overall = trung bình band 4 tiêu chí, áp trần theo completion (guardrail), rồi
    làm tròn 0.5 và kẹp [0, 9].
    """
    if not result.criteria:
        return 0.0
    mean = sum(c.score for c in result.criteria) / len(result.criteria)
    cap = min(
        _IELTS_LEVEL_CAP.get(result.task_completion, 9.0),
        _IELTS_LEVEL_CAP.get(result.content_relevance, 9.0),
    )
    capped = min(mean, cap)
    return max(0.0, min(9.0, _round_half(capped)))
