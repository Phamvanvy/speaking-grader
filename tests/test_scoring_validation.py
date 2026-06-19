"""Test offline cho lưới validation kết quả chấm (_validate_result).

Bắt các output 'hợp lệ schema nhưng rác' mà Pydantic không chặn được — tái hiện
glitch JSON thực tế từng quan sát (thiếu tiêu chí, suggestions lẫn tên key, text
bị cắt). Không cần API key hay audio.
"""

from __future__ import annotations

from src.rubrics.toeic import get_question_type
from src.schema import CompletionLevel, CriterionScore, SpeakingResult
from src.scoring import _is_truncated, _validate_result

_QT = get_question_type("read_aloud")  # cần [pronunciation, intonation_stress]


def _result(criteria, *, rationale="Lập luận đầy đủ và hoàn chỉnh.",
            summary="Nhận xét tổng kết hoàn chỉnh."):
    return SpeakingResult(
        question_type="read_aloud",
        task_completion=CompletionLevel.high,
        content_relevance=CompletionLevel.high,
        criteria=criteria,
        estimated_toeic_score=120,
        score_rationale=rationale,
        summary_feedback=summary,
    )


def _crit(key, *, justification="Lý do chấm đầy đủ.", suggestions=None):
    return CriterionScore(
        criterion=key, score=2.0, justification=justification,
        suggestions=suggestions or [],
    )


def test_is_truncated():
    assert _is_truncated("")
    assert _is_truncated("   ")
    assert _is_truncated("đọc số dưới dạng chữ (")
    assert _is_truncated("danh sách [")
    assert not _is_truncated("hoàn chỉnh.")
    assert not _is_truncated("có (score 2) ở giữa câu.")


def test_clean_result_passes():
    result = _result([_crit("pronunciation"), _crit("intonation_stress")])
    assert _validate_result(result, _QT) == []


def test_empty_suggestions_allowed():
    # Model trả thiếu suggestions (mặc định []) vẫn hợp lệ — KHÔNG được gắn cờ.
    result = _result([
        _crit("pronunciation", suggestions=[]),
        _crit("intonation_stress", suggestions=[]),
    ])
    assert _validate_result(result, _QT) == []


def test_missing_required_criterion_flagged():
    result = _result([_crit("pronunciation")])  # thiếu intonation_stress
    problems = _validate_result(result, _QT)
    assert any("intonation_stress" in p and "thiếu" in p for p in problems)


def test_suggestions_polluted_with_criterion_keys_flagged():
    result = _result([
        _crit("pronunciation", suggestions=["pronunciation", "intonation_stress"]),
        _crit("intonation_stress"),
    ])
    problems = _validate_result(result, _QT)
    assert any("chứa tên tiêu chí" in p for p in problems)


def test_truncated_text_flagged():
    result = _result(
        [
            _crit("pronunciation", justification="WER thấp nhưng đọc số ("),
            _crit("intonation_stress"),
        ],
        summary="Thí sinh đọc đầy đủ chữ số dưới dạng từ (",
    )
    problems = _validate_result(result, _QT)
    assert any("justification" in p and "bị cắt" in p for p in problems)
    assert any("summary_feedback bị cắt" in p for p in problems)


def test_real_broken_sample_caught():
    # Tái hiện đúng outputs/sample__q1_read_aloud.json (glitch thực tế):
    # thiếu intonation_stress + suggestions=tên key + justification/summary cụt.
    broken = _result(
        [_crit(
            "pronunciation",
            justification="Dựa trên WER thấp ... thay vì chữ viết (",
            suggestions=["pronunciation", "intonation_stress"],
        )],
        summary="Thí sinh đã hoàn thành tốt ... đọc đầy đủ chữ số dưới dạng từ (",
    )
    problems = _validate_result(broken, _QT)
    assert len(problems) >= 3  # cả 3 dạng hỏng đều bị bắt
