"""Test offline cho rubric + chấm điểm IELTS (và đối xứng với TOEIC).

Bao gồm: registry IELTS (4 tiêu chí / 3 Part), resolver chống cross-exam, làm
tròn band chuẩn IELTS, _compute_ielts_band (mean + cap + clamp), tính đối xứng
"backend tính điểm tổng, không tin số LLM bốc", và prompt building exam-aware.
Không cần API key hay audio.
"""

from __future__ import annotations

import pytest

from src.rubrics import (
    EXAM_REGISTRIES,
    list_question_types,
    resolve_question_type,
)
from src.rubrics.base import Exam
from src.rubrics.ielts import IELTS_QUESTION_TYPES
from src.schema import CompletionLevel, CriterionScore, SpeakingResult
from src.scoring import (
    _build_system_prompt,
    _build_user_prompt,
    _compute_ielts_band,
    _compute_toeic_score,
    _round_half,
)
from src.asr import Transcription, Word
from src.features import Features
from src.gating import GatingResult

_IELTS_CRITERIA = {
    "fluency_coherence",
    "lexical_resource",
    "grammatical_range",
    "pronunciation",
}


# --- Registry & resolver -----------------------------------------------------

def test_ielts_registry_has_three_parts():
    assert set(IELTS_QUESTION_TYPES) == {
        "part1_interview",
        "part2_long_turn",
        "part3_discussion",
    }


def test_ielts_parts_use_four_official_criteria():
    for qt in IELTS_QUESTION_TYPES.values():
        assert {c.key for c in qt.criteria} == _IELTS_CRITERIA, qt.key
        assert qt.exam == Exam.IELTS.value


def test_part2_uses_provided_info():
    assert IELTS_QUESTION_TYPES["part2_long_turn"].uses_provided_info is True


def test_resolver_dispatches_by_exam():
    assert resolve_question_type("part2_long_turn", exam="ielts").key == "part2_long_turn"
    assert resolve_question_type("read_aloud", exam="toeic").key == "read_aloud"


def test_resolver_blocks_cross_exam():
    # IELTS không có read_aloud; TOEIC không có part3_discussion.
    with pytest.raises(KeyError):
        resolve_question_type("read_aloud", exam="ielts")
    with pytest.raises(KeyError):
        resolve_question_type("part3_discussion", exam="toeic")


def test_resolver_rejects_unknown_exam():
    with pytest.raises(KeyError):
        resolve_question_type("part1_interview", exam="ielt")


def test_exam_registries_keyed_by_enum_values():
    # Registry key = ĐÚNG tập giá trị Exam enum (nguồn hằng duy nhất) — derive
    # từ enum thay vì pin cứng danh sách để thêm kỳ thi không phải sửa test này.
    assert set(EXAM_REGISTRIES) == {e.value for e in Exam}
    assert set(list_question_types("ielts")) == set(IELTS_QUESTION_TYPES)


# --- Làm tròn band chuẩn IELTS (round-half-UP, sạch nhiễu float) -------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        (6.124, 6.0),
        (6.25, 6.5),
        (6.74, 6.5),
        (6.75, 7.0),
        (6.0, 6.0),
        (8.9, 9.0),
        (0.24, 0.0),
        (0.25, 0.5),
    ],
)
def test_round_half_boundaries(raw, expected):
    assert _round_half(raw) == expected


# --- _compute_ielts_band -----------------------------------------------------

def _ielts_result(bands, *, task=CompletionLevel.high, content=CompletionLevel.high):
    return SpeakingResult(
        question_type="part2_long_turn",
        task_completion=task,
        content_relevance=content,
        criteria=[
            CriterionScore(criterion=k, score=s, justification="ok", suggestions=[])
            for k, s in zip(sorted(_IELTS_CRITERIA), bands)
        ],
        score_rationale="Lập luận đầy đủ.",
        summary_feedback="Nhận xét đầy đủ.",
    )


def test_compute_ielts_band_simple_mean():
    assert _compute_ielts_band(_ielts_result([6.0, 6.0, 6.0, 6.0])) == 6.0


def test_compute_ielts_band_rounds_quarter_up():
    # mean = 6.25 → 6.5
    assert _compute_ielts_band(_ielts_result([6.0, 6.0, 6.0, 7.0])) == 6.5
    # mean = 6.75 → 7.0
    assert _compute_ielts_band(_ielts_result([7.0, 7.0, 7.0, 6.0])) == 7.0


def test_compute_ielts_band_capped_by_low_completion():
    # Band tiêu chí cao nhưng task_completion very_low → bị cap về 3.0.
    band = _compute_ielts_band(
        _ielts_result([8.0, 8.0, 8.0, 8.0], task=CompletionLevel.very_low)
    )
    assert band == 3.0


def test_compute_ielts_band_clamped():
    assert _compute_ielts_band(_ielts_result([9.0, 9.0, 9.0, 9.0])) == 9.0
    assert _compute_ielts_band(_ielts_result([0.0, 0.0, 0.0, 0.0])) == 0.0


# --- Đối xứng: backend tính điểm tổng, KHÔNG tin số LLM bốc ------------------

def test_backend_overwrites_llm_ielts_band():
    # LLM "bốc" band 9.0 nhưng tiêu chí thực tế chỉ 6.0 → tính lại 6.0.
    result = _ielts_result([6.0, 6.0, 6.0, 6.0])
    result.estimated_ielts_band = 9.0  # giả lập LLM trả bừa
    assert _compute_ielts_band(result) == 6.0


def test_backend_overwrites_llm_toeic_score():
    # LLM "bốc" 200 nhưng tiêu chí 1/3 (→60đ mỗi tiêu chí) → tính lại 60.
    result = SpeakingResult(
        question_type="read_aloud",
        task_completion=CompletionLevel.high,
        content_relevance=CompletionLevel.high,
        criteria=[
            CriterionScore(criterion="pronunciation", score=1.0, justification="ok"),
            CriterionScore(criterion="intonation_stress", score=1.0, justification="ok"),
        ],
        estimated_toeic_score=200,  # giả lập LLM trả bừa
        score_rationale="r",
        summary_feedback="s",
    )
    assert _compute_toeic_score(result) == 60


# --- Prompt building exam-aware ----------------------------------------------

def _transcription() -> Transcription:
    return Transcription(
        text="I went to Japan last summer.",
        words=[Word("I", 0.0, 0.2, 0.9)],
        language="en",
        duration=12.0,
    )


def _features() -> Features:
    return Features(
        speech_rate_wpm=120.0,
        word_count=6,
        speaking_duration_sec=12.0,
        audio_duration_sec=12.0,
        silence_sec=0.0,
        pause_count=0,
        total_pause_sec=0.0,
        longest_pause_sec=0.0,
        filler_count=0,
        avg_word_probability=0.9,
        min_word_probability=0.8,
    )


def _gating() -> GatingResult:
    return GatingResult(is_empty=False, task_completion_floor=None, reasons=[])


def test_system_prompt_ielts_mentions_band_and_criteria():
    qt = resolve_question_type("part2_long_turn", exam="ielts")
    sys_prompt = _build_system_prompt(qt, "vi")
    assert "IELTS" in sys_prompt
    assert "0-9" in sys_prompt
    for c in qt.criteria:
        assert c.key in sys_prompt
    # Không nhắc thang TOEIC 0-200 trong prompt IELTS.
    assert "0-200" not in sys_prompt


def test_system_prompt_toeic_still_uses_0_3():
    qt = resolve_question_type("read_aloud", exam="toeic")
    sys_prompt = _build_system_prompt(qt, "vi")
    assert "TOEIC" in sys_prompt
    assert "0-3" in sys_prompt


def test_user_prompt_opening_is_exam_aware():
    ielts_qt = resolve_question_type("part1_interview", exam="ielts")
    prompt = _build_user_prompt(
        ielts_qt, "What do you do?", None, _transcription(), _features(), _gating()
    )
    assert "IELTS Speaking response" in prompt
