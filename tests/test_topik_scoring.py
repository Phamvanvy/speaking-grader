"""Test M3 TOPIK: registry 6 dạng câu + _compute_topik_score + overall weighted.

Mirror test_ielts_rubric.py: điểm tổng do CODE tính tất định (không tin số LLM),
trần theo mức câu, gộp cả đề có trọng số. Cut-lines chính thức: 1급 20-49,
2급 50-89, 3급 90-109, 4급 110-129, 5급 130-159, 6급 160-200.
"""

from __future__ import annotations

import pytest

from src.questions import _load_set
from src.rubrics import list_question_types, resolve_question_type
from src.rubrics.topik import (
    TOPIK_LEVEL_CAP,
    TOPIK_OVERALL_WEIGHT,
    TOPIK_QUESTION_TYPES,
)
from src.schema import CompletionLevel, CriterionScore, SpeakingResult
from src.scoring import _compute_topik_score, compute_exam_overall
from src.scoring.prompts import _build_system_prompt

_OFFICIAL_KEYS = [
    "q1_answer_question",
    "q2_role_play",
    "q3_picture_story",
    "q4_complete_dialogue",
    "q5_interpret_data",
    "q6_present_opinion",
]
_TOPIK_CRITERIA = ["content_task", "language_use", "delivery"]


# --- Registry ----------------------------------------------------------------

def test_topik_registry_has_six_official_types_plus_read_aloud():
    assert set(list_question_types("topik")) == set(_OFFICIAL_KEYS) | {"read_aloud"}


def test_topik_official_types_use_three_official_criteria():
    for key in _OFFICIAL_KEYS:
        qt = resolve_question_type(key, exam="topik")
        assert qt.exam == "topik"
        assert [c.key for c in qt.criteria] == _TOPIK_CRITERIA


def test_topik_input_shapes_match_official_format():
    # q2/q3/q5 dùng ảnh; q4/q5 dùng provided_info; q1/q6 text-only.
    assert "image" in resolve_question_type("q2_role_play", "topik").display_inputs
    assert "image" in resolve_question_type("q3_picture_story", "topik").display_inputs
    q4 = resolve_question_type("q4_complete_dialogue", "topik")
    assert q4.uses_provided_info and "provided_info" in q4.required_inputs
    q5 = resolve_question_type("q5_interpret_data", "topik")
    assert q5.uses_provided_info
    assert resolve_question_type("q6_present_opinion", "topik").required_inputs == (
        "prompt",
    )


def test_cap_and_weight_tables_cover_every_registered_type():
    # Trần điểm phải có cho MỌI dạng (kể cả read_aloud); weight chỉ cần cho 6
    # dạng official (dạng lạ → default 1 trong compute_exam_overall).
    assert set(TOPIK_LEVEL_CAP) == set(TOPIK_QUESTION_TYPES)
    assert set(TOPIK_OVERALL_WEIGHT) == set(_OFFICIAL_KEYS)
    # Mức câu tăng dần: sơ cấp < trung cấp < cao cấp.
    assert TOPIK_LEVEL_CAP["q1_answer_question"] < TOPIK_LEVEL_CAP["q3_picture_story"]
    assert TOPIK_LEVEL_CAP["q3_picture_story"] < TOPIK_LEVEL_CAP["q6_present_opinion"]
    assert (
        TOPIK_OVERALL_WEIGHT["q1_answer_question"]
        < TOPIK_OVERALL_WEIGHT["q4_complete_dialogue"]
        < TOPIK_OVERALL_WEIGHT["q6_present_opinion"]
    )


# --- _compute_topik_score ------------------------------------------------------

def _topik_result(
    scores,
    *,
    question_type="q6_present_opinion",
    task=CompletionLevel.high,
    content=CompletionLevel.high,
):
    return SpeakingResult(
        question_type=question_type,
        task_completion=task,
        content_relevance=content,
        criteria=[
            CriterionScore(criterion=k, score=s, justification="ok", suggestions=[])
            for k, s in zip(_TOPIK_CRITERIA, scores)
        ],
        score_rationale="Lập luận đầy đủ.",
        summary_feedback="Nhận xét đầy đủ.",
    )


def _qt(key):
    return resolve_question_type(key, exam="topik")


def test_topik_anchor_points():
    # 3/5 đều → sàn 4급 (110); 4/5 đều → giữa 5급 (155); 5/5 đều → 200 (câu cao cấp).
    assert _compute_topik_score(_topik_result([3, 3, 3]), _qt("q6_present_opinion")) == 110
    assert _compute_topik_score(_topik_result([4, 4, 4]), _qt("q6_present_opinion")) == 155
    assert _compute_topik_score(_topik_result([5, 5, 5]), _qt("q6_present_opinion")) == 200
    assert _compute_topik_score(_topik_result([0, 0, 0]), _qt("q6_present_opinion")) == 0


def test_topik_beginner_question_capped():
    # Câu sơ cấp làm hoàn hảo KHÔNG phải bằng chứng 6급 → trần 130 (vừa chạm 5급).
    perfect = _topik_result([5, 5, 5], question_type="q1_answer_question")
    assert _compute_topik_score(perfect, _qt("q1_answer_question")) == 130
    assert _compute_topik_score(perfect, _qt("q2_role_play")) == 130
    # Câu trung cấp trần 170; read_aloud (practice, delivery thuần) không trần.
    assert _compute_topik_score(perfect, _qt("q4_complete_dialogue")) == 170
    assert _compute_topik_score(perfect, _qt("read_aloud")) == 200


def test_topik_penalized_by_low_completion():
    # 5/5 nhưng lạc đề (content very_low) → 200 × 0.35 = 70, dưới trần nên giữ 70.
    result = _topik_result([5, 5, 5], content=CompletionLevel.very_low)
    assert _compute_topik_score(result, _qt("q6_present_opinion")) == 70


def test_topik_deterministic_and_overwrites_llm():
    # LLM "bốc" 200 nhưng tiêu chí 3/5 → tính lại 110; hai lần chạy ra cùng số.
    result = _topik_result([3, 3, 3])
    result.estimated_topik_score = 200  # giả lập LLM trả bừa
    qt = _qt("q6_present_opinion")
    assert _compute_topik_score(result, qt) == _compute_topik_score(result, qt) == 110


def test_topik_empty_criteria_zero():
    result = _topik_result([3, 3, 3])
    result.criteria = []
    assert _compute_topik_score(result, _qt("q6_present_opinion")) == 0


# --- compute_exam_overall (topik: trung bình có trọng số theo mức câu) --------

def test_topik_overall_weighted_mean():
    # q1 (w=1) 100đ, q4 (w=2) 120đ, q6 (w=3) 150đ
    # → (100·1 + 120·2 + 150·3) / 6 = 790/6 ≈ 131.67 → 132.
    out = compute_exam_overall(
        "topik",
        [
            {"estimated_topik_score": 100, "question_type": "q1_answer_question"},
            {"estimated_topik_score": 120, "question_type": "q4_complete_dialogue"},
            {"estimated_topik_score": 150, "question_type": "q6_present_opinion"},
        ],
    )
    assert out == 132


def test_topik_overall_unknown_type_defaults_weight_one():
    # Dạng câu lạ / thiếu question_type → weight 1 (không crash, không loại câu).
    out = compute_exam_overall(
        "topik",
        [
            {"estimated_topik_score": 100, "question_type": "mystery"},
            {"estimated_topik_score": 200},
        ],
    )
    assert out == 150


def test_topik_overall_skips_unscored_and_handles_empty():
    out = compute_exam_overall(
        "topik",
        [
            None,
            {},
            {"estimated_topik_score": None},
            {"estimated_topik_score": 90, "question_type": "q1_answer_question"},
        ],
    )
    assert out == 90
    assert compute_exam_overall("topik", [None, {}]) is None


def test_toeic_ielts_overall_unchanged_by_topik_branch():
    # Nhánh topik không được đổi hành vi EN: TOEIC mean bội 10, IELTS mean 0.5.
    assert compute_exam_overall(
        "toeic", [{"estimated_toeic_score": 110}, {"estimated_toeic_score": 120}]
    ) == 120
    assert compute_exam_overall(
        "ielts", [{"estimated_ielts_band": 6.0}, {"estimated_ielts_band": 6.5}]
    ) == 6.5


# --- Prompt + ngân hàng đề ------------------------------------------------------

def test_topik_system_prompt_scale_and_no_self_scoring():
    prompt = _build_system_prompt(_qt("q6_present_opinion"), "vi")
    assert "KOREAN" in prompt
    assert "estimated_topik_score" in prompt
    assert "0-5" in prompt
    # Không rơi nhầm sang nhánh TOEIC/IELTS.
    assert "estimated_toeic_score" not in prompt
    assert "estimated_ielts_band" not in prompt


@pytest.mark.parametrize("set_id", ["set1"])
def test_topik_question_bank_loads_and_types_resolve(set_id):
    title, questions = _load_set("topik", set_id)
    assert title
    assert questions
    for q in questions.values():
        qt = resolve_question_type(q.type, exam="topik")
        # Câu text-only v1: không được đòi input ảnh khi bộ đề chưa kèm ảnh.
        if "image" in qt.required_inputs and "provided_info" not in qt.required_inputs:
            assert q.image_path, f"{q.id}: dạng {q.type} cần image_path"
        if qt.uses_provided_info and "provided_info" in qt.required_inputs:
            assert q.provided_info, f"{q.id}: dạng {q.type} cần provided_info"
