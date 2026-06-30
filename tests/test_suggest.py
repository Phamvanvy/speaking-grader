"""Test offline cho tính năng gợi ý bài nói mẫu (/suggest).

Kiểm tra: dựng system/user prompt (nhúng guidance + band + tên dạng câu), mức band
mặc định theo kỳ thi, và schema SampleAnswer parse được. Không gọi LLM thật.
"""

from __future__ import annotations

from src.rubrics import resolve_question_type
from src.rubrics.base import Exam
from src.schema import SampleAnswer
from src.suggest import (
    _build_suggest_system_prompt,
    _build_suggest_user_prompt,
    default_target_band,
)


def test_default_target_band_per_exam() -> None:
    assert default_target_band(Exam.IELTS.value) == "9.0"
    assert "TOEIC" in default_target_band(Exam.TOEIC.value)


def test_system_prompt_ielts_embeds_band_and_guidance() -> None:
    qt = resolve_question_type("part2_long_turn", exam=Exam.IELTS.value)
    prompt = _build_suggest_system_prompt(qt, "9.0", "vi")
    assert "IELTS" in prompt
    assert "band 9.0" in prompt
    assert qt.label in prompt
    # Guidance riêng của dạng câu phải được nhúng (để bài mẫu khớp kỳ vọng).
    assert "cue card" in prompt
    # answer tiếng Anh, highlights theo feedback_lang.
    assert "ENGLISH" in prompt
    assert "Vietnamese" in prompt


def test_system_prompt_toeic_uses_top_score_role() -> None:
    qt = resolve_question_type("describe_picture", exam=Exam.TOEIC.value)
    prompt = _build_suggest_system_prompt(qt, "TOEIC mức cao nhất (~200)", "en")
    assert "TOEIC Speaking coach" in prompt
    assert qt.label in prompt


def test_user_prompt_includes_inputs() -> None:
    up = _build_suggest_user_prompt(
        "Describe a memorable trip.",
        "You should say: where you went...",
        120,
        has_image=False,
    )
    assert "Describe a memorable trip." in up
    assert "where you went" in up
    assert "120 seconds" in up


def test_user_prompt_mentions_image_when_present() -> None:
    up = _build_suggest_user_prompt("", None, None, has_image=True)
    assert "image is attached" in up


def test_sample_answer_schema_roundtrip() -> None:
    raw = (
        '{"answer": "Well, I\'d like to talk about...", "target_band": "9.0", '
        '"highlights": ["dùng collocation strong feelings"], "outline": ["mở bài"]}'
    )
    sa = SampleAnswer.model_validate_json(raw)
    assert sa.target_band == "9.0"
    assert sa.highlights and sa.outline
    # highlights/outline mặc định rỗng nếu thiếu.
    minimal = SampleAnswer.model_validate_json('{"answer": "x", "target_band": "9.0"}')
    assert minimal.highlights == [] and minimal.outline == []
