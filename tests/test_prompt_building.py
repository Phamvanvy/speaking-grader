"""Test offline cho dựng prompt chấm (_build_system_prompt / _build_user_prompt).

Xác minh: (1) tài liệu cho sẵn (provided_info) chỉ vào payload với dạng câu
uses_provided_info; (2) system prompt chứa tiêu chí + guidance riêng từng dạng;
(3) ghi chú ảnh khác nhau giữa Describe Picture và Respond with info. Không cần
API key hay audio.
"""

from __future__ import annotations

import json

from src.asr import Transcription, Word
from src.features import Features
from src.gating import GatingResult
from src.rubrics.ielts import get_question_type as get_ielts_question_type
from src.rubrics.toeic import get_question_type
from src.scoring import (
    _build_system_prompt,
    _build_user_prompt,
    _local_response_schema,
)

_INFO = "9:00 AM Opening Keynote (Room A); 10:30 AM Session 1 (Room B, Mark Lee)"


def _transcription() -> Transcription:
    return Transcription(
        text="The first session starts at nine.",
        words=[Word("The", 0.0, 0.2, 0.9)],
        language="en",
        duration=10.0,
    )


def _features() -> Features:
    return Features(
        speech_rate_wpm=110.0,
        word_count=6,
        speaking_duration_sec=10.0,
        audio_duration_sec=10.0,
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


def _user_payload(qt_key: str, *, provided_info=None, has_image=False) -> dict:
    qt = get_question_type(qt_key)
    prompt = _build_user_prompt(
        qt,
        "What time does the first session start?",
        None,
        _transcription(),
        _features(),
        _gating(),
        has_image=has_image,
        provided_info=provided_info,
    )
    # Phần JSON nằm sau đoạn note; parse object đầu tiên từ dấu '{' đầu tiên
    # (raw_decode bỏ qua khối language reminder nằm SAU JSON).
    payload, _ = json.JSONDecoder().raw_decode(prompt, prompt.index("{"))
    return payload


def test_provided_info_included_for_respond_with_info():
    payload = _user_payload("respond_with_info", provided_info=_INFO)
    assert payload.get("provided_info") == _INFO


def test_provided_info_omitted_for_types_without_flag():
    # respond_questions không có uses_provided_info → bỏ qua dù truyền vào.
    payload = _user_payload("respond_questions", provided_info=_INFO)
    assert "provided_info" not in payload


def test_provided_info_omitted_when_not_supplied():
    payload = _user_payload("respond_with_info", provided_info=None)
    assert "provided_info" not in payload


def test_user_prompt_carries_output_language():
    # Ba lớp chống model nhỏ trả sai ngôn ngữ: output_language trong payload +
    # reminder ở CUỐI prompt (sau JSON, sát điểm model bắt đầu sinh).
    qt = get_question_type("respond_questions")
    prompt = _build_user_prompt(
        qt,
        "What time does the first session start?",
        None,
        _transcription(),
        _features(),
        _gating(),
        feedback_lang="vi",
    )
    payload, end = json.JSONDecoder().raw_decode(prompt, prompt.index("{"))
    # resolve_language_name("vi") = "Vietnamese (tiếng Việt)" — kèm tên bản địa.
    assert payload["output_language"].startswith("Vietnamese")
    tail = prompt[end:]
    assert "IMPORTANT LANGUAGE REQUIREMENT" in tail
    assert "Vietnamese" in tail
    # Reminder là text thuần sau JSON — không được chứa brace làm hỏng parse.
    assert "{" not in tail and "}" not in tail


def test_system_prompt_contains_criteria_and_guidance():
    for key in ("respond_questions", "respond_with_info", "express_opinion"):
        qt = get_question_type(key)
        sys_prompt = _build_system_prompt(qt, "vi")
        for c in qt.criteria:
            assert c.key in sys_prompt, (key, c.key)
        # Guidance đã được làm dày (không còn một dòng cụt) → đủ dài.
        assert len(qt.guidance) > 200, key
    # Organization chỉ là tiêu chí của Q11.
    assert "organization" in _build_system_prompt(
        get_question_type("express_opinion"), "vi"
    )


def test_local_schema_constrains_criteria_count_and_keys():
    # IELTS (4 tiêu chí) và TOEIC (số tiêu chí khác nhau theo dạng câu): schema
    # gửi backend local phải ép đúng số lượng + enum đúng tập key của qt.
    for qt in (
        get_ielts_question_type("part2_long_turn"),
        get_question_type("express_opinion"),
    ):
        schema = _local_response_schema(qt)
        n = len(qt.criteria)
        keys = [c.key for c in qt.criteria]
        crit = schema["properties"]["criteria"]
        assert crit["minItems"] == n
        assert crit["maxItems"] == n
        assert schema["$defs"]["CriterionScore"]["properties"]["criterion"]["enum"] == keys


def test_local_schema_does_not_mutate_shared_schema():
    # model_json_schema() trả dict mới mỗi lần → siết theo qt không rò rỉ ra
    # schema mặc định (vẫn là array không min/max).
    from src.schema import SpeakingResult

    _local_response_schema(get_ielts_question_type("part1_interview"))
    base = SpeakingResult.model_json_schema()
    assert "minItems" not in base["properties"]["criteria"]
    assert "enum" not in base["$defs"]["CriterionScore"]["properties"]["criterion"]


def test_system_prompt_demands_exact_criteria():
    qt = get_ielts_question_type("part3_discussion")
    sys_prompt = _build_system_prompt(qt, "vi")
    assert f"EXACTLY {len(qt.criteria)}" in sys_prompt
    for c in qt.criteria:
        assert c.key in sys_prompt


def test_image_note_differs_describe_vs_respond_with_info():
    describe = _build_user_prompt(
        get_question_type("describe_picture"),
        "Describe the picture.",
        None,
        _transcription(),
        _features(),
        _gating(),
        has_image=True,
    )
    respond = _build_user_prompt(
        get_question_type("respond_with_info"),
        "Which sessions is Mark Lee leading?",
        None,
        _transcription(),
        _features(),
        _gating(),
        has_image=True,
        provided_info=_INFO,
    )
    assert "describe" in describe.lower()
    assert "source document" in respond.lower()
    assert "source document" not in describe.lower()
