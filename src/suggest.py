"""Sinh bài nói MẪU chất lượng cao cho các dạng câu mở (IELTS, tả tranh...).

Khác `src/scoring/` (chấm điểm): ở đây ta YÊU CẦU LLM tạo một bài nói mẫu để
người học tham khảo/luyện theo. Tái dùng nguyên hạ tầng gọi LLM của scoring
(`_generate_anthropic` / `_generate_local`) — chỉ khác phần soạn prompt + schema
output (`SampleAnswer`).

Điểm vào: `suggest_answer(config, qt, ...)`. Endpoint /suggest gọi hàm này.
"""

from __future__ import annotations

import logging

from .config import Config, resolve_language_name
from .rubrics.base import Exam, QuestionType
from .schema import SampleAnswer, WordInfo
from .scoring.backends import _generate_anthropic, _generate_local

logger = logging.getLogger("toeic.suggest")


def default_target_band(exam: str) -> str:
    """Mức nhắm tới mặc định khi client không gửi target_band."""
    return "9.0" if exam == Exam.IELTS.value else "TOEIC mức cao nhất (~200)"


def _build_suggest_system_prompt(
    qt: QuestionType, target_band: str, feedback_lang: str
) -> str:
    is_ielts = qt.exam == Exam.IELTS.value
    exam_label = "IELTS" if is_ielts else "TOEIC"
    examiner_role = (
        "an expert IELTS Speaking examiner and tutor"
        if is_ielts
        else "an expert TOEIC Speaking coach"
    )
    target_label = (
        f"IELTS band {target_band}" if is_ielts else f"a top-scoring TOEIC response ({target_band})"
    )
    language_name = resolve_language_name(feedback_lang)

    return (
        f"You are {examiner_role}. Your job is to produce ONE model spoken answer "
        f"for the following {exam_label} Speaking task ({qt.label}) that would "
        f"achieve {target_label}.\n\n"
        "TASK TYPE GUIDANCE (write an answer that fully satisfies these "
        "expectations):\n"
        f"{qt.guidance}\n\n"
        "SCORING SCALE (calibrate the quality to the target level):\n"
        f"{qt.scale_description}\n\n"
        "REQUIREMENTS for the model answer:\n"
        "- Write the `answer` in natural SPOKEN English (as a strong test-taker "
        "would actually speak it), not formal written prose. Use natural discourse "
        "markers, but stay coherent and well-organized.\n"
        "- Match the expected length/duration of this task type (cover all cue-card "
        "points / describe the picture fully / develop ideas with reasons and "
        "examples as the task demands).\n"
        "- Showcase the lexical range, collocations, and grammatical structures "
        f"expected at {target_label}, while keeping it realistic and on-topic.\n"
        "- Provide `highlights`: 3-5 concrete, learnable features from your answer "
        "(strong collocations, complex structures, linking devices, idea-development "
        "techniques) the learner can borrow.\n"
        "- Provide a short `outline` (key points / opening-body-closing) for longer "
        "monologue tasks; it may be empty for short answers.\n"
        f"- Write `answer` in ENGLISH. Write `highlights` and `outline` in "
        f"{language_name}.\n"
        f"- Echo the target level in `target_band` (e.g. '{target_band}')."
    )


def _build_suggest_user_prompt(
    prompt_text: str,
    provided_info: str | None,
    expected_duration_sec: float | None,
    has_image: bool,
) -> str:
    parts: list[str] = []
    if prompt_text and prompt_text.strip():
        parts.append(f"TASK / QUESTION:\n{prompt_text.strip()}")
    if provided_info and provided_info.strip():
        parts.append(f"PROVIDED MATERIAL (cue card / context):\n{provided_info.strip()}")
    if has_image:
        parts.append(
            "An image is attached above — base your description on what is actually "
            "shown in it."
        )
    if expected_duration_sec:
        parts.append(
            f"Target speaking duration: about {int(expected_duration_sec)} seconds "
            "— size the answer accordingly."
        )
    if not parts:
        parts.append(
            "No explicit prompt was provided; produce a strong general model answer "
            "appropriate for this task type."
        )
    parts.append("Now produce the model answer as structured JSON.")
    return "\n\n".join(parts)


def suggest_answer(
    config: Config,
    qt: QuestionType,
    *,
    prompt_text: str = "",
    provided_info: str | None = None,
    image_b64: str | None = None,
    image_media_type: str | None = None,
    target_band: str = "",
    expected_duration_sec: float | None = None,
) -> SampleAnswer:
    """Sinh một SampleAnswer cho dạng câu `qt` qua backend LLM đã cấu hình."""
    target = target_band.strip() or default_target_band(qt.exam)
    system_prompt = _build_suggest_system_prompt(qt, target, config.feedback_lang)
    user_prompt = _build_suggest_user_prompt(
        prompt_text, provided_info, expected_duration_sec, has_image=bool(image_b64)
    )

    if config.is_local:
        result = _generate_local(
            config,
            system_prompt,
            user_prompt,
            SampleAnswer,
            SampleAnswer.model_json_schema(),
            "SampleAnswer",
            image_b64,
            image_media_type,
        )
    else:
        result = _generate_anthropic(
            config, system_prompt, user_prompt, SampleAnswer, image_b64, image_media_type
        )

    assert isinstance(result, SampleAnswer)
    # Bảo đảm target_band luôn có giá trị có nghĩa kể cả khi model bỏ trống.
    if not (result.target_band or "").strip():
        result.target_band = target
    return result


def word_info(config: Config, word: str, lang: str) -> WordInfo:
    """Sinh định nghĩa EN + ví dụ + nghĩa (lang) cho 1 từ — popup luyện phát âm.

    Caller (endpoint /word-info) cache kết quả theo (word, lang) trong
    src/words.py nên mỗi từ chỉ tốn 1 call LLM.
    """
    language_name = resolve_language_name(lang)
    system_prompt = (
        "You are an English learner's dictionary editor. For the given English "
        "word, produce structured JSON with:\n"
        "- `definition_en`: ONE short, learner-friendly English definition "
        "(Oxford Learner's style) for the word's MOST COMMON sense.\n"
        "- `example_en`: ONE natural English example sentence (≤20 words) using "
        "that sense.\n"
        f"- `meaning`: the word's meaning in {language_name}, dictionary-style "
        "and concise, matching the same sense.\n"
        "- `word`: echo the word in lowercase."
    )
    user_prompt = f"WORD: {word}\n\nNow produce the entry as structured JSON."

    if config.is_local:
        result = _generate_local(
            config, system_prompt, user_prompt, WordInfo,
            WordInfo.model_json_schema(), "WordInfo", None, None,
        )
    else:
        result = _generate_anthropic(
            config, system_prompt, user_prompt, WordInfo, None, None
        )
    assert isinstance(result, WordInfo)
    result.word = word
    return result
