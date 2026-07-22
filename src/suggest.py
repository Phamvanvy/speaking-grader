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
from .schema import PracticeTask, RolePlayScript, SampleAnswer, StoryQuest, WordInfo
from .scoring.backends import generate

logger = logging.getLogger("toeic.suggest")


def default_target_band(exam: str) -> str:
    """Mức nhắm tới mặc định khi client không gửi target_band."""
    if exam == Exam.IELTS.value:
        return "9.0"
    if exam == Exam.TOPIK.value:
        return "TOPIK 말하기 mức cao nhất (~200)"
    return "TOEIC mức cao nhất (~200)"


def _build_suggest_system_prompt(
    qt: QuestionType, target_band: str, feedback_lang: str
) -> str:
    is_ielts = qt.exam == Exam.IELTS.value
    is_topik = qt.exam == Exam.TOPIK.value
    exam_label = (
        "IELTS" if is_ielts
        else "TOPIK Speaking (한국어 말하기)" if is_topik
        else "TOEIC"
    )
    examiner_role = (
        "an expert IELTS Speaking examiner and tutor"
        if is_ielts
        else "an expert TOPIK Speaking (Korean) examiner and tutor"
        if is_topik
        else "an expert TOEIC Speaking coach"
    )
    target_label = (
        f"IELTS band {target_band}" if is_ielts
        else f"a top-scoring TOPIK Speaking response ({target_band})" if is_topik
        else f"a top-scoring TOEIC response ({target_band})"
    )
    # Bài mẫu viết bằng NGÔN NGỮ NÓI của kỳ thi (topik → tiếng Hàn), không phải
    # feedback_lang (feedback_lang chỉ áp cho highlights/outline).
    answer_language = "KOREAN (한국어)" if is_topik else "ENGLISH"
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
        f"- Write the `answer` in natural SPOKEN {answer_language} (as a strong "
        "test-taker would actually speak it), not formal written prose. Use natural "
        "discourse markers, but stay coherent and well-organized.\n"
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
        f"- Write `answer` in {answer_language}. Write `highlights` and `outline` "
        f"in {language_name}.\n"
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

    # generate() dispatch theo config.backend (anthropic/local/openrouter, kèm
    # fallback) — meta chỉ để log nội bộ, bài mẫu không cần telemetry.
    result, _meta = generate(
        config,
        system_prompt,
        user_prompt,
        SampleAnswer,
        image_b64=image_b64,
        image_media_type=image_media_type,
    )

    assert isinstance(result, SampleAnswer)
    # Bảo đảm target_band luôn có giá trị có nghĩa kể cả khi model bỏ trống.
    if not (result.target_band or "").strip():
        result.target_band = target
    return result


def suggest_practice_task(config: Config, qt: QuestionType) -> PracticeTask:
    """Sinh MỘT đề luyện tập (task-context) cho dạng câu `qt` — để chấm thật lesson.

    Khác `suggest_answer` (sinh bài MẪU để tham khảo): ở đây sinh ĐỀ BÀI cho học
    viên tự trả lời rồi hệ thống chấm. Điền field theo dạng câu:
    - read_aloud → `reference` (đoạn văn để đọc to), `prompt` để trống.
    - respond_with_info → `prompt` (câu hỏi) + `provided_info` (tài liệu nguồn).
    - dạng mở khác → chỉ `prompt`.
    Caller (course/practice.py) chỉ dùng cho dạng câu KHÔNG cần ảnh.
    """
    is_topik = qt.exam == Exam.TOPIK.value
    is_ielts = qt.exam == Exam.IELTS.value
    exam_label = (
        "IELTS" if is_ielts
        else "TOPIK Speaking (한국어 말하기)" if is_topik
        else "TOEIC"
    )
    task_language = "KOREAN (한국어)" if is_topik else "ENGLISH"

    if qt.key == "read_aloud":
        field_instr = (
            "This is a READ-ALOUD task. Put a short passage (2-4 sentences, "
            f"natural {task_language}, suitable to read aloud in ~30-45s) in "
            "`reference`. Leave `prompt` and `provided_info` EMPTY."
        )
    elif qt.uses_provided_info or "provided_info" in qt.required_inputs:
        field_instr = (
            "This task gives the test-taker source material to answer FROM. Put a "
            f"compact piece of source material (a schedule / agenda / info table, "
            f"in {task_language}, as plain text lines) in `provided_info`, and ONE "
            "specific question that must be answered USING that material in "
            "`prompt`. Leave `reference` EMPTY."
        )
    else:
        field_instr = (
            "Put ONE realistic test question/instruction the test-taker must "
            f"answer by SPEAKING in `prompt` (natural {task_language}). Leave "
            "`reference` and `provided_info` EMPTY."
        )

    system_prompt = (
        f"You are an expert {exam_label} Speaking test writer. Produce ONE fresh, "
        f"realistic PRACTICE TASK for the task type ({qt.label}) that a learner "
        "will attempt by speaking, and that our system will then grade.\n\n"
        "TASK TYPE EXPECTATIONS (so the task fits the type):\n"
        f"{qt.guidance}\n\n"
        "OUTPUT FIELDS:\n"
        f"{field_instr}\n\n"
        "Keep it self-contained (no image needed), unambiguous, and answerable in "
        "one short spoken turn. Output structured JSON only."
    )
    user_prompt = (
        f"Create one {exam_label} '{qt.label}' practice task now, as structured JSON."
    )
    result, _meta = generate(config, system_prompt, user_prompt, PracticeTask)
    assert isinstance(result, PracticeTask)
    return result


def suggest_roleplay(config: Config, exam: str, setting: str) -> RolePlayScript:
    """Sinh MỘT kịch bản hội thoại nhập vai (Role-play Quest, Phase 3B).

    LLM CHỈ sinh nội dung — chấm nói vẫn qua gradePronunciation dùng chung. Cache
    user-agnostic theo (exam, topic) ở src/course/quests.py (mỗi chủ đề 1 call).
    Kịch bản viết bằng NGÔN NGỮ NÓI của kỳ thi (EN cho TOEIC/IELTS).
    """
    is_ielts = exam == Exam.IELTS.value
    exam_label = "IELTS" if is_ielts else "TOEIC"
    level_label = (
        "an upper-intermediate IELTS candidate (band ~6.5-7)"
        if is_ielts
        else "an intermediate TOEIC learner"
    )

    system_prompt = (
        f"You are an expert {exam_label} speaking coach writing a short, realistic "
        "ROLE-PLAY conversation for a learner to practice speaking. The learner will "
        "READ their lines aloud and we grade PRONUNCIATION only, so every learner "
        "line must be natural, self-contained, and easy to read aloud.\n\n"
        "REQUIREMENTS:\n"
        "- Set `scenario` (1-2 sentences), `role_user` (the learner's role) and "
        "`role_npc` (the other speaker's role).\n"
        "- Produce 4-6 `turns`. In each turn the NPC speaks first (`npc`), then give "
        "the learner's ideal reply in `expected_user` (ONE natural spoken sentence, "
        "≤20 words) and a SHORT `hint` telling the learner WHAT to say (not the exact "
        "words), e.g. 'Greet back and ask about the price.'\n"
        f"- Write `npc` and `expected_user` in natural spoken ENGLISH suited to "
        f"{level_label}. Write `hint` in Vietnamese.\n"
        "- Keep the conversation coherent and progressing; end it naturally.\n"
        "- Output structured JSON only."
    )
    user_prompt = (
        f"Create one {exam_label} role-play conversation for this situation:\n"
        f"{setting}\n\nNow produce it as structured JSON."
    )
    result, _meta = generate(config, system_prompt, user_prompt, RolePlayScript)
    assert isinstance(result, RolePlayScript)
    return result


def suggest_story(config: Config, exam: str, setting: str) -> StoryQuest:
    """Sinh MỘT truyện đọc-to tuyến tính (Story Quest, Phase 3C).

    LLM CHỈ sinh nội dung — chấm nói vẫn qua gradePronunciation dùng chung (đọc to
    từng đoạn). Cache user-agnostic theo (exam, topic) ở src/course/quests.py.
    Truyện viết bằng NGÔN NGỮ NÓI của kỳ thi (EN cho TOEIC/IELTS).
    """
    is_ielts = exam == Exam.IELTS.value
    exam_label = "IELTS" if is_ielts else "TOEIC"
    level_label = (
        "an upper-intermediate IELTS candidate (band ~6.5-7)"
        if is_ielts
        else "an intermediate TOEIC learner"
    )

    system_prompt = (
        f"You are an expert {exam_label} speaking coach writing a SHORT read-aloud "
        "story for a learner to practice pronunciation. The learner reads each "
        "segment aloud and we grade PRONUNCIATION only, so every segment must be "
        "natural and easy to read aloud.\n\n"
        "REQUIREMENTS:\n"
        "- Set a short `title`.\n"
        "- Produce 4-6 `segments`, each ONE natural sentence (4-25 words) that "
        "together tell a coherent, complete little story with a clear beginning and "
        "end.\n"
        f"- Write in natural spoken ENGLISH suited to {level_label}. No dialogue "
        "tags or quotation marks that are awkward to read aloud.\n"
        "- Output structured JSON only."
    )
    user_prompt = (
        f"Write one short {exam_label} read-aloud story on this theme:\n"
        f"{setting}\n\nNow produce it as structured JSON."
    )
    result, _meta = generate(config, system_prompt, user_prompt, StoryQuest)
    assert isinstance(result, StoryQuest)
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

    result, _meta = generate(config, system_prompt, user_prompt, WordInfo)
    assert isinstance(result, WordInfo)
    result.word = word
    return result
