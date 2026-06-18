"""Chấm điểm bằng LLM với structured output.

Gửi đề bài + (script) + transcript + số liệu khách quan + cờ gating cho model,
nhận về SpeakingResult đúng schema (không phải tự parse JSON).

Hai backend (xem Config.backend):
- "anthropic": Claude qua Anthropic SDK (messages.parse + adaptive thinking).
- "local": model local (vd Qwen3 qua llama.cpp server) qua API
  OpenAI-compatible, ép schema bằng response_format json_schema.
"""

from __future__ import annotations

import json
import logging
import time

from .asr import Transcription
from .config import Config, resolve_language_name
from .features import Features
from .gating import GatingResult
from .rubrics.toeic import QuestionType
from .schema import SpeakingResult

logger = logging.getLogger("toeic.scoring")


def _build_system_prompt(qt: QuestionType, feedback_lang: str) -> str:
    criteria_lines = "\n".join(
        f"- {c.key} ({c.label}): {c.description}" for c in qt.criteria
    )
    language_name = resolve_language_name(feedback_lang)
    return f"""You are an experienced TOEIC Speaking examiner. Score one spoken \
response for the task type: {qt.label}.

TASK GUIDANCE:
{qt.guidance}

CRITERIA TO SCORE (only these):
{criteria_lines}

SCORING SCALE:
{qt.scale_description}

EVIDENCE RULES (important):
- Use the OBJECTIVE METRICS provided (speech_rate_wpm, pause_count, \
longest_pause_sec, filler_count, and for read-aloud the accuracy_metrics: wer, \
substitutions, insertions, deletions) as PRIMARY evidence.
- Use analysis of the transcript text as SECONDARY evidence.
- Do NOT rely solely on ASR confidence (avg_word_probability / \
min_word_probability). It is affected by microphone quality, accent, and \
background noise — treat it ONLY as weak supporting evidence, never as the \
pronunciation score itself.
- For read-aloud, accuracy_metrics.word_issues lists places where the ASR \
transcript diverged from the script (substitution / insertion / deletion), e.g. \
expected "morning" but recognized "warning". These are NOT confirmed \
mispronunciations — the ASR may have misheard due to noise, accent, or its own \
limits. Use them ONLY as "words worth reviewing": you may say the ASR may have \
misheard a word and suggest the test-taker double-check it. NEVER state with \
certainty that the test-taker pronounced a specific word wrong based on a \
word_issue alone.

TASK COMPLETION:
- task_completion reflects whether the response actually fulfils the prompt \
(answered fully, long enough, on-topic). A grammatically perfect but far too \
short or off-topic answer must get a LOW task_completion.
- If a completion floor is provided by upstream rule-based checks, do not score \
task_completion higher than that floor.

Map the per-criterion scores to estimated_toeic_score on the 0-200 TOEIC \
Speaking scale (TOEIC does NOT use IELTS bands). Be consistent and calibrated.
Give concrete, actionable suggestions for each criterion.

EXPLAIN YOUR REASONING (important):
- Each criterion's `justification` must be a clear, logical chain: cite the \
specific objective metric or transcript evidence, say what it implies, then why \
that lands the criterion at this 0-3 score and not one higher or lower.
- `score_rationale` must explain step by step how the per-criterion scores \
combine into the final estimated_toeic_score: which criteria pulled the score \
up or down, how task_completion / content_relevance and any gating floor were \
applied, and why the result falls in this 0-200 band rather than higher/lower. \
Do not just restate the number — justify it.

OUTPUT LANGUAGE (important):
- Write ALL human-readable text — every `justification`, every entry in \
`suggestions`, `score_rationale`, and `summary_feedback` — in {language_name}.
- Keep machine fields unchanged and in English: the `criterion` field must stay \
the lowercase English key (e.g. "pronunciation", "intonation_stress"), and the \
enum values for task_completion / content_relevance (very_low/low/medium/high) \
stay as-is. Only the explanatory prose is translated."""


def _build_user_prompt(
    qt: QuestionType,
    prompt_text: str,
    reference_script: str | None,
    transcription: Transcription,
    features: Features,
    gating: GatingResult,
    has_image: bool = False,
) -> str:
    payload = {
        "task_prompt": prompt_text,
        "reference_script": reference_script if qt.uses_reference_script else None,
        "transcript": transcription.text,
        "objective_metrics": features.to_dict(),
        "rule_based_gating": {
            "task_completion_floor": gating.task_completion_floor,
            "reasons": gating.reasons,
            "reference_coverage": gating.reference_coverage,
            "fail_reference_match": gating.fail_reference_match,
        },
    }
    image_note = (
        "An IMAGE of the picture the test-taker was asked to describe is attached "
        "to this message. Judge whether the spoken transcript accurately and "
        "completely describes what is actually in the picture (objects, people, "
        "actions, setting). A description that does not match the picture must "
        "lower content_relevance / relevance.\n\n"
        if has_image
        else ""
    )
    return (
        "Score the following TOEIC Speaking response. All numeric metrics are "
        "pre-computed and objective.\n\n"
        + image_note
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def score(
    config: Config,
    qt: QuestionType,
    prompt_text: str,
    reference_script: str | None,
    transcription: Transcription,
    features: Features,
    gating: GatingResult,
    image_b64: str | None = None,
    image_media_type: str | None = None,
) -> SpeakingResult:
    """Gọi LLM (Claude hoặc model local) và trả về SpeakingResult.

    image_b64/image_media_type: ảnh đề bài (vd Describe Picture) gửi kèm dạng
    vision. Cả hai backend đều hỗ trợ; bỏ trống nếu không có ảnh.
    """
    system_prompt = _build_system_prompt(qt, config.feedback_lang)
    user_prompt = _build_user_prompt(
        qt,
        prompt_text,
        reference_script,
        transcription,
        features,
        gating,
        has_image=bool(image_b64),
    )

    if config.is_local:
        return _score_local(
            config, system_prompt, user_prompt, image_b64, image_media_type
        )
    return _score_anthropic(
        config, system_prompt, user_prompt, image_b64, image_media_type
    )


def _score_anthropic(
    config: Config,
    system_prompt: str,
    user_prompt: str,
    image_b64: str | None = None,
    image_media_type: str | None = None,
) -> SpeakingResult:
    if not config.has_api_key:
        raise RuntimeError(
            "Thiếu ANTHROPIC_API_KEY. Đặt trong .env, dùng TOEIC_BACKEND=local "
            "để chấm bằng model local, hoặc chạy với --no-ai để chỉ lấy "
            "transcript + features."
        )

    import anthropic

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    # Không có ảnh → giữ nguyên content dạng chuỗi (hành vi cũ). Có ảnh → khối
    # image (base64) đứng trước, rồi khối text để Claude nhìn tranh trước khi đọc.
    if image_b64:
        content: object = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_media_type or "image/jpeg",
                    "data": image_b64,
                },
            },
            {"type": "text", "text": user_prompt},
        ]
    else:
        content = user_prompt

    t0 = time.monotonic()
    response = client.messages.parse(
        model=config.model,
        max_tokens=config.max_tokens,
        thinking={"type": "adaptive"},
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
        output_format=SpeakingResult,
    )
    latency = time.monotonic() - t0

    usage = response.usage
    logger.info(
        "Claude chấm xong | model=%s | latency=%.2fs | "
        "input_tokens=%s output_tokens=%s",
        config.model,
        latency,
        getattr(usage, "input_tokens", "?"),
        getattr(usage, "output_tokens", "?"),
    )

    result = response.parsed_output
    if result is None:
        # stop_reason refusal / max_tokens → parsed_output có thể None
        hint = (
            f" JSON bị cắt vì chạm trần max_tokens={config.max_tokens} — "
            f"tăng TOEIC_MAX_TOKENS."
            if response.stop_reason == "max_tokens"
            else ""
        )
        raise RuntimeError(
            f"Claude không trả về kết quả đúng schema "
            f"(stop_reason={response.stop_reason}).{hint}"
        )
    return result


def _score_local(
    config: Config,
    system_prompt: str,
    user_prompt: str,
    image_b64: str | None = None,
    image_media_type: str | None = None,
) -> SpeakingResult:
    """Chấm bằng model local qua API OpenAI-compatible (vd llama.cpp server).

    Ép đúng schema bằng response_format json_schema — llama.cpp chuyển schema
    thành GBNF grammar nên JSON trả về luôn hợp lệ. Không có 'thinking' của
    Claude; nếu model hỗ trợ reasoning (Qwen3) có thể bật qua chat template.
    """
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover - phụ thuộc tuỳ chọn
        raise RuntimeError(
            "Backend local cần gói 'openai'. Cài: pip install openai"
        ) from e

    client = OpenAI(base_url=config.local_base_url, api_key=config.local_api_key)

    # Định dạng vision OpenAI-compatible: data URI base64. Cần model local có
    # thị giác (vd Qwen2.5-VL); model thuần text sẽ bỏ qua/lỗi khối ảnh.
    if image_b64:
        data_uri = f"data:{image_media_type or 'image/jpeg'};base64,{image_b64}"
        user_content: object = [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": user_prompt},
        ]
    else:
        user_content = user_prompt

    # Tắt reasoning cho model kiểu Qwen3 trừ khi bật rõ ràng. Truyền qua
    # chat_template_kwargs (llama.cpp với --jinja sẽ áp dụng vào chat template;
    # các server khác bỏ qua key lạ). Tắt thinking nhanh ~6.7× (xem Config).
    extra_body = {
        "chat_template_kwargs": {"enable_thinking": config.local_enable_thinking}
    }

    t0 = time.monotonic()
    response = client.chat.completions.create(
        model=config.local_model,
        max_tokens=config.max_tokens,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "SpeakingResult",
                "schema": SpeakingResult.model_json_schema(),
                "strict": True,
            },
        },
        extra_body=extra_body,
    )
    latency = time.monotonic() - t0

    usage = response.usage
    logger.info(
        "Model local chấm xong | model=%s | base_url=%s | latency=%.2fs | "
        "prompt_tokens=%s completion_tokens=%s",
        config.local_model,
        config.local_base_url,
        latency,
        getattr(usage, "prompt_tokens", "?"),
        getattr(usage, "completion_tokens", "?"),
    )

    finish = response.choices[0].finish_reason
    content = response.choices[0].message.content
    if finish == "length":
        raise RuntimeError(
            f"Model local bị cắt vì chạm trần max_tokens={config.max_tokens} "
            f"(finish_reason=length) → JSON dở dang. Tăng TOEIC_MAX_TOKENS, "
            f"hoặc giảm độ dài nhận xét."
        )
    if not content:
        raise RuntimeError(
            f"Model local không trả về nội dung (finish_reason={finish})."
        )
    try:
        return SpeakingResult.model_validate_json(content)
    except Exception as e:  # noqa: BLE001 - bọc lỗi parse cho rõ
        raise RuntimeError(
            f"Model local trả JSON không đúng schema SpeakingResult: {e}\n"
            f"Nội dung: {content[:500]}"
        ) from e
